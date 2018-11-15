from __future__ import print_function
import sys
from rirnet_database import RirnetDatabase
import torch

import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
from datetime import datetime, timedelta
from torch.autograd import Variable
import numpy as np
from importlib import import_module
from glob import glob
import signal


# -------------  Initialization  ------------- #
class Model:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        sys.path.append(model_dir)
        autoenc = import_module('autoencoder')
        discriminator = import_module('discriminator_ae')
        self.autoenc = autoenc.Net()
        self.D = discriminator.Net()
        self.autoenc_args = self.autoenc.args()
        self.D_args = self.D.args()
        use_cuda = not self.autoenc_args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.autoenc.to(self.device)
        self.D.to(self.device)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


        x = torch.linspace(-2, 2, 256)
        weight = (1-torch.exp(2*x))/(4*(1+torch.exp(2*x)))
        weight += 1 - weight[0]
        weight = weight.repeat(self.autoenc_args.batch_size, 2, 1)
        self.mse_weight = weight.cuda()

        list_epochs = glob('*_autoenc.pth')
        list_epochs = [ x for x in list_epochs if "opt_autoenc" not in x ]

        self.autoenc_optimizer = optim.Adam(self.autoenc.parameters(), lr=self.autoenc_args.lr, betas=(0.9, 0.99), eps=1e-5, weight_decay=0, amsgrad=False)
        self.D_optimizer = optim.SGD(self.D.parameters(), lr=self.D_args.lr, momentum=self.D_args.momentum, nesterov=True)

        if list_epochs == []:
            epoch = 0
        else:
            epoch = max([int(e.split('_')[0]) for e in list_epochs])
            self.autoenc_optimizer.load_state_dict(torch.load(os.path.join(model_dir, '{}_opt_autoenc.pth'.format(str(epoch)))))
            self.autoenc.load_state_dict(torch.load(os.path.join(model_dir, '{}_autoenc.pth'.format(str(epoch)))))
            #self.D_optimizer.load_state_dict(torch.load(os.path.join(model_dir, '{}_Do.pth'.format(str(epoch)))))
            #self.D.load_state_dict(torch.load(os.path.join(model_dir, '{}_D.pth'.format(str(epoch)))))
            for g in self.autoenc_optimizer.param_groups:
                g['lr'] = self.autoenc_args.lr
                g['momentum'] = self.autoenc_args.momentum
            #for g in self.D_optimizer.param_groups:
            #    g['lr'] = self.D_args.lr
            #    g['momentum'] = self.D_args.momentum

        self.epoch = epoch
        self.csv_path = os.path.join(self.autoenc_args.db_path, 'db.csv')
        data_transform = self.autoenc.data_transform()
        target_transform = self.autoenc.target_transform()

        train_db = RirnetDatabase(is_training = True, args = self.autoenc_args, data_transform = data_transform, target_transform = target_transform)
        eval_db = RirnetDatabase(is_training = False, args = self.autoenc_args, data_transform = data_transform, target_transform = target_transform)
        self.train_loader = torch.utils.data.DataLoader(train_db, batch_size=self.autoenc_args.batch_size, shuffle=True, **self.kwargs)
        self.eval_loader = torch.utils.data.DataLoader(eval_db, batch_size=self.autoenc_args.batch_size, shuffle=True, **self.kwargs)

        self.autoenc_mean_train_loss = 0
        self.autoenc_mean_eval_loss = 0


        try:
            getattr(F, self.autoenc_args.loss_function)
        except AttributeError:
            print('AttributeError! {} is not a valid loss function. The string must exactly match a pytorch loss '
                  'function'.format(self.autoenc_args.loss_function))
            sys.exit()


    def train(self):

        for g in self.autoenc_optimizer.param_groups:
            g['lr'] = g['lr']*0.99

        #if self.epoch%120 == 0:
        #    for g in self.autoenc_optimizer.param_groups:
        #        g['lr'] = self.autoenc_args.lr
        #        plt.title(g['lr'])

        self.autoenc.train()
        self.D.train()
        autoenc_loss_list = []
        for batch_idx, (source, target) in enumerate(self.train_loader):
            source, target = source.to(self.device), target.to(self.device)
            self.D_optimizer.zero_grad()

            with torch.no_grad():
                output = self.autoenc(target, encode=True, decode=True)
            
            # Train discriminator
            target_verdict = self.D(target)
            D_loss_1 = getattr(F, self.D_args.loss_function)(target_verdict, torch.ones(self.autoenc_args.batch_size, 1).float().cuda())
            D_loss_1.backward()

            output_verdict = self.D(output)
            D_loss_2 = getattr(F, self.D_args.loss_function)(output_verdict, torch.zeros(self.autoenc_args.batch_size, 1).float().cuda())
            D_loss_2.backward()

            self.D_optimizer.step()

            # Train autoencoder
            self.autoenc_optimizer.zero_grad()
            output = self.autoenc(target, encode=True, decode=True)
            autoenc_verdict = self.D(output)
            autoenc_loss_1 = getattr(F, self.D_args.loss_function)(autoenc_verdict, torch.ones(self.autoenc_args.batch_size, 1).float().cuda())
            autoenc_loss_1.backward(retain_graph=True)
            #autoenc_loss_2 = getattr(F, self.autoenc_args.loss_function)(output, target)
            autoenc_loss_2 = self.mse_weighted(output, target, self.mse_weight)
            autoenc_loss_2.backward(retain_graph=True)
            autoenc_loss_3 = self.hausdorff(output, target)
            autoenc_loss_3.backward()
            self.autoenc_optimizer.step()

            autoenc_loss_list.append(autoenc_loss_3.item())

            if batch_idx % self.autoenc_args.log_interval == 0:
                print('Train Epoch: {:5d} [{:5d}/{:5d} ({:4.1f}%)]\tLoss: {:.4f}\t{:.4f}\t{:.4f}'.format(
                    self.epoch + 1, batch_idx * len(source), len(self.train_loader.dataset), 
                    100. * batch_idx / len(self.train_loader), autoenc_loss_3.item(), autoenc_loss_1.item(), autoenc_loss_2.item()))

        self.autoenc_mean_train_loss = np.mean(autoenc_loss_list)

        self.target_im_train = target.cpu().detach().numpy()[0]
        self.output_im_train = output.cpu().detach().numpy()[0]

    def evaluate(self):
        self.autoenc.eval()
        eval_loss_list = []
        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.eval_loader):
                source, target = source.to(self.device), target.to(self.device)
                output = self.autoenc(target, encode=True, decode=True)
                eval_loss = self.hausdorff(output, target)
                #eval_loss += getattr(F, self.autoenc_args.loss_function)(output, target).item()
                eval_loss_list.append(eval_loss)

        self.target_im_eval = target.cpu().detach().numpy()[0]
        self.output_im_eval = output.cpu().detach().numpy()[0]

        self.mean_eval_loss = np.mean(eval_loss_list)
        print(self.mean_eval_loss)

    def mse_weighted(self, output, target, weight):
        return torch.sum(weight * (output - target)**2)/output.numel()

    def hausdorff(self, output, target):
        res = 0
        for i, sample in enumerate(output):
            x = output[i].t()
            y = target[i].t()

            x_norm = (x**2).sum(1).view(-1,1)
            y_norm = (y**2).sum(1).view(1,-1)

            dist = x_norm+y_norm-2*torch.mm(x, torch.transpose(y,0,1))

            mean_1 = torch.mean(torch.min(dist, 0)[0])
            mean_2 = torch.mean(torch.min(dist, 1)[0])
            res += mean_1 + mean_2
        return res / 10

    def save_model(self):
        print(' '+'-'*64, '\nSaving\n', '-'*64)
        model_full_path = os.path.join(self.model_dir, '{}_autoenc.pth'.format(str(self.epoch)))
        optimizer_full_path = os.path.join(self.model_dir, '{}_opt_autoenc.pth'.format(str(self.epoch)))
        torch.save(self.autoenc.state_dict(), model_full_path)
        torch.save(self.autoenc_optimizer.state_dict(), optimizer_full_path)

        model_full_path = os.path.join(self.model_dir, '{}_D.pth'.format(str(self.epoch)))
        optimizer_full_path = os.path.join(self.model_dir, '{}_Do.pth'.format(str(self.epoch)))
        torch.save(self.D.state_dict(), model_full_path)
        torch.save(self.D_optimizer.state_dict(), optimizer_full_path)

    def loss_to_file(self):
        with open('loss_over_epochs_autoenc.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([self.epoch, self.autoenc_mean_train_loss, self.mean_eval_loss, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    def generate_plot(self):
        frmt = "%Y-%m-%d %H:%M:%S"
        plot_data = pd.read_csv('loss_over_epochs_autoenc.csv', header=None)
        epochs_raw, train_losses_raw, eval_losses_raw, times_raw = plot_data.values.T

        epochs = [int(epoch) for epoch in list(epochs_raw) if is_number(epoch)]
        train_losses = [float(loss) for loss in list(train_losses_raw) if is_number(loss)]
        eval_losses = [float(loss) for loss in list(eval_losses_raw) if is_number(loss)]

        if self.autoenc_args.save_timestamps:
            total_time = timedelta(0, 0, 0)
            if np.size(times_raw) > 1:
                start_times = times_raw[epochs_raw == 'started']
                stop_times = times_raw[epochs_raw == 'stopped']
                for i_stop_time, stop_time in enumerate(stop_times):
                    total_time += datetime.strptime(stop_time, frmt) - datetime.strptime(start_times[i_stop_time], frmt)
                total_time += datetime.now() - datetime.strptime(start_times[-1], frmt)
                plt.title('Trained for {} hours and {:2d} minutes'.format(int(total_time.days/24 + total_time.seconds//3600), (total_time.seconds//60)%60))

        plt.figure(figsize=(16,9), dpi=110)
        
        plt.subplot(2,1,1)
        plt.xlabel('Epochs')
        plt.ylabel('Loss ({})'.format(self.autoenc_args.loss_function))
        plt.semilogy(epochs, train_losses, label='Train Loss')
        plt.semilogy(epochs, eval_losses, label='Eval Loss')
        plt.legend()
        plt.grid(True, 'both')
        plt.title('Loss')

        plt.subplot(2,2,3)
        plt.plot(self.target_im_train[0,:], self.target_im_train[1,:], 'o', linewidth=0.5, markersize=2, label='target')
        plt.plot(self.output_im_train[0,:], self.output_im_train[1,:], 'x', linewidth=0.5, markersize=2, label='output')
        plt.grid(True)
        plt.legend()
        plt.title('Train output')
        #plt.axis([-0.25, 3.25, -0.25, 3.25])

        plt.subplot(2,2,4)
        plt.plot(self.target_im_eval[0,:], self.target_im_eval[1,:], 'o', linewidth=0.5, markersize=2, label='target')
        plt.plot(self.output_im_eval[0,:], self.output_im_eval[1,:], 'x', linewidth=0.5, markersize=2, label='output')
        plt.grid(True)
        plt.legend()
        plt.title('Eval output')
        #plt.axis([-0.25, 3.25, -0.25, 3.25])

        plt.tight_layout()
        plt.savefig('autoenc.png')
        plt.close()


    def stop_session(self):
        with open('loss_over_epochs_autoenc.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['stopped', 'stopped', 'stopped', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


    def start_session(self):
        with open('loss_over_epochs_autoenc.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['started', 'started', 'started', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])



def main(model_dir):
    global interrupted
    interrupted = False
    signal.signal(signal.SIGINT, signal_handler)

    model = Model(model_dir)
    model.start_session()

    for epoch in range(model.epoch, model.autoenc_args.epochs + 1):
        model.train()
        model.evaluate()
        model.epoch = epoch+1
        model.loss_to_file()
        model.generate_plot()

        if interrupted:
            print(' '+'-'*64, '\nEarly stopping\n', '-'*64)
            model.stop_session()
            model.save_model()
            break


def signal_handler(signal, frame):
    print(' '+'-'*64, '\nTraining will stop after this epoch\n', '-'*64)
    global interrupted
    interrupted = True


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    main(sys.argv[1])
