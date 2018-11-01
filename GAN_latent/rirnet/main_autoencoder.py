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
#from pyroomacoustics.build_rir import fast_rir_builder


# -------------  Initialization  ------------- #
class Model:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        sys.path.append(model_dir)
        autoenc = import_module('autoencoder')
        self.autoenc = autoenc.Net()
        self.autoenc_args = self.autoenc.args()

        use_cuda = not self.autoenc_args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.autoenc.to(self.device)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        list_epochs = glob('*.pth')
        list_epochs = [ x for x in list_epochs if "opt_autoenc" not in x ]
        #self.autoenc_optimizer = optim.SDG(self.autoenc.parameters(), lr=self.autoenc_args.lr, momentum=self.autoenc_args.momentum, nesterov=True)

        self.autoenc_optimizer = optim.Adam(self.autoenc.parameters(), lr=self.autoenc_args.lr, betas=(0.9, 0.99), eps=1e-5, weight_decay=0, amsgrad=False)

        if list_epochs == []:
            epoch = 0
        else:
            epoch = max([int(e.split('_')[0]) for e in list_epochs])
            self.autoenc_optimizer.load_state_dict(torch.load(os.path.join(model_dir, '{}_opt_autoenc.pth'.format(str(epoch)))))
            self.autoenc.load_state_dict(torch.load(os.path.join(model_dir, '{}_autoenc.pth'.format(str(epoch)))))
            self.autoenc.cuda()
            for g in self.autoenc_optimizer.param_groups:
                g['lr'] = self.autoenc_args.lr
                g['momentum'] = self.autoenc_args.momentum


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
            plt.title(g['lr'])

        if self.epoch%50 == 0:
            for g in self.autoenc_optimizer.param_groups:
                g['lr'] = self.autoenc_args.lr
                plt.title(g['lr'])

        self.autoenc.train()
        autoenc_loss_list = []
        autoenc_loss_h_list = []
        for batch_idx, (source, target) in enumerate(self.train_loader):
            torch.cuda.empty_cache()
            self.autoenc_optimizer.zero_grad()
            source, target = source.to(self.device), target.to(self.device)
            output = self.autoenc(target, encode=True, decode=True)

            x = torch.linspace(-2,2,924).unsqueeze(0)
            slope = ((-(torch.exp(2*x)-1)/(torch.exp(2*x)+1)*0.25))
            slope = slope +(1 - slope[0,0])
            weight = torch.cat(( torch.ones(1,100), slope), 1)
            weight = weight.unsqueeze(0).repeat(100,2,1).cuda()
            autoenc_loss = self.mse_weighted(output, target, weight)

            #autoenc_loss = getattr(F, self.autoenc_args.loss_function)(output[:,:,:], target[:,:,:])
            autoenc_loss_h = self.hausdorff(output[:,:,:], target[:,:,:])
            autoenc_loss_h.backward(retain_graph=True)
            autoenc_loss.backward()
            self.autoenc_optimizer.step()

            autoenc_loss_list.append(autoenc_loss.item())
            #autoenc_loss_h_list.append(autoenc_loss_h.item())

            if batch_idx % self.autoenc_args.log_interval == 0:
                print('Train Epoch: {:5d} [{:5d}/{:5d} ({:4.1f}%)]\tLoss: {:.6f}, {:.6f}'.format(
                    self.epoch + 1, batch_idx * len(source), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), autoenc_loss.item(), 0))

        self.autoenc_mean_train_loss = np.mean(autoenc_loss_list)

        target_im = target.cpu().detach().numpy()
        output_im = output.cpu().detach().numpy()
        plt.plot(output_im[0, 0, :].T, output_im[0, 1, :].T, '--x', linewidth=0.5, markersize=1.2, label='output')
        plt.plot(target_im[0, 0, :].T, target_im[0, 1, :].T, '--o', linewidth=0.5, markersize=0.7, label='target')
        plt.grid(True)
        plt.legend()
        plt.savefig('example_output_train.png')
        plt.close()

    def evaluate(self):
        self.autoenc.eval()
        eval_loss_list = []
        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.eval_loader):
                source, target = source.to(self.device), target.to(self.device)
                output = self.autoenc(target, encode=True, decode=True)
                eval_loss = getattr(F, self.autoenc_args.loss_function)(output[:,:,:], target[:,:,:]).item()
                eval_loss_list.append(eval_loss)

        target_im = target.cpu().detach().numpy()
        output_im = output.cpu().detach().numpy()

        plt.plot(output_im[0, 0, :].T, output_im[0, 1, :].T, 'x', linewidth=0.5, markersize=1.2, label='output')
        plt.plot(target_im[0, 0, :].T, target_im[0, 1, :].T, 'o', linewidth=0.5, markersize=0.7, label='target')
        plt.title('eval')
        plt.legend()
        plt.grid(True)
        plt.savefig('example_output_eval.png')
        plt.close()
        self.mean_eval_loss = np.mean(eval_loss_list)
        print(self.mean_eval_loss)


    def mse_weighted(self, output, target, weight):
        return torch.sum(weight * (output - target)**2)/output.numel()

    def hausdorff(self, output, target):


        res = 0
        for i, sample in enumerate(output):
            x = output[i].t()
            y = target[i].t()

            x_norm = (x**2).sum(1).view(-1, 1)
            y_norm = (y**2).sum(1).view(1, -1)
            dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))

            mean_1 = torch.mean(torch.min(dist, 0)[0])
            mean_2 = torch.mean(torch.min(dist, 1)[0])
            res += mean_1 + mean_2
        return res/50



    def save_model(self):
        print(' '+'-'*64, '\nSaving\n', '-'*64)
        model_full_path = os.path.join(self.model_dir, '{}_autoenc.pth'.format(str(self.epoch)))
        optimizer_full_path = os.path.join(self.model_dir, '{}_opt_autoenc.pth'.format(str(self.epoch)))
        torch.save(self.autoenc.state_dict(), model_full_path)
        torch.save(self.autoenc_optimizer.state_dict(), optimizer_full_path)

    def loss_to_file(self):
        with open('loss_over_epochs_ae.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([self.epoch, self.autoenc_mean_train_loss, self.autoenc_mean_train_loss, self.mean_eval_loss, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


    def generate_plot(self):
        frmt = "%Y-%m-%d %H:%M:%S"
        plot_data = pd.read_csv('loss_over_epochs_ae.csv', header=None)
        epochs_raw, train_l1_raw, train_l3_raw, eval_losses_raw, times_raw = plot_data.values.T

        epochs = [int(epoch) for epoch in list(epochs_raw) if is_number(epoch)]
        l1_train_losses = [float(loss) for loss in list(train_l1_raw) if is_number(loss)]
        l3_train_losses = [float(loss) for loss in list(train_l3_raw) if is_number(loss)]
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
        plt.xlabel('Epochs')
        plt.ylabel('Loss ({})'.format(self.autoenc_args.loss_function))
        plt.semilogy(epochs, l1_train_losses, label='Abstract Train Loss')
        plt.semilogy(epochs, l3_train_losses, label='Real Train Loss')
        plt.semilogy(epochs, eval_losses, label='Real Eval Loss')
        plt.legend()
        plt.grid(True, 'both')
        plt.savefig('loss_over_epochs.png')
        plt.close()

    def stop_session(self):
        with open('loss_over_epochs_ae.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['stopped', 'stopped', 'stopped', 'stopped', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


    def start_session(self):
        with open('loss_over_epochs_ae.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['started', 'started', 'started', 'started', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])



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
