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
        generator = import_module('generator')
        discriminator = import_module('discriminator')
        refiner = import_module('refiner')
        self.G = generator.Net()
        self.D = discriminator.Net()
        self.R = refiner.Net()
        self.G_args = self.G.args()
        self.D_args = self.D.args()
        self.R_args = self.R.args()

        use_cuda = not self.G_args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.G.to(self.device)
        self.D.to(self.device)
        self.R.to(self.device)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        list_epochs = glob('*.pth')
        list_epochs = [ x for x in list_epochs if "Go" not in x ]
        self.G_optimizer = optim.SGD(self.G.parameters(), lr=self.G_args.lr, momentum=self.G_args.momentum, nesterov=True)
        self.D_optimizer = optim.SGD(self.D.parameters(), lr=self.D_args.lr, momentum=self.D_args.momentum, nesterov=True)
        self.R_optimizer = optim.SGD(self.R.parameters(), lr=self.R_args.lr, momentum=self.R_args.momentum, nesterov=True)
        if list_epochs == []:
            epoch = 0
        else:
            epoch = max([int(e.split('_')[0]) for e in list_epochs])
            self.G_optimizer.load_state_dict(torch.load(os.path.join(model_dir, '{}_Go.pth'.format(str(epoch)))))
            self.G.load_state_dict(torch.load(os.path.join(model_dir, '{}_G.pth'.format(str(epoch)))))
            self.D_optimizer.load_state_dict(torch.load(os.path.join(model_dir, '{}_Do.pth'.format(str(epoch)))))
            self.D.load_state_dict(torch.load(os.path.join(model_dir, '{}_D.pth'.format(str(epoch)))))
            self.R_optimizer = optim.SGD(self.R.parameters(), lr=self.R_args.lr, momentum=self.R_args.momentum, nesterov=True)
            #self.R_optimizer.load_state_dict(torch.load(os.path.join(model_dir, '{}_Ro.pth'.format(str(epoch)))))
            #self.R.load_state_dict(torch.load(os.path.join(model_dir, '{}_R.pth'.format(str(epoch)))))
            self.G.cuda()
            self.D.cuda()
            self.R.cuda()
            for g in self.G_optimizer.param_groups:
                g['lr'] = self.G_args.lr
                g['momentum'] = self.G_args.momentum
            for g in self.D_optimizer.param_groups:
                g['lr'] = self.D_args.lr
                g['momentum'] = self.D_args.momentum
            for g in self.R_optimizer.param_groups:
                g['lr'] = self.R_args.lr
                g['momentum'] = self.R_args.momentum


        self.epoch = epoch
        self.csv_path = os.path.join(self.G_args.db_path, 'db.csv')
        data_transform = self.G.data_transform()
        target_transform = self.G.target_transform()

        train_db = RirnetDatabase(is_training = True, args = self.G_args, data_transform = data_transform, target_transform = target_transform)
        eval_db = RirnetDatabase(is_training = False, args = self.G_args, data_transform = data_transform, target_transform = target_transform)
        self.train_loader = torch.utils.data.DataLoader(train_db, batch_size=self.G_args.batch_size, shuffle=True, **self.kwargs)
        self.eval_loader = torch.utils.data.DataLoader(eval_db, batch_size=self.G_args.batch_size, shuffle=True, **self.kwargs)

        self.G_mean_train_loss = 0
        self.G_mean_eval_loss = 0

        try:
            getattr(F, self.G_args.loss_function)
        except AttributeError:
            print('AttributeError! {} is not a valid loss function. The string must exactly match a pytorch loss '
                  'function'.format(self.G_args.loss_function))
            sys.exit()


    def train(self):
        self.G.train()
        self.D.train()
        self.R.train()
        G_loss_list = []
        D_loss_list = []
        R_loss_list = []
        for batch_idx, (source, target) in enumerate(self.train_loader):
            torch.cuda.empty_cache()
            self.D_optimizer.zero_grad()

            source, target = source.to(self.device), target.to(self.device)
            real_decision = self.D(target)
            train_loss = getattr(F, self.D_args.loss_function)(real_decision.squeeze(), Variable(torch.ones(source.data.size()[0]).cuda()))
            train_loss.backward()

            fake = self.G(source).detach()
            fake_decision = self.D(fake)
            train_loss = getattr(F, self.D_args.loss_function)(fake_decision.squeeze(), Variable(torch.zeros(source.data.size()[0]).cuda()))
            train_loss.backward()
            self.D_optimizer.step()
            #D_loss_list.append(train_loss.item())

            self.G_optimizer.zero_grad()
            fake = self.G(source)
            #train_loss_G = getattr(F, self.G_args.loss_function)(fake, target)
            train_loss_G = self.hausdorff(fake, target)
            train_loss_G.backward(retain_graph=True)
            decision = self.D(fake)
            train_loss_D = getattr(F, self.G_args.loss_function)(decision.squeeze(), Variable(torch.ones(source.data.size()[0]).cuda()))
            train_loss_D.backward()
            self.G_optimizer.step()

            self.R_optimizer.zero_grad()
            refined = self.R(fake.detach())
            refined_d = self.D(refined)
            train_loss_R = self.hausdorff(refined, target)
            train_loss_RD = getattr(F, self.G_args.loss_function)(self.D(refined).squeeze(), torch.ones(source.data.size()[0]).cuda())
            train_lossRRD = train_loss_R + train_loss_RD
            train_lossRRD.backward()
            self.R_optimizer.step()

            G_loss_list.append(train_loss_G.item())
            D_loss_list.append(train_loss_D.item())
            R_loss_list.append(train_loss_R.item())

            if batch_idx % self.G_args.log_interval == 0:
                print('Train Epoch: {:5d} [{:5d}/{:5d} ({:4.1f}%)]\tLoss: {:.6f}, {:.6f}, {:.6f}'.format(
                    self.epoch + 1, batch_idx * len(source), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), train_loss_G.item(), train_loss_D.item(), train_loss_R.item()))
        self.G_mean_train_loss = np.mean(G_loss_list)
        self.D_mean_train_loss = np.mean(D_loss_list)
        self.R_mean_train_loss = np.mean(R_loss_list)

        target_im = target.cpu().detach().numpy()
        output_im = fake.cpu().detach().numpy()
        refined_im = refined.cpu().detach().numpy()


        plt.plot(output_im[0, 0, :], output_im[0, 1, :], '-x', linewidth=0.5, markersize=0.7)
        plt.plot(target_im[0, 0, :], target_im[0, 1, :], '-o', linewidth=0.5, markersize=0.7)
        plt.plot(refined_im[0, 0, :], refined_im[0, 1, :], '-o', linewidth=0.5, markersize=0.7)
        plt.title('train')
        plt.savefig('example_output_train.png')
        plt.close()


    def evaluate(self):
        self.G.eval()
        eval_loss_list = []
        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.eval_loader):
                source, target = source.to(self.device), target.to(self.device)
                output = self.R(self.G(source))
                eval_loss = self.hausdorff(output, target).item()
                eval_loss_list.append(eval_loss)

        target_im = target.cpu().detach().numpy()
        output_im = output.cpu().detach().numpy()

        plt.plot(output_im[0, 0, :], output_im[0, 1, :], '-x', linewidth=0.5, markersize=0.7)
        plt.plot(target_im[0, 0, :], target_im[0, 1, :], '-o', linewidth=0.5, markersize=0.7)
        plt.title('eval')
        plt.savefig('example_output.png')
        plt.close()
        self.mean_eval_loss = np.mean(eval_loss_list)
        print(self.mean_eval_loss)


    def save_model(self):
        print(' '+'-'*64, '\nSaving\n', '-'*64)
        model_full_path = os.path.join(self.model_dir, '{}_G.pth'.format(str(self.epoch)))
        optimizer_full_path = os.path.join(self.model_dir, '{}_Go.pth'.format(str(self.epoch)))
        torch.save(self.G.state_dict(), model_full_path)
        torch.save(self.G_optimizer.state_dict(), optimizer_full_path)

        model_full_path = os.path.join(self.model_dir, '{}_D.pth'.format(str(self.epoch)))
        optimizer_full_path = os.path.join(self.model_dir, '{}_Do.pth'.format(str(self.epoch)))
        torch.save(self.D.state_dict(), model_full_path)
        torch.save(self.D_optimizer.state_dict(), optimizer_full_path)


    def loss_to_file(self):
        with open('loss_over_epochs.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([self.epoch, self.G_mean_train_loss, self.mean_eval_loss, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


    def generate_plot(self):
        frmt = "%Y-%m-%d %H:%M:%S"
        plot_data = pd.read_csv('loss_over_epochs.csv', header=None)
        epochs_raw, train_losses_raw, eval_losses_raw, times_raw = plot_data.values.T

        epochs = [int(epoch) for epoch in list(epochs_raw) if is_number(epoch)]
        train_losses = [float(loss) for loss in list(train_losses_raw) if is_number(loss)]
        eval_losses = [float(loss) for loss in list(eval_losses_raw) if is_number(loss)]

        if self.G_args.save_timestamps:
            total_time = timedelta(0, 0, 0)
            if np.size(times_raw) > 1:
                start_times = times_raw[epochs_raw == 'started']
                stop_times = times_raw[epochs_raw == 'stopped']
                for i_stop_time, stop_time in enumerate(stop_times):
                    total_time += datetime.strptime(stop_time, frmt) - datetime.strptime(start_times[i_stop_time], frmt)
                total_time += datetime.now() - datetime.strptime(start_times[-1], frmt)
                plt.title('Trained for {} hours and {:2d} minutes'.format(int(total_time.days/24 + total_time.seconds//3600), (total_time.seconds//60)%60))
        plt.xlabel('Epochs')
        plt.ylabel('Loss ({})'.format(self.G_args.loss_function))
        plt.semilogy(epochs, train_losses, label='Training Loss')
        plt.semilogy(epochs, eval_losses, label='Evaluation Loss')
        plt.legend()
        plt.savefig('loss_over_epochs.png')
        plt.close()

    def stop_session(self):
        with open('loss_over_epochs.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['stopped', 'stopped', 'stopped', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


    def start_session(self):
        with open('loss_over_epochs.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['started', 'started', 'started', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    def hausdorff(self, output, target):
        res = 0
        for i, _ in enumerate(output):
            x = output[i]
            y = target[i]

            n = x.size(0)
            d = x.size(1)
            x = x.expand(n,n,d)
            y = y.expand(n,n,d)
            dist = torch.pow(x - y, 2).sum(2)

            mean_1 = torch.mean(torch.min(dist, 0)[0])
            mean_2 = torch.mean(torch.min(dist, 1)[0])
            res += mean_1 + mean_2
        return res/100000


def main(model_dir):
    # noinspection PyGlobalUndefined
    global interrupted
    interrupted = False
    signal.signal(signal.SIGINT, signal_handler)

    model = Model(model_dir)
    model.start_session()

    for epoch in range(model.epoch, model.G_args.epochs + 1):
        model.train()
        model.evaluate()
        model.epoch = epoch+1
        model.loss_to_file()
        model.generate_plot()
        if epoch % model.G_args.save_interval == 0:
            model.save_model()

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
