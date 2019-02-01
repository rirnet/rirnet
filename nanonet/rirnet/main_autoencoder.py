from __future__ import print_function
from rirnet_database import RirnetDatabase
from datetime import datetime, timedelta
from torch.autograd import Variable
from importlib import import_module
from glob import glob

import sys
import torch
import os
import csv

import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import signal
import rirnet.misc as misc

class Model:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        sys.path.append(model_dir)
        self.autoencoder, self.epoch = misc.load_latest(model_dir, 'autoencoder')
        self.autoencoder_args = self.autoencoder.args()
        use_cuda = not self.autoencoder_args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.autoencoder.to(self.device)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        self.autoencoder_optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.autoencoder_args.lr, betas=(0.9, 0.99), eps=1e-5, weight_decay=0, amsgrad=False)

        self.best_loss = np.inf

        if self.epoch != 0:
            self.autoencoder_optimizer.load_state_dict(torch.load(os.path.join(model_dir, '{}_opt_autoencoder.pth'.format(self.epoch))))

            for g in self.autoencoder_optimizer.param_groups:
                g['lr'] = self.autoencoder_args.lr

        data_transform = self.autoencoder.data_transform()
        target_transform = self.autoencoder.target_transform()

        train_db = RirnetDatabase(is_training = True, args = self.autoencoder_args, data_transform = data_transform, target_transform = target_transform)
        eval_db = RirnetDatabase(is_training = False, args = self.autoencoder_args, data_transform = data_transform, target_transform = target_transform)
        self.train_loader = torch.utils.data.DataLoader(train_db, batch_size=self.autoencoder_args.batch_size, shuffle=True, **self.kwargs)
        self.eval_loader = torch.utils.data.DataLoader(eval_db, batch_size=self.autoencoder_args.batch_size, shuffle=True, **self.kwargs)

        self.autoencoder_mean_train_loss = 0
        self.autoencoder_mean_eval_loss = 0

    def train(self):

        autoencoder_loss_list = []
        self.autoencoder.train()
        for batch_idx, (source, target) in enumerate(self.train_loader):
            if self.epoch < 10:
                target += torch.rand(self.autoencoder_args.batch_size, 2, 1)
            source, target = source.to(self.device), target.to(self.device)

            self.autoencoder_optimizer.zero_grad()
            output = self.autoencoder(target, encode=True, decode=True)

            autoencoder_loss = self.chamfer_loss(output, target)
            autoencoder_loss.backward()
            self.autoencoder_optimizer.step()

            autoencoder_loss_list.append(autoencoder_loss.item())

            print('Train Epoch: {:5d} [{:5d}/{:5d} ({:4.1f}%)]\tLoss: {:.4f}'.format(
                self.epoch + 1, batch_idx * len(source), len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), autoencoder_loss.item()))

        self.autoencoder_mean_train_loss = np.mean(autoencoder_loss_list)

        self.target_im_train = target.cpu().detach().numpy()[0]
        self.output_im_train = output.cpu().detach().numpy()[0]

    def evaluate(self):
        self.autoencoder.eval()
        eval_loss_list = []
        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.eval_loader):
                source, target = source.to(self.device), target.to(self.device)
                output = self.autoencoder(target, encode=True, decode=True)
                eval_loss = self.chamfer_loss(output, target)
                eval_loss_list.append(eval_loss.item())

        self.mean_eval_loss = np.mean(eval_loss_list)

        if self.mean_eval_loss < self.best_loss:
            self.best_loss = self.mean_eval_loss
            #ding()

        self.target_im_eval = target.cpu().detach().numpy()[0]
        self.output_im_eval = output.cpu().detach().numpy()[0]

        print('Current eval loss: \t{}'.format(self.mean_eval_loss))
        print('Best eval loss: \t{}'.format(self.best_loss))

        f = open('ae_results', 'w')
        f.write('AE val loss\n')
        f.write('{}'.format(self.best_loss))

    def chamfer_loss(self, output, target):
        x,y = output.permute(0,2,1), target.permute(0,2,1)
        B, N, D = x.size()
        xx = torch.bmm(x, x.transpose(2,1))
        yy = torch.bmm(y, y.transpose(2,1))
        zz = torch.bmm(x, y.transpose(2,1))
        diag_ind = torch.arange(0, N).type(torch.cuda.LongTensor)
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = (rx.transpose(2, 1) + ry - 2*zz)
        l1 = torch.mean(P.min(1)[0])
        l2 = torch.mean(P.min(2)[0])
        return 10*(l1 + l2)

    def save_model(self):
        print(' '+'-'*64, '\nSaving\n', '-'*64)
        model_full_path = os.path.join(self.model_dir, '{}_autoencoder.pth'.format(str(self.epoch)))
        optimizer_full_path = os.path.join(self.model_dir, '{}_opt_autoencoder.pth'.format(str(self.epoch)))
        torch.save(self.autoencoder.state_dict(), model_full_path)
        torch.save(self.autoencoder_optimizer.state_dict(), optimizer_full_path)

    def loss_to_file(self):
        with open('loss_over_epochs_autoencoder.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([self.epoch, self.autoencoder_mean_train_loss, self.mean_eval_loss, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    def generate_plot(self):
        frmt = "%Y-%m-%d %H:%M:%S"
        plot_data = pd.read_csv('loss_over_epochs_autoencoder.csv', header=None)
        epochs_raw, train_losses_raw, eval_losses_raw, times_raw = plot_data.values.T

        epochs = [int(epoch) for epoch in list(epochs_raw) if is_number(epoch)]
        train_losses = [float(loss) for loss in list(train_losses_raw) if is_number(loss)]
        eval_losses = [float(loss) for loss in list(eval_losses_raw) if is_number(loss)]

        total_time = timedelta(0, 0, 0)
        if np.size(times_raw) > 1:
            start_times = times_raw[epochs_raw == 'started']
            stop_times = times_raw[epochs_raw == 'stopped']
            for i_stop_time, stop_time in enumerate(stop_times):
                total_time += datetime.strptime(stop_time, frmt) - datetime.strptime(start_times[i_stop_time], frmt)
            total_time += datetime.now() - datetime.strptime(start_times[-1], frmt)
            plt.title('Trained for {} hours and {:2d} minutes'.format(int(total_time.days/24 + total_time.seconds//3600), (total_time.seconds//60)%60))


        max_plot_length = 900

        fig = plt.figure()
        plt.semilogy(epochs[-max_plot_length:], train_losses[-max_plot_length:], label='Train Loss')
        plt.semilogy(epochs[-max_plot_length:], eval_losses[-max_plot_length:], label='Eval Loss')
        plt.legend()
        plt.grid(True, 'both')
        plt.title('Autoencoder Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig('fig/ae_loss.png')
        plt.savefig('fig/ae_loss.eps')

        fig = plt.figure()
        plt.plot(self.target_im_train[0,:], self.target_im_train[1,:], 'o-', linewidth=0.05, markersize=2, label='Target')
        plt.plot(self.output_im_train[0,:], self.output_im_train[1,:], 'x-', linewidth=0.05, markersize=2, label='Output')
        plt.grid(True)
        plt.legend()
        plt.title('Autoencoder Train Output')
        plt.xlabel('Timing')
        plt.ylabel('-log(Amplitude)')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig('fig/ae_train_output.png')
        plt.savefig('fig/ae_train_output.eps')

        fig = plt.figure()
        plt.plot(self.target_im_eval[0,:], self.target_im_eval[1,:], 'o-', linewidth=0.05, markersize=2, label='Target')
        plt.plot(self.output_im_eval[0,:], self.output_im_eval[1,:], 'x-', linewidth=0.05, markersize=2, label='Output')
        plt.grid(True)
        plt.legend()
        plt.title('Autoencoder Eval Output')
        plt.xlabel('Timing')
        plt.ylabel('-log(Amplitude)')
        plt.ylabel('Loss')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig('fig/ae_eval_output.png')
        plt.savefig('fig/ae_eval_output.eps')
        plt.close()

    def stop_session(self):
        with open('loss_over_epochs_autoencoder.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['stopped', 'stopped', 'stopped', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    def start_session(self):
        with open('loss_over_epochs_autoencoder.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['started', 'started', 'started', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


def main(model_dir):
    global interrupted
    interrupted = False
    signal.signal(signal.SIGINT, signal_handler)

    model = Model(model_dir)
    model.start_session()

    for epoch in range(model.epoch, model.autoencoder_args.epochs + 1):
        model.train()
        model.evaluate()
        model.epoch = epoch+1
        model.loss_to_file()
        model.generate_plot()

        if interrupted:
            break

    model.stop_session()
    model.save_model()
    print(' '+'-'*64, '\nStopping\n', '-'*64)

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

def ding():
    print('Ding!')
    os.system('cvlc ~/ding.wav --play-and-exit > /dev/null')


if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    misc.set_fig()
    main(sys.argv[1])
