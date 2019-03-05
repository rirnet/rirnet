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
        self.net, self.epoch = misc.load_latest(model_dir, 'net')
        self._args = self.net.args()
        use_cuda = not self._args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.net.to(self.device)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        self.net_optimizer = optim.Adam(self.net.parameters(), lr=self._args.lr, betas=(0.9, 0.99), eps=1e-5, weight_decay=0, amsgrad=False)

        if self.epoch != 0:
            self.net_optimizer.load_state_dict(torch.load(os.path.join(model_dir, '{}_opt.pth'.format(self.epoch))))

            for g in self.net_optimizer.param_groups:
                g['lr'] = self._args.lr
                g['momentum'] = self._args.momentum

        data_transform = self.net.data_transform()
        target_transform = self.net.target_transform()

        train_db = RirnetDatabase(is_training = True, args = self._args, data_transform = data_transform, target_transform = target_transform)
        eval_db = RirnetDatabase(is_training = False, args = self._args, data_transform = data_transform, target_transform = target_transform)
        self.train_loader = torch.utils.data.DataLoader(train_db, batch_size=self._args.batch_size, shuffle=True, **self.kwargs)
        self.eval_loader = torch.utils.data.DataLoader(eval_db, batch_size=self._args.batch_size, shuffle=True, **self.kwargs)

        self.net_mean_train_loss = 0
        self.net_mean_eval_loss = 0

    def train(self):
        self.net.train()
        loss_list = []
        for batch_idx, (source, target) in enumerate(self.train_loader):
            source, target = source.to(self.device), target.to(self.device)
            self.net_optimizer.zero_grad()
            loss = 0
            output = self.net(source).squeeze()
            loss = F.mse_loss(output, target[:,:,:95])*100 
            loss.backward()
            self.net_optimizer.step()

            loss_list.append(loss.item())

            if batch_idx % self._args.log_interval == 0:
                print('Train Epoch: {:5d} [{:5d}/{:5d} ({:4.1f}%)]\tLoss: {:.7f}'.format(
                    self.epoch + 1, batch_idx * len(source), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))

        self.net_mean_train_loss = np.mean(loss_list)

        self.target_im_train = target.cpu().detach().numpy()[0]
        self.source_im_train = source.cpu().detach().numpy()[0]
        self.output_im_train = output.cpu().detach().numpy()[0]

    def evaluate(self):
        self.net.eval()
        eval_loss_list = []
        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.eval_loader):
                source, target = source.to(self.device), target.to(self.device)
                output = self.net(source).squeeze()
                eval_loss = F.mse_loss(output, target[:,:,:95])*100
                eval_loss_list.append(eval_loss.item())

        self.target_im_eval = target.cpu().detach().numpy()[0]
        self.output_im_eval = output.cpu().detach().numpy()[0]

        self.mean_eval_loss = np.mean(eval_loss_list)
        print(self.mean_eval_loss)

    def mse_weighted(self, output, target, weight):
        return torch.sum(weight * (output - target)**2)/output.numel()

    def save_model(self):
        print(' '+'-'*64, '\nSaving\n', '-'*64)
        model_full_path = os.path.join(self.model_dir, '{}_net.pth'.format(str(self.epoch)))
        optimizer_full_path = os.path.join(self.model_dir, '{}_opt.pth'.format(str(self.epoch)))
        torch.save(self.net.state_dict(), model_full_path)
        torch.save(self.net_optimizer.state_dict(), optimizer_full_path)

    def loss_to_file(self):
        with open('loss_over_epochs.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([self.epoch, self.net_mean_train_loss, self.mean_eval_loss, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    def generate_plot(self):
        frmt = "%Y-%m-%d %H:%M:%S"
        plot_data = pd.read_csv('loss_over_epochs.csv', header=None)
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

        plt.figure(figsize=(16,9), dpi=110)

        plt.subplot(2,3,1)
        plt.xlabel('Epochs')
        plt.ylabel('Loss mse')
        plt.semilogy(epochs, train_losses, label='Train Loss')
        plt.semilogy(epochs, eval_losses, label='Eval Loss')
        plt.legend()
        plt.grid(True, 'both')
        plt.title('Loss')

        print(np.max(self.output_im_train))
        
        plt.subplot(2,3,2)
        plt.imshow(self.output_im_train, label='output', vmin = 0, vmax = 1)
        plt.title('Train output')
        
        plt.subplot(2,3,3)
        plt.imshow(self.target_im_train, label='target', vmin = 0, vmax = 1)
        plt.title('Train target')

        plt.subplot(2,3,4)
        plt.imshow(self.source_im_train, label='target', vmin = 0, vmax = 1)
        plt.title('Train input')

        plt.subplot(2,3,5)
        plt.imshow(self.output_im_eval, label='output', vmin = 0, vmax = 1)
        plt.title('Eval output')
        
        plt.subplot(2,3,6)
        plt.imshow(self.target_im_eval, label='target', vmin = 0, vmax = 1)
        plt.title('Eval target')

        plt.tight_layout()
        plt.savefig('net.png')
        plt.close()

    def stop_session(self):
        with open('loss_over_epochs.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['stopped', 'stopped', 'stopped', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    def start_session(self):
        with open('loss_over_epochs.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['started', 'started', 'started', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


def main(model_dir):
    global interrupted
    interrupted = False
    signal.signal(signal.SIGINT, signal_handler)

    model = Model(model_dir)
    model.start_session()

    for epoch in range(model.epoch, model._args.epochs + 1):
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
