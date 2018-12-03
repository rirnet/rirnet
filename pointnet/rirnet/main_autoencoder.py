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
        self.discriminator, _ = misc.load_latest(model_dir, 'discriminator')
        self.autoencoder_args = self.autoencoder.args()
        self.discriminator_args = self.discriminator.args()
        use_cuda = not self.autoencoder_args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.autoencoder.to(self.device)
        self.discriminator.to(self.device)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        x = torch.linspace(-2, 2, 256)
        weight = (1-torch.exp(2*x))/(4*(1+torch.exp(2*x)))
        weight += 1 - weight[0]
        weight = weight.repeat(self.autoencoder_args.batch_size, 2, 1)
        self.mse_weight = weight.cuda()

        self.autoencoder_optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.autoencoder_args.lr, betas=(0.9, 0.99), eps=1e-5, weight_decay=0, amsgrad=False)
        self.discriminator_optimizer = optim.SGD(self.discriminator.parameters(), lr=self.discriminator_args.lr, momentum=self.discriminator_args.momentum, nesterov=True)

        if self.epoch != 0:
            self.autoencoder_optimizer.load_state_dict(torch.load(os.path.join(model_dir, '{}_opt_autoencoder.pth'.format(self.epoch))))

            for g in self.autoencoder_optimizer.param_groups:
                g['lr'] = self.autoencoder_args.lr
                g['momentum'] = self.autoencoder_args.momentum

        data_transform = self.autoencoder.data_transform()
        target_transform = self.autoencoder.target_transform()

        train_db = RirnetDatabase(is_training = True, args = self.autoencoder_args, data_transform = data_transform, target_transform = target_transform)
        eval_db = RirnetDatabase(is_training = False, args = self.autoencoder_args, data_transform = data_transform, target_transform = target_transform)
        self.train_loader = torch.utils.data.DataLoader(train_db, batch_size=self.autoencoder_args.batch_size, shuffle=True, **self.kwargs)
        self.eval_loader = torch.utils.data.DataLoader(eval_db, batch_size=self.autoencoder_args.batch_size, shuffle=True, **self.kwargs)

        self.autoencoder_mean_train_loss = 0
        self.autoencoder_mean_eval_loss = 0

    def train(self):

        for g in self.autoencoder_optimizer.param_groups:
            g['lr'] = g['lr']*0.99

        self.autoencoder.train()
        self.discriminator.train()
        autoencoder_loss_list = []
        for batch_idx, (source, target) in enumerate(self.train_loader):
            source, target = source.to(self.device), target.to(self.device)
            self.discriminator_optimizer.zero_grad()

            with torch.no_grad():
                output = self.autoencoder(target, encode=True, decode=True)

            # Train discriminator
            target_verdict = self.discriminator(target)
            discriminator_loss_1 = getattr(F, self.discriminator_args.loss_function)(target_verdict, torch.ones(self.autoencoder_args.batch_size, 1).float().cuda())
            discriminator_loss_1.backward()

            output_verdict = self.discriminator(output)
            discriminator_loss_2 = getattr(F, self.discriminator_args.loss_function)(output_verdict, torch.zeros(self.autoencoder_args.batch_size, 1).float().cuda())
            discriminator_loss_2.backward()

            self.discriminator_optimizer.step()

            # Train autoencoder
            self.autoencoder_optimizer.zero_grad()
            output = self.autoencoder(target, encode=True, decode=True)
            autoencoder_verdict = self.discriminator(output)
            autoencoder_loss_1 = getattr(F, self.discriminator_args.loss_function)(autoencoder_verdict, torch.ones(self.autoencoder_args.batch_size, 1).float().cuda())
            autoencoder_loss_1.backward(retain_graph=True)
            #autoencoder_loss_2 = getattr(F, self.autoencoder_args.loss_function)(output, target)
            #autoencoder_loss_2 = self.mse_weighted(output, target, self.mse_weight)
            #autoencoder_loss_2.backward(retain_graph=True)
            autoencoder_loss_3 = self.hausdorff(output, target)
            autoencoder_loss_3.backward()
            self.autoencoder_optimizer.step()

            autoencoder_loss_list.append(autoencoder_loss_3.item())

            if batch_idx % self.autoencoder_args.log_interval == 0:
                print('Train Epoch: {:5d} [{:5d}/{:5d} ({:4.1f}%)]\tLoss: {:.4f}\t{:.4f}\t{:.4f}'.format(
                    self.epoch + 1, batch_idx * len(source), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), autoencoder_loss_3.item(), autoencoder_loss_3.item(), autoencoder_loss_1.item()))#autoencoder_loss_1.item(), autoencoder_loss_2.item()))

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
                eval_loss = self.hausdorff(output, target)
                #eval_loss += getattr(F, self.autoencoder_args.loss_function)(output, target).item()
                eval_loss_list.append(eval_loss)

        self.target_im_eval = target.cpu().detach().numpy()[0]
        self.output_im_eval = output.cpu().detach().numpy()[0]

        self.mean_eval_loss = np.mean(eval_loss_list)
        print(self.mean_eval_loss)

    def mse_weighted(self, output, target, weight):
        return torch.sum(weight * (output - target)**2)/output.numel()

    def hausdorff(self, output, target):
        B, _, _ = output.size()
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
        return 10* res / B

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

        plt.figure(figsize=(16,9), dpi=110)

        plt.subplot(2,1,1)
        plt.xlabel('Epochs')
        plt.ylabel('Loss ({})'.format(self.autoencoder_args.loss_function))
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
        plt.savefig('autoencoder.png')
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
