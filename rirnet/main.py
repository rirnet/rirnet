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
import numpy as np
from importlib import import_module
from glob import glob
import signal


# -------------  Initialization  ------------- #
class Model:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        sys.path.append(model_dir)
        net = import_module('net')
        self.model = net.Net()
        self.args = self.model.args()
        torch.manual_seed(self.args.seed)

        use_cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(self.device)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        list_epochs = glob('*.pth')
        list_epochs = [ x for x in list_epochs if "_" not in x ]
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        if not list_epochs:
            epoch = 0
        else:
            epoch = max([int(e.split('.')[0]) for e in list_epochs])
            self.optimizer.load_state_dict(torch.load(os.path.join(model_dir, 'o_{}.pth'.format(str(epoch)))))
            self.model.load_state_dict(torch.load(os.path.join(model_dir, '{}.pth'.format(str(epoch)))))
            self.model.cuda()
            for g in self.optimizer.param_groups:
                g['lr'] = self.args.lr
                g['momentum'] = self.args.momentum
        self.epoch = epoch
        self.csv_path = os.path.join(self.args.db_path, 'db.csv')
        data_transform = self.model.transform()

        train_db = RirnetDatabase(is_training = True, args = self.args, transform = data_transform)
        eval_db = RirnetDatabase(is_training = False, args = self.args, transform = data_transform)
        self.train_loader = torch.utils.data.DataLoader(train_db, batch_size=self.args.batch_size, shuffle=True,
                                                        **self.kwargs)
        self.eval_loader = torch.utils.data.DataLoader(eval_db, batch_size=self.args.batch_size, shuffle=True,
                                                       **self.kwargs)

        self.mean_train_loss = 0
        self.mean_eval_loss = 0

        try:
            getattr(F, self.args.loss_function)
        except AttributeError:
            print('AttributeError! {} is not a valid loss function. The string must exactly match a pytorch loss '
                  'function'.format(self.args.loss_function))
            sys.exit()


    def train(self):
        self.model.train()
        loss_list = []
        for batch_idx, (source, target) in enumerate(self.train_loader):
            torch.cuda.empty_cache()
            source, target = source.to(self.device), target.to(self.device)
            output = self.model(source)
            self.optimizer.zero_grad()
            train_loss = getattr(F, self.args.loss_function)(output, target)

            train_loss.backward()
            self.optimizer.step()
            loss_list.append(train_loss.item())
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {:5d} [{:5d}/{:5d} ({:4.1f}%)]\tLoss: {:.6f}'.format(
                    self.epoch + 1, batch_idx * len(source), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), train_loss.item()))
        self.mean_train_loss = np.mean(loss_list)


    def evaluate(self):
        self.model.eval()
        eval_loss_list = []
        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.eval_loader):
                source, target = source.to(self.device), target.to(self.device)
                output = self.model(source)
                eval_loss = getattr(F, self.args.loss_function)(output, target).item()
                eval_loss_list.append(eval_loss)
            plt.subplot(2, 2, 1)
            plt.imshow(output.cpu().detach().numpy()[0, :, :], vmin=-3, vmax=3)
            plt.title('Output')

            plt.subplot(2, 2, 2)
            plt.imshow(target.cpu().detach().numpy()[0, :, :], vmin=-3, vmax=3)
            plt.title('Target')

            plt.subplot(2, 2, 3)
            plt.imshow(output.cpu().detach().numpy()[0, :, 0:40], vmin=-3, vmax=3)
            plt.title('Output')

            plt.subplot(2, 2, 4)
            plt.imshow(target.cpu().detach().numpy()[0, :, 0:40], vmin=-3, vmax=3)
            plt.title('Target')

            plt.savefig('example_output.png')
            plt.close()
        self.mean_eval_loss = np.mean(eval_loss_list)
        print(self.mean_eval_loss)


    def save_model(self):
        print(' '+'-'*64, '\nSaving\n', '-'*64)
        model_full_path = os.path.join(self.model_dir, '{}.pth'.format(str(self.epoch)))
        optimizer_full_path = os.path.join(self.model_dir, 'o_{}.pth'.format(str(self.epoch)))
        torch.save(self.model.state_dict(), model_full_path)
        torch.save(self.optimizer.state_dict(), optimizer_full_path)


    def loss_to_file(self):
        with open('loss_over_epochs.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([self.epoch, self.mean_train_loss, self.mean_eval_loss, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


    def generate_plot(self):
        frmt = "%Y-%m-%d %H:%M:%S"
        plot_data = pd.read_csv('loss_over_epochs.csv', header=None)
        epochs_raw, train_losses_raw, eval_losses_raw, times_raw = plot_data.values.T

        epochs = [int(epoch) for epoch in list(epochs_raw) if is_number(epoch)]
        train_losses = [float(loss) for loss in list(train_losses_raw) if is_number(loss)]
        eval_losses = [float(loss) for loss in list(eval_losses_raw) if is_number(loss)]

        if self.args.save_timestamps:
            total_time = timedelta(0, 0, 0)
            if np.size(times_raw) > 1:
                start_times = times_raw[epochs_raw == 'started']
                stop_times = times_raw[epochs_raw == 'stopped']
                for i_stop_time, stop_time in enumerate(stop_times):
                    total_time += datetime.strptime(stop_time, frmt) - datetime.strptime(start_times[i_stop_time], frmt)
                total_time += datetime.now() - datetime.strptime(start_times[-1], frmt)
                plt.title('Trained for {} hours and {:2d} minutes'.format(int(total_time.days/24 + total_time.seconds//3600), (total_time.seconds//60)%60))
        plt.xlabel('Epochs')
        plt.ylabel('Loss ({})'.format(self.args.loss_function))
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


def main(model_dir):
    # noinspection PyGlobalUndefined
    global interrupted
    interrupted = False
    signal.signal(signal.SIGINT, signal_handler)

    model = Model(model_dir)
    model.start_session()

    for epoch in range(model.epoch, model.args.epochs + 1):
        model.train()
        model.evaluate()
        model.epoch = epoch+1
        model.loss_to_file()
        model.generate_plot()
        if epoch % model.args.save_interval == 0:
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
