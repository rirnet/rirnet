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
        generator_rir = import_module('generator_rir')
        generator_sig = import_module('generator_sig')
        refiner = import_module('decoder')
        self.GS = generator_sig.Net()
        self.GR = generator_rir.Net()
        self.R = refiner.Net()
        self.GS_args = self.GS.args()
        self.GR_args = self.GR.args()
        self.R_args = self.R.args()

        use_cuda = not self.GS_args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.GS.to(self.device)
        self.GR.to(self.device)
        self.R.to(self.device)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        list_epochs = glob('*.pth')
        list_epochs = [ x for x in list_epochs if "GSo" not in x ]
        self.GS_optimizer = optim.SGD(self.GS.parameters(), lr=self.GS_args.lr, momentum=self.GS_args.momentum, nesterov=True)
        self.GR_optimizer = optim.SGD(self.GR.parameters(), lr=self.GR_args.lr, momentum=self.GR_args.momentum, nesterov=True)
        self.R_optimizer = optim.SGD(self.R.parameters(), lr=self.R_args.lr, momentum=self.R_args.momentum, nesterov=True)
        if list_epochs == []:
            epoch = 0
        else:
            epoch = max([int(e.split('_')[0]) for e in list_epochs])
            self.GS_optimizer.load_state_dict(torch.load(os.path.join(model_dir, '{}_GSo.pth'.format(str(epoch)))))
            self.GS.load_state_dict(torch.load(os.path.join(model_dir, '{}_GS.pth'.format(str(epoch)))))
            self.GR_optimizer.load_state_dict(torch.load(os.path.join(model_dir, '{}_GRo.pth'.format(str(epoch)))))
            self.GR.load_state_dict(torch.load(os.path.join(model_dir, '{}_GR.pth'.format(str(epoch)))))
            self.R_optimizer = optim.SGD(self.R.parameters(), lr=self.R_args.lr, momentum=self.R_args.momentum, nesterov=True)
            #self.R_optimizer.load_state_dict(torch.load(os.path.join(model_dir, '{}_Ro.pth'.format(str(epoch)))))
            #self.R.load_state_dict(torch.load(os.path.join(model_dir, '{}_R.pth'.format(str(epoch)))))
            self.GS.cuda()
            self.GR.cuda()
            self.R.cuda()
            for g in self.GS_optimizer.param_groups:
                g['lr'] = self.GS_args.lr
                g['momentum'] = self.GS_args.momentum
            for g in self.GR_optimizer.param_groups:
                g['lr'] = self.GR_args.lr
                g['momentum'] = self.GR_args.momentum
            for g in self.R_optimizer.param_groups:
                g['lr'] = self.R_args.lr
                g['momentum'] = self.R_args.momentum


        self.epoch = epoch
        self.csv_path = os.path.join(self.GS_args.db_path, 'db.csv')
        data_transform = self.GS.data_transform()
        target_transform = self.GS.target_transform()

        train_db = RirnetDatabase(is_training = True, args = self.GS_args, data_transform = data_transform, target_transform = target_transform)
        eval_db = RirnetDatabase(is_training = False, args = self.GS_args, data_transform = data_transform, target_transform = target_transform)
        self.train_loader = torch.utils.data.DataLoader(train_db, batch_size=self.GS_args.batch_size, shuffle=True, **self.kwargs)
        self.eval_loader = torch.utils.data.DataLoader(eval_db, batch_size=self.GS_args.batch_size, shuffle=True, **self.kwargs)

        self.GS_mean_train_loss = 0
        self.GS_mean_eval_loss = 0

        try:
            getattr(F, self.GS_args.loss_function)
        except AttributeError:
            print('AttributeError! {} is not a valid loss function. The string must exactly match a pytorch loss '
                  'function'.format(self.GS_args.loss_function))
            sys.exit()


    def train(self):
        self.GS.train()
        self.GR.train()
        self.R.train()
        l1_loss_list = []
        l2_loss_list = []
        l3_loss_list = []
        for batch_idx, (source, target) in enumerate(self.train_loader):
            torch.cuda.empty_cache()

            source, target = source.to(self.device), target.to(self.device)

            self.GS_optimizer.zero_grad()
            self.GR_optimizer.zero_grad()
            self.R_optimizer.zero_grad()

            ##Forward and update GS
            abs_gen_rir = self.GS(source)
            abs_real_rir = self.GR(target).detach()
            l1 = self.hausdorff(abs_gen_rir, abs_real_rir)
            l1.backward(retain_graph = True)
            self.GS_optimizer.step()

            ##Forward and update GR
            abs_real_rir = self.GR(target)
            abs_gen_rir = self.GS(source).detach()
            l2 = self.hausdorff(abs_real_rir, abs_gen_rir)
            l2.backward(retain_graph = True)
            self.GR_optimizer.step()

            ##Forward through GS, GR and R, update all
            abs_gen_rir = self.GS(source)
            abs_real_rir = self.GR(target)
            l1 = self.hausdorff(abs_gen_rir, abs_real_rir)
            l2 = self.hausdorff(abs_real_rir, abs_gen_rir)
            if batch_idx%2 == 0:
                gen_rir = self.R(abs_gen_rir)
            else:
                gen_rir = self.R(abs_real_rir)
            gen_rir[:,0] = (gen_rir[:,0].clone().t() - gen_rir[:,0,0].clone()).t()
            l3 = self.hausdorff(gen_rir, target)
            l1.backward(retain_graph = True)
            l2.backward(retain_graph = True)
            l3.backward()

            self.GS_optimizer.step()
            self.GR_optimizer.step()
            self.R_optimizer.step()

            l1_loss_list.append(l1.item())
            l2_loss_list.append(l2.item())
            l3_loss_list.append(l3.item())

            if batch_idx % self.GS_args.log_interval == 0:
                print('Train Epoch: {:5d} [{:5d}/{:5d} ({:4.1f}%)]\tLoss: {:.6f}, {:.6f}, {:.6f}'.format(
                    self.epoch + 1, batch_idx * len(source), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), l1.item(), l2.item(), l3.item()))

        self.l1_mean_train_loss = np.mean(l1_loss_list)
        self.l2_mean_train_loss = np.mean(l2_loss_list)
        self.l3_mean_train_loss = np.mean(l3_loss_list)

        target_im = target.cpu().detach().numpy()
        output_im = gen_rir.cpu().detach().numpy()

        abs_gen_rir_im = abs_gen_rir.cpu().detach().numpy()
        abs_real_rir_im = abs_real_rir.cpu().detach().numpy()

        plt.plot(output_im[0, 0, :], output_im[0, 1, :], '--x', linewidth=0.5, markersize=0.7, label='gen_rir')
        plt.plot(target_im[0, 0, :], target_im[0, 1, :], '--o', linewidth=0.5, markersize=0.7, label='target')
        plt.title('train')
        plt.grid(True)
        plt.legend()
        plt.savefig('example_output_train.png')
        plt.close()

        plt.plot(abs_gen_rir_im[0, 0, :], abs_gen_rir_im[0, 1, :], '--x', linewidth=0.5, markersize=0.7, label='gen_abs')
        plt.plot(abs_real_rir_im[0, 0, :], abs_real_rir_im[0, 1, :], '--o', linewidth=0.5, markersize=0.7, label='real_abs')
        plt.title('train')
        plt.grid(True)
        plt.legend()
        plt.savefig('example_output_train_abs.png')
        plt.close()

    def evaluate(self):
        self.G.eval()
        eval_loss_list = []
        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.eval_loader):
                source, target = source.to(self.device), target.to(self.device)
                output = self.R(self.GS(source))
                output[:,0] = (output[:,0].clone().t() - output[:,0,0].clone()).t()
                eval_loss = self.hausdorff(output, target).item()
                eval_loss_list.append(eval_loss)

        target_im = target.cpu().detach().numpy()
        output_im = output.cpu().detach().numpy()

        plt.plot(output_im[0, 0, :], output_im[0, 1, :], '--x', linewidth=0.5, markersize=0.7)
        plt.plot(target_im[0, 0, :], target_im[0, 1, :], '--o', linewidth=0.5, markersize=0.7)
        plt.title('eval')
        plt.grid(True)
        plt.savefig('example_output.png')
        plt.close()
        self.mean_eval_loss = np.mean(eval_loss_list)
        print(self.mean_eval_loss)


    def save_model(self):
        print(' '+'-'*64, '\nSaving\n', '-'*64)
        model_full_path = os.path.join(self.model_dir, '{}_GS.pth'.format(str(self.epoch)))
        optimizer_full_path = os.path.join(self.model_dir, '{}_GSo.pth'.format(str(self.epoch)))
        torch.save(self.GS.state_dict(), model_full_path)
        torch.save(self.GS_optimizer.state_dict(), optimizer_full_path)

        model_full_path = os.path.join(self.model_dir, '{}_GR.pth'.format(str(self.epoch)))
        optimizer_full_path = os.path.join(self.model_dir, '{}_GRo.pth'.format(str(self.epoch)))
        torch.save(self.GR.state_dict(), model_full_path)
        torch.save(self.GR_optimizer.state_dict(), optimizer_full_path)


    def loss_to_file(self):
        with open('loss_over_epochs.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([self.epoch, self.l1_mean_train_loss, self.l2_mean_train_loss, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


    def generate_plot(self):
        frmt = "%Y-%m-%d %H:%M:%S"
        plot_data = pd.read_csv('loss_over_epochs.csv', header=None)
        epochs_raw, train_losses_raw, eval_losses_raw, times_raw = plot_data.values.T

        epochs = [int(epoch) for epoch in list(epochs_raw) if is_number(epoch)]
        train_losses = [float(loss) for loss in list(train_losses_raw) if is_number(loss)]
        eval_losses = [float(loss) for loss in list(eval_losses_raw) if is_number(loss)]

        if self.GS_args.save_timestamps:
            total_time = timedelta(0, 0, 0)
            if np.size(times_raw) > 1:
                start_times = times_raw[epochs_raw == 'started']
                stop_times = times_raw[epochs_raw == 'stopped']
                for i_stop_time, stop_time in enumerate(stop_times):
                    total_time += datetime.strptime(stop_time, frmt) - datetime.strptime(start_times[i_stop_time], frmt)
                total_time += datetime.now() - datetime.strptime(start_times[-1], frmt)
                plt.title('Trained for {} hours and {:2d} minutes'.format(int(total_time.days/24 + total_time.seconds//3600), (total_time.seconds//60)%60))
        plt.xlabel('Epochs')
        plt.ylabel('Loss ({})'.format(self.GS_args.loss_function))
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

    for epoch in range(model.epoch, model.GS_args.epochs + 1):
        model.train()
        #model.evaluate()
        model.epoch = epoch+1
        model.loss_to_file()
        model.generate_plot()
        if epoch % model.G_args.save_interval == 0:
            model.save_model()

        if interrupted:
            print(' '+'-'*64, '\nEarly stopping\n', '-'*64)
            model.stop_session()
            #model.save_model()
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
