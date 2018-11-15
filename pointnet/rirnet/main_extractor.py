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
from pyroomacoustics.utilities import fractional_delay


# -------------  Initialization  ------------- #
class Model:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        sys.path.append(model_dir)
        extractor = import_module('extractor')
        autoenc = import_module('autoencoder')
        self.extractor = extractor.Net()
        self.autoenc = autoenc.Net()
        self.extractor_args = self.extractor.args()

        use_cuda = not self.extractor_args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.extractor.to(self.device)
        self.autoenc.to(self.device)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        list_epochs_extractor = glob('*_extractor.pth')
        list_epochs_autoenc = glob('*_autoenc.pth')
        list_epochs_discriminator = glob('*_disc.pth')
        list_epochs_extractor = [ x for x in list_epochs_extractor if "opt_extractor" not in x ]
        list_epochs_autoenc = [ x for x in list_epochs_autoenc if "opt_autoenc" not in x ]
        list_epochs_disc = [ x for x in list_epochs_autoenc if "opt_disc" not in x ]

        self.extractor_optimizer = optim.Adam(self.extractor.parameters(), lr=self.extractor_args.lr, betas=(0.9, 0.99), eps=1e-5, weight_decay=0, amsgrad=False)

        epoch = max([int(e.split('_')[0]) for e in list_epochs_autoenc])
        self.autoenc.load_state_dict(torch.load(os.path.join(model_dir, '{}_autoenc.pth'.format(str(epoch)))))

        if list_epochs_extractor == []:
            epoch = 0
        else:
            epoch = max([int(e.split('_')[0]) for e in list_epochs_extractor])
            self.extractor_optimizer.load_state_dict(torch.load(os.path.join(model_dir, '{}_opt_extractor.pth'.format(str(epoch)))))
            self.extractor.load_state_dict(torch.load(os.path.join(model_dir, '{}_extractor.pth'.format(str(epoch)))))
            for g in self.extractor_optimizer.param_groups:
                g['lr'] = self.extractor_args.lr
                g['momentum'] = self.extractor_args.momentum

        self.epoch = epoch
        data_transform = self.extractor.data_transform()
        target_transform = self.extractor.target_transform()

        train_db = RirnetDatabase(is_training = True, args = self.extractor_args, data_transform = data_transform, target_transform = target_transform)
        eval_db = RirnetDatabase(is_training = False, args = self.extractor_args, data_transform = data_transform, target_transform = target_transform)
        self.train_loader = torch.utils.data.DataLoader(train_db, batch_size=self.extractor_args.batch_size, shuffle=True, **self.kwargs)
        self.eval_loader = torch.utils.data.DataLoader(eval_db, batch_size=self.extractor_args.batch_size, shuffle=True, **self.kwargs)

        self.extractor_mean_train_loss = 0
        self.extractor_mean_eval_loss = 0

        try:
            getattr(F, self.extractor_args.loss_function)
        except AttributeError:
            print('AttributeError! {} is not a valid loss function. The string must exactly match a pytorch loss '
                  'function'.format(self.extractor_args.loss_function))
            sys.exit()


    def train(self):

        for g in self.extractor_optimizer.param_groups:
            g['lr'] = g['lr']*0.95
            plt.title(g['lr'])

        if self.epoch%50 == 0:
            for g in self.extractor_optimizer.param_groups:
                g['lr'] = self.extractor_args.lr
                plt.title(g['lr'])

        self.extractor.train()
        extractor_loss_list = []
        for batch_idx, (source, target) in enumerate(self.train_loader):
            torch.cuda.empty_cache()
            source, target = source.to(self.device), target.to(self.device)

            latent_target = self.autoenc(target, encode=True, decode=False)
            latent_source = self.extractor(source)
            self.extractor_optimizer.zero_grad()
            extractor_loss = self.mse_weighted(latent_source, latent_target, 1)
            extractor_loss.backward(retain_graph=True)

            output = self.autoenc(latent_source, encode=False, decode=True)
            extractor_loss_output = self.hausdorff(output, target)
            #extractor_loss_output = self.mse_weighted(output, target, 1)

            #extractor_loss_output = torch.tensor(0)
            extractor_loss_output.backward()
            self.extractor_optimizer.step()

            extractor_loss_list.append(extractor_loss.item())

            if batch_idx % self.extractor_args.log_interval == 0:
                print('Train Epoch: {:5d} [{:5d}/{:5d} ({:4.1f}%)]\tLoss: {:.6f}, {:.6f}'.format(
                    self.epoch + 1, batch_idx * len(source), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), extractor_loss.item(), extractor_loss_output.item()))
        self.extractor_mean_train_loss = np.mean(extractor_loss_list)

        self.latent_target_im_train = latent_target.cpu().detach().numpy()[0]
        self.latent_output_im_train = latent_source.cpu().detach().numpy()[0]

    def evaluate(self):
        self.extractor.eval()
        eval_loss_list = []
        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.eval_loader):
                torch.cuda.empty_cache()
                source, target = source.to(self.device), target.to(self.device)

                latent_target = self.autoenc(target, encode=True, decode=False)
                latent_source = self.extractor(source)
                self.extractor_optimizer.zero_grad()
                output = self.autoenc(latent_source, encode=False, decode=True)
                extractor_loss = self.hausdorff(output[:,:,:], target[:,:,:])

                self.rir_im = []
                self.extractor_optimizer.step()
                eval_loss_list.append(extractor_loss.item())
        self.latent_target_im_eval = latent_target.cpu().detach().numpy()[0]
        self.latent_output_im_eval = latent_source.cpu().detach().numpy()[0]

        self.target_im = target.cpu().detach().numpy()[0].T
        self.output_im = output.cpu().detach().numpy()[0].T

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

    def reconstruct_rir(self, output):
        fdl = 81
        fdl2 = (fdl-1) // 2
        time = (output[:,0].astype('double')+1)*1024
        peaks = np.exp(-output[:,1]).astype('double')
        ir = np.arange(np.ceil((1.05*time.max()) + fdl))*0
        for i in range(time.shape[0]):
            time_ip = int(np.round(time[i]))
            time_fp = time[i] - time_ip
            ir[time_ip-fdl2:time_ip+fdl2+1] += peaks[i]*fractional_delay(time_fp)
        start_ind = min(np.where(ir != 0)[0])
        ir = ir[start_ind:-3000]
        return ir

    def save_model(self):
        print(' '+'-'*64, '\nSaving\n', '-'*64)
        model_full_path = os.path.join(self.model_dir, '{}_extractor.pth'.format(str(self.epoch)))
        optimizer_full_path = os.path.join(self.model_dir, '{}_opt_extractor.pth'.format(str(self.epoch)))
        torch.save(self.extractor.state_dict(), model_full_path)
        torch.save(self.extractor_optimizer.state_dict(), optimizer_full_path)

    def loss_to_file(self):
        with open('loss_over_epochs_ex.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([self.epoch, self.extractor_mean_train_loss, self.extractor_mean_train_loss, self.mean_eval_loss, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    def generate_plot(self):
        frmt = "%Y-%m-%d %H:%M:%S"
        plot_data = pd.read_csv('loss_over_epochs_ex.csv', header=None)
        epochs_raw, train_l1_raw, train_l3_raw, eval_losses_raw, times_raw = plot_data.values.T

        epochs = [int(epoch) for epoch in list(epochs_raw) if is_number(epoch)]
        l1_train_losses = [float(loss) for loss in list(train_l1_raw) if is_number(loss)]
        l3_train_losses = [float(loss) for loss in list(train_l3_raw) if is_number(loss)]
        eval_losses = [float(loss) for loss in list(eval_losses_raw) if is_number(loss)]

        if self.extractor_args.save_timestamps:
            total_time = timedelta(0, 0, 0)
            if np.size(times_raw) > 1:
                start_times = times_raw[epochs_raw == 'started']
                stop_times = times_raw[epochs_raw == 'stopped']
                for i_stop_time, stop_time in enumerate(stop_times):
                    total_time += datetime.strptime(stop_time, frmt) - datetime.strptime(start_times[i_stop_time], frmt)
                total_time += datetime.now() - datetime.strptime(start_times[-1], frmt)
                plt.title('Trained for {} hours and {:2d} minutes'.format(int(total_time.days/24 + total_time.seconds//3600), (total_time.seconds//60)%60))
        plt.figure(figsize=(16,9), dpi=110)
        plt.subplot(3,1,1)
        plt.xlabel('Epochs')
        plt.ylabel('Loss ({})'.format(self.extractor_args.loss_function))
        plt.semilogy(epochs, l1_train_losses, label='Abstract Train Loss')
        plt.semilogy(epochs, l3_train_losses, label='Real Train Loss')
        plt.semilogy(epochs, eval_losses, label='Real Eval Loss')
        plt.legend()
        plt.grid(True, 'both')
        plt.title('Loss')

        plt.subplot(3,2,3)
        plt.plot(self.latent_target_im_train, '--ok', linewidth=0.5, markersize=3, label='target0')
        plt.plot(self.latent_output_im_train, '-o', linewidth=0.5, markersize=3, label='output0')
        plt.grid(True)
        plt.legend()
        plt.title('Train latent')

        plt.subplot(3,2,4)
        plt.plot(self.latent_target_im_eval, '--ok', linewidth=0.5, markersize=3, label='target0')
        plt.plot(self.latent_output_im_eval, '-o', linewidth=0.5, markersize=3, label='output0')
        plt.grid(True)
        plt.legend()
        plt.title('Eval latent')

        plt.subplot(3,2,5)
        plt.plot(self.target_im[:,0], self.target_im[:,1], '--o', linewidth=0.5, markersize=2, label='target')
        plt.plot(self.output_im[:,0], self.output_im[:,1], '--x', linewidth=0.5, markersize=2, label='output')
        plt.grid(True)
        plt.legend()
        plt.title('Eval output')

        plt.subplot(3,2,6)
        plt.plot(self.rir_im)
        plt.grid(True)
        plt.title('Output rir')

        plt.tight_layout()
        plt.savefig('complete_pipeline.png')
        plt.close()

    def stop_session(self):
        with open('loss_over_epochs_ex.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['stopped', 'stopped', 'stopped', 'stopped', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


    def start_session(self):
        with open('loss_over_epochs_ex.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['started', 'started', 'started', 'started', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])



def main(model_dir):
    global interrupted
    interrupted = False
    signal.signal(signal.SIGINT, signal_handler)

    model = Model(model_dir)
    model.start_session()

    for epoch in range(model.epoch, model.extractor_args.epochs + 1):
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

