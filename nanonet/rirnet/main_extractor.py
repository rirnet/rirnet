from __future__ import print_function
from rirnet_database import RirnetDatabase
from datetime import datetime, timedelta
from torch.autograd import Variable
from importlib import import_module
from glob import glob
from pyroomacoustics.utilities import fractional_delay

import sys
import torch
import os
import csv
import signal
import scipy.stats

import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import rirnet.misc as misc
import scipy as sp

# -------------  Initialization  ------------- #
class Model:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        sys.path.append(model_dir)
        extractor = import_module('extractor')
        self.extractor, self.epoch = misc.load_latest(model_dir, 'extractor')
        self.autoenc, _ = misc.load_latest(model_dir, 'autoencoder')
        self.extractor_args = self.extractor.args()

        use_cuda = not self.extractor_args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.extractor.to(self.device)
        self.autoenc.to(self.device)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        self.extractor_optimizer = optim.Adam(self.extractor.parameters(), lr=self.extractor_args.lr, betas=(0.9, 0.99), eps=1e-5, weight_decay=0, amsgrad=False)

        self.best_eval_loss = np.inf
        self.best_eval_loss_latent = np.inf

        if self.epoch != 0:
            self.extractor_optimizer.load_state_dict(torch.load(os.path.join(model_dir, '{}_opt_extractor.pth'.format(self.epoch))))
            for g in self.extractor_optimizer.param_groups:
                g['lr'] = self.extractor_args.lr

        data_transform = self.extractor.data_transform()
        target_transform = self.extractor.target_transform()

        train_db = RirnetDatabase(is_training = True, args = self.extractor_args, data_transform = data_transform, target_transform = target_transform)
        eval_db = RirnetDatabase(is_training = False, args = self.extractor_args, data_transform = data_transform, target_transform = target_transform)
        self.train_loader = torch.utils.data.DataLoader(train_db, batch_size=self.extractor_args.batch_size, shuffle=True, **self.kwargs)
        self.eval_loader = torch.utils.data.DataLoader(eval_db, batch_size=self.extractor_args.batch_size, shuffle=True, **self.kwargs)

        self.extractor_mean_train_loss = 0
        self.extractor_mean_eval_loss = 0


    def train(self):
        self.extractor.train()
        extractor_loss_latent_list = []
        extractor_loss_output_list = []
        for batch_idx, (source, target) in enumerate(self.train_loader):
            torch.cuda.empty_cache()

            source, target = source.to(self.device), target.to(self.device)

            latent_target = self.autoenc(target, encode=True, decode=False)
            latent_source = self.extractor(source)
            extractor_loss = 0
            self.extractor_optimizer.zero_grad()
            extractor_loss_latent = self.mse_weighted(latent_source, latent_target, 10)
            extractor_loss += extractor_loss_latent

            output = self.autoenc(latent_source, encode=False, decode=True)
            extractor_loss_output = self.chamfer_loss(output, target)
            extractor_loss += extractor_loss_output
            extractor_loss_latent.backward()
            self.extractor_optimizer.step()

            extractor_loss_latent_list.append(extractor_loss_latent.item())
            extractor_loss_output_list.append(extractor_loss_output.item())

            print('Train Epoch: {:5d} [{:5d}/{:5d} ({:4.1f}%)]\tLatent loss: {:.6f}\t Output loss: {:.6f}'.format(
                self.epoch + 1, batch_idx * len(source), len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), extractor_loss_latent.item(), extractor_loss_output.item()))
        self.extractor_mean_train_loss_latent = np.mean(extractor_loss_latent_list)
        self.extractor_mean_train_loss_output = np.mean(extractor_loss_output_list)

        self.latent_target_im_train = latent_target.cpu().detach().numpy()[0]
        self.latent_output_im_train = latent_source.cpu().detach().numpy()[0]

    def evaluate(self):
        self.extractor.eval()
        eval_loss_list = []
        eval_loss_list_latent = []

        n_samples = len(self.eval_loader)

        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.eval_loader):
                torch.cuda.empty_cache()
                source, target = source.to(self.device), target.to(self.device)

                latent_target = self.autoenc(target, encode=True, decode=False)
                latent_source = self.extractor(source)

                eval_loss_list_latent.append(self.mse_weighted(latent_source, latent_target, 10).item())

                output = self.autoenc(latent_source, encode=False, decode=True)
                output_source = self.autoenc(latent_target, encode=False, decode=True)
                extractor_loss = self.chamfer_loss(output, target)

                self.rir_im = []
                eval_loss_list.append(extractor_loss.item())

                #jsd, _, _ = self.calculate_metrics(output, target)

        self.jsd_mean = 1 #jsd/1000
        self.coverage_mean = 1 #coverage/1000
        self.mmd_mean = 1 #mmd/1000

        self.latent_target_im_eval = latent_target.cpu().detach().numpy()[0]
        self.latent_output_im_eval = latent_source.cpu().detach().numpy()[0]

        self.target_im = target.cpu().detach().numpy()[0].T
        self.output_im = output.cpu().detach().numpy()[0].T
        self.source_im = output_source.cpu().detach().numpy()[0].T

        self.mean_eval_loss = np.mean(eval_loss_list)
        self.mean_eval_loss_latent = np.mean(eval_loss_list_latent)

        if self.mean_eval_loss < self.best_eval_loss:
            self.best_eval_loss = self.mean_eval_loss
        if self.mean_eval_loss_latent < self.best_eval_loss_latent:
            self.best_eval_loss_latent = self.mean_eval_loss_latent

        print('Best Latent loss eval:', self.best_eval_loss_latent)
        print('Best Output loss eval:', self.best_eval_loss)

        f = open('ex_results', 'w')
        f.write('Ex output loss    --- Ex latent loss\n')
        f.write('{} --- {}'.format(self.best_eval_loss, self.best_eval_loss_latent))


    def kldiv(self, A, B):
        """
        Calculates the Kullback–Leibler divergence
        of numpy-arrays A and B
        """
        a = A.copy()
        b = B.copy()
        index = np.logical_and(a>0, b>0)
        a = a[index]
        b = b[index]
        return np.sum([v for v in a*np.log2(a/b)])


    def jsdiv(self, A, B, bins):
        """
        Calculates the Jensen-Shannon divergence
        based on the resolution 'bins' for two
        pytorch tensors of size [b, 2, n] with
        values within the unit square
        """

        [b, d, n] = A.size()

        A_np = A.transpose(1,2).contiguous().view(b*n,2).cpu().numpy()
        B_np = B.transpose(1,2).contiguous().view(b*n,2).cpu().numpy()
        hist_A, _, _ = np.histogram2d(A_np[:,0], A_np[:,1], range=[[0, 1],[0, 1]], bins=bins)
        hist_B, _, _ = np.histogram2d(B_np[:,0], B_np[:,1], range=[[0, 1],[0, 1]], bins=bins)

        P = hist_A/hist_A.sum()
        Q = hist_B/hist_B.sum()
        M = 1/2*(P+Q)
        return (self.kldiv(P,M)+self.kldiv(Q,M))/2

    def MR(self, A, B):
        """
        Calculates the coverage of tensors A and B
        """

        return 1

    def mmd(self, A, B):
        """
        Calculates the Minimum Matching Distance
        of tensors A and B
        """
        return 1

    def calculate_metrics(self, output, target):
        """
        Calculates the Jensen-Shannon divergence,
        Coverage and Minimum Matching distance
        of tensors output and target of size
        [b, 2, n]
        """
        jsd = self.jsdiv(output/10, target/10, 40)
        coverage = self.coverage(output, target)
        mmd = self.mmd(output, target)
        return jsd, coverage, mmd

    def mse_weighted(self, output, target, weight):
        return torch.sum(weight * (output - target)**2)/output.numel()

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

        #Pxtoy = P.min(1)[1]
        #Pytox = P.min(2)[1]

        #c1 = 0
        #c2 = 0
        #for i in range(B):
        #    c1 += len(Pxtoy[i].unique())
        #    c2 += len(Pytox[i].unique())
        #print((c1+c2)/(2*B*N))

        return 10*(l1 + l2)

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
            writer.writerow([
                self.epoch,
                self.extractor_mean_train_loss_latent,
                self.extractor_mean_train_loss_output,
                self.mean_eval_loss,
                self.mean_eval_loss_latent,
                self.jsd_mean,
                self.coverage_mean,
                self.mmd_mean,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    def generate_plot(self):
        frmt = "%Y-%m-%d %H:%M:%S"
        plot_data = pd.read_csv('loss_over_epochs_ex.csv', header=None)
        epochs_raw, train_l1_raw, train_l2_raw, eval_losses_raw, eval_losses_raw_latent, jsd_raw, coverage_raw, mmd_raw, times_raw = plot_data.values.T

        epochs = [int(epoch) for epoch in list(epochs_raw) if is_number(epoch)]
        l1_train_losses = [float(loss) for loss in list(train_l1_raw) if is_number(loss)]
        l2_train_losses = [float(loss) for loss in list(train_l2_raw) if is_number(loss)]
        eval_losses = [float(loss) for loss in list(eval_losses_raw) if is_number(loss)]
        eval_losses_latent = [float(loss) for loss in list(eval_losses_raw_latent) if is_number(loss)]
        jsd = [float(loss) for loss in list(jsd_raw) if is_number(loss)]
        coverage = [float(loss) for loss in list(coverage_raw) if is_number(loss)]
        mmd = [float(loss) for loss in list(mmd_raw) if is_number(loss)]

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
        plt.xlabel('Epochs')
        plt.ylabel('Extractor Loss')
        plt.semilogy(epochs[-max_plot_length:], l1_train_losses[-max_plot_length:], label='Latent Train Loss')
        plt.semilogy(epochs[-max_plot_length:], l2_train_losses[-max_plot_length:], label='Output Train Loss')
        plt.semilogy(epochs[-max_plot_length:], eval_losses_latent[-max_plot_length:], label='Latent Eval Loss')
        plt.semilogy(epochs[-max_plot_length:], eval_losses[-max_plot_length:], label='Output Eval Loss')
        plt.legend()
        plt.grid(True, 'both')
        plt.title('Loss')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig('fig/ex_loss.png')
        plt.savefig('fig/ex_loss.eps')

        fig = plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.plot(epochs[-max_plot_length:], jsd[-max_plot_length:], label='JSD')
        plt.plot(epochs[-max_plot_length:], coverage[-max_plot_length:], label='COVERAGE')
        plt.plot(epochs[-max_plot_length:], mmd[-max_plot_length:], label='MMD')
        plt.legend()
        plt.grid(True, 'both')
        plt.title('Metrics')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig('fig/ex_metrics.png')
        plt.savefig('fig/ex_metrics.eps')

        fig = plt.figure()
        plt.plot(self.latent_target_im_train, '--ok', linewidth=0.5, markersize=3, label='Target')
        plt.plot(self.latent_output_im_train, '-o', linewidth=0.5, markersize=3, label='Output')
        plt.grid(True)
        plt.legend()
        plt.title('Latent space Training')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig('fig/ex_train_latent.png')
        plt.savefig('fig/ex_train_latent.eps')

        fig = plt.figure()
        plt.plot(self.latent_target_im_eval, '--ok', linewidth=0.5, markersize=3, label='Target')
        plt.plot(self.latent_output_im_eval, '-o', linewidth=0.5, markersize=3, label='Output')
        plt.grid(True)
        plt.legend()
        plt.title('Latent space Evaluation')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig('fig/ex_eval_latent.png')
        plt.savefig('fig/ex_eval_latent.eps')

        fig = plt.figure()
        plt.plot(self.target_im[:,0], self.target_im[:,1], 'o', linewidth=0.5, markersize=2, label='Target')
        plt.plot(self.output_im[:,0], self.output_im[:,1], 'x', linewidth=0.5, markersize=2, label='Output')
        plt.grid(True)
        plt.legend()
        plt.title('Extractor Eval Output')
        plt.xlabel('Timing')
        plt.ylabel('-log(Amplitude)')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig('fig/ex_eval_output.png')
        plt.savefig('fig/ex_eval_output.eps')

        fig = plt.figure()
        plt.plot(self.target_im[:,0], self.target_im[:,1], 'o', linewidth=0.5, markersize=2, label='Target')
        plt.plot(self.source_im[:,0], self.source_im[:,1], 'x', linewidth=0.5, markersize=2, label='Output')
        plt.grid(True)
        plt.legend()
        plt.title('Extractor Eval Target Output')
        plt.xlabel('Timing')
        plt.ylabel('-log(Amplitude)')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig('fig/ex_eval_target_output.png')
        plt.savefig('fig/ex_eval_target_output.eps')
        plt.close()

    def stop_session(self):
        with open('loss_over_epochs_ex.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['stopped', 'stopped', 'stopped', 'stopped', 'stopped', 'stopped', 'stopped', 'stopped', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


    def start_session(self):
        with open('loss_over_epochs_ex.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['started', 'started', 'started', 'started', 'started', 'started', 'started', 'started', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])



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


if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    misc.set_fig()
    main(sys.argv[1])
