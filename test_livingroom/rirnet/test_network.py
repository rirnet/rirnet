from __future__ import print_function
import sys
import torch
from rirnet_database import RirnetDatabase
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
from datetime import datetime, timedelta
from torch.autograd import Variable
import numpy as np
from glob import glob
from importlib import import_module
from pyroomacoustics.utilities import fractional_delay
from scipy.optimize import curve_fit
import rirnet.acoustic_utils as au


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

        self.autoenc.load_state_dict(torch.load('../models/323_autoenc.pth'))
        self.extractor.load_state_dict(torch.load('../models/138_extractor.pth'))

        data_transform = self.extractor.data_transform()

        self.audio_anechoic, self.fs = au.read_wav('../../audio/stuf/hardvard.wav')

        eval_db = RirnetDatabase(args = self.extractor_args, data_transform = data_transform)
        self.eval_loader = torch.utils.data.DataLoader(eval_db, batch_size=self.extractor_args.batch_size, shuffle=True, **self.kwargs)


    def run(self):
        self.extractor.eval()
        with torch.no_grad():
            for batch_idx, source in enumerate(self.eval_loader):
                source = source.to(self.device)

                latent_source = self.extractor(source)
                output = self.autoenc(latent_source, encode=False, decode=True)[0].cpu().numpy()

                filled_times_output, filled_alphas_output = self.fill_peaks(output[0,:], output[1,:])
                
                output_rir = self.reconstruct_rir(filled_times_output, filled_alphas_output)
               
                rev_signal_output = au.convolve(self.audio_anechoic, output_rir)

                plt.subplot(313)
                plt.plot(output_rir)
                plt.show()

                au.save_wav('output.wav', rev_signal_output, self.fs, 1)

                au.play_file('output.wav')


    def reconstruct_rir(self, time, alpha):
        '''
        Construct a room impulse response from the negative log version that the
        networks use. Uses the sinc function to approximate a physical impulse response.
        A random subset of the reflections are negated (helps with removing artefacts).
        Adopted from pyroomacoustics.
        '''
        fdl = 81
        fdl2 = (fdl-1) // 2
        time = (time.astype('double')+1)*1024
        alpha = np.exp(-alpha).astype('double')
        signs = np.random.randint(0,2, len(alpha))*2-1
        alpha *= signs
        ir = np.arange(np.ceil((1.05*time.max()) + fdl))*0
        for i in range(time.shape[0]):
            time_ip = int(np.round(time[i]))
            time_fp = time[i] - time_ip
            ir[time_ip-fdl2:time_ip+fdl2+1] += alpha[i]*fractional_delay(time_fp)
        start_ind = min(np.where(ir != 0)[0])
        ir = ir[start_ind:]
        return ir

    def fill_peaks(self, times, alphas):
        '''
        Approximate amplitudes and times for late reverb as simulations fails to 
        decrease time spacings indefinitely. The amplitudes are assumed to follow
        the decay defined in func method.
        '''
        def func(t, a, b, c, d):
            return a*np.log(b*(t+c))+d

        coeff, _ = curve_fit(func, times, alphas)
        rir_max_time = 1/coeff[1]*np.exp((6-coeff[3])/coeff[0])-coeff[2]
        rir_max_time = np.min((rir_max_time, max(times)*2))
        t = np.linspace(0, rir_max_time, 1000)
        n_in_bins, bin_edges = np.histogram(times, 25)
        ind_max_bin = np.argmax(n_in_bins)
        time_max_bin = bin_edges[ind_max_bin]
        time_simulation_limit = times[np.argmin(abs(times-time_max_bin))]
        n_early_reflections = np.sum(times < time_simulation_limit)
        area_early = time_simulation_limit*n_in_bins[ind_max_bin]/2
        density_early = n_early_reflections/area_early
        area_late = n_in_bins[ind_max_bin]/time_simulation_limit*rir_max_time*(rir_max_time-time_simulation_limit)/2
        n_late_reflections = int(area_late*density_early)
        
        new_times = np.random.triangular(time_simulation_limit, rir_max_time, rir_max_time, n_late_reflections)
         
        std = np.std(alphas-func(times, *coeff))
        new_deviations = np.random.normal(scale=std, size=n_late_reflections)
        new_alphas = func(new_times, *coeff) + new_deviations

        filled_times = np.concatenate((times, new_times))
        filled_alphas = np.concatenate((alphas, new_alphas))
        plt.subplot(311)
        plt.plot(times, alphas, '.')
        plt.plot(new_times, new_alphas, '.')
        plt.plot(t, func(t, *coeff))
        plt.subplot(312)
        plt.plot(bin_edges[:-1], n_in_bins)
        return filled_times, filled_alphas


def main():
    model = Model('../models')
    model.run()

if __name__ == '__main__':
    main()

