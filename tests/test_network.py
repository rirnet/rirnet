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

        self.autoenc.load_state_dict(torch.load('2567_autoenc.pth'))
        self.extractor.load_state_dict(torch.load('13_extractor.pth'))

        data_transform = self.extractor.data_transform()
        target_transform = self.extractor.target_transform()

        eval_db = RirnetDatabase(is_training = False, args = self.extractor_args, data_transform = data_transform, target_transform = target_transform)
        self.eval_loader = torch.utils.data.DataLoader(eval_db, batch_size=self.extractor_args.batch_size, shuffle=True, **self.kwargs)

        self.audio_anechoic, self.fs = au.read_wav('hardvard.wav')

    def run(self):
        self.extractor.eval()
        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.eval_loader):
                source, target = source.to(self.device), target[0].numpy()

                latent_source = self.extractor(source)
                output = self.autoenc(latent_source, encode=False, decode=True)[0].cpu().numpy()
                times = output[0,:]
                alphas = output[1,:]

                filled_times, filled_alphas = self.fill_peaks(times, alphas)

                output_rir = self.reconstruct_rir(filled_times, filled_alphas)
                target_rir = self.reconstruct_rir(target[0,:], target[1,:])

                rev_signal_output = au.convolve(self.audio_anechoic, output_rir)
                rev_signal_target = au.convolve(self.audio_anechoic, target_rir)

                au.save_wav('output.wav', rev_signal_output, self.fs, 1)
                au.save_wav('target.wav', rev_signal_target, self.fs, 1)


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
        n_in_bins, bin_edges = np.histogram(times, 25)
        ind_max_bin = np.argmax(n_in_bins)
        time_max_bin = bin_edges[ind_max_bin]
        time_simulation_limit = times[np.argmin(abs(times-time_max_bin))]
        n_early_reflections = np.sum(times < time_simulation_limit)
        area_early = time_simulation_limit*n_in_bins[ind_max_bin]/2
        density_early = n_early_reflections/area_early
        area_late = n_in_bins[ind_max_bin]/time_simulation_limit*max(times)*(max(times)-time_simulation_limit)/2
        n_late_reflections = int(area_late*density_early)
        
        new_times = np.random.triangular(time_simulation_limit, max(times), max(times), n_late_reflections)
        
        std = np.std(alphas-func(times, *coeff))
        new_deviations = np.random.normal(scale=std, size=n_late_reflections)
        new_alphas = func(new_times, *coeff) + new_deviations
        
        filled_times = np.concatenate((times, new_times))
        filled_alphas = np.concatenate((alphas, new_alphas))
        return filled_times, filled_alphas


def main():
    model = Model('../pointnet/models')
    model.run()

if __name__ == '__main__':
    main()

