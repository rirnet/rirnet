#from __future__ import print_function
from rirnet_database import RirnetDatabase
#from datetime import datetime, timedelta
#from torch.autograd import Variable
#from importlib import import_module
#from pyroomacoustics.utilities import fractional_delay
#from scipy.optimize import curve_fit

import sys
import torch
import os
#import csv

#import torch.nn.functional as F
#import torch.optim as optim
import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
import rirnet.acoustic_utils as au
import rirnet.misc as misc

class Model:
    def __init__(self, model_dir):
        self.model_dir = model_dir

        self.extractor, _ = misc.load_latest(model_dir, 'extractor')
        self.autoencoder, _ = misc.load_latest(model_dir, 'autoencoder')

        self.extractor_args = self.extractor.args()

        use_cuda = not self.extractor_args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.extractor.to(self.device)
        self.autoencoder.to(self.device)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        data_transform = self.extractor.data_transform()
        target_transform = self.extractor.target_transform()

        eval_db = RirnetDatabase(is_training = False, args = self.extractor_args, data_transform = data_transform, target_transform = target_transform)
        self.eval_loader = torch.utils.data.DataLoader(eval_db, batch_size=self.extractor_args.batch_size, shuffle=True, **self.kwargs)

        self.audio_anechoic, self.fs = au.read_wav('../../audio/harvard/male.wav')

    def run(self):
        self.extractor.eval()
        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.eval_loader):
                source, target = source.to(self.device), target[0].numpy()

                latent_source = self.extractor(source)
                output = self.autoencoder(latent_source, encode=False, decode=True)[0].cpu().numpy()

                filled_times_output, filled_alphas_output = misc.fill_peaks(output[0,:], output[1,:], 10)
                filled_times_target, filled_alphas_target = misc.fill_peaks(target[0,:], target[1,:], 10)

                output_rir = misc.reconstruct_rir(filled_times_output, filled_alphas_output)
                output_rir_conv = misc.reconstruct_rir_conv(filled_times_output, filled_alphas_output)
                target_rir = misc.reconstruct_rir(filled_times_target, filled_alphas_target)


                plt.plot(output_rir)
                plt.plot(output_rir_conv)
                plt.show()

                rev_signal_output = au.convolve(self.audio_anechoic, output_rir)
                rev_signal_target = au.convolve(self.audio_anechoic, target_rir)

                au.save_wav('output.wav', rev_signal_output, self.fs, 1)
                au.save_wav('target.wav', rev_signal_target, self.fs, 1)

                au.play_file('output.wav')
                au.play_file('target.wav')

def main():
    model = Model('../models')
    model.run()


if __name__ == '__main__':
    main()

