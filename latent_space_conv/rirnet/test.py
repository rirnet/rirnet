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
import rirnet.acoustic_utils as au

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
        list_epochs_extractor = [ x for x in list_epochs_extractor if "opt_extractor" not in x ]
        list_epochs_autoenc = [ x for x in list_epochs_autoenc if "opt_autoenc" not in x ]
        epoch = max([int(e.split('_')[0]) for e in list_epochs_autoenc])
        self.autoenc.load_state_dict(torch.load(os.path.join(model_dir, '{}_autoenc.pth'.format(str(epoch)))))

        epoch = max([int(e.split('_')[0]) for e in list_epochs_extractor])
        self.extractor.load_state_dict(torch.load(os.path.join(model_dir, '{}_extractor.pth'.format(str(epoch)))))

        self.epoch = epoch
        data_transform = self.extractor.data_transform()
        target_transform = self.extractor.target_transform()

        eval_db = RirnetDatabase(is_training = False, args = self.extractor_args, data_transform = data_transform, target_transform = target_transform)
        self.eval_loader = torch.utils.data.DataLoader(eval_db, batch_size=self.extractor_args.batch_size, shuffle=True, **self.kwargs)

    def test(self):
        audio_list = glob('../../audio/mousepatrol/m.wav')
        wav = au.normalize(au.read_wav(audio_list[0], 44100)[0])

        self.extractor.eval()
        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.eval_loader):
                torch.cuda.empty_cache()
                source, target = source.to(self.device), target.to(self.device)

                latent_source = self.extractor(source)
                output = self.autoenc(latent_source, encode=False, decode=True)

                out = output[0].cpu().numpy().T
                tar = target[0].cpu().numpy().T

                rir_out = self.reconstruct_rir(out)
                rir_tar = self.reconstruct_rir(tar)
                plt.subplot(2,1,1)
                plt.plot(rir_out)
                #plt.subplot(2,1,2)
                #plt.plot(rir_tar)
                yo = au.convolve(wav, rir_out)
                plt.subplot(2,1,2)
                plt.plot(au.normalize(yo))
                plt.show()
                yt = au.convolve(wav, rir_tar)
                au.save_wav('yo{}'.format(batch_idx), yo, 44100, True)
                au.save_wav('yt{}'.format(batch_idx), yt, 44100, True)

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

def main(model_dir):
    model = Model(model_dir)
    model.test()

if __name__ == '__main__':
    main(sys.argv[1])

