#!/usr/bin/env python3

import argparse
import logging
import librosa
import sys
import torch

import numpy as np
import scipy as sp
import rirnet.misc as misc

from multiprocessing import Process, Queue
from rirnet.transforms import ToTensor
import matplotlib.pyplot as plt

class Model:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        sys.path.append(model_dir)
        self.net, self.epoch = misc.load_latest(model_dir, 'net')
        self._args = self.net.args()
        #use_cuda = not self._args.no_cuda and torch.cuda.is_available()
        #self.device = torch.device("cuda" if use_cuda else "cpu")
        #self.net.to(self.device)
        #self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    def forward(self, spectrogram):
        nw_input = torch.from_numpy(-np.log(np.abs(spectrogram)))
        with torch.no_grad():
            nw_output = self.net(nw_input.unsqueeze(0)).squeeze().numpy()
        return nw_output

def multiProc():
    model_dir = '/home/felix/rirnet/foorir_felix/models'
    model = Model(model_dir)

    def f(indata, q, model):
        indata = indata[:,0]
        fs = 16384
        indata = librosa.core.resample(indata, 44100, fs)
        n_fft = 128
        _,_,spectrogram = sp.signal.stft(indata, fs=fs, nfft=n_fft, nperseg=n_fft)
        spectrogram = spectrogram[:,:225]
        rir = model.forward(spectrogram)
        q.put(rir)
        print('left result')

    def clean(processes):
        for i in range(len(processes)):
            if processes and not processes[0].is_alive():
                p = processes.pop(0)
                p.terminate()
        return processes

    try:
        import sounddevice as sd
        q = Queue()
        q.put([[0], [0]])
        processes = []
        plt.ion()
        callback_status = sd.CallbackFlags()

        def callback(indata, outdata, frames, time, status):
            rir = q.get()
            print(rir)
            plt.imshow(rir)
            plt.draw()
            plt.pause(0.01)
            global callback_status
            p = Process(target=f, args=(indata,q,model,))
            processes.append(p)
            p.start()
            clean(processes)

        with sd.Stream(callback=callback, channels = 1, samplerate = 44100, blocksize = 44100):
            print("#" * 80)
            print("press Return to quit")
            print("#" * 80)
            input()

        if callback_status:
            logging.warning(str(callback_status))
    except BaseException as e:
        # This avoids printing the traceback, especially if Ctrl-C is used.
        raise SystemExit(str(e))

if __name__ == "__main__":
    multiProc()
