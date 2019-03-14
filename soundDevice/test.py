#!/usr/bin/env python3
import argparse
import logging
import librosa

import numpy as np
import scipy as sp

from multiprocessing import Process, Queue
import matplotlib.pyplot as plt

def multiProc():
    def f(indata, q):
        indata = indata[:,0]
        indata = librosa.core.resample(indata, 44100, 16000)
        fs = 16000
        nfft = 512
        n_fft = 128
        _,_,spectrogram = sp.signal.stft(indata, fs=fs, nfft=n_fft, nperseg=n_fft)
        q.put(np.abs(spectrogram))

    def clean(processes):
        for i in range(len(processes)):
            if processes and not processes[0].is_alive():
                p = processes.pop(0)
                p.terminate()
        return processes


    try:
        import sounddevice as sd
        q = Queue()
        processes = []
        plt.ion()
        callback_status = sd.CallbackFlags()

        def callback(indata, outdata, frames, time, status):
            global callback_status
            #callback_status |= status
            p = Process(target=f, args=(indata,q,))
            processes.append(p)
            p.start()
            plt.imshow(q.get())
            plt.show()
            plt.pause(0.0001)
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
