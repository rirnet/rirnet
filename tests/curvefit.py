from __future__ import print_function
import time
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.optimize import curve_fit
import rirnet.roomgen as rg

def compute_peaks(room):
    if room.visibility is None:
        room.image_source_model()
    peaks = []
    for m, mic in enumerate(room.mic_array.R.T):
        distances = room.sources[0].distance(mic)
        times = distances/343.0*room.fs
        alphas = room.sources[0].damping / (4.*np.pi*distances)
        slice = tuple(np.where(room.visibility[0][m] == 1))
        alphas = alphas[slice]
        times = times[slice]
        peaks.append([times - min(times), alphas])
    return peaks

fs, audio_anechoic = wavfile.read('../audio/chamber/train/ch_100.wav')

room_dim = [5, 4, 6]
shoebox = rg.generate(3, 7, 2, 3, 1, fs=16000, max_order=5, min_abs=0.01, max_abs=0.5)

# run sim
shoebox.compute_rir()
peaks = compute_peaks(shoebox)

def func(x,a,b,c):
    return a**(-(x+b)**c)

peaks = np.squeeze(peaks)

# sort peaks
idx = np.argsort(peaks[0])
peaks = np.array(peaks)[:,idx]

x = np.squeeze(peaks[0])
y = np.squeeze(peaks[1])
coeff, _ = curve_fit(func, x[:256]/1000, y[:256])

plt.plot(x, y, 'x')
plt.plot(x, (func(x/1000, *coeff)))
plt.show()

