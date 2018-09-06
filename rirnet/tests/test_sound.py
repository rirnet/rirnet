import sys
#sys.path.append('../')
import rirnet.roomgen as rg
import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

room = rg.generate(max_order = 10, min_side=5, max_side=25, min_height=2, max_height=3, n_mics=1, fs=44100, absorption = 0.5)
room.compute_rir()
rate, x = sp.io.wavfile.read('/home/felix/Downloads/drums.wav')
h = room.rir[0][0]

y = sp.signal.fftconvolve(x,h)
y /= (2*np.max(np.abs(y)))
sp.io.wavfile.write('drums_convolved.wav', rate, y)
