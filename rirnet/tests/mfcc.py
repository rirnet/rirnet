import rirnet.acoustic_utils as au
import rirnet.roomgen as rg
import pyroomacoustics as pra
import numpy as np

c=340
max_order=11
max_side=30
fs=44100

x, rate = au.read_wav('../../../Downloads/drums.wav')

room = rg.generate(max_order = max_order, min_side=10, max_side=max_side, min_height=2, max_height=3, n_mics=1, fs=44100, absorption = 0.1)
room.compute_rir()

x = au.normalize(x)

h = room.rir[0][0]
waveform_length = au.next_power_of_two(np.size(h))
h = np.pad(h, (0, waveform_length-np.size(h)), 'edge')

mfcc = au.waveform_to_mfcc(h, rate, n_mfcc = 30)
recon = au.mfcc_to_waveform(mfcc, rate, waveform_length)

output = au.convolve(x,h)
output = au.normalize(output)
output_mfcc = au.convolve(x,recon)
output_mfcc = au.normalize(output_mfcc)

au.save_wav('test_no_mfcc.wav', output, rate)
au.save_wav('test_mfcc.wav', output_mfcc, rate)

au.play_file('test_mfcc.wav')
au.play_file('test_no_mfcc.wav')
