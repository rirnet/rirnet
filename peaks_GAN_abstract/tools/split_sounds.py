import rirnet.acoustic_utils as au
import librosa
import numpy as np
import matplotlib.pyplot as plt

out_path = '../audio/chamber/'
in_path = '../audio/chamber/full/full.wav'
rate = 44100
data, rate = au.read_wav(in_path, rate=rate)
sound_starts = librosa.onset.onset_detect(data, sr=rate, backtrack=True)*512
for i, start in enumerate(sound_starts):
    stop = start + au.next_power_of_two(int(rate/4))
    energy = np.sum(np.abs(data[stop-100:stop]))
    if energy < 0.01:
        au.save_wav(out_path + 'ch_{}.wav'.format(i), data[start:stop], rate)
