import rirnet.acoustic_utils as au
import librosa
import numpy as np
import matplotlib.pyplot as plt

out_path = '../../audio/livingroom/test/'
in_path = '../../audio/livingroom/full/mario.wav'
rate = 44100
data, rate = au.read_wav(in_path, rate=rate)
sound_starts = librosa.onset.onset_detect(data, sr=rate, backtrack=True)*512
plt.plot(data)
plt.vlines(sound_starts, -1, 1)
#plt.show()
for i, start in enumerate(sound_starts):
    stop = start + au.next_power_of_two(int(rate/4))
    plt.plot(data[start:stop])
    energy = np.sum(np.abs(data[stop-1000:stop]))
    plt.title(energy)
    #plt.show()
    if energy < 8:
        au.save_wav(out_path + 'mario_{}.wav'.format(i), data[start:stop], rate)
