import rirnet.acoustic_utils as au
import librosa
import matplotlib.pyplot as plt

path_saverz = '../../audio/'

rate = 44100
data, rate = au.read_wav('../../audio/sticks.wav', rate=rate)
sound_starts = librosa.onset.onset_detect(data, sr=rate, backtrack=True)*512

for i, start in enumerate(sound_starts):
    stop = start + au.next_power_of_two(int(rate/8))
    plt.plot(data[start:stop])
    plt.title('sticks_%02d.wav' %i)
    plt.show()
    au.save_wav(path_saverz + 'sticks_%02d.wav' %i, data[start:stop], rate)
