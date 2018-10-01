import rirnet.acoustic_utils as au
import librosa

path_saverz = '../../audio/'

rate = 44100 
data, rate = au.read_wav('../../audio/claps.wav', rate=rate)
sound_starts = librosa.onset.onset_detect(data, sr=rate, backtrack=True)*512

for i, start in enumerate(sound_starts):
    stop = start + au.next_power_of_two(int(rate/8))
    au.save_wav(path_saverz + 'clap_%02d.wav' %i, data[start:stop], rate)


