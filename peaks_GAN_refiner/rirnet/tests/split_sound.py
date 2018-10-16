import matplotlib.pyplot as plt
import rirnet.acoustic_utils as au
import librosa
import os

audio_folder_path = os.path.abspath('../../audio')

data, rate = au.read_wav(os.path.join(audio_folder_path, 'macclaps.wav'))
onsets = librosa.onset.onset_detect(data, sr=rate, backtrack=True)*512

plt.plot(data)

for i, onset in enumerate(onsets):
    offset = onset + 2**15
    plt.axvline(onset, color='g')
    plt.axvline(offset, color='r')
    file_path = os.path.join(audio_folder_path, 'macclap_{}.wav'.format(i))
    sound = data[onset:offset]
    au.save_wav(file_path, sound, rate, norm=1)

plt.show()
