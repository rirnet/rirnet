import numpy as np
import rirnet.acoustic_utils as au
import matplotlib.pyplot as plt
import librosa

n_mfcc = 40
n_fft = 2048
hop_length = 512

x, sr = au.read_wav('../../audio/claps.wav')
P, mfcc = au.waveform_to_mfcc(x, sr, n_mfcc)
y = au.mfcc_to_waveform(mfcc, sr, np.size(x))
y2 = y


D = librosa.core.stft(y2, n_fft=n_fft, hop_length=hop_length)
amp = np.abs(D)
P = np.angle(D)

for i in range(10000):
    D = librosa.core.stft(y2, n_fft=n_fft, hop_length=hop_length)
    D = amp*np.exp(1j*np.angle(D))
    y2 = librosa.core.istft(D)

plt.subplot(2,1,1)
plt.plot(y)
plt.subplot(2,1,2)
plt.plot(y2)
plt.show()

au.save_wav('orig.wav', y, sr, True)
au.save_wav('iter.wav', y2, sr, True)
