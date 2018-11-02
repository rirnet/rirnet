import librosa
import matplotlib.pyplot as plt
import rirnet.acoustic_utils as au
import numpy as np
import scipy

def _spectrogram(y, n_fft=2048, hop_length=512, power=1):
    complex_spectrum = (librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length))
    amplitud = np.abs(complex_spectrum)
    S = amplitud**power
    P = np.angle(complex_spectrum)
    print(P)
    #plt.subplot(2,1,1)
    P = np.random.random(np.shape(P))*2*np.pi-np.pi
    #######THIS IS HOW YOU DO IT#########
    #y = librosa.core.istft(amplitud*np.exp(1j*P))
    #au.save_wav('donald.wav', y, sr, norm=True)
    return P, S, n_fft


def melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                   power=2.0, **kwargs):
    P, S, n_fft = _spectrogram(y=y, n_fft=n_fft, hop_length=hop_length,
                            power=power)
    # Build a Mel filter
    mel_basis = librosa.filters.mel(sr, n_fft, **kwargs)
    melspect = np.dot(mel_basis, S)

    return P, melspect

def i_melspectrogram(P, mfcc, sr, n_fft):
    dctm = librosa.filters.dct(n_mfcc, 128)
    mel_basis = librosa.filters.mel(sr, n_fft)
    mfcc_to_power = 10.**(np.dot(dctm.T, mfcc)/10.)
    S = np.dot(mel_basis.T, mfcc_to_power)
    plt.subplot(2,1,1)
    plt.plot(mfcc)

    #excitation = np.random.randn(458752)
    #E = librosa.stft(excitation)
    #E = E/np.abs(E)

    amplitud = np.sqrt(S)
    plt.plot(amplitud)
    plt.show()
    #y = librosa.core.istft(E*amplitud)
    y = librosa.core.istft(amplitud*np.exp(1j*P))
    return y


n_mfcc = 40
n_fft = 2048
dct_type = 2
norm = 'ortho'

y, sr = au.read_wav('../../audio/clap_00.wav')
P, melspect = melspectrogram(y=y, sr=sr)
S_db = librosa.core.power_to_db(melspect)
mfcc = scipy.fftpack.dct(S_db, axis=0, type=2, norm=norm)[:n_mfcc]
S_db = scipy.fftpack.idct(mfcc, n=n_mfcc, axis=0, type=2, norm=norm)
S = librosa.core.db_to_power(S_db)  #, ref=0.1)
y2 = i_melspectrogram(P, mfcc, sr, n_fft)
au.save_wav('donald.wav', y2, sr, norm=True)

phase_data, mfcc = au.waveform_to_mfcc(y, sr, n_mfcc)
y3 = au.mfcc_to_waveform(mfcc, sr, np.size(y))
au.save_wav('linkolin.wav', y3, sr, norm=True)

plt.subplot(3,1,1)
plt.plot(y3)
plt.title('linkolin')
plt.subplot(3,1,2)
plt.plot(y2)
plt.title('dramp')
plt.subplot(3,1,3)
plt.plot(y)
plt.title('chers vagina')
plt.show()
