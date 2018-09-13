import numpy as np
import librosa
import scipy as sp
import os


def mfcc_to_waveform(mfcc, rate, waveform_length):
    n_mel = 128
    n_fft = 2048
    n_mfcc = mfcc.shape[0]

    dctm = librosa.filters.dct(n_mfcc, n_mel)
    mel_basis = librosa.filters.mel(rate, n_fft)

    bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis),
    axis=0))

    dctmmfcc = 10.**(np.dot(dctm.T, mfcc)/10.)
    recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T, dctmmfcc)

    excitation = np.random.randn(waveform_length)
    E = librosa.stft(excitation)
    recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))
    return recon


def waveform_to_mfcc(waveform, rate, n_mfcc):
    return librosa.feature.mfcc(waveform, sr=rate, n_mfcc=n_mfcc)


def convolve(x, h):
    return sp.signal.fftconvolve(x,h)


def next_power_of_two(x):
    return 1 if x == 0 else 2**(x-1).bit_length()


def pad_to(x, length):
    return np.pad(x, (0, length - np.size(x)), 'edge')


def normalize(x):
    return x / np.max(np.abs(x))


def resample(waveform, rate, target_rate):
    return librosa.core.resample(waveform, rate, target_rate)


def play_file(path):
    os.system('cvlc '+ path +' --play-and-exit')


def save_wav(path, data, rate):
    librosa.output.write_wav(path, data, rate)


def read_wav(path, rate=None):
    data, rate = librosa.load(path, sr=rate)
    return data, rate
