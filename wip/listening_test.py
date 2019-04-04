import rirnet.roomgen as rg
import rirnet.acoustic_utils as au
import rirnet.misc as misc
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import torch
import glob
import os
import csv
import random
import pyroomacoustics

class SoundEngine:

    def __init__(self, audio_folder_path, fs):
        self.audio_list = self.load_audio(audio_folder_path, fs)
        
    def random(self):
        return random.choice(self.audio_list)

    def load_audio(self, audio_folder_path, fs):
        audio_list = []
        audio_filename_list = glob.glob(os.path.join(audio_folder_path, '*.wav'))
        for audio_filename in audio_filename_list:
            audio_file_path = os.path.join(audio_folder_path, audio_filename)
            audio = au.normalize(au.read_wav(audio_file_path, fs)[0])
            audio_list.append(audio)
        return audio_list


def main():
    net, _ = misc.load_latest('/home/eriklarsson/rirnet/timeconv/models', 'net')
    
    fs = 16384
    n_fft = 128

    sound_engine = SoundEngine('/home/eriklarsson/rirnet/audio/chamber/val', 44100)
    anechoic_signal = sound_engine.random()

    rir_real, _ = au.read_wav('/home/eriklarsson/rirnet/audio/rirs/lecture.wav', 44100)
    rir_real = rir_real[:44100//2]
    rev_real = au.resample(au.convolve(rir_real, anechoic_signal), 44100, fs)
    
    _, _, rev_spectrogram = sp.signal.stft(rev_real, fs=fs, nfft=n_fft, nperseg=n_fft)
    net_input = torch.from_numpy(-np.log(np.abs(rev_spectrogram))).unsqueeze(0).float()

    with torch.no_grad():
        net_output = net(net_input).squeeze().numpy()
    phase = np.exp(1j*np.random.uniform(low = -np.pi, high = np.pi, size = np.shape(net_output)))
    _, rir_net = sp.signal.istft(net_output*phase, fs, nfft=n_fft, nperseg=n_fft)
    plt.imshow(net_output)
    plt.show()
    rir_net = au.resample(rir_net, fs, 44100)

    anechoic_test, _ = au.read_wav('/home/eriklarsson/rirnet/audio/harvard/male.wav')
    anechoic_test = anechoic_test[250000:400000,0]

    rev_real_test = au.convolve(rir_real, anechoic_test)
    rev_net_test = au.convolve(rir_net, anechoic_test)
    
    au.save_wav('real.wav', rev_real_test, 44100, True)
    au.save_wav('net.wav', rev_net_test, 44100, True)
    
if __name__ == '__main__':
    main()
