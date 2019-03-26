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


class MaterialEngine:

    def __init__(self, materials_csv_path, surfaces_csv_path):
        self.materials = {}
        self.wall = []
        self.floor = []
        self.ceiling = []
        self.add_materials_from_csv(materials_csv_path)
        self.add_materials_to_surfaces(surfaces_csv_path)

    def add_materials_from_csv(self, csv_path):
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                data = np.array(row[1:], float)
                assert len(data) is 7, '{} does not have the correct number of coefficients'.format(row[0])
                self.materials[row[0]] = data

    def add_materials_to_surfaces(self, csv_path):
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for material in reader:
                for surface in material[1:]:
                    if surface == 'wall':
                        self.wall.append(material[0])
                    if surface == 'ceiling':
                        self.ceiling.append(material[0])
                    if surface == 'floor':
                        self.floor.append(material[0])

    def random(self):
        absorption = {'floor': self.materials[random.choice(self.floor)],
                      'ceiling': self.materials[random.choice(self.ceiling)]}

        for wall in ['east', 'west', 'north', 'south']:
            absorption[wall] = self.materials[random.choice(self.wall)]

        return absorption

def generate_monoband_rir(x, y, z, mic_pos, source_pos, fs, max_order, abs_coeffs):
    absorption = rg.get_absorption_by_index(abs_coeffs, 3)
    room = pyroomacoustics.room.ShoeBox([x, y, z], fs, absorption=absorption, max_order=max_order)
    room.add_source(source_pos)
    room.add_microphone_array(pyroomacoustics.MicrophoneArray(mic_pos.T, fs=fs))
    room.compute_rir()
    rir = room.rir[0][0]
    ind_1st_nonzero = next((i for i, x in enumerate(rir) if x), None)
    rir = au.pad_to(rir[ind_1st_nonzero:], 2**15)
    return rir

def preprocess_peaks(signal, fs):
    mfcc = au.waveform_to_mfcc(signal, fs, 40)[1][:,:-1]
    delta_1, delta_2 = au.calculate_delta_features(mfcc)
    mean = np.load('/home/felix/rirnet/database/mean.npy').T
    std = np.load('/home/felix/rirnet/database/std.npy').T
    normalized = np.nan_to_num((np.array([mfcc, delta_1, delta_2]).T - mean)/std).T
    return torch.tensor(normalized).unsqueeze(0).float()
    
def main():
    net_timeconv, _ = misc.load_latest('/home/felix/rirnet/timeconv/models', 'net')
    net_peaks_ae, _ = misc.load_latest('/home/felix/rirnet/nanonet/models/16', 'autoencoder')
    net_peaks_ext, _ = misc.load_latest('/home/felix/rirnet/nanonet/models/16', 'extractor')

    x, y, z = 6, 9, 3
    mic_pos = rg.generate_pos_in_rect(x, y, z, 1)
    source_pos = rg.generate_pos_in_rect(x, y, z, 1)[0]
    fs_peaks = 44100
    fs_timeconv = 16384
    n_fft = 128
    

    sound_engine = SoundEngine('/home/felix/rirnet/audio/chamber/val', fs_peaks)
    material_engine = MaterialEngine('/home/felix/rirnet/wip/materials.csv', '/home/felix/rirnet/wip/surfaces.csv')
    abs_coeffs = material_engine.random()

    multiband_rir = rg.generate_multiband_rirs(x, y, z, mic_pos, source_pos, fs_timeconv, 60, abs_coeffs)[0]
    monoband_rir = generate_monoband_rir(x, y, z, mic_pos, source_pos, fs_peaks, 8, abs_coeffs)

    an_sig_peaks = sound_engine.random()
    an_sig_timeconv = au.resample(an_sig_peaks, fs_peaks, fs_timeconv)

    rev_sig_multi = au.convolve(multiband_rir, an_sig_timeconv)
    _, _, rev_sig_multi_spectrogram = sp.signal.stft(rev_sig_multi, fs=fs_timeconv, nfft=n_fft, nperseg=n_fft)
    input_timeconv = torch.from_numpy(-np.log(np.abs(rev_sig_multi_spectrogram))).unsqueeze(0).float()

    rev_sig_mono = au.pad_to(au.convolve(monoband_rir, an_sig_peaks), 2**16)
    input_peaks = preprocess_peaks(rev_sig_mono, fs_peaks)

    with torch.no_grad():
        output_timeconv = net_timeconv(input_timeconv).squeeze().numpy()
        output_peaks = net_peaks_ae(net_peaks_ext(input_peaks), decode=True).squeeze().numpy()
        plt.figure()
        plt.imshow(output_timeconv)
        plt.show()
    phase = np.exp(1j*np.random.uniform(low = -np.pi, high = np.pi, size = np.shape(output_timeconv)))
    _, output_timeconv = sp.signal.istft(output_timeconv*phase, fs_timeconv, nfft=n_fft, nperseg=n_fft)
    
    
    plt.subplot(221)
    plt.plot(output_timeconv)
    plt.subplot(222)
    rev_output = au.convolve(output_timeconv, an_sig_timeconv)
    plt.plot(rev_output/np.max(np.abs(rev_output)))
    #plt.scatter(output_peaks[0], output_peaks[1])
    plt.subplot(223)
    plt.plot(multiband_rir)
    plt.subplot(224)
    plt.plot(rev_sig_multi/np.max(np.abs(rev_sig_multi)))
    plt.show()

    au.save_wav('synthetic.wav', rev_output, fs_timeconv, True)
    au.save_wav('tru.wav', rev_sig_multi, fs_timeconv, True)
    
if __name__ == '__main__':
    main()
