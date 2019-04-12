import rirnet.roomgen as rg
import rirnet.acoustic_utils as au
import rirnet.misc as misc
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import datetime
import librosa.output
import torch
import glob
import os
import csv
import random
import pyroomacoustics
import sys

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
        info = []
        floor_material = random.choice(self.floor)
        ceiling_material = random.choice(self.ceiling)
        info.append(floor_material)
        info.append(ceiling_material)
        absorption = {'floor': self.materials[floor_material],
                      'ceiling': self.materials[ceiling_material]}

        for wall in ['east', 'west', 'north', 'south']:
            wall_material = random.choice(self.wall)
            info.append(wall_material)
            absorption[wall] = self.materials[wall_material]

        return absorption, info

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

    net_timeconv, _ = misc.load_latest('/home/felix/rirnet/timeconv_felix/models', 'net')
    net_peaks_ae, _ = misc.load_latest('/home/felix/rirnet/nanonet/models/16', 'autoencoder')
    net_peaks_ext, _ = misc.load_latest('/home/felix/rirnet/nanonet/models/16', 'extractor')

    fs_peaks = 44100
    fs_timeconv = 44100
    n_fft = 128

    sound_engine = SoundEngine('/home/felix/rirnet/audio/chamber/val', fs_peaks)
    material_engine = MaterialEngine('/home/felix/rirnet/wip/materials.csv', '/home/felix/rirnet/wip/surfaces.csv')
    for i in range(15):
        x = np.random.uniform(3, 15)
        y = np.random.uniform(3, 15)
        z = np.random.uniform(2, 4)
        mic_pos = rg.generate_pos_in_rect(x, y, z, 1)
        source_pos = rg.generate_pos_in_rect(x, y, z, 1)[0]

        abs_coeffs, info = material_engine.random()
        info.append(str(x))
        info.append(str(y))
        info.append(str(z))

        with open('cases_synthetic/info_{}.txt'.format(i), "w") as text_file:
            for elem in info:
                text_file.write(elem + '\n')

        multiband_rir = rg.generate_multiband_rirs(x, y, z, mic_pos, source_pos, fs_timeconv, 60, abs_coeffs)[0]
        monoband_rir = generate_monoband_rir(x, y, z, mic_pos, source_pos, fs_peaks, 8, abs_coeffs)

        an_sig_peaks = sound_engine.random()
        an_sig_timeconv = au.resample(an_sig_peaks, 44100, fs_timeconv)

        rev_sig_multi = sp.signal.fftconvolve(multiband_rir, an_sig_timeconv)
        _, _, rev_sig_multi_spectrogram = sp.signal.stft(rev_sig_multi, fs=fs_timeconv, nfft=n_fft, nperseg=n_fft)
        _, _, multiband_rir_spectrogram = sp.signal.stft(multiband_rir, fs=fs_timeconv, nfft=n_fft, nperseg=n_fft)
        input_timeconv = torch.from_numpy(-np.log(np.abs(rev_sig_multi_spectrogram))).unsqueeze(0).float()

        multiband_rir_spectrogram = np.abs(multiband_rir_spectrogram)
        with torch.no_grad():
            output_timeconv = net_timeconv(input_timeconv).squeeze().numpy()

        print(np.max(output_timeconv))
        output_timeconv /= np.max(output_timeconv)
        print(np.max(multiband_rir_spectrogram))
        multiband_rir_spectrogram /= np.max(multiband_rir_spectrogram)

        plt.subplot(221)
        plt.imshow(np.abs(output_timeconv))
        plt.subplot(222)
        plt.imshow(np.abs(multiband_rir_spectrogram))

        phase = np.exp(1j*np.random.uniform(low=-np.pi, high=np.pi, size=np.shape(output_timeconv)))
        _, output_timeconv = sp.signal.istft(np.abs(output_timeconv)*phase, fs=44100, nperseg=128, noverlap = 64)
        phase = np.exp(1j*np.random.uniform(low=-np.pi, high=np.pi, size=np.shape(multiband_rir_spectrogram)))
        _, multiband_rir = sp.signal.istft(np.abs(multiband_rir_spectrogram)*phase, fs=44100, nperseg=128, noverlap = 64)

        plt.subplot(223)
        plt.plot(output_timeconv)
        plt.subplot(224)
        plt.plot(multiband_rir)

        sounds = glob.glob("/home/felix/rirnet/audio/harvard/cases/*.wav")
        random_sound1, random_sound2 = random.sample(set(sounds), 2)
        test_sound,_ = librosa.core.load(random_sound1, sr=44100)
        ref_sound,_ = librosa.core.load(random_sound2, sr=44100)

        test_output = sp.signal.fftconvolve(test_sound, output_timeconv)
        test_output /= np.max(np.abs(test_output))
        test_output *= 2147483647
        test_output = np.asarray(test_output, dtype=np.int32)

        test_input = sp.signal.fftconvolve(test_sound, multiband_rir)
        test_input /= np.max(np.abs(test_input))
        test_input *= 2147483647
        test_input = np.asarray(test_input, dtype=np.int32)

        ref_sound_rev = sp.signal.fftconvolve(ref_sound, multiband_rir)
        ref_sound_rev /= np.max(np.abs(ref_sound_rev))
        ref_sound_rev *= 2147483647
        ref_sound_rev = np.asarray(ref_sound_rev, dtype=np.int32)

        #test_output = au.resample(test_output, fs_timeconv, 44100)
        #test_input = au.resample(test_input, fs_timeconv, 44100)
        plt.savefig('spects_{}.png'.format(i))
        plt.close()
        sp.io.wavfile.write('cases_synthetic/test_output_{}.wav'.format(i), fs_timeconv, test_output)
        sp.io.wavfile.write('cases_synthetic/test_input_{}.wav'.format(i), fs_timeconv, test_input)
        sp.io.wavfile.write('cases_synthetic/test_reference_{}.wav'.format(i), fs_timeconv, ref_sound_rev)

if __name__ == '__main__':
    main()
