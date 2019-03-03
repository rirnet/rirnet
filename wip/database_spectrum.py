import csv
import random

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import acoustics as ac
import scipy as sp


def generate_pos_in_rect(x, y, z, n_pos):
    """
    Generates and returns n_pos positions 3d positions that are guaranteed to be within the given rectangle and at
    least 0.5 units from surfaces. Assumes rectangles are [x,y,z] > [2.5,2.5,2.5].

    Returns list of np.arrays of 3d positions
    """

    return np.random.rand(n_pos, 3)*[x, y, z]*0.6+0.5


def get_absorption_by_index(abs_coeffs, i):
    coeffs = {'east': abs_coeffs['east'][i], 'west': abs_coeffs['west'][i], 'north': abs_coeffs['north'][i],
              'south': abs_coeffs['south'][i], 'floor': abs_coeffs['floor'][i], 'ceiling': abs_coeffs['ceiling'][i]}
    return coeffs


def generate_multiband_spectrum(x, y, z, n_mics, fs, max_order, abs_coeffs, n_fft):

    source_pos = generate_pos_in_rect(x, y, z, 1)[0]

    mic_pos = generate_pos_in_rect(x, y, z, n_mics)
    mic_array = pra.MicrophoneArray(mic_pos.T, fs=fs)

    multiband_spectrum_batch = []
    multiband_rir_batch = np.zeros([n_mics, fs//2])

    center_freqs = [125, 250, 500, 1000, 2000, 4000, 4000*np.sqrt(2)]
    for i in range(7):
        coeffs = get_absorption_by_index(abs_coeffs, i)
        room = pra.ShoeBox([x, y, z], fs=fs, max_order=max_order, absorption=coeffs)
        room.add_source(source_pos)
        room.add_microphone_array(mic_array)
        room.compute_rir()
        rir_batch = []

        for j, rir in enumerate(room.rir):
            rir = rir[0]
            if(i < 6):
                rir = ac.signal.octavepass(rir, center_freqs[i], fs, 1, order=8)
            else:
                rir = ac.signal.highpass(rir, center_freqs[i], fs, order=8)
            rir_batch.append(rir[:fs//2])

        multiband_rir_batch += np.array(rir_batch)

    for rir in multiband_rir_batch:
        f_bins, t_bins, stft = sp.signal.stft(rir, fs=fs, nfft=n_fft, nperseg=n_fft)
        print(t_bins)
        print(f_bins)
        multiband_spectrum_batch.append(np.abs(stft))

    return multiband_spectrum_batch


class Materials:

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
                self.materials[row[0]] = np.array(row[1:], float)

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

    def generate(self):
        absorption = {'floor': self.materials[random.choice(self.floor)],
                      'ceiling': self.materials[random.choice(self.ceiling)]}

        for wall in ['east', 'west', 'north', 'south']:
            absorption[wall] = self.materials[random.choice(self.wall)]

        return absorption


def generate_spectrum(x_max, y_max, z_max, batch, mat_gen, fs=2**14, max_order=80, n_fft=512):
    assert all(dim > 2.5 for dim in [x_max, y_max, z_max]), "All dimensions must be > 2.5"
    x, y, z = np.random.rand(3)*(np.array([x_max, y_max, z_max])-2.5)+2.5

    absorption = mat_gen.generate()

    spectrum = generate_multiband_spectrum(x, y, z, batch, fs, max_order, absorption, n_fft)
    return spectrum


materials = Materials('materials.csv', 'surfaces.csv')
plt.imshow(generate_spectrum(10, 5, 3, 1, materials, n_fft=128)[0])
plt.show()

