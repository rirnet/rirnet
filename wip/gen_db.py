import multiprocessing as mp
import numpy as np
import rirnet.roomgen as rg
import rirnet.acoustic_utils as au
import scipy as sp
import matplotlib.pyplot as plt
from queue import Empty

import time
import csv
import random
import os
import glob

class SoundEngine:
    def __init__(self, audio_folder_path):
        self.audio_list = self.load_audio(audio_folder_path)

    def random(self):
        return random.choice(self.audio_list)

    def load_audio(self, audio_folder_path):
        audio_list = []
        audio_filename_list = glob.glob(os.path.join(audio_folder_path, '*.wav'))
        for audio_filename in audio_filename_list:
            audio_file_path = os.path.join(audio_folder_path, audio_filename)
            audio = au.normalize(au.read_wav(audio_file_path)[0])
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

    def random(self):
        absorption = {'floor': self.materials[random.choice(self.floor)],
                      'ceiling': self.materials[random.choice(self.ceiling)]}

        for wall in ['east', 'west', 'north', 'south']:
            absorption[wall] = self.materials[random.choice(self.wall)]

        return absorption


def generate_spectrograms(queue, args):
    x_max, y_max, z_max, n_mics, n_fft, max_order, fs, material_engine, sound_engine = args
    np.random.seed()
    x, y, z = np.random.rand(3)*(np.array([x_max, y_max, z_max])-2.5)+2.5
    absorption = material_engine.random()
    an_sig = sound_engine.random()

    rir_list = rg.generate_multiband_rirs(x, y, z, n_mics, fs, max_order, absorption, n_fft)
    rev_sig_spectrograms = []
    rir_spectrograms = []
    for rir in rir_list:
        rev_sig = au.convolve(rir, an_sig)
        _,_,rir_spectrogram = sp.signal.stft(rir, fs=fs, nfft=n_fft, nperseg=n_fft)
        _,_,rev_sig_spectrogram = sp.signal.stft(rev_sig, fs=fs, nfft=n_fft, nperseg=n_fft)
        rev_sig_spectrograms.append(np.abs(rev_sig_spectrogram))
        rir_spectrograms.append(np.abs(rir_spectrogram))

    queue.put([rev_sig_spectrograms, rir_spectrograms])


def clean(processes):
    if processes and not processes[0].is_alive():
        p = processes.pop(0)
        p.terminate()
    return processes


def init_queue(n_proc, args):
    queue = mp.Queue()
    processes = [mp.Process(target=generate_spectrograms, args=(queue, args,)) for x in range(n_proc)]
    for p in processes:
        p.start()
    return queue, processes


def main():
    n_proc = 6
    x_max = 10
    y_max = 10
    z_max = 3
    n_mics = 10
    n_fft = 128
    max_order = 80
    fs = 16000
    material_engine = MaterialEngine('materials.csv', 'surfaces.csv')
    sound_engine = SoundEngine('/home/felix/rirnet/audio/chamber/train')
    args = [x_max, y_max, z_max, n_mics, n_fft, max_order, fs,  material_engine, sound_engine]
    queue, processes = init_queue(n_proc, args)
    runs = 0

    with open('list.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        while(1):
            try:
                rev_sig_spectrograms, rir_spectrograms = queue.get(timeout=20)
                p = processes.pop(0)
                p.terminate()
                processes.append(mp.Process(target=generate_spectrograms, args=(queue, args,)))
                p = processes[-1]
                p.start()

                n_per_run = np.shape(rir_spectrograms)[0]
                for i in range(n_per_run):
                    rev_filename = '{}_rev.npy'.format(n_per_run*runs+i)
                    rir_filename = '{}_rir.npy'.format(n_per_run*runs+i)
                    np.save(rev_filename, rev_sig_spectrograms[i])
                    np.save(rir_filename, rir_spectrograms[i])
                    writer.writerow([rev_filename, rir_filename])
                runs += 1
                print('Produced: ', runs*n_per_run, end="\r")

            except Empty or TimeoutError:
                print('overlong time, resetting queue')
                p = processes.pop(0)
                p.terminate()
                queue, processes = init_queue(n_proc, args)

if __name__ == '__main__':
    main()