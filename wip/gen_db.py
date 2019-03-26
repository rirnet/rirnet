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


def generate_spectrograms(queue, args):
    x_max, y_max, z_max, n_mics, n_per_seg, max_order, fs, material_engine, sound_engine = args
    np.random.seed()
    x, y, z = np.random.rand(3)*(np.array([x_max, y_max, z_max])-2.5)+2.5
    mic_pos = rg.generate_pos_in_rect(x, y, z, n_mics)
    source_pos = rg.generate_pos_in_rect(x, y, z, 1)[0]
    absorption = material_engine.random()
    an_sig = sound_engine.random()

    rir_list = rg.generate_multiband_rirs(x, y, z, mic_pos, source_pos, fs, max_order, absorption)
    rev_sig_spectrograms = []
    rir_spectrograms = []
    for rir in rir_list:
        rev_sig = au.convolve(rir, an_sig)
        _,_,rir_spectrogram = sp.signal.stft(rir, fs=fs, nperseg=n_per_seg)
        _,_,rev_sig_spectrogram = sp.signal.stft(rev_sig, fs=fs, nperseg=n_per_seg)
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
    sort = 'val'
    n_samples = 1000
    n_proc = 6
    x_max = 10
    y_max = 10
    z_max = 3
    n_mics = 10
    n_per_seg = 128
    max_order = 80
    fs = 16384
    material_engine = MaterialEngine('materials.csv', 'surfaces.csv')
    sound_engine = SoundEngine('/home/felix/rirnet/audio/chamber/'+sort, fs)
    save_folder_path = '/home/felix/rirnet/db_fft/'+sort
    csv_filename = sort+'.csv'
    args = [x_max, y_max, z_max, n_mics, n_per_seg, max_order, fs,  material_engine, sound_engine]
    queue, processes = init_queue(n_proc, args)

    with open(os.path.join(save_folder_path,'../',csv_filename), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for run in range(n_samples//n_mics):
            try:
                rev_sig_spectrograms, rir_spectrograms = queue.get(timeout=20)
                p = processes.pop(0)
                p.terminate()
                processes.append(mp.Process(target=generate_spectrograms, args=(queue, args,)))
                p = processes[-1]
                p.start()

                for sample in range(n_mics):
                    rev_filename = '{}_rev.npy'.format(n_mics*run+sample)
                    rir_filename = '{}_rir.npy'.format(n_mics*run+sample)
                    rev_path = os.path.join(save_folder_path,rev_filename)
                    rir_path = os.path.join(save_folder_path,rir_filename)
                    np.save(rev_path, rev_sig_spectrograms[sample])
                    np.save(rir_path, rir_spectrograms[sample])
                    writer.writerow([rev_path, rir_path])
                print('Produced: ', run*n_mics, end="\r")

            except Empty or TimeoutError:
                print('overlong time, resetting queue')
                p = processes.pop(0)
                p.terminate()
                queue, processes = init_queue(n_proc, args)

    for p in processes:
        p.terminate()
    print('\nAll done!')

if __name__ == '__main__':
    main()
