#!/usr/bin/env python3

import rirnet.roomgen as rg
import rirnet.acoustic_utils as au
import numpy as np
import yaml
import sys
import csv
import os
import random

filename_db_setup = 'db_setup.yaml'
audio_path_rel = '../../audio'
data_folder = 'data'
header=['path_data', 'path_target', 'mean_target', 'std_target',
        'room_corners', 'room_absorption', 'room_mics', 'room_source']


class RirGenerator:
    def __init__(self, db_setup):
        self.i_total = 0
        self.n_total = db_setup['n_samples']
        self.db_setup = db_setup
        self.h_length = None
        self.discarded = 0


    def __iter__(self):
        return self


    def __next__(self):
        if self.i_total == self.n_total:
            raise StopIteration
        i_produced = 0
        h_list = []
        info_list = []
        room = rg.generate_from_dict(self.db_setup)
        room.compute_rir()
        for i_rir, rir in enumerate(room.rir):
            rir_length = len(rir[0])
            if not self.h_length:
                self.h_length = au.next_power_of_two(rir_length)
            if rir_length > self.h_length:
                self.discarded += 1
                return self.__next__()
            else:
                rir = au.pad_to(rir[0], self.h_length)
                h_list.append(rir)
                info_list.append([room.corners, room.absorption, room.mic_array.R[:, i_rir], 
                                room.sources[0].position])
                i_produced += 1
                if self.i_total + i_produced == self.n_total:
                    break
        self.i_total += i_produced
        return h_list, info_list


def generate_waveforms(wav, h_list, db_setup):
    data_list = []
    target_list = []

    for i_h, h in enumerate(h_list):
        y = au.convolve(wav, h)
        y_length = au.next_power_of_two(np.size(y))
        data = au.pad_to(y, y_length)
        target = au.pad_to(h, y_length)
        target_list.append(target)
        data_list.append(data)
    return np.array(target_list), np.array(data_list)


def load_wavs(audio_path_folder, db_setup):
    audio_list = db_setup['source_audio']
    rate = db_setup['fs']
    wav_list = []
    for audio_path in audio_list:
        wav_path = os.path.join(audio_path_folder, audio_path)
        wav, _ = au.read_wav(wav_path, rate)
        wav_list.append(wav)
    return wav_list


def waveforms_to_mfccs(waveforms, db_setup):
    fs = db_setup['fs']
    n_mfcc = db_setup['n_mfcc']
    return [au.waveform_to_mfcc(waveform, fs, n_mfcc) for waveform  in waveforms]


def pad_list_to_pow2(h_list):
    longest_irf = len(max(h_list, key=len))
    target_length = au.next_power_of_two(longest_irf)
    h_list = [au.pad_to(h, target_length) for h in h_list]
    return h_list


def repeat_list(x,n):
    return x*n


def parse_yaml(filename):
    with open(filename, 'r') as stream:
        db_setup = yaml.load(stream)
    return db_setup


def build_db(root):
    root = os.path.abspath(root)
    path_db_setup = os.path.join(root, filename_db_setup)
    db_setup = parse_yaml(path_db_setup)
    audio_path = os.path.join(root, audio_path_rel)
    data_folder_path = os.path.join(root, data_folder)
    if not os.path.exists(data_folder_path):
        os.mkdir(data_folder_path)

    wav_list = load_wavs(audio_path, db_setup)

    rir_generator = RirGenerator(db_setup) 
    with open(os.path.join(root, 'db.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)

    while rir_generator.i_total < rir_generator.n_total:
        for h_list, info_list in rir_generator:
            counter = rir_generator.i_total/rir_generator.n_total*100
            print('Progress: {:5.01f}%, Discarded {} times.'.format(counter, rir_generator.discarded), end="\r")
            
            wav = random.choice(wav_list)
            target_list, data_list = generate_waveforms(wav, h_list, db_setup)    
            if target_list.size > 0:
                if db_setup['data_format'] == 'mfcc':
                    data_list = waveforms_to_mfccs(data_list, db_setup)
                elif db_setup['data_format'] == 'waveform':
                    data_list = data_list[:, None]
                else:
                    print('No valid data format specified in db_setup.yaml')
                    sys.exit()
                target_list = waveforms_to_mfccs(target_list, db_setup)
                target_mean = np.mean(target_list, axis=(0,2))
                target_std = np.std(target_list, axis=(0,2))
                target_list = (target_list - target_mean[:,None])/target_std[:,None]

                data_mean = np.mean(data_list, axis=(0,2))
                data_std = np.std(data_list, axis=(0,2))
                data_list = np.squeeze((data_list - data_mean[:,None])/data_std[:,None])

                info_list = repeat_list(info_list, len(db_setup['source_audio']))


                with open(os.path.join(root, 'db.csv'), 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    n_saved = rir_generator.i_total - len(data_list)
                    for i, data in enumerate(data_list):
                        try:
                            target = target_list[i]
                        except:
                            print(np.shape(data), np.shape(target))
                        corners, absorption, mics, sources = info_list[i]
                        name_data = '{}_d.npy'.format(n_saved + i)
                        name_target = '{}_t.npy'.format(n_saved + i)
                        path_data = os.path.join(data_folder_path, name_data)
                        path_target = os.path.join(data_folder_path, name_target)
                        np.save(path_data, data)
                        np.save(path_target, target)
                        writer.writerow([path_data, path_target, target_mean, target_std, corners,
                                        absorption, mics, sources])
    print('\nBirth Complet')
    print('It\'s {}'.format(random.choice(['a Boy! Yay!', '... a Grill :('])))

if __name__ == "__main__":
    try:
        build_db(sys.argv[1])
    except FileNotFoundError:
        print('FileNotFoundError! Did you set the database structure up correctly? (See help-file in database folder).')
    except IndexError:
        print('IndexError! This script takes an input; the path to the preferred database.')
