#!/usr/bin/env python3

import rirnet.roomgen as rg
import rirnet.acoustic_utils as au
import numpy as np
import yaml
import sys
import csv
import os
import random
import pandas as pd

db_setup_filename = 'db_setup.yaml'
db_mean_filename = 'mean.npy'
db_std_filename = 'std.npy'
db_csv_filename = 'db.csv'
audio_rel_path = '../../audio'
data_folder = 'data'
header=['data_path', 'target_path', 'room_corners', 'room_absorption', 'room_mics', 'room_source', 'mean_path', 'std_path']


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


def load_wavs(audio_folder, db_setup):
    audio_list = db_setup['source_audio']
    rate = db_setup['fs']
    wav_list = []
    for audio_filename in audio_list:
        wav_path = os.path.join(audio_folder, audio_filename)
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


def normalize_dataset(db_csv_path, data_mean, target_mean):
    df = pd.read_csv(db_csv_path)
    n_rows = df.shape[0]
    data_std = np.std(data_mean, axis=0)
    target_std = np.std(target_mean, axis=0)
    for i in range(n_rows):
        data_path = df.iloc[i, 0]
        target_path = df.iloc[i, 1]
        data = np.load(data_path)
        data = np.nan_to_num((data-data_mean)/data_std)
        target = np.load(target_path)
        target = np.nan_to_num((target-target_mean)/target_std)
        np.save(data_path, data)
        np.save(target_path, target)


def build_db(root):
    root = os.path.abspath(root)
    db_mean_path = os.path.join(root, db_mean_filename)
    db_std_path = os.path.join(root, db_std_filename)
    db_csv_path = os.path.join(root, db_csv_filename)
    db_setup_path = os.path.join(root, db_setup_filename)
    db_setup = parse_yaml(db_setup_path)
    audio_path = os.path.join(root, audio_rel_path)
    data_folder_path = os.path.join(root, data_folder)
    if not os.path.exists(data_folder_path):
        os.mkdir(data_folder_path)

    wav_list = load_wavs(audio_path, db_setup)

    rir_generator = RirGenerator(db_setup)
    with open(db_csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)

    db_target_mean = np.array([])
    db_data_mean = np.array([])

    while rir_generator.i_total < rir_generator.n_total:
        for h_list, info_list in rir_generator:
            counter = rir_generator.i_total/rir_generator.n_total*100
            print('Progress: {:5.01f}%, Discarded {} times.'.format(counter, rir_generator.discarded), end="\r")

            wav = random.choice(wav_list)
            target_list, data_list = generate_waveforms(wav, h_list, db_setup)

            if db_setup['data_format'] == 'mfcc':
                data_list = waveforms_to_mfccs(data_list, db_setup)
            elif db_setup['data_format'] == 'waveform':
                data_list = data_list[:, None]
            else:
                print('No valid data format specified in db_setup.yaml')
                sys.exit()


            target_list = waveforms_to_mfccs(target_list, db_setup)

            if np.size(db_target_mean) == 0:
                db_target_mean = np.empty_like(target_list[0])
            if np.size(db_data_mean) == 0:
                db_data_mean = np.empty_like(data_list[0])

            n = db_setup['n_samples']
            db_target_mean += np.sum(target_list, axis=0)/n
            db_data_mean += np.sum(data_list, axis=0)/n

            info_list = repeat_list(info_list, len(db_setup['source_audio']))


            with open(db_csv_path, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                n_saved = rir_generator.i_total - len(data_list)
                for i, data in enumerate(data_list):
                    try:
                        target = target_list[i]
                    except:
                        print(np.shape(data), np.shape(target))
                    corners, absorption, mics, sources = info_list[i]
                    data_filename = '{}_d.npy'.format(n_saved + i)
                    target_filename = '{}_t.npy'.format(n_saved + i)
                    data_path = os.path.join(data_folder_path, data_filename)
                    target_path = os.path.join(data_folder_path, target_filename)
                    np.save(data_path, data)
                    np.save(target_path, target)
                    writer.writerow([data_path, target_path, corners,
                                        absorption, mics, sources, db_mean_path, db_std_path])
    print('\ndatabase generated, normalizing')
    normalize_dataset(db_csv_path, db_data_mean, db_target_mean)
    np.save(db_mean_path, db_target_mean)
    np.save(db_std_path, np.std(db_target_mean, axis=0))
    print('Done')

if __name__ == "__main__":
    try:
        build_db(sys.argv[1])
    except FileNotFoundError:
        print('FileNotFoundError! Did you set the database structure up correctly? (See help-file in database folder).')
    except IndexError:
        print('IndexError! This script takes an input; the path to the preferred database.')
