#!/usr/bin/env python3

import rirnet.roomgen as rg
import rirnet.acoustic_utils as au
import numpy as np
import yaml
import sys
import csv
import os

filename_db_setup = 'db_setup.yaml'
audio_path_rel = '../../audio'
header=['path_data', 'path_target', 'mean_target', 'std_target',
        'room_corners', 'room_absorption', 'room_mics', 'room_source']


def generate_rooms(db_setup):
    n_rooms = db_setup['n_rooms']
    h_list = []
    info = []
    for i_room in range(n_rooms):
        counter = i_room/n_rooms*100
        print('Generating Rooms: {:4.1f}%'.format(counter))
        room = rg.generate_from_dict(db_setup)
        room.compute_rir()
        for rir in room.rir:
            h_list.append(rir[0])
            info.append([room.corners, room.absorption, room.mic_array.R,
                         room.sources[0].position])

    h_list = pad_list_to_pow2(h_list)
    return h_list, info


def generate_waveforms(audio_path_folder, h_list, db_setup):
    data_list = []
    target_list = []
    audio_list = db_setup['source_audio']
    rate = db_setup['fs']
    #n_mfcc = db_setup['n_mfcc']

    for i_audio, audio_path in enumerate(audio_list):
        try:
            wav_path = os.path.join(audio_path_folder, audio_path)
            x, _ = au.read_wav(wav_path, rate)
        except FileNotFoundError:
            print('FileNotFoundError! The audio file {} cannot be found.'.format(wav_path))
            sys.exit()

        for i_h, h in enumerate(h_list):
            y = au.convolve(x, h)
            y_length = au.next_power_of_two(np.size(y))
            data = au.pad_to(y, y_length)

            counter = (i_audio*len(h_list)+i_h)/(len(h_list)*len(audio_list))*100
            print('Convolving: {:4.1f}%'.format(counter))
            target = au.pad_to(h, y_length)
            target_list.append(target)
            data_list.append(data)
    return np.array(target_list), np.array(data_list)


def waveforms_to_mfccs(waveforms, db_setup):
    fs = db_setup['fs']
    n_mfcc = db_setup['n_mfcc']
    return [au.waveform_to_mfcc(waveform, fs, n_mfcc) for waveform  in waveforms]


def pad_list_to_pow2(h_list):
    longest_irf = len(max(h_list, key=len))
    target_length = au.next_power_of_two(longest_irf)
    h_list = [au.pad_to(h, target_length) for h in h_list]
    return h_list


def normalize(lists):
    mean = np.mean(lists, axis=(0,2))
    std = column_std(lists, axis=(0,2))
    return (lists - mean[:,None])/std[:,None]


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

    irf_list, info_list = generate_rooms(db_setup)
    data_list, target_list = generate_waveforms(audio_path, irf_list, db_setup)

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


    with open(os.path.join(root, 'db.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        for i in range(len(data_list)):
            data = data_list[i]
            target = target_list[i]
            corners, absorption, mics, sources = info_list[i]
            name_data = '{}_d.npy'.format(i)
            name_target = '{}_t.npy'.format(i)
            path_data = os.path.join(root, name_data)
            path_target = os.path.join(root, name_target)
            np.save(path_data, data)
            np.save(path_target, target)
            writer.writerow([path_data, path_target, target_mean, target_std, corners,
                             absorption, mics, sources])
    print('db buil cmplet')

if __name__ == "__main__":
    try:
        build_db(sys.argv[1])
    except FileNotFoundError:
        print('FileNotFoundError! Did you set the database structure up correctly? (See help-file in database folder).')
    except IndexError:
        print('IndexError! This script takes an input; the path to the preferred database.')
