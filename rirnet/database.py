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


def build_db(root):
    root = os.path.abspath(root)
    path_db_setup = os.path.join(root, filename_db_setup)
    db_setup = parse_yaml(path_db_setup)
    audio_path = os.path.join(root, audio_path_rel)
    
    h_list, info_list = generate_rooms(db_setup)

    mfcc_y_list, mfcc_h_list = generate_mfccs(audio_path, h_list, db_setup)
    info_list = repeat_list(info_list, len(db_setup['source_audio']))
    mfcc_h_mean = column_mean(mfcc_h_list)
    mfcc_h_std = column_std(mfcc_h_list)
    mfcc_y_list = normalize(mfcc_y_list)
    mfcc_h_list = normalize(mfcc_h_list)

    with open(os.path.join(root, 'db.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        for i in range(len(mfcc_y_list)):
            mfcc_y = mfcc_y_list[i]
            mfcc_h = mfcc_h_list[i]
            corners, absorption, mics, sources = info_list[i]
            name_d = '{}_d.npy'.format(i)
            name_t = '{}_t.npy'.format(i)
            path_d = os.path.join(root, name_d)
            path_t = os.path.join(root, name_t)
            np.save(path_d, mfcc_y)
            np.save(path_t, mfcc_h)
            writer.writerow([path_d, path_t, mfcc_h_mean, mfcc_h_std, corners, 
                             absorption, mics, sources])


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

    h_list = fix_lengths(h_list)
    return h_list, info


def generate_mfccs(audio_path, h_list, db_setup):
    mfcc_y_list = []
    mfcc_h_list = []
    wav_list = db_setup['source_audio']
    rate = db_setup['fs']
    n_mfcc = db_setup['n_mfcc']

    for i_audio, wav_path in enumerate(wav_list):
        x, _ = au.read_wav(os.path.join(audio_path, wav_path), rate)
        for i_h, h in enumerate(h_list):
            y = au.convolve(x, h)
            y_length = au.next_power_of_two(np.size(y))
            y = au.pad_to(y, y_length)

            counter = (i_audio*len(h_list)+i_h)/(len(h_list)*len(wav_list))*100
            print('Convolving: {:4.1f}%'.format(counter))
            h = au.pad_to(h, y_length)
            mfcc_y = au.waveform_to_mfcc(y, rate, n_mfcc)
            mfcc_h = au.waveform_to_mfcc(h, rate, n_mfcc)
            mfcc_y_list.append(mfcc_y)
            mfcc_h_list.append(mfcc_h)
    return mfcc_y_list, mfcc_h_list


def fix_lengths(h_list):
    longest_irf = len(max(h_list, key=len))
    target_length = au.next_power_of_two(longest_irf)
    h_list = [au.pad_to(h, target_length) for h in h_list]
    return h_list


def column_mean(lists):
    return np.mean(lists, axis=(0,2))


def column_std(lists):
    return np.std(lists, axis=(0,2))


def normalize(lists):
    mean = column_mean(lists)
    std = column_std(lists)
    return (lists - mean[:,None])/std[:,None]


def repeat_list(x,n):
    return x*n


def parse_yaml(filename):
    with open(filename, 'r') as stream:
        db_setup = yaml.load(stream)
    return db_setup


if __name__ == "__main__":
    try:
        build_db(sys.argv[1])
    except FileNotFoundError:
        print('FileNotFoundError! Did you set the database structure up correctly? (See help-file in database folder)')
    except IndexError:
        print('IndexError! This script takes an input; the path to the preferred database')
