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
header=['path_data', 'path_target', 'room_corners', 'room_absorption', 'room_mics', 'room_source']


def build_db(root):
    root = os.path.abspath(root)
    path_db_setup = os.path.join(root, filename_db_setup)
    db_setup = parse_yaml(path_db_setup)
    n_rooms = db_setup['n_rooms']
    source_audio = db_setup['source_audio']
    n_mfcc = db_setup['n_mfcc']
    rate = db_setup['fs']

    h_list = []
    info = []
    room_pos_number = []
    for i_room in range(n_rooms):
        room = rg.generate_from_dict(db_setup)
        room.compute_rir()
        print('Generating rooms: ' + str(np.round(i_room/n_rooms*100, 3)) + '%')
        for pos, rir in enumerate(room.rir):
            h_list.append(rir[0])
            info.append([room.corners, room.absorption, room.mic_array.R, room.sources[0].position])
            room_pos_number.append([i_room, pos])

    longest_irf = au.next_power_of_two(len(max(h_list, key=len)))
    h_list = [au.pad_to(h, longest_irf) for h in h_list]

    with open(os.path.join(root, 'db.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        for i_audio, audio_path in enumerate(source_audio):
            x, _ = au.read_wav(os.path.join(root, audio_path_rel, audio_path), rate)

            for i_h, h in enumerate(h_list):
                y = au.convolve(x, h)
                y_length = au.next_power_of_two(np.size(y))
                y = au.pad_to(y, y_length)
                print('Convolving: ' + str(np.round(i_h/len(h_list)*100, 3)) + '%')
                h = au.pad_to(h, y_length)
                mfcc_y = np.log(au.waveform_to_mfcc(y, rate, n_mfcc) + 2**14)
                mfcc_h = np.log(au.waveform_to_mfcc(h, rate, n_mfcc) + 2**14)
                name_d = 'room{:04d}_pos{:04d}_audio{:02d}_data.npy'.format(room_pos_number[i_h][0],
                                                                            room_pos_number[i_h][1], i_audio)
                name_t = 'room{:04d}_pos{:04d}_audio{:02d}_target.npy'.format(room_pos_number[i_h][0],
                                                                              room_pos_number[i_h][1], i_audio)
                path_d = os.path.join(root, name_d)
                path_t = os.path.join(root, name_t)
                np.save(path_d, mfcc_y)
                np.save(path_t, mfcc_h)
                writer.writerow([path_d, path_t, info[i_h]])


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
