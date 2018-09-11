#!/usr/bin/env python3

import rirnet.roomgen as rg
import rirnet.acoustic_utils as au
import numpy as np
import yaml
import sys
import csv
import os
import pickle

filename_db_setup = 'db_setup.yaml'
audio_path_rel = '../../audio'
header=['path_data', 'path_target', 'room_corners', 'room_absorption', 'room_mics', 'room_source']

def build_db(root):
    root = os.path.abspath(root)
    path_db_setup = os.path.join(root, filename_db_setup)
    db_setup = parse_yaml(path_db_setup)
    source_audio = db_setup['source_audio']
    n_mfcc = db_setup['n_mfcc']
    x, rate = au.read_wav(os.path.join(root, audio_path_rel, source_audio))
    db_setup['fs'] = rate
    n_rooms = db_setup['n_rooms']

    with open(os.path.join(root, 'db.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(header)
        for i_room in range(n_rooms):
            room = rg.generate_from_dict(db_setup)
            room.compute_rir()
            for i_rir, rir in enumerate(room.rir):
                y = au.convolve(x, rir[0])
                mfcc_y = au.waveform_to_mfcc(y, rate, n_mfcc)
                mfcc_h = au.waveform_to_mfcc(rir[0], rate, n_mfcc)
                name_d = 'room%04d_pos%04d_data.npy' % (i_room, i_rir)
                name_t = 'room%04d_pos%04d_target.npy' % (i_room, i_rir)
                path_d = os.path.join(root, name_d)
                path_t = os.path.join(root, name_t)
                np.save(path_d, mfcc_y)
                np.save(path_t, mfcc_h)
                writer.writerow([path_d, path_t, room.corners, room.absorption, room.mic_array.R, room.sources[0].position])


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
