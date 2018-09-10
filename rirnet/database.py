import rirnet.roomgen as rg
import rirnet.acoustic_utils as au
import numpy as np
import yaml
import sys
import csv
import os

filename_db_setup = 'db_setup.yaml'


def build_db(root):
    path_db_setup = os.path.join(root, filename_db_setup)
    db_setup = parse_yaml(path_db_setup)
    source_audio = db_setup['source_audio']
    n_mfcc = db_setup['n_mfcc']
    x, rate = au.read_wav(source_audio)
    db_setup['fs'] = rate
    n_rooms = db_setup['n_rooms']
    n_mics = db_setup['n_mics']
    for i_room in range(n_rooms):
        room = rg.generate_from_dict(db_setup)
        room.compute_rir()
        for i_rir, rir in enumerate(room.rir):
            y = au.convolve(x, rir[0])
            mfcc_y = au.waveform_to_mfcc(y, rate, n_mfcc, db_setup['n_fft'], db_setup['hop_length'])
            mfcc_h = au.waveform_to_mfcc(rir[0], rate, n_mfcc, db_setup['n_fft'], db_setup['hop_length'])
            room_string = str(i_room).zfill(len(str(n_rooms)))
            rir_string = str(i_rir).zfill(len(str(n_mics)))
            name_string = room_string + '_' + rir_string
            np.save(os.path.join(root, name_string+'_d.npy'), mfcc_y)
            np.save(os.path.join(root, name_string+'_t.npy'), mfcc_h)
            


def parse_yaml(filename):
    with open(filename, 'r') as stream:
        db_setup = yaml.load(stream)
    return db_setup


if __name__ == "__main__":
    build_db(sys.argv[1])
