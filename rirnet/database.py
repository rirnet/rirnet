import rirnet.roomgen as rg
import rirnet.acoustic_utils as au
import numpy as np
import yaml
import sys
import csv


def build_db(filename_db_setup):
    db_setup = parse_yaml(filename_db_setup)
    source_audio = db_setup['source_audio']
    n_mfcc = db_setup['n_mfcc']
    x, rate = au.read_wav(source_audio)
    db_setup['fs'] = rate
    room = rg.generate_from_dict(db_setup)


def parse_yaml(filename):
    with open(filename, 'r') as stream:
        db_setup = yaml.load(stream)
    return db_setup


if __name__ == "__main__":
    build_db(sys.argv[1])
