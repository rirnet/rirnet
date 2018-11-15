#!/usr/bin/env python3

import multiprocessing as mp
import rirnet.roomgen as rg
import rirnet.acoustic_utils as au
import numpy as np
import yaml
import sys
import csv
import os
import random
import pandas as pd
import glob
import librosa
import matplotlib.pyplot as plt
import operator
import signal
import sys
from scipy.spatial.distance import cdist as cdist
from sklearn.utils import shuffle

db_setup_filename = 'db_setup.yaml'
audio_rel_path = '../../audio/livingroom/test'
db_rel_path = '../database'
header=['data_path']


def remove_leading_zeros(rir):
    ind_1st_nonzero = next((i for i, x in enumerate(rir) if x), None)
    rir[0:ind_1st_nonzero] = []
    return np.array(rir)


def load_wavs(audio_folder, db_setup):
    audio_list = db_setup['source_audio']
    rate = db_setup['fs']
    wav_list = []
    if audio_list == ['']:
        audio_list = glob.glob(os.path.join(audio_folder, '*.wav'))
    for audio_filename in audio_list:
        wav_path = os.path.join(audio_folder, audio_filename)
        wav = au.normalize(au.read_wav(wav_path, rate)[0])
        wav_list.append(wav)
    return wav_list


def waveforms_to_mfccs(waveforms, db_setup):
    fs = db_setup['fs']
    n_mfcc = db_setup['n_mfcc']
    mfccs = [au.waveform_to_mfcc(waveform, fs, n_mfcc)[1][:,:-1] for waveform in waveforms]
    return mfccs


def calculate_delta_features(data_list):
    delta_list = []
    delta_2_list = []
    for data in data_list:
        delta_list.append(librosa.feature.delta(data))
        delta_2_list.append(librosa.feature.delta(data, order=2))
    return delta_list, delta_2_list


def parse_yaml(filename):
    with open(filename, 'r') as stream:
        db_setup = yaml.load(stream)
    return db_setup


def build(wav_list, db_csv_path, data_folder_path, db_setup):
    with open(db_csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
    data_list_all = []
    for wav in wav_list:
        wav = au.pad_to(wav, 2**16, 0)
        _, mfcc = au.waveform_to_mfcc(wav, db_setup['fs'], db_setup['n_mfcc'])
        mfcc = mfcc[:,:-1]
        delta_1 = librosa.feature.delta(mfcc)
        delta_2 = librosa.feature.delta(mfcc, order=2)
        data_list = [mfcc, delta_1, delta_2]
        
        n = db_setup['n_samples_val']
        #db_data_mean = np.sum(data_list, axis=(0, 3))/(n*np.shape(data_list)[-1])
        data_list_all.append(data_list)
    

    with open(db_csv_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i, data in enumerate(data_list_all):
            data_filename = '{}_d.npy'.format(i)
            data_path = os.path.join(data_folder_path, data_filename)
            np.save(data_path, data)
            writer.writerow([data_path])

def main():
    root = os.path.abspath(db_rel_path)
    data_folder_path = os.path.join(root, 'data')

    db_csv_path = os.path.join(root, 'db.csv')

    audio_path = os.path.join(root, audio_rel_path)

    db_setup_path = os.path.join(root, db_setup_filename)
    db_setup = parse_yaml(db_setup_path)
    
    wav_list = load_wavs(audio_path, db_setup)

    build(wav_list, db_csv_path, data_folder_path, db_setup)


if __name__ == "__main__":
    main()
