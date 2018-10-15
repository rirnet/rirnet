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

db_setup_filename = 'db_setup.yaml'
db_csv_filename = 'db.csv'
audio_rel_path = '../../audio/mousepatrol/'
data_folder = 'data'
header=['data_path']


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def load_wavs(audio_folder, db_setup):
    audio_list = db_setup['source_audio']
    rate = db_setup['fs']
    wav_list = []
    if audio_list == ['']:
        audio_list = glob.glob(os.path.join(audio_folder, '*.wav'))
    for audio_filename in audio_list:
        wav_path = os.path.join(audio_folder, audio_filename)
        wav, _ = au.read_wav(wav_path, rate)
        wav /= np.max(np.abs(wav))
        wav_list.append(wav)
    return wav_list


def parse_yaml(filename):
    with open(filename, 'r') as stream:
        db_setup = yaml.load(stream)
    return db_setup


def normalize_dataset(db_csv_path, mean):
    db_csv_folder, _ = os.path.split(db_csv_path)

    df = pd.read_csv(db_csv_path)
    n_rows = df.shape[0]
    std_sum = np.zeros_like(mean)

    for i in range(n_rows):
        path = df.iloc[i,0]
        print(path)
        data = np.load(path)
        std_sum += (data-mean)**2

    std = np.sqrt(std_sum/(n_rows-1))
    print(np.shape(std))
    np.save(os.path.join(db_csv_folder, 'std.npy'), std)
    np.save(os.path.join(db_csv_folder, 'mean.npy'), mean)

    np.seterr(invalid='ignore')
    for i in range(n_rows):
        path = df.iloc[i,0]
        data = np.load(path)
        data = (data-mean)/std
        data = np.nan_to_num(data)
        np.save(path, data)
        au.save_wav('{}.wav'.format(i),wav,44100, norm=True)


def build_db(root):
    root = os.path.abspath(root)
    db_csv_path = os.path.join(root, db_csv_filename)
    db_setup_path = os.path.join(root, db_setup_filename)
    db_setup = parse_yaml(db_setup_path)
    audio_path = os.path.join(root, audio_rel_path)
    data_folder_path = os.path.join(root, data_folder)
    if not os.path.exists(data_folder_path):
        os.mkdir(data_folder_path)

    wav_list = load_wavs(audio_path, db_setup)

    with open(db_csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)

    db_data_mean = np.array([])
    data_list=[]

    for i, wav in enumerate(wav_list):
        if np.size(db_data_mean) == 0:
            db_data_mean = np.zeros_like(wav)

        #n = db_setup['n_samples']
        #db_data_mean += np.sum(data_list)/n

        for c, chunk in enumerate(chunks(wav, 1024*20)):
            print(c)
            if len(chunk) == 1024*20:
                with open(db_csv_path, 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    data_filename = '{}_{}_d.npy'.format(i,c)
                    data_path = os.path.join(data_folder_path, data_filename)
                    np.save(data_path, chunk)
                    writer.writerow([data_path])
    #normalize_dataset(db_csv_path, db_data_mean)

if __name__ == "__main__":
    try:
        build_db(sys.argv[1])
    except FileNotFoundError:
        print('FileNotFoundError! Did you set the database structure up correctly? (See help-file in database folder).')
    except IndexError:
        print('IndexError! This script takes an input; the path to the preferred database.')
