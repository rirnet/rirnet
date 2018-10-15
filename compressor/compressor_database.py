#!/usr/bin/env python3

import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import rirnet.acoustic_utils as au


class RirnetDatabase(Dataset):
    """ Data-Target dataset to use with rirnet"""

    def __init__(self, is_training, args, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with data-target pairs.
            root_dir (string): Database directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        csv_file = os.path.join(args.db_path, 'db.csv')
        database = pd.read_csv(csv_file)
        n_total = len(database)
        indices = np.arange(n_total)
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        if is_training:
            positions = np.arange(int(np.floor(n_total * args.db_ratio)))
        else:
            positions = np.arange(int(np.ceil(n_total * args.db_ratio)), n_total)

        indices = indices[positions]
        self.dataset = database.iloc[indices, :]
        self.root_dir = args.db_path
        self.transform = transform


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        y_path = os.path.join(self.root_dir, self.dataset.iloc[idx, 0])
        h_path = os.path.join(self.root_dir, self.dataset.iloc[idx, 1])
        x_path = os.path.join(self.root_dir, self.dataset.iloc[idx, 2])
        y = self.split(np.load(y_path))
        h = self.split(np.load(h_path))
        x = self.split(np.load(x_path))
        if self.transform:
            y = self.transform(y)
            h = self.transform(h)
            x = self.transform(x)
        return y, h, x


    def split(self, signal):
        signal = au.pad_to(signal, au.next_power_of_two(np.size(signal)))
        n_sub = np.size(signal)//1024
        return np.reshape(signal, (n_sub,1024))

















