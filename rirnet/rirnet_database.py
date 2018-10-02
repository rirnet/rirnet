#!/usr/bin/env python3

#from __future__ import print_function, division
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

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
        data_path = os.path.join(self.root_dir, self.dataset.iloc[idx, 0])
        target_path = os.path.join(self.root_dir, self.dataset.iloc[idx, 1])
        data = np.load(data_path)
        target = np.load(target_path)
        if self.transform:
            data = self.transform(data)
            target = self.transform(target)
        return data, target
