#!/usr/bin/env python3

import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch

class RirnetDatabase(Dataset):
    """ Data-Target dataset to use with rirnet"""

    def __init__(self, is_training, args, data_transform=None, target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with data-target pairs.
            root_dir (string): Database directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if is_training:
            csv_file = os.path.join(args.db_path, 'db-train.csv')
            self.root_dir = os.path.join(args.db_path, 'train_data')
        else:
            csv_file = os.path.join(args.db_path, 'db-val.csv')
            self.root_dir = os.path.join(args.db_path, 'val_data')

        database = pd.read_csv(csv_file)

        self.dataset = database
        self.data_transform = data_transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        data_path = os.path.join(self.root_dir, self.dataset.iloc[idx, 0])
        target_path = os.path.join(self.root_dir, self.dataset.iloc[idx, 1])
        data = np.load(data_path)
        target = np.load(target_path) 
        order = np.argsort(target[0])
        target = target[:, order]

        if self.data_transform and self.target_transform:
            data = self.data_transform(data)
            target = self.target_transform(target)[:, :256]
        return data, target
