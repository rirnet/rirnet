#!/usr/bin/env python3

import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch

class RirnetDatabase(Dataset):
    """ Data-Target dataset to use with rirnet"""

    def __init__(self, args, data_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with data-target pairs.
            root_dir (string): Database directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        csv_file = os.path.join(args.db_path, 'db.csv')
        self.root_dir = os.path.join(args.db_path, 'data')

        database = pd.read_csv(csv_file)

        self.dataset = database
        self.data_transform = data_transform


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        data_path = os.path.join(self.root_dir, self.dataset.iloc[idx, 0])
        data = np.load(data_path)
        data = self.data_transform(data)
        return data
