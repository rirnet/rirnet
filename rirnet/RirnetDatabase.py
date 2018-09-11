#!/usr/bin/env python3

#from __future__ import print_function, division
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class FrontRearDataset(Dataset):
    """ Data-Target dataset to use with rirnet"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with data-target pairs.
            root_dir (string): Database directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.database_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.database_csv)

    def __getitem__(self, idx):
        data_path = os.path.join(self.root_dir, self.database_csv.iloc[idx, 0])
        target_path = os.path.join(self.root_dir, self.database_csv.iloc[idx, 1])
        data = np.load(data_path)
        target = np.load(target_path)
        if self.transform:
            data = self.transform(data)
            target = self.transform(target)

        return data, target
