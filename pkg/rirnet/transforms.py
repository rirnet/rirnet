import torch
import numpy as np
import os

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample).float()


class ToNormalized(object):
    """Normalize."""

    def __init__(self, mean_path, std_path):
        self.mean = np.load(mean_path).T
        self.std = np.load(std_path).T


    def __call__(self, sample):
        np.seterr(divide='ignore', invalid='ignore')
        return np.nan_to_num((sample.T-self.mean)/self.std).T


class ToNegativeLog(object):
    """Convert to -Log"""

    def __call__(self, sample):
        return np.array([sample[0]/1024, -np.log(sample[1])])


class ToUnitNorm(object):
    """Normalization for peaks data"""

    def __call__(self, sample):
        data = sample[1]
        data -= np.min(data)
        return np.array([sample[0], data])
