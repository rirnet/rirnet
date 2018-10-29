import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from rirnet.transforms import ToUnitNorm, ToTensor, ToNormalized, ToNegativeLog
import numpy as np
import torch

# -------------  Network class  ------------- #
class Net(nn.Module):

        # -------------  Model Layers  ------------- #
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.AvgPool1d(2)

        self.map1 = nn.Linear(32, 16)
        self.bnm1 = nn.BatchNorm1d(16)

        self.map2 = nn.Linear(16, 1)

        # -------------  Forward Pass  ------------- #
    def forward(self, x, encode=True, decode=True):
        p = x.size()
        (_, C, W) = x.data.size()
        x = x.view(-1, C * W)
        x = F.relu(self.map1(x))
        x = F.tanh(self.map2(x))
        return x


        # -------------  Training settings  ------------- #
    def args(self):
        parser = argparse.ArgumentParser(description='PyTorch rirnet')
        parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                            help='learning rate (default: 0.005)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                            help='SGD momentum (default: 0.5)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--loss_function', type=str, default='mse_loss',
                            help='the loss function to use. Must be EXACTLY as the function is called in pytorch docs')
        parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--save-interval', type=int, default=1000000000,
                            help='how many batches to wait before saving network')
        parser.add_argument('--plot', type=bool, default=True,
                            help='show plot while training (turn off if using ssh)')
        parser.add_argument('--db_path', type=str, default='../database',
                            help='path to folder that contains database csv')
        parser.add_argument('--db_ratio', type=float, default=0.9,
                            help='ratio of the db to use for training')
        parser.add_argument('--save_timestamps', type=bool, default=True,
                            help='enables saving of timestamps to csv')
        self.args, unknown = parser.parse_known_args()
        return self.args


        # -------------  Transform settings  ------------- #
    def data_transform(self):
        data_transform = transforms.Compose([ToNormalized(self.args.db_path, 'mean_data.npy', 'std_data.npy'), ToTensor()])
        return data_transform

    def target_transform(self):
        target_transform = transforms.Compose([ToNegativeLog(), ToUnitNorm(),  ToTensor()])
        return target_transform
