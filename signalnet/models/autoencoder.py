import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
from rirnet.transforms import ToUnitNorm, ToTensor, ToNormalized, ToNegativeLog
import matplotlib.pyplot as plt
import rirnet.acoustic_utils as au

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.bottleneck = 16

        self.conv1 = nn.Conv1d(2, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, self.bottleneck, 1)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(self.bottleneck)

        self.fc1 = nn.Linear(self.bottleneck, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 512)


    def forward(self, x, encode=False, decode=False):
        if encode:
            x = self.encode(x)
        if decode:
            x = self.decode(x)
        return x

    def encode(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.conv4(x)
        x, _ = x.max(2)
        #m = int(bin(2**(epoch//20+1)-1)[2:])
        #arr = [int(i) for i in str(m)]
        #arr = au.pad_to(arr, 32)
        #x = torch.tensor(arr).float().cuda()*x
        return x

    def decode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 2, 256)
        return x


    def args(self):
        parser = argparse.ArgumentParser(description='PyTorch rirnet')
        parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.00005, metavar='LR',
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
        parser.add_argument('--save-interval', type=int, default=100,
                            help='how many batches to wait before saving network')
        parser.add_argument('--plot', type=bool, default=True,
                            help='show plot while training (turn off if using ssh)')
        parser.add_argument('--train_db_path', type=str, default='../../database/db-train.csv',
                            help='path to train csv')
        parser.add_argument('--val_db_path', type=str, default='../../database/db-val.csv',
                            help='path to val csv')
        parser.add_argument('--mean_path', type=str, default='../../database/mean_data.npy',
                            help='path to dataset mean')
        parser.add_argument('--std_path', type=str, default='../../database/std_data.npy',
                            help='path to dataset std')
        parser.add_argument('--n_peaks', type=int, default=256,
                            help='number of points that the network uses')
        self.args, unknown = parser.parse_known_args()
        return self.args


        # -------------  Transform settings  ------------- #
    def data_transform(self):
        data_transform = transforms.Compose([ToNormalized(self.args.mean_path, self.args.std_path), ToTensor()])
        return data_transform

    def target_transform(self):
        target_transform = transforms.Compose([ToNegativeLog(), ToUnitNorm(),  ToTensor()])
        return target_transform
