import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from rirnet.transforms import ToTensor, ToNormalized, ToNegativeLog, ToUnitNorm
import numpy as np
import torch

# -------------  Network class  ------------- #
class Net(nn.Module):

        # -------------  Model Layers  ------------- #
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 1, stride = 1, padding=0)
        self.bn1 = nn.BatchNorm1d(2)
        self.conv2 = nn.Conv1d(2, 4, 1, stride = 1, padding=0)
        self.conv3 = nn.Conv1d(4, 8, 1, stride = 1, padding=0)
        self.conv4 = nn.Conv1d(8, 16, 1, stride = 1, padding=0)
        self.conv5 = nn.Conv1d(16, 32, 1, stride = 1, padding=0)
        self.conv6 = nn.Conv1d(32, 64, 1, stride = 1, padding=0)
        self.conv7 = nn.Conv1d(64, 64, 1, stride = 1, padding=0)

        self.fc1 = nn.Linear(64, 64)
        self.bnf1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(64, 16)

        self.pool = nn.MaxPool1d(2)

        # -------------  Forward Pass  ------------- #
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = F.relu(self.conv7(x))
        x, _ = x.max(2)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


        # -------------  Training settings  ------------- #
    def args(self):
        parser = argparse.ArgumentParser(description='PyTorch rirnet')
        parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                            help='learning rate (default: 0.001)')
        parser.add_argument('--momentum', type=float, default=0.1, metavar='M',
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
        #data_transform = transforms.Compose([ToNormalized(self.args.mean_path, self.args.std_path), ToTensor()])
        data_transform = transforms.Compose([ToTensor()])
        return data_transform

    def target_transform(self):
        target_transform = transforms.Compose([ToNegativeLog(), ToUnitNorm(), ToTensor()])
        return target_transform
