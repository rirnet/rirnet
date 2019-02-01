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

        self.bottleneck = 64

        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 256, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(256)

        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(5,16)

        self.dropout = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.bottleneck)

        # -------------  Forward Pass  ------------- #
    def forward(self, x):

        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = (F.relu(self.conv3(x)))
        x = (F.relu(self.conv4(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = (F.relu(self.conv5(x)))
        x = (F.relu(self.conv6(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = (F.relu(self.conv7(x)))
        x = self.avgpool(x)

        (_, C, W, H) = x.size()
        x = x.view(-1, C * W * H)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


        # -------------  Training settings  ------------- #
    def args(self):
        parser = argparse.ArgumentParser(description='PyTorch rirnet')
        parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                            help='learning rate (default: 0.05)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--train_db_path', type=str, default='../../../database/db-train.csv',
                            help='path to train csv')
        parser.add_argument('--val_db_path', type=str, default='../../../database/db-val.csv',
                            help='path to val csv')
        parser.add_argument('--mean_path', type=str, default='../../../database/mean.npy',
                            help='path to dataset mean')
        parser.add_argument('--std_path', type=str, default='../../../database/std.npy',
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
        target_transform = transforms.Compose([ToNegativeLog(), ToUnitNorm(), ToTensor()])
        return target_transform
