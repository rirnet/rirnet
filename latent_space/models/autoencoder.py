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
        self.conv1a = nn.Conv1d(2, 4, 513, stride=1, padding=256)
        self.conv1b = nn.Conv1d(2, 4, 257, stride=1, padding=128)
        self.bn1a = nn.BatchNorm1d(4)
        self.bn1b = nn.BatchNorm1d(4)

        self.conv2a = nn.Conv1d(4, 64, 5, stride=1, padding=2)
        self.conv2b = nn.Conv1d(4, 64, 3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm1d(64)
        self.bn2b = nn.BatchNorm1d(64)

        self.conv3a = nn.Conv1d(64, 128, 5, stride=1, padding=2)
        self.conv3b = nn.Conv1d(64, 128, 3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm1d(128)
        self.bn3b = nn.BatchNorm1d(128)

        self.conv4a = nn.Conv1d(128, 128, 5, stride=1, padding=2)
        self.conv4b = nn.Conv1d(128, 128, 3, stride=1, padding=1)
        self.bn4a = nn.BatchNorm1d(128)
        self.bn4b = nn.BatchNorm1d(128)

        self.conv5a = nn.Conv1d(128, 128, 3, stride=1, padding=1)
        self.conv5b = nn.Conv1d(128, 128, 1, stride=1, padding=0)
        self.bn5a = nn.BatchNorm1d(128)
        self.bn5b = nn.BatchNorm1d(128)

        self.pool = nn.AvgPool1d(4)
        self.map1x1 = nn.Linear(128, 16)
        self.map2x1 = nn.Linear(16, 1024)

        self.map1x2 = nn.Linear(128, 16)
        self.map2x2 = nn.Linear(16, 1024)

        self.map = nn.Linear(2048, 2048)


        # -------------  Forward Pass  ------------- #
    def forward(self, x, encode=True, decode=True):
        if encode:
            x1 = F.relu(self.bn1a(self.conv1a(x)))
            x2 = F.relu(self.bn1b(self.conv1b(x)))
            x1 = self.pool(x1)
            x2 = self.pool(x2)

            x1 = F.relu(self.bn2a(self.conv2a(x1)))
            x2 = F.relu(self.bn2b(self.conv2b(x2)))
            x1 = self.pool(x1)
            x2 = self.pool(x2)

            x1 = F.relu(self.bn3a(self.conv3a(x1)))
            x2 = F.relu(self.bn3b(self.conv3b(x2)))
            x1 = self.pool(x1)
            x2 = self.pool(x2)

            x1 = F.relu(self.bn4a(self.conv4a(x1)))
            x2 = F.relu(self.bn4b(self.conv4b(x2)))
            x1 = self.pool(x1)
            x2 = self.pool(x2)

            x1 = F.relu(self.bn5a(self.conv5a(x1)))
            x2 = F.relu(self.bn5b(self.conv5b(x2)))
            x1 = self.pool(x1)
            x2 = self.pool(x2)

            p = x1.size()
            (_, C, W) = x1.data.size()
            x1 = x1.view(-1, C * W)
            x1 = F.relu(self.map1x1(x1))

            x2 = x2.view( -1, C * W)
            x2 = F.relu(self.map1x2(x2))
            x = torch.cat((x1.unsqueeze(2),x2.unsqueeze(2)), 2)
        if decode:
            x1 = x[:, :, 0]
            x2 = x[:, :, 1]

            x1 = F.relu(self.map2x1(x1))
            x1 = x1.view(-1, 1024)

            x2 = F.relu(self.map2x2(x2))
            x2 = x2.view(-1, 1024)

            x = torch.cat((x1,x2), 1)
            x = (self.map(x))
            x = x.view(-1, 2, 1024)

        return x


        # -------------  Training settings  ------------- #
    def args(self):
        parser = argparse.ArgumentParser(description='PyTorch rirnet')
        parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
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
