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

        self.conv1a = nn.Conv2d(3, 4, 65, stride=2, padding=32)
        self.conv1b = nn.Conv2d(3, 4, 33, stride=2, padding=16)
        self.conv1c = nn.Conv2d(3, 4, 17, stride=2, padding=8)
        self.conv1d = nn.Conv2d(3, 4, 9, stride=2, padding=4)
        self.conv1e = nn.Conv2d(3, 4, 5, stride=2, padding=2)
        self.conv1f = nn.Conv2d(3, 4, 3, stride=2, padding=1)

        self.bn1a = nn.BatchNorm2d(4)
        self.bn1b = nn.BatchNorm2d(4)
        self.bn1c = nn.BatchNorm2d(4)
        self.bn1d = nn.BatchNorm2d(4)
        self.bn1e = nn.BatchNorm2d(4)
        self.bn1f = nn.BatchNorm2d(4)

        self.conv2a = nn.Conv2d(4, 2, 65, stride=2, padding=32)
        self.conv2b = nn.Conv2d(4, 2, 33, stride=2, padding=16)
        self.conv2c = nn.Conv2d(4, 2, 17, stride=2, padding=8)
        self.conv2d = nn.Conv2d(4, 2, 9, stride=2, padding=4)
        self.conv2e = nn.Conv2d(4, 2, 5, stride=2, padding=2)
        self.conv2f = nn.Conv2d(4, 2, 3, stride=2, padding=1)

        self.bn2a = nn.BatchNorm2d(2)
        self.bn2b = nn.BatchNorm2d(2)
        self.bn2c = nn.BatchNorm2d(2)
        self.bn2d = nn.BatchNorm2d(2)
        self.bn2e = nn.BatchNorm2d(2)
        self.bn2f = nn.BatchNorm2d(2)

        self.pool = nn.AvgPool2d(2)
        self.map1x1 = nn.Linear(192, 8)
        self.bn1x = nn.BatchNorm1d(8)
        self.map1x2 = nn.Linear(192, 8)
        self.bn2x = nn.BatchNorm1d(8)

        self.dropout = nn.Dropout(p=0.2)

        self.map2 = nn.Linear(16, 16)

        # -------------  Forward Pass  ------------- #
    def forward(self, x):

        x1 = self.bn1a(F.relu(self.conv1a(x)))
        x2 = self.bn1b(F.relu(self.conv1b(x)))
        x3 = self.bn1c(F.relu(self.conv1c(x)))
        x4 = self.bn1d(F.relu(self.conv1d(x)))
        x5 = self.bn1e(F.relu(self.conv1e(x)))
        x6 = self.bn1f(F.relu(self.conv1f(x)))

        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)
        x4 = self.pool(x4)
        x5 = self.pool(x5)
        x6 = self.pool(x6)

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x3 = self.dropout(x3)
        x4 = self.dropout(x4)
        x5 = self.dropout(x5)
        x6 = self.dropout(x6)

        x1 = self.bn2a(F.relu(self.conv2a(x1)))
        x2 = self.bn2b(F.relu(self.conv2b(x2)))
        x3 = self.bn2c(F.relu(self.conv2c(x3)))
        x4 = self.bn2d(F.relu(self.conv2d(x4)))
        x5 = self.bn2e(F.relu(self.conv2e(x5)))
        x6 = self.bn2f(F.relu(self.conv2f(x6)))

        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)
        x4 = self.pool(x4)
        x5 = self.pool(x5)
        x6 = self.pool(x6)


        x = torch.cat((x1,x2,x3,x4,x5,x6), 2)

        p = x.size()
        (_, C, W, H) = x.data.size()
        x1 = x.view(-1, C * W * H)
        x1 = F.relu(self.map1x1(x1))


        x2 = x.view(-1, C * W * H)
        x2 = F.relu(self.map1x2(x2))


        x = torch.cat((x1,x2), 1)
        x = self.map2(x)
        x = x.view(-1, 16)
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
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                            help='learning rate (default: 0.05)')
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
        parser.add_argument('--train_db_path', type=str, default='../database/db-train.csv',
                            help='path to train csv')
        parser.add_argument('--val_db_path', type=str, default='../database/db-val.csv',
                            help='path to val csv')
        parser.add_argument('--mean_path', type=str, default='../database/mean_data.npy',
                            help='path to dataset mean')
        parser.add_argument('--std_path', type=str, default='../database/std_data.npy',
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
