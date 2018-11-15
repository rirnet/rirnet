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

        #self.conv1a = nn.Conv2d(3, 4, 513, stride=2, padding=256)
        #self.conv1b = nn.Conv2d(3, 4, 257, stride=2, padding=128)
        #self.conv1c = nn.Conv2d(3, 4, 129, strid=2, padding=64)
        self.conv1d = nn.Conv2d(3, 4, 65, stride=2, padding=32)
        self.conv1e = nn.Conv2d(3, 4, 33, stride=2, padding=16)
        self.conv1f = nn.Conv2d(3, 4, 17, stride=2, padding=8)
        self.conv1g = nn.Conv2d(3, 4, 9, stride=2, padding=4)
        self.conv1h = nn.Conv2d(3, 4, 5, stride=2, padding=2)
        self.conv1i = nn.Conv2d(3, 4, 3, stride=2, padding=1)

        #self.bn1a = nn.BatchNorm2d(4)
        #self.bn1b = nn.BatchNorm2d(4)
        #self.bn1c = nn.BatchNorm2d(4)
        self.bn1d = nn.BatchNorm2d(4)
        self.bn1e = nn.BatchNorm2d(4)
        self.bn1f = nn.BatchNorm2d(4)
        self.bn1g = nn.BatchNorm2d(4)
        self.bn1h = nn.BatchNorm2d(4)
        self.bn1i = nn.BatchNorm2d(4)

        #self.conv2a = nn.Conv2d(4, 2, 513, stride=2, padding=256)
        #self.conv2b = nn.Conv2d(4, 2, 257, stride=2, padding=128)
        #self.conv2c = nn.Conv2d(4, 2, 129, stride=2, padding=64)
        self.conv2d = nn.Conv2d(4, 2, 65, stride=2, padding=32)
        self.conv2e = nn.Conv2d(4, 2, 33, stride=2, padding=16)
        self.conv2f = nn.Conv2d(4, 2, 17, stride=2, padding=8)
        self.conv2g = nn.Conv2d(4, 2, 9, stride=2, padding=4)
        self.conv2h = nn.Conv2d(4, 2, 5, stride=2, padding=2)
        self.conv2i = nn.Conv2d(4, 2, 3, stride=2, padding=1)

        #self.bn2a = nn.BatchNorm2d(2)
        #self.bn2b = nn.BatchNorm2d(2)
        #self.bn2c = nn.BatchNorm2d(2)
        self.bn2d = nn.BatchNorm2d(2)
        self.bn2e = nn.BatchNorm2d(2)
        self.bn2f = nn.BatchNorm2d(2)
        self.bn2g = nn.BatchNorm2d(2)
        self.bn2h = nn.BatchNorm2d(2)
        self.bn2i = nn.BatchNorm2d(2)

        self.pool = nn.AvgPool2d(2)
        self.map1x1 = nn.Linear(192, 8)
        self.bn1x = nn.BatchNorm1d(8)
        self.map1x2 = nn.Linear(192, 8)
        self.bn2x = nn.BatchNorm1d(8)
        
        self.dropout = nn.Dropout(p=0.3)

        self.map2 = nn.Linear(16, 16)

        # -------------  Forward Pass  ------------- #
    def forward(self, x):

        #x1 = self.bn1a(F.relu(self.conv1a(x)))
        #x2 = self.bn1b(F.relu(self.conv1b(x)))
        #x3 = self.bn1c(F.relu(self.conv1c(x)))
        x4 = self.bn1d(F.relu(self.conv1d(x)))
        x5 = self.bn1e(F.relu(self.conv1e(x)))
        x6 = self.bn1f(F.relu(self.conv1f(x)))
        x7 = self.bn1g(F.relu(self.conv1g(x)))
        x8 = self.bn1h(F.relu(self.conv1h(x)))
        x9 = self.bn1i(F.relu(self.conv1i(x)))

        #x1 = self.pool(x1)
        #x2 = self.pool(x2)
        #x3 = self.pool(x3)
        x4 = self.pool(x4)
        x5 = self.pool(x5)
        x6 = self.pool(x6)
        x7 = self.pool(x7)
        x8 = self.pool(x8)
        x9 = self.pool(x9)

        x4 = self.dropout(x4)
        x5 = self.dropout(x5)
        x6 = self.dropout(x6)
        x7 = self.dropout(x7)
        x8 = self.dropout(x8)
        x9 = self.dropout(x9)

        #x1 = self.bn2a(F.relu(self.conv2a(x1)))
        #x2 = self.bn2b(F.relu(self.conv2b(x2)))
        #x3 = self.bn2c(F.relu(self.conv2c(x3)))
        x4 = self.bn2d(F.relu(self.conv2d(x4)))
        x5 = self.bn2e(F.relu(self.conv2e(x5)))
        x6 = self.bn2f(F.relu(self.conv2f(x6)))
        x7 = self.bn2g(F.relu(self.conv2g(x7)))
        x8 = self.bn2h(F.relu(self.conv2h(x8)))
        x9 = self.bn2i(F.relu(self.conv2i(x9)))

        #x1 = self.pool(x1)
        #x2 = self.pool(x2)
        #x3 = self.pool(x3)
        x4 = self.pool(x4)
        x5 = self.pool(x5)
        x6 = self.pool(x6)
        x7 = self.pool(x7)
        x8 = self.pool(x8)
        x9 = self.pool(x9)


        x = torch.cat((x4,x5,x6,x7,x8,x9), 2)

        p = x.size()
        (_, C, W, H) = x.data.size()
        x1 = x.view(-1, C * W * H)
        x1 = (self.bn1x(self.map1x1(x1)))


        x2 = x.view(-1, C * W * H)
        x2 = (self.bn2x(self.map1x2(x2)))


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
        parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
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
        target_transform = transforms.Compose([ToNegativeLog(), ToUnitNorm(), ToTensor()])
        return target_transform
