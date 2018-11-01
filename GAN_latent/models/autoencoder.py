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
        self.conv1a = nn.Conv1d(2, 4, 513, stride=2, padding=256)
        self.conv1b = nn.Conv1d(2, 4, 257, stride=2, padding=128)
        self.conv1c = nn.Conv1d(2, 4, 129, stride=2, padding=64)
        self.conv1d = nn.Conv1d(2, 4, 65, stride=2, padding=32)
        self.conv1e = nn.Conv1d(2, 4, 33, stride=2, padding=16)
        self.conv1f = nn.Conv1d(2, 4, 17, stride=2, padding=8)
        self.conv1g = nn.Conv1d(2, 4, 9, stride=2, padding=4)
        self.conv1h = nn.Conv1d(2, 4, 5, stride=2, padding=2)
        self.conv1i = nn.Conv1d(2, 4, 3, stride=2, padding=1)

        self.bn1a = nn.BatchNorm1d(4)
        self.bn1b = nn.BatchNorm1d(4)
        self.bn1c = nn.BatchNorm1d(4)
        self.bn1d = nn.BatchNorm1d(4)
        self.bn1e = nn.BatchNorm1d(4)
        self.bn1f = nn.BatchNorm1d(4)
        self.bn1g = nn.BatchNorm1d(4)
        self.bn1h = nn.BatchNorm1d(4)
        self.bn1i = nn.BatchNorm1d(4)

        self.conv2a = nn.Conv1d(4, 2, 513, stride=2, padding=256)
        self.conv2b = nn.Conv1d(4, 2, 257, stride=2, padding=128)
        self.conv2c = nn.Conv1d(4, 2, 129, stride=2, padding=64)
        self.conv2d = nn.Conv1d(4, 2, 65, stride=2, padding=32)
        self.conv2e = nn.Conv1d(4, 2, 33, stride=2, padding=16)
        self.conv2f = nn.Conv1d(4, 2, 17, stride=2, padding=8)
        self.conv2g = nn.Conv1d(4, 2, 9, stride=2, padding=4)
        self.conv2h = nn.Conv1d(4, 2, 5, stride=2, padding=2)
        self.conv2i = nn.Conv1d(4, 2, 3, stride=2, padding=1)

        self.bn2a = nn.BatchNorm1d(2)
        self.bn2b = nn.BatchNorm1d(2)
        self.bn2c = nn.BatchNorm1d(2)
        self.bn2d = nn.BatchNorm1d(2)
        self.bn2e = nn.BatchNorm1d(2)
        self.bn2f = nn.BatchNorm1d(2)
        self.bn2g = nn.BatchNorm1d(2)
        self.bn2h = nn.BatchNorm1d(2)
        self.bn2i = nn.BatchNorm1d(2)

        #self.conv2a = nn.Conv1d(4, 64, 5, stride=1, padding=2)
        #self.conv2b = nn.Conv1d(4, 64, 1, stride=1, padding=0)
        #self.bn2a = nn.BatchNorm1d(64)
        #self.bn2b = nn.BatchNorm1d(64)

        self.conv3a = nn.Conv1d(64, 128, 5, stride=1, padding=2)
        self.conv3b = nn.Conv1d(64, 128, 1, stride=1, padding=0)
        self.bn3a = nn.BatchNorm1d(128)
        self.bn3b = nn.BatchNorm1d(128)

        self.conv4a = nn.Conv1d(128, 128, 5, stride=1, padding=2)
        self.conv4b = nn.Conv1d(128, 128, 1, stride=1, padding=0)
        self.bn4a = nn.BatchNorm1d(128)
        self.bn4b = nn.BatchNorm1d(128)

        self.conv5a = nn.Conv1d(128, 128, 5, stride=1, padding=2)
        self.conv5b = nn.Conv1d(128, 128, 1, stride=1, padding=0)
        self.bn5a = nn.BatchNorm1d(128)
        self.bn5b = nn.BatchNorm1d(128)

        self.pool = nn.AvgPool1d(2)
        self.map1x1 = nn.Linear(1152, 16)
        self.bnm1 = nn.BatchNorm1d(16)
        self.map2x1 = nn.Linear(16, 1024)

        self.map1x2 = nn.Linear(1152, 16)
        self.bnm2 = nn.BatchNorm1d(16)
        self.map2x2 = nn.Linear(16, 1024)

        self.map = nn.Linear(2048, 2048)


        # -------------  Forward Pass  ------------- #
    def forward(self, x, encode=True, decode=True):
        if encode:
            #torch.manual_seed()
            x1 = self.bn1a(F.relu(self.conv1a(x)))
            x2 = self.bn1b(F.relu(self.conv1b(x)))
            x3 = self.bn1c(F.relu(self.conv1c(x)))
            x4 = self.bn1d(F.relu(self.conv1d(x)))
            x5 = self.bn1e(F.relu(self.conv1e(x)))
            x6 = self.bn1f(F.relu(self.conv1f(x)))
            x7 = self.bn1g(F.relu(self.conv1g(x)))
            x8 = self.bn1h(F.relu(self.conv1h(x)))
            x9 = self.bn1i(F.relu(self.conv1i(x)))

            x1 = self.pool(x1)
            x2 = self.pool(x2)
            x3 = self.pool(x3)
            x4 = self.pool(x4)
            x5 = self.pool(x5)
            x6 = self.pool(x6)
            x7 = self.pool(x7)
            x8 = self.pool(x8)
            x9 = self.pool(x9)

            x1 = self.bn2a(F.relu(self.conv2a(x1)))
            x2 = self.bn2b(F.relu(self.conv2b(x2)))
            x3 = self.bn2c(F.relu(self.conv2c(x3)))
            x4 = self.bn2d(F.relu(self.conv2d(x4)))
            x5 = self.bn2e(F.relu(self.conv2e(x5)))
            x6 = self.bn2f(F.relu(self.conv2f(x6)))
            x7 = self.bn2g(F.relu(self.conv2g(x7)))
            x8 = self.bn2h(F.relu(self.conv2h(x8)))
            x9 = self.bn2i(F.relu(self.conv2i(x9)))

            x1 = self.pool(x1)
            x2 = self.pool(x2)
            x3 = self.pool(x3)
            x4 = self.pool(x4)
            x5 = self.pool(x5)
            x6 = self.pool(x6)
            x7 = self.pool(x7)
            x8 = self.pool(x8)
            x9 = self.pool(x9)

            x = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8,x9), 2)

            p = x.size()
            (_, C, W) = x.data.size()
            x1 = x.view(-1, C * W)
            x1 = self.bnm1(F.tanh(self.map1x1(x1)))

            x2 = x.view(-1, C * W)
            x2 = self.bnm1(F.tanh(self.map1x2(x2)))

            #x1 = self.bn2a(F.relu(self.conv2a(x1)))
            #x2 = self.bn2b(F.relu(self.conv2b(x2)))
            #x1 = self.pool(x1)
            #x2 = self.pool(x2)

            #print(x1.size())
            #x1 = self.bn3a(F.relu(self.conv3a(x1)))
            #x2 = self.bn3b(F.relu(self.conv3b(x2)))
            #x1 = self.pool(x1)
            #x2 = self.pool(x2)

            #print(x1.size())
            #x1 = self.bn4a(F.relu(self.conv4a(x1)))
            #x2 = self.bn4b(F.relu(self.conv4b(x2)))
            #x1 = self.pool(x1)
            #x2 = self.pool(x2)

            #print(x1.size())
            #x1 = self.bn5a(F.relu(self.conv5a(x1)))
            #x2 = self.bn5b(F.relu(self.conv5b(x2)))
            #x1 = self.pool(x1)
            #x2 = self.pool(x2)


            #print(x1.size())
            #p = x1.size()
            #(_, C, W) = x1.data.size()
            #x1 = x1.view(-1, C * W)
            #x1 = self.bnm1(F.tanh(self.map1x1(x1)))

            #x2 = x2.view( -1, C * W)
            #x2 = self.bnm2(F.tanh(self.map1x2(x2)))
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
