import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
from rirnet.transforms import ToUnitNorm, ToTensor, ToNormalized, ToNegativeLog
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1_t = nn.Conv2d(1, 8, (65,1))
        self.conv1_f = nn.Conv2d(1, 8, (1,225))
        self.conv2_t = nn.Conv1d(8, 16, 3, padding = 1)
        self.conv2_f = nn.Conv1d(8, 16, 3, padding = 1)
        self.conv3_t = nn.Conv1d(16, 32, 3, padding = 1)
        self.conv3_f = nn.Conv1d(16, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(1, 4, 3, padding = 1)
        self.conv3 = nn.Conv2d(4, 8, 3, padding = 1)
        self.conv6 = nn.Conv2d(4, 1, 3, padding = 1)

        self.bn1_t = nn.BatchNorm2d(8)
        self.bn1_f = nn.BatchNorm2d(8)
        self.bn2_t = nn.BatchNorm1d(16)
        self.bn2_f = nn.BatchNorm1d(16)
        self.bn3_t = nn.BatchNorm1d(32)
        self.bn3_f = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(8)

        self.cbn1 = nn.BatchNorm2d(64)
        self.cbn2 = nn.BatchNorm2d(32)
        self.cbn3 = nn.BatchNorm2d(16)
        self.cbn4 = nn.BatchNorm2d(4)

        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, 4096)
        
        self.pool = nn.MaxPool2d(2)
        self.pool_t = nn.MaxPool1d(3)
        self.pool_f = nn.MaxPool1d(2)

        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, 2, padding=[1,1])
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, 2)
        self.deconv3 = nn.ConvTranspose2d(32, 16, 3, 2, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(16, 4, 3, 2)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        #print(x.size())
        #x = torch.narrow(x, -1, 0, 362)
        x = x.unsqueeze(1)
        x_t = F.relu(self.bn1_t(self.conv1_t(x))).squeeze(2)
        x_f = F.relu(self.bn1_f(self.conv1_f(x))).squeeze(3)
        x_t = self.pool_t(x_t)
        x_f = self.pool_f(x_f)

        x_t = self.dropout(x_t)
        x_f = self.dropout(x_f)

        x_t = F.relu(self.bn2_t(self.conv2_t(x_t)))
        x_f = F.relu(self.bn2_f(self.conv2_f(x_f)))
        
        x_t = self.pool_t(x_t)
        x_f = self.pool_f(x_f)

        x_t = self.dropout(x_t)
        x_f = self.dropout(x_f)

        x_t = F.relu(self.bn3_t(self.conv3_t(x_t)))
        x_f = F.relu(self.bn3_f(self.conv3_f(x_f)))
        
        x_t = self.pool_t(x_t)
        x_f = self.pool_f(x_f)

        x_t = self.dropout(x_t)
        x_f = self.dropout(x_f)

        x = torch.cat((x_t, x_f), 2)
        

        (B, W, H) = x.size()
        x = x.view(B, W * H)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(B,128,4,8)
        x = F.relu(self.cbn1(self.deconv1(x)))
        x = self.dropout(x)
        x = F.relu(self.cbn2(self.deconv2(x)))
        x = self.dropout(x)
        x = F.relu(self.cbn3(self.deconv3(x)))
        x = self.dropout(x)
        x = F.relu(self.cbn4(self.deconv4(x)))
        x = self.dropout(x)
        x = self.conv6(x)

        return x


    def args(self):
        parser = argparse.ArgumentParser(description='PyTorch rirnet')
        parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                            help='learning rate (default: 0.005)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--train_db_path', type=str, default='../../db_fft_horder/train.csv',
                            help='path to train csv')
        parser.add_argument('--val_db_path', type=str, default='../../db_fft_horder/val.csv',
                            help='path to val csv')
        parser.add_argument('--log_interval', type=int, default=10,
                            help='log interval')
        self.args, unknown = parser.parse_known_args()
        return self.args


        # -------------  Transform settings  ------------- #
    def data_transform(self):
        data_transform = transforms.Compose([ToTensor()])
        return data_transform

    def target_transform(self):
        target_transform = transforms.Compose([ToTensor()])
        return target_transform
