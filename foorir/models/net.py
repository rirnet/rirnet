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

        self.conv1 = nn.Conv2d(1, 4, 15, padding = 7)
        self.conv2 = nn.Conv2d(4, 8, 5, padding = 2)
        self.conv3 = nn.Conv2d(8, 16, 3, padding = 1)
        self.conv4 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv5 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv6 = nn.Conv2d(1, 1, 1)

        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)

        self.cbn1 = nn.BatchNorm2d(64)
        self.cbn2 = nn.BatchNorm2d(32)
        self.cbn3 = nn.BatchNorm2d(16)
        self.cbn4 = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(1408, 32)
        self.fc2 = nn.Linear(32, 3200)
        
        self.pool = nn.MaxPool2d(2)

        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, 2, padding=[1,0])#, output_padding=[0,1])
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, 2, padding=[1,0])#, output_padding=[0,1])
        self.deconv3 = nn.ConvTranspose2d(32, 16, 3, 2, padding=[1,0])#, output_padding=[0,1])
        self.deconv4 = nn.ConvTranspose2d(16, 1, 3, 2, padding=[1,0])#, output_padding=[0,1])
        #target size = 65, 126 = 3276 


    def forward(self, x, encode=False, decode=False):
        #print('in-size;', x.size())
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        #print(x.size())
        (_, C, W, H) = x.size()
        x = x.view(-1, C * W * H)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1,128,5,5)
        #print(x.size())
        x = F.relu(self.cbn1(self.deconv1(x)))
        #print(x.size())
        x = F.relu(self.cbn2(self.deconv2(x)))
        #print(x.size())
        x = F.relu(self.cbn3(self.deconv3(x)))
        #print(x.size())
        x = F.relu(self.cbn4(self.deconv4(x)))
        #print(x.size())
        x = self.conv6(x)
        #print(x.size())

        #target size = 65, 126 
        return F.sigmoid(x)


    def args(self):
        parser = argparse.ArgumentParser(description='PyTorch rirnet')
        parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                            help='learning rate (default: 0.005)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--train_db_path', type=str, default='../../db_fft/val.csv',
                            help='path to train csv')
        parser.add_argument('--val_db_path', type=str, default='../../db_fft/val.csv',
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
