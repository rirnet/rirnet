import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from rirnet.transforms import ToTensor, ToNormalized, ToNegativeLog, ToUnitNorm
import numpy as np

# -------------  Network class  ------------- #
class Net(nn.Module):

        # -------------  Model Layers  ------------- #
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(256, 512, 1, stride=1, padding=0)
        self.conv10 = nn.Conv1d(2, 10, 5, stride=1, padding=2)
        self.conv11 = nn.Conv1d(10, 2, 5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(128, affine=True)
        self.bn2 = nn.BatchNorm2d(128, affine=True)
        self.bn3 = nn.BatchNorm2d(128, affine=True)
        self.bn4 = nn.BatchNorm2d(128, affine=True)
        self.bn5 = nn.BatchNorm2d(128, affine=True)
        self.bn6 = nn.BatchNorm2d(128, affine=True)
        self.bn7 = nn.BatchNorm2d(128, affine=True)
        self.bn8 = nn.BatchNorm2d(256, affine=True)
        self.bn9 = nn.BatchNorm2d(512, affine=True)
        self.bn10 = nn.BatchNorm1d(10, affine=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.map1 = nn.Linear(8192, 8192)
        self.map2 = nn.Linear(8192, 2048)


        # -------------  Forward Pass  ------------- #
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.pool(x)
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool(x)
        x = F.relu(self.bn9(self.conv9(x)))
        p = x.size()
        (_, C, H, W) = x.data.size()
        x = x.view( -1 , C * H * W)
        x = self.map1(x)
        x = self.map2(x)
        x = x.view(-1, 2, 1024)
        x = self.bn10(self.conv10(x))
        x = self.conv11(x)
        return x


        # -------------  Training settings  ------------- #
    def args(self):
        parser = argparse.ArgumentParser(description='PyTorch rirnet')
        parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
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
