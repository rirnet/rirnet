import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv1d(2, 2, 129, padding=64)
        self.map1 = nn.Linear(512, 64)
        self.map2 = nn.Linear(64, 1)

    def forward(self, x):
        (_, D, N) = x.size()
        x = self.conv(x)
        x = x.view(-1, D * N)
        x = F.relu(self.map1(x))
        x = torch.sigmoid(self.map2(x))
        return x

    def args(self):
        parser = argparse.ArgumentParser(description='PyTorch rirnet')
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: 0.005)')
        parser.add_argument('--momentum', type=float, default=0.00001, metavar='M',
                            help='SGD momentum (default: 0.5)')
        parser.add_argument('--loss_function', type=str, default='mse_loss',
                            help='the loss function to use. Must be EXACTLY as the function is called in pytorch docs')
        self.args, unknown = parser.parse_known_args()
        return self.args
