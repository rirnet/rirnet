import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
from rirnet.transforms import ToUnitNorm, ToTensor, ToNormalized, ToNegativeLog

def farthest_point_sampling(points, n_centroids):
    device = points.device
    B, D, N = points.size()
    C = n_centroids
    centroids = torch.zeros(B, D, C, dtype=points.dtype).to(device)
    ind = torch.randint(0, N, (B,), dtype=torch.long)
    batch_indices = torch.arange(B)
    centroids[batch_indices, :, 0] = points[batch_indices, :, ind]
    distances = dist_to_point(centroids[:, :, 0].unsqueeze(2), points)
    for i in range(1, n_centroids):
        ind = torch.argmax(distances, dim=1)
        centroids[batch_indices, :, i] = points[batch_indices, :, ind]
        distances = torch.min(distances, dist_to_point(centroids[batch_indices, :, i].unsqueeze(2), points))
    return centroids


def dist_to_point(p0, points):
    return torch.sum((p0 - points)**2, 1)


def square_dist_mat(centroids, points):
    B, _, K = centroids.size()
    _, _, N = points.size()
    dist = -2 * torch.matmul(centroids.permute(0, 2, 1), points)
    dist += torch.sum(centroids ** 2, 1).view(B, K, 1)
    dist += torch.sum(points ** 2, 1).view(B, 1, N)
    return dist


def group_points(points, n_centroids, radius, max_in_group, relative_pos):
    B, D, N = points.size()
    C = n_centroids
    K = max_in_group
    centroids = farthest_point_sampling(points, C)
    group_idx = torch.arange(N, dtype=torch.long).view(1, 1, N).repeat([B, C, 1])
    sqrdists = square_dist_mat(centroids, points)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :K]
    group_first = group_idx[:, :, 0].view(B, C, 1).repeat([1, 1, K])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    grouped_points = index_points(points, group_idx).view(B, D, C, K)
    if relative_pos:
        grouped_points -= centroids.unsqueeze(3).repeat([1, 1, 1, K])
    return grouped_points


def index_points(points, group_idx):
    B, _, _ = points.size()
    view_shape = list(group_idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(group_idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, :, group_idx]


class SetAbstraction(nn.Module):
    def __init__(self, C, D, K, bottle_neck, r, relative_pos):
        super(SetAbstraction, self).__init__()
        self.C = C
        self.D = D
        self.K = K
        self.bottle_neck = bottle_neck
        self.r = r
        self.relative_pos = relative_pos
        self.conv1 = nn.Conv2d(self.D, 64, 1)
        self.conv2 = nn.Conv2d(64, 128, 1)
        self.conv3 = nn.Conv2d(128, 256, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(self.C*256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, self.bottle_neck)

        #init.constant_(self.transform.weight, 0)
        #init.eye_(self.transform.bias.view(self.input_features, self.input_features))

    def forward(self, x):
        x = group_points(x, self.C, self.r, self.K, self.relative_pos)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = nn.MaxPool2d((1, self.K))(x)
        x = x.view(-1, 256 * self.C)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1, self.bottle_neck)
        # add identity matrix?
        return x


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.C = 32
        self.D = 2
        self.K = 32
        self.bottle_neck = 8
        #self.sa1 = SetAbstraction(self.C, self.D, self.K, self.bottle_neck, 0.5)
        self.sa2 = SetAbstraction(self.C, self.D, self.K, self.bottle_neck, 1, False)
        #self.sa3 = SetAbstraction(self.C, self.D, self.K, self.bottle_neck, 2)
        self.conv = nn.Conv1d(1, self.bottle_neck, 1, padding=0)
        self.bn1 = nn.BatchNorm1d(self.bottle_neck)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.fc1 = nn.Linear(self.bottle_neck ** 2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(self.D * self.K, self.D * self.K)

    def forward(self, x):
        view = x.size()
        x = self.encode(x)
        x = self.decode(x, view)
        return x

    def encode(self, x):
        #x1 = self.sa1(x).unsqueeze(1)
        x = self.sa2(x).unsqueeze(1)
        #x3 = self.sa3(x).unsqueeze(1)
        #x = torch.cat((x1, x2, x3), dim=1)
        x = F.relu(self.bn1(self.conv(x)))
        _, D, K = x.size()
        x = x.view(-1, D * K)
        x = self.fc4(x)
        return x

    def decode(self, x, view):
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(view)
        return x


    def args(self):
        parser = argparse.ArgumentParser(description='PyTorch rirnet')
        parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
        parser.add_argument('--save-interval', type=int, default=100,
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
