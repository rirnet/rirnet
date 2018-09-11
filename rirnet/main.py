from __future__ import print_function
import sys
from RirnetDatabase import RirnetDatabase
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
from importlib import import_module
from glob import glob
from skimage.color import rgb2luv, luv2rgb

#TODO this code needs love, 'plt_loss_vector' and similar needs better names.
#TODO constants should be added to self.args in net_master.py

# -------------  Initialization  ------------- #
class Model:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        sys.path.append(model_dir)
        net = import_module('net')
        self.model = net.Net()
        self.args = self.model.args()
        torch.manual_seed(self.args.seed)

        use_cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(self.device)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        list_epochs = glob('*.pth')
        if list_epochs == []:
            start_epoch = 1
        else:
            start_epoch = max([int(e.split('.')[0]) for e in list_epochs])
            self.model.load_state_dict(torch.load(os.path.join(model_dir, '{}.pth'.format(str(start_epoch)))))
            start_epoch += 1

        self.epoch = start_epoch
        self.csv_path = os.path.join(self.args.db_path, 'db.csv')
        data_transform = self.model.transform()
        train_db = RirnetDatabase(csv_file=self.csv_path, root_dir=self.args.db_path, transform=data_transform)
        self.train_loader = torch.utils.data.DataLoader(train_db, batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        #self.test_loader = torch.utils.data.DataLoader(TigernetDataset(
        #    csv_file='../../code/train.csv', root_dir='../../dataset', transform=data_transform),
        #    batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)


    def train(self):
        self.model.train()
        for batch_idx, (source, target) in enumerate(self.train_loader):
            source, target = source.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(source)
            loss = F.mse_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * len(source), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))


    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.test_loader):
                source, target = source.to(self.device), target.to(self.device)
                output = self.model(source)
                # test_loss += F.mse_loss(output, target)
                # correct += output.eq(target.data).sum().item()
                # test_loss /= len(test_loader.dataset)
                # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                #    test_loss, test_loss, len(test_loader.dataset),
                #    100. * test_loss / len(test_loader.dataset)))
                # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                #    test_loss, correct, len(test_loader.dataset),
                #    100. * correct / len(test_loader.dataset)))


    def save(self):
        full_path = os.path.join(self.model_dir, '{}.pth'.format(str(self.epoch)))
        torch.save(self.model.state_dict(), full_path)


def main(model_dir):
    model = Model(model_dir)

    for epoch in range(model.epoch, model.args.epochs + 1):
        model.epoch = epoch
        model.train()
        if epoch % model.args.save_interval == 0:
            model.save()
        # test()


if __name__ == '__main__':
    main(sys.argv[1])
