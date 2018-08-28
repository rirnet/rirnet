from __future__ import print_function
import sys
from TigernetDataset import TigernetDataset
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
        self.args = net.args()
        use_cuda = not self.args.no_cuda and torch.cuda.is_available()
        torch.manual_seed(self.args.seed)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        self.model = net.Net().to(self.device)
        list_epochs = glob('*.pth')

        if list_epochs == []:
            start_epoch = 1
        else:
            start_epoch = max([int(e.split('.')[0]) for e in list_epochs])
            self.model.load_state_dict(torch.load(os.path.join(model_dir, '{}.pth'.format(str(start_epoch)))))
            start_epoch += 1

        self.epoch = start_epoch
        data_transform = net.transform()
        self.train_loader = torch.utils.data.DataLoader(TigernetDataset(
            csv_file='../../code/train.csv', root_dir='../../dataset',
            transform=data_transform), batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        self.test_loader = torch.utils.data.DataLoader(TigernetDataset(
            csv_file='../../code/train.csv', root_dir='../../dataset', transform=data_transform),
            batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.loss_vector = []
        self.plt_loss_vector = []
        if self.args.plot:
            self.fig1 = plt.figure(1)
            self.fig2 = plt.figure(2)
            self.axplot = self.fig1.add_subplot(111)
            self.ax1 = self.fig2.add_subplot(131)
            self.ax2 = self.fig2.add_subplot(132)
            self.ax3 = self.fig2.add_subplot(133)


    def train(self):
        #TODO l2r does not belong here and has a bad name
        def l2r(im):
            im = im.double().detach().cpu().numpy()[0,:,:,:].T
            im = luv2rgb((im)*255-128)
            return np.rot90(im, k=1, axes=(1,0))

        self.model.train()
        for batch_idx, (source, target) in enumerate(self.train_loader):
            source, target = source.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(source)
            loss = F.mse_loss(output, target)
            loss.backward()
            self.optimizer.step()
            self.loss_vector.append(loss.item())
            if batch_idx % self.args.log_interval == 0:
                self.plt_loss_vector.append(np.mean(self.loss_vector))
                self.loss_vector[:] = []
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * len(source), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), self.plt_loss_vector[-1]))
                self.axplot.plot(np.arange(0, len(self.plt_loss_vector)*self.args.log_interval, self.args.log_interval), self.plt_loss_vector,'ko-')
                self.axplot.set_title('Epoch: %i' %self.epoch)
                self.axplot.set_ylabel('MSE loss')
                self.axplot.set_xlabel('Batch no. (batch size is %i)' %self.args.batch_size)
                if self.args.plot:
                    plt.draw()
                    plt.pause(0.01)
                    self.fig1.tight_layout()
                    imout = l2r(output)
                    imin = l2r(source)
                    imtar = l2r(target)

                    self.ax1.imshow(imout)
                    self.ax1.set_title('Network Output')
                    self.ax2.imshow(imin)
                    self.ax2.set_title('Network Input')
                    self.ax3.imshow(imtar)
                    self.ax3.set_title('Target')

                    plt.draw()
                    plt.pause(0.001)
                    self.ax1.clear()
                    self.ax2.clear()
                    self.ax3.clear()


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


def main():
    model = Model(os.getcwd())

    for epoch in range(model.epoch, model.args.epochs + 1):
        model.epoch = epoch
        model.train()
        if epoch % model.args.save_interval == 0:
            model.save()
        # test()


if __name__ == '__main__':
    main()
