import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
#from Transforms import ToLab, ToTensor

# -------------  Network class  ------------- #
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 9, padding=5)
        self.conv2 = nn.Conv2d(5, 10, 7, padding=3)
        self.conv3 = nn.Conv2d(10, 10, 7, padding=3)
        self.conv4 = nn.Conv2d(10, 10, 7, padding=3)
        self.conv5 = nn.Conv2d(10, 10, 7, padding=3)
        self.conv6 = nn.Conv2d(10, 10, 7, padding=3)
        self.conv7 = nn.Conv2d(10, 10, 3, padding=1)
        self.conv8 = nn.Conv2d(10, 5, 3, padding=1)
        self.conv9 = nn.Conv2d(5, 1, 3, padding=0)
        self.ct1 = nn.ConvTranspose2d(10, 10, 4, stride=2, padding=1, output_padding=(1,0))
        self.ct2 = nn.ConvTranspose2d(10, 10, 4, stride=2, padding=1)
        self.ct3 = nn.ConvTranspose2d(10, 10, 4, stride=2, padding=1, output_padding=(1,1))
        self.ct4 = nn.ConvTranspose2d(10, 10, 4, stride=2, padding=1, output_padding=(0,1))
        self.bn1 = nn.BatchNorm2d(5, affine=False)
        self.bn2 = nn.BatchNorm2d(10, affine=False)
        self.bn3 = nn.BatchNorm2d(10, affine=False)
        self.bn4 = nn.BatchNorm2d(10, affine=False)
        self.bn5 = nn.BatchNorm2d(10, affine=False)
        self.bn6 = nn.BatchNorm2d(10, affine=False)
        self.bn7 = nn.BatchNorm2d(10, affine=False)
        self.bn8 = nn.BatchNorm2d(5, affine=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.map = nn.Linear(80, 80)


    def forward(self, x):
        x = x.unsqueeze(1).float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool4(x)
        x = F.relu(self.bn6(self.conv6(x)))
        p = x.size()
        (_, C, H, W) = x.data.size()
        x = x.view( -1 , C * H * W)
        x= F.relu(self.map(x))
        x = x.view(p)
        x = F.relu((self.ct1(x)))
        x = F.relu((self.ct2(x)))
        x = F.relu((self.ct3(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu((self.ct4(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu((self.conv9(x)))
        return x.squeeze().double()


        # -------------  Training settings  ------------- #
    def args(self):
        parser = argparse.ArgumentParser(description='PyTorch rirnet')
        parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', 
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=500, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                            help='learning rate (default: 0.005)')
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
        parser.add_argument('--save-interval', type=int, default=10, 
                            help='how many batches to wait before saving network')
        parser.add_argument('--plot', type=bool, default=True, 
                            help='show plot while training (turn off if using ssh)')
        parser.add_argument('--db_path', type=str, default='FIXME', 
                            help='path to folder that contains database csv')
        parser.add_argument('--db_ratio', type=float, default=0.9, 
                            help='ratio of the db to use for training')
        parser.add_argument('--save_timestamps', type=bool, default=True,
                            help='enables saving of timestamps to csv')
        args, unknown = parser.parse_known_args()
        return args


    def transform(self):
        pass
        #    data_transform = transforms.Compose([
#        ToLab(),
#        ToNormalized(),
#        ToTensor(),
#        ])
#       return data_transform
