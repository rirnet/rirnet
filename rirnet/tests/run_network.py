from __future__ import print_function
from torch.autograd import Variable
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
import numpy as np
from importlib import import_module
from glob import glob
import rirnet.acoustic_utils as au

# -------------  Initialization  ------------- #
class Model:
    def __init__(self, network_path, model_dir):
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

        self.model.load_state_dict(torch.load(network_path))


    def forward(self, data):
        data = self.transform(data)
        data = data.to(self.device)
        output = self.model(data)
        return output


    def transform(self, data):
        return torch.from_numpy(data).unsqueeze(0).float()


def main(network_path, data_csv_path):
    model_dir,_ = os.path.split(network_path)
    data_dir,_ = os.path.split(data_csv_path)
    model = Model(network_path, model_dir)

    csv_data = pd.read_csv(data_csv_path)

    source_path = csv_data['data_path'][0]
    target_path = csv_data['target_path'][0]

    mean_target = np.load(os.path.join(data_dir, 'mean_target.npy'))
    std_target = np.load(os.path.join(data_dir, 'std_target.npy'))
    mean_data = np.load(os.path.join(data_dir, 'mean_data.npy'))
    std_data = np.load(os.path.join(data_dir, 'std_data.npy'))

    source = np.load(source_path)
    target = np.load(target_path)

    output = model.forward(source)/3
    output = output.cpu().detach().numpy()

    output = output*std_target+mean_target
    target = target*std_target+mean_target

    irf_output = au.mfcc_to_waveform(output, 44100, 2**16)
    irf_target =  au.mfcc_to_waveform(target, 44100, 2**16)

    x,rate = au.read_wav('../../audio/cher.mp3')
    y = au.convolve(x, irf_output)
    au.save_wav('test.wav', y, rate)
    au.play_file('test.wav')

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(221)
    ax2 = fig1.add_subplot(222)
    ax3 = fig1.add_subplot(223)
    ax4 = fig1.add_subplot(224)
    ax1.plot(output)
    ax1.set_title('Output')
    ax2.plot(irf_output)
    ax2.set_title('Output IRF')
    ax3.plot(target)
    ax3.set_title('Target')
    ax4.plot(irf_target)
    ax4.set_title('Target IRF')
    plt.show()


if __name__ == '__main__':
    network_path =  sys.argv[1]
    data_csv_path = sys.argv[2]
    main(network_path, data_csv_path)

