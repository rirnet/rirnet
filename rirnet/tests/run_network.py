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
    model = Model(network_path, model_dir)

    csv_data = pd.read_csv(data_csv_path)

    data_input_path = csv_data['path_data'][0]
    data_target_path = csv_data['path_target'][0]
    mean_y = csv_data['mean_data'][0].strip('[]').split()
    mean_h = csv_data['mean_target'][0].strip('[]').split()
    std_y = csv_data['std_data'][0].strip('[]').split()
    std_h = csv_data['std_target'][0].strip('[]').split()

    mean_h = np.array([float(x) for x in mean_h])
    mean_h = mean_h[:, None]
    std_h = np.array([float(x) for x in std_h])
    std_h = std_h[:, None]


    data_input = np.load(data_input_path)
    data_target = np.load(data_target_path)

    output = model.forward(data_input)

    output = output.detach().cpu().numpy()
    print(np.shape(output))
    output = output*std_h+mean_h
    data_target = data_target*std_h+mean_h
    irf_output = au.mfcc_to_waveform(output, 44100, 2**16)
    irf_data_target =  au.mfcc_to_waveform(data_target, 44100, 2**16)

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
    ax3.plot(data_target)
    ax3.set_title('Target')
    ax4.plot(irf_data_target)
    ax4.set_title('Target IRF')
    plt.show()


if __name__ == '__main__':
    network_path =  sys.argv[1]
    data_csv_path = sys.argv[2]
    main(network_path, data_csv_path)

