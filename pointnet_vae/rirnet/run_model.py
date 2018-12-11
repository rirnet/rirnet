import rirnet.acoustic_utils as au
import rirnet.misc as misc
import numpy as np
import matplotlib.pyplot as plt

import torch
import os

from glob import glob

class Model:
    def __init__(self, model_dir):
        model_dir = os.path.abspath(model_dir)
        self.extractor, _ = misc.load_latest(model_dir, 'extractor')
        self.autoencoder, _ = misc.load_latest(model_dir, 'autoencoder')
        self.extractor = self.extractor.double().eval()
        self.autoencoder = self.autoencoder.double().eval()

    def forward(self, nw_input):
        with torch.no_grad():
            nw_output = self.autoencoder(self.extractor(nw_input), decode=True).numpy()
        return nw_output

def preprocess(mfccs):
    delta_1_list, delta_2_list = au.calculate_delta_features(mfccs)
    data_list = [[mfcc, delta_1, delta_2] for mfcc, delta_1, delta_2 in zip(mfccs, delta_1_list, delta_2_list)]

    mean = np.load('../database/mean_data.npy').T
    std = np.load('../database/std_data.npy').T
    nw_input = [np.nan_to_num((np.array(data).T - mean)/std).T for data in data_list]

    return torch.tensor(nw_input)


def postprocess(nw_output, points_omitted, debug=False):
    batch_size = np.shape(nw_output)[0]
    rir_list = []
    for i in range(batch_size):
        filled_times_output, filled_alphas_output = misc.fill_peaks(nw_output[i,0,:], nw_output[i,1,:], points_omitted, debug)
        output_rir = misc.reconstruct_rir(filled_times_output, filled_alphas_output)
        rir_list.append(output_rir)
    return rir_list

def main():
    n_mfcc = 40
    model_dir = '../models'
    model = Model(model_dir)

    signal, rate = au.read_wav('../../audio/trapphus.wav')
    signal_segment_list = au.split_signal(signal, rate = rate, segment_length = 60000, min_energy = 100, max_energy=4, debug=False)
    signal_segment_list = [au.pad_to(segment, 2**16) for segment in signal_segment_list]
    mfccs = [au.waveform_to_mfcc(segment, rate, n_mfcc)[1][:,:-1] for segment in signal_segment_list]
    nw_input = preprocess(mfccs)
    nw_output = model.forward(nw_input)
    rir_list = postprocess(nw_output, 0, True)
    rir_list_2 = postprocess(nw_output, 20, True)

    plt.show()

if __name__ == "__main__":
    main()
