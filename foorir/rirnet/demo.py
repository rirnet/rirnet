import rirnet.acoustic_utils as au
import rirnet.misc as misc
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy as sp

fs = 16384

signal, _ = au.read_wav('../../audio/livingroom/full/mario.wav', fs)
start = np.max(np.random.randint(signal.shape[0]-fs), 0)
snippet = signal[start:start+fs]

net, _ = misc.load_latest('../models', 'net')
net.to("cuda")


a = True

while a:
    start = np.max(np.random.randint(signal.shape[0]-fs), 0)
    snippet = signal[start:start+fs]

    output = au.split_signal(signal, rate = fs, segment_length = fs//4, min_energy = 10, max_energy = 20, hop_length=128, debug=False)
    if len(output) > 0:
        segment = output[0]

        _, _, rev_stft = sp.signal.stft(segment, fs=fs, nperseg=128)
        print(np.shape(rev_stft))
        _, sig = sp.signal.istft(rev_stft, fs=fs, nperseg=128)

        print(len(sig))
        plt.plot(sig)
        plt.show()

        #print(rev_stft.shape)
        #rev_stft = torch.from_numpy(-np.log(np.abs(rev_stft))).float().unsqueeze(0).cuda()
        
        #print(rev_stft.size())
        a = False
        #rir_stft = net(rev_stft)

        #plt.imshow(rir_stft)
        #plt.pause(0.001)
