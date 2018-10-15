import sys
import sounddevice as sd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import rirnet.acoustic_utils as au
import os
from importlib import import_module
import torch
import queue

class Model:
    def __init__(self, model_path):
        model_dir, model_name = os.path.split(model_path)
        self.model_dir = model_dir
        sys.path.append(model_dir)
        net = import_module('net')
        self.model = net.Net()
        self.args = self.model.args()
        #torch.manual_seed(self.args.seed)


        use_cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(self.device)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        temp = torch.load(model_path, map_location='cpu')

        for key, val in temp.items():
            print(key)

        self.model.load_state_dict(temp)
        data_transform = self.model.transform()


    def evaluate(self, source):
        self.model.eval()
        with torch.no_grad():
            source = source.to(self.device)
            output = self.model(source)
            eval_loss = getattr(F, self.args.loss_function)(output, target).item()
            eval_loss_list.append(eval_loss)


def audio_callback(indata, frame, time, status):
    if status:
        print(status)
    q.put(indata)
    

def main(model_path):
    global q 
    q = queue.Queue()
    duration = 2
    fs = 44100
    array = np.zeros([duration * fs, 1])

    sd.default.samplerate = 44100
    sd.default.channels = 1

    #model = Model(model_path)


    recording = sd.rec(duration * fs)
    sd.wait()

    plt.plot(recording)
    plt.show()


    with sd.InputStream(blocksize=64, callback=audio_callback):

    #mfcc = au.waveform_to_mfcc(recording)
        while(True):
            print(q.get())
            recording = q.get()
            print(np.shape(recording))
            mfcc = au.waveform_to_mfcc(recording, fs, 40)
            print(mfcc)
    #output = model.evaluate(mfcc)

    #plt.plot(recording)
    #plt.show()

if __name__ == '__main__':
    main(sys.argv[1])
