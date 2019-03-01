#!/usr/bin/env python3
import layout
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing as mp
import rirnet.roomgen as rg
import rirnet.acoustic_utils as au
import numpy as np
import yaml
import sys
import csv
import os
import random
import pandas as pd
import glob
import librosa
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import operator
import signal
import sys
from scipy.spatial.distance import cdist as cdist
from sklearn.utils import shuffle

class RirGenerator:
    def __init__(self, db_setup, n_total):
        self.n_total = n_total
        self.i_total = 0
        self.db_setup = db_setup
        self.h_length = db_setup['mfcc_length']*256
        self.min_peaks_length = db_setup['min_peaks_length']
        self.discarded = 0
        self.output = mp.Queue()
        self.processes = [mp.Process(target=self.compute_room_proc) for x in range(db_setup['n_proc'])]
        for p in self.processes:
            p.start()

    def __iter__(self):
        return self

    def __next__(self):
        if self.i_total == self.n_total:
            raise StopIteration

        i_produced = 0
        h_list = []
        info_list = []
        peaks_list = []
        self.terminate_dead_proc()
        room = self.output.get()
        self.processes.append(mp.Process(target=self.compute_room_proc))
        new_proc = self.processes[-1]
        new_proc.start()

        for i_rir, rir in enumerate(room.rir):
            cut_rir = remove_leading_zeros(list(rir[0]))
            rir_length = len(cut_rir)
            peaks = room.peaks[i_rir]
            peaks_length = len(peaks[0])

            if rir_length > self.h_length:
                self.discarded += 1
                return self.__next__()
            else:
                rir = au.pad_to(cut_rir, self.h_length, 0)

                h_list.append(rir)
                if peaks_length < self.min_peaks_length:
                    self.discarded += 1
                    return self.__next__()
                #else:
                #    times = au.pad_to(peaks[0], self.peaks_length, np.max(peaks[0]))
                #    alphas = au.pad_to(peaks[1], self.peaks_length, np.min(peaks[1]))
                #    peaks = [times, alphas]

                peaks_list.append(peaks)
                info_list.append([room.corners, room.absorption, room.mic_array.R[:, i_rir], room.sources[0].position])

                interrupted = False

                i_produced += 1
                if self.i_total + i_produced == self.n_total:
                    for process in self.processes:
                        process.terminate()
                    break
                if interrupted:
                    for process in self.processes:
                        process.terminate()
                    print('Terminated processes')
                    sys.exit()
        self.i_total += i_produced
        return h_list, peaks_list, info_list

    def terminate_dead_proc(self):
        while self.processes and not self.processes[0].is_alive():
            p = self.processes.pop(0)
            p.terminate()

    def compute_room_proc(self):
        room = rg.generate_from_dict(self.db_setup)
        room.compute_rir()
        peaks = self.compute_peaks(room)
        room.peaks = peaks
        self.output.put(room)

    def compute_peaks(self, room):
        if room.visibility is None:
            room.image_source_model()
        peaks = []
        for m, mic in enumerate(room.mic_array.R.T):
            distances = room.sources[0].distance(mic)
            times = distances/343.0*room.fs
            alphas = room.sources[0].damping / (4.*np.pi*distances)
            slice = tuple(np.where(room.visibility[0][m] == 1))
            alphas = alphas[slice]
            times = times[slice]

            if self.db_setup['order_sorting']:
                orders = room.sources[0].orders[slice]

                ordered_inds = []
                for order in range(min(orders), max(orders)):
                    order_inds = np.where(orders == order)[0]
                    time_inds = np.argsort(times[order_inds])
                    for ind in order_inds[time_inds]:
                        ordered_inds.append(ind)
                peaks.append([times[ordered_inds] - min(times[ordered_inds]), alphas[ordered_inds]])
            else:
                peaks.append([times - min(times), alphas])
        return peaks


def remove_leading_zeros(rir):
    ind_1st_nonzero = next((i for i, x in enumerate(rir) if x), None)
    rir[0:ind_1st_nonzero] = []
    return np.array(rir)


def convolve_and_pad(wav, h_list):
    data_list = []

    for i_h, h in enumerate(h_list):
        y = au.convolve(wav, h)
        y_length = au.next_power_of_two(np.size(y))
        data = au.pad_to(y, y_length, 0)
        data_list.append(data)

    return np.array(data_list)


def load_wavs(audio_folder, db_setup):
    audio_list = db_setup['source_audio']
    rate = db_setup['fs']
    wav_list = []
    if audio_list == ['']:
        audio_list = glob.glob(os.path.join(audio_folder, '*.wav'))
    for audio_filename in audio_list[:10]:
        wav_path = audio_filename
        wav = au.normalize(au.read_wav(wav_path, rate)[0])
        wav_list.append(wav)
    return wav_list


def waveforms_to_mfccs(waveforms, db_setup):
    fs = db_setup['fs']
    n_mfcc = db_setup['n_mfcc']
    mfccs = [au.waveform_to_mfcc(waveform, fs, n_mfcc)[1][:,:-1] for waveform in waveforms]
    return mfccs


def pad_list_to_pow2(h_list):
    longest_irf = len(max(h_list, key=len))
    target_length = au.next_power_of_two(longest_irf)
    h_list = [au.pad_to(h, target_length) for h in h_list]
    return h_list


def calculate_delta_features(data_list):
    delta_list = []
    delta_2_list = []
    for data in data_list:
        delta_list.append(librosa.feature.delta(data))
        delta_2_list.append(librosa.feature.delta(data, order=2))
    return delta_list, delta_2_list


def parse_yaml(filename):
    with open(filename, 'r') as stream:
        db_setup = yaml.load(stream)
    return db_setup


def build(wav_list):
    db_setup = parse_yaml('db_setup.yaml')
    rir_generator = RirGenerator(db_setup, 1)

    for h_list, peaks_list, info_list in rir_generator:
        wav = wav_list[2]
        waveform_list = convolve_and_pad(wav, h_list)

        plt.figure()
        plt.plot(wav_list[2][:10000]/np.max(np.abs(wav_list[2])))
        plt.title('Waveform anechoic signal')
        plt.xlabel('Sample (n)')
        plt.ylabel('Amplitude')
        plt.grid()
        #plt.yticks([])
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig('signal.eps')
        plt.savefig('signal.png')
        print('saved wav plot')

        plt.figure()
        plt.plot(waveform_list[0][:10000]/np.max(np.abs(waveform_list[0])))
        plt.title('Waveform reverberant signal')
        plt.xlabel('Sample (n)')
        plt.ylabel('Amplitude')
        plt.grid()
        #plt.yticks([])
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig('rev_signal.eps')
        plt.savefig('rev_signal.png')
        print('saved rev signal plot')


        plt.figure()
        plt.plot(h_list[0][:10000]/max(h_list[0]))
        plt.title('Impulse response function')
        plt.xlabel('Sample (n)')
        plt.ylabel('Amplitude')
        plt.grid()
        #plt.yticks([])
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig('irf.eps')
        plt.savefig('irf.png')
        print('saved irf signal plot')

        print(np.shape(peaks_list))

        plt.figure()
        plt.plot(peaks_list[0][0]/8000*44100/1000, (-np.log(peaks_list[0][1])-min(-np.log(peaks_list[0][1])))/6, '.')
        plt.title('Peak data')
        plt.xlabel('Time [ms]')
        plt.ylabel('-log(Amplitude)')
        plt.grid()
        #plt.yticks([])
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig('peaks.eps')
        plt.savefig('peaks.png')
        print('saved irf signal plot')

        mfcc_list = waveforms_to_mfccs(waveform_list, db_setup)
        delta_1_list, delta_2_list = calculate_delta_features(mfcc_list)
        plt.close('all')


        mfcc = mfcc_list[0][2:33,:33]
        delta_1 = delta_1_list[0][2:33,:33]
        delta_2 = delta_2_list[0][2:33,:33]


        mfcc_im = mfcc - np.min(mfcc)
        mfcc_im /= np.max(mfcc_im)
        delta_1_im = delta_1 - np.min(delta_1)
        delta_1_im /= np.max(delta_1_im)
        delta_2_im = delta_2 - np.min(delta_2)
        delta_2_im /= np.max(delta_2_im)

        image = np.dstack([mfcc_im, delta_1_im, delta_2_im])

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey = True, sharex = True)

        ax1.title.set_text('MFCC')
        ax2.title.set_text('Delta 1')
        ax3.title.set_text('Delta 2')
        ax4.title.set_text('RGB')

        im1 = ax1.imshow(mfcc, origin='lower', interpolation='None', extent=[0,33,2,33])
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
        cbar.set_ticks([np.ceil(np.min(mfcc)),0,np.floor(np.max(mfcc))])
        ax1.tick_params(axis=u'both', which=u'both',length=0)
        ax1.set_xticks([0, 32], minor=False)
        ax1.set_yticks([2, 32], minor=False)

        im2 = ax2.imshow(delta_1, origin='lower', interpolation='None', extent=[0,33,2,33])
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im2, cax=cax, orientation='vertical')
        cbar.set_ticks([np.ceil(np.min(delta_1)),0,np.floor(np.max(delta_1))])
        ax2.tick_params(axis=u'both', which=u'both',length=0)
        ax2.set_xticks([0, 32], minor=False)
        ax2.set_yticks([2, 32], minor=False)

        im3 = ax3.imshow(delta_2, origin='lower', interpolation='None', extent=[0,33,2,33])
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im3, cax=cax, orientation='vertical')
        cbar.set_ticks([np.ceil(np.min(delta_2)),0,np.floor(np.max(delta_2))])
        ax3.tick_params(axis=u'both', which=u'both',length=0)
        ax3.set_xticks([0, 32], minor=False)
        ax3.set_yticks([2, 32], minor=False)

        im4 = ax4.imshow(image, origin='lower', interpolation='None', extent=[0,33,2,33])
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im4, cax=cax, orientation='vertical')
        cbar.set_ticks([int(0),int(1)])
        ax4.tick_params(axis=u'both', which=u'both',length=0)
        ax4.set_xticks([0, 32], minor=False)
        ax4.set_yticks([2, 32], minor=False)

        plt.tight_layout()

        plt.savefig('mfcc.eps')

def main():

    fig_width_pt = 213.4*2 # full width 426.8, half width 213.4  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0      # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]

    params = {'backend': 'ps',
              'axes.labelsize': 12,
              'font.size': 12,
              'font.family': 'STIXGeneral',
              #'figure.autolayout' : True,
              'legend.fontsize': 12,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'text.usetex': True,
              'figure.figsize': fig_size}
    pylab.rcParams.update(params)

    db_setup = parse_yaml('db_setup.yaml')
    path_audio = '../../audio/chamber/train'
    wav_list = load_wavs(path_audio, db_setup)

    _ = build(wav_list)

if __name__ == "__main__":
    main()
