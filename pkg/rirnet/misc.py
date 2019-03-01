from glob import glob
from importlib import import_module
from pyroomacoustics.utilities import fractional_delay
from scipy.optimize import curve_fit

import torch
import sys
import os
import pylab

import matplotlib.pyplot as plt
import numpy as np
import rirnet.acoustic_utils as au

def load_latest(abspath, identifier):
    '''
    Initialize network and load the latest saved state dict based on epoch numbers.
    Returns network and epoch number of latest network
    '''
    sys.path.append(abspath)
    model = import_module(identifier).Net()
    list_pth = glob(os.path.join(abspath, '*[!_opt]_{}.pth'.format(identifier)))
    if list_pth:
        max_epoch = max([int(os.path.basename(e).split('_')[0]) for e in list_pth])
        filename = '{}_{}.pth'.format(max_epoch, identifier)
        model.load_state_dict(torch.load(os.path.join(abspath, filename), map_location='cpu'))
    else:
        max_epoch = 0
    return model, max_epoch

def reconstruct_rir_conv(time, alpha):
    fdl = 81
    fdl2 = (fdl-1) // 2
    time = (time.astype('double')+1)*1024
    alpha = np.exp(-alpha).astype('double')
    signs = np.random.randint(0,2, len(alpha))*2-1
    #alpha *= signs
    ir = np.arange(np.ceil((1.05*time.max()) + fdl))*0

    inds = np.argsort(time)
    time = np.round(time[inds]).astype(int)
    alpha = alpha[inds]

    print(time)

    peaks = np.zeros(np.max(time)+1)
    for n, t in enumerate(time):
        peaks[t] += alpha[n]

    #peaks[time] = alpha

    ir = au.convolve(peaks, np.hanning(fdl)*np.sinc(np.linspace(-41, 41, 81)))

    start_ind = min(np.where(ir > 10**(-10))[0])
    ir = ir[start_ind:]

    return ir

def reconstruct_rir(time, alpha):
    '''
    Construct a room impulse response from the negative log version that the
    networks use. Uses the sinc function to approximate a physical impulse response.
    A random subset of the reflections are negated (helps with removing artefacts).
    Adopted from pyroomacoustics.
    '''
    fdl = 81
    fdl2 = (fdl-1) // 2
    time = (time.astype('double')+1)*1024
    alpha = np.exp(-alpha).astype('double')
    signs = np.random.randint(0,2, len(alpha))*2-1
    #alpha *= signs
    ir = np.arange(np.ceil((1.05*time.max()) + fdl))*0
    for i in range(time.shape[0]):
        time_ip = int(np.round(time[i]))
        time_fp = time[i] - time_ip
        ir[time_ip-fdl2:time_ip+fdl2+1] += alpha[i]*fractional_delay(time_fp)
    start_ind = min(np.where(ir != 0)[0])
    ir = ir[start_ind:]
    return ir

def fill_peaks(times, alphas, points_omitted, debug=False):
    '''
    Approximate amplitudes and times for late reverb as simulations fails to
    decrease time spacings indefinitely. The amplitudes are assumed to follow
    the decay defined in func method.
    '''
    def func(t, a, b, c, d):
        return a*np.log(b*(t+c))+d

    times[times < 0] = 0
    omit_slice = np.argsort(times)[points_omitted:]
    sliced_sorted_times = times[omit_slice]
    sliced_sorted_alphas = alphas[omit_slice]

    coeff, _ = curve_fit(func, sliced_sorted_times, sliced_sorted_alphas)
    rir_max_time = max(times)*5
    t = np.linspace(0, rir_max_time, 1000)
    n_in_bins, bin_edges = np.histogram(times, 25)
    ind_max_bin = np.argmax(n_in_bins)
    time_max_bin = bin_edges[ind_max_bin]
    time_simulation_limit = times[np.argmin(abs(times-time_max_bin))]
    n_early_reflections = np.sum(times < time_simulation_limit)
    area_early = time_simulation_limit*n_in_bins[ind_max_bin]/2
    density_early = n_early_reflections/area_early
    area_late = n_in_bins[ind_max_bin]/time_simulation_limit*rir_max_time*(rir_max_time-time_simulation_limit)/2
    n_late_reflections = int(area_late * density_early)

    new_times = np.random.triangular(time_simulation_limit, rir_max_time, rir_max_time, n_late_reflections)

    std = np.std(sliced_sorted_alphas-func(sliced_sorted_times, *coeff))
    new_deviations = np.random.normal(scale=std, size=n_late_reflections)
    new_alphas = func(new_times, *coeff) + new_deviations

    filled_times = np.concatenate((times, new_times))
    filled_alphas = np.concatenate((alphas, new_alphas))

    if debug:
        plt.figure()
        plt.plot(times, alphas, 'o', label='Network output', markersize=0.5)
        plt.plot(new_times, new_alphas, 'o', label='Data extrapolation', markersize=0.5)
        plt.plot(filled_times, func(filled_times, *coeff), '.', label='Fitted line', markersize=0.5)
        legend = plt.legend(prop={'size': 12})
        legend.legendHandles[0]._legmarker.set_markersize(6)
        legend.legendHandles[1]._legmarker.set_markersize(6)
        legend.legendHandles[2]._legmarker.set_markersize(6)

        plt.ylabel(r'$-\log{\alpha}$', fontsize=16)
        plt.xlabel('time (s)', fontsize=16)

        plt.ylim(bottom = -0.25)

        plt.figure()
        plt.plot(bin_edges[:-1], n_in_bins)
        plt.plot([0, time_simulation_limit],[0,n_in_bins[ind_max_bin]])
        plt.plot([time_simulation_limit, rir_max_time],[0, n_in_bins[ind_max_bin]/time_simulation_limit*rir_max_time])
    return filled_times, filled_alphas

def set_fig(xscale = 1, yscale = 1):
    fig_width_pt = 426.8  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0      # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width*xscale,fig_height*yscale]

    params = {'backend': 'ps',
              'axes.labelsize': 12,
              'font.size': 12,
              'font.family': 'STIXGeneral',
              'legend.fontsize': 12,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'text.usetex': True,
              'figure.figsize': fig_size}
    pylab.rcParams.update(params)
