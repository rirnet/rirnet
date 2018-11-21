from glob import glob
from importlib import import_module
from pyroomacoustics.utilities import fractional_delay
from scipy.optimize import curve_fit

import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

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
    alpha *= signs
    ir = np.arange(np.ceil((1.05*time.max()) + fdl))*0
    for i in range(time.shape[0]):
        time_ip = int(np.round(time[i]))
        time_fp = time[i] - time_ip
        ir[time_ip-fdl2:time_ip+fdl2+1] += alpha[i]*fractional_delay(time_fp)
    start_ind = min(np.where(ir != 0)[0])
    ir = ir[start_ind:]
    return ir

def fill_peaks(times, alphas, debug=False):
    '''
    Approximate amplitudes and times for late reverb as simulations fails to
    decrease time spacings indefinitely. The amplitudes are assumed to follow
    the decay defined in func method.
    '''
    def func(t, a, b, c, d):
        return a*np.log(b*(t+c))+d

    coeff, _ = curve_fit(func, times, alphas)
    rir_max_time = 1/coeff[1]*np.exp((6-coeff[3])/coeff[0])-coeff[2]
    rir_max_time = np.min((rir_max_time, max(times)*2))
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

    std = np.std(alphas-func(times, *coeff))
    new_deviations = np.random.normal(scale=std, size=n_late_reflections)
    new_alphas = func(new_times, *coeff) + new_deviations

    filled_times = np.concatenate((times, new_times))
    filled_alphas = np.concatenate((alphas, new_alphas))

    if debug:
        plt.plot(times, alphas, 'x', label='input')
        plt.plot(new_times, new_alphas, 'o', label='filling')
        plt.plot(filled_times, func(filled_times, *coeff), '.', label='fit')
        plt.show()

        plt.plot(bin_edges[:-1], n_in_bins)
        plt.plot([0, time_simulation_limit],[0,n_in_bins[ind_max_bin]])
        plt.plot([time_simulation_limit, rir_max_time],[0, n_in_bins[ind_max_bin]/time_simulation_limit*rir_max_time])
        plt.show()
    return filled_times, filled_alphas
