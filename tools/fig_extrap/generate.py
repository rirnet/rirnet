import sys
import torch
import os
import matplotlib.pyplot as plt
import pylab
import numpy as np
import rirnet.acoustic_utils as au
import rirnet.misc as misc
from scipy.optimize import curve_fit
from torchvision import transforms
from rirnet.transforms import ToTensor, ToNormalized, ToNegativeLog, ToUnitNorm


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
    timespan = np.linspace(np.min(times), np.max(new_times))
    return new_times, new_alphas, timespan, func(timespan, *coeff)


class Model:
    def __init__(self, model_dir):
        sys.path.append('../../nanonet/rirnet')
        from rirnet_database import RirnetDatabase
        print(sys.path)
        self.model_dir = model_dir

        self.extractor, _ = misc.load_latest(model_dir, 'extractor')
        self.autoencoder, _ = misc.load_latest(model_dir, 'autoencoder')

        self.extractor_args = self.extractor.args()

        use_cuda = not self.extractor_args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.extractor.to(self.device)
        self.autoencoder.to(self.device)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        data_transform = transforms.Compose([ToNormalized('../../database/mean.npy', '../../database/std.npy')])
        target_transform = transforms.Compose([ToNegativeLog(), ToUnitNorm(), ToTensor()])

        self.extractor_args.val_db_path = '../../database/db-val.csv'

        eval_db = RirnetDatabase(is_training = False, args = self.extractor_args, data_transform = data_transform, target_transform = target_transform)
        self.eval_loader = torch.utils.data.DataLoader(eval_db, batch_size=self.extractor_args.batch_size, shuffle=True, **self.kwargs)

        self.audio_anechoic, self.fs = au.read_wav('../../audio/harvard/male.wav')

    def run(self):
        self.extractor.eval()
        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.eval_loader):
                source, target = source.to(self.device).float(), target[0].numpy()

                latent_source = self.extractor(source)
                output = self.autoencoder(latent_source, encode=False, decode=True)[0].cpu().numpy()



                filled_times_output, filled_alphas_output, linex, liney = fill_peaks(output[0,:], output[1,:], 10)
                filled_times_target, filled_alphas_target, _,  _ = fill_peaks(target[0,:], target[1,:], 10)

                output_rir = misc.reconstruct_rir(filled_times_output, filled_alphas_output)
                output_rir_conv = misc.reconstruct_rir_conv(filled_times_output, filled_alphas_output)
                target_rir = misc.reconstruct_rir(filled_times_target, filled_alphas_target)

                print(np.shape(output))
                plt.plot(output[0], output[1], 'o', markersize = 0.5, label='Network output')
                plt.plot(filled_times_output, filled_alphas_output, 'o', markersize = 0.5, label='Extrapolated data')
                plt.plot(linex, liney, label='Approximated function')
                plt.grid()
                plt.xlabel('Timings')
                plt.ylabel('-log(Amplitudes)')
                plt.ylim(0, np.max(filled_alphas_output))
                plt.legend()
                plt.gcf().subplots_adjust(bottom=0.15)
                plt.savefig('extrap.eps')
                plt.show()

                #plt.plot(output_rir)
                #plt.plot(output_rir_conv)
                #plt.show()

                #rev_signal_output = au.convolve(self.audio_anechoic, output_rir)
                #rev_signal_target = au.convolve(self.audio_anechoic, target_rir)

                #au.save_wav('output.wav', rev_signal_output, self.fs, 1)
                #au.save_wav('target.wav', rev_signal_target, self.fs, 1)

                #au.play_file('output.wav')
                #au.play_file('target.wav')

def main():

    fig_width_pt = 426.8  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0      # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]

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

    plt.figure()

    model = Model('../../nanonet/models/32')
    model.run()


if __name__ == '__main__':
    main()

