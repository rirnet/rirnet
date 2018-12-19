import sys
import librosa

import numpy as np
import matplotlib.pyplot as plt
import rirnet.roomgen as rg
import rirnet.acoustic_utils as au
import rirnet.database as db

from librosa.display import specshow

def main(audio_path):
    room = rg.generate(4, 10, 2, 3, 10, max_order=8)
    room.plot(mic_marker_size=30)

    room.compute_rir()
    rir = room.rir[0][0]
    first_index = next((i for i, x in enumerate(rir) if x), None)
    rir = rir[first_index:]/max(abs(rir))
    t_rir = np.arange(len(rir))/44100.

    sound, rate = au.read_wav(audio_path)
    t_sound = np.arange(len(sound))/44100.

    signal = au.convolve(sound, rir)
    signal /= max(abs(signal))
    t_signal = np.arange(len(signal))/44100.

    mic = room.mic_array.R.T[0]
    distances = room.sources[0].distance(mic)
    times = distances/343.0*room.fs
    alphas = room.sources[0].damping / (4.*np.pi*distances)
    slice = tuple(np.where(room.visibility[0][0] == 1))
    alphas = -np.log(alphas[slice])
    alphas -= min(alphas)
    times = (times[slice] - min(times[slice]))/44100.
    right_lim = max(times)

    mfcc = librosa.feature.mfcc(y=signal, sr=44100., n_mels=40)


    eps = 0.1

    plt.figure()

    ax = plt.subplot(2,2,1)
    plt.plot(t_sound, sound)
    plt.title('Anechoic sound')
    plt.xlabel('Time (s)')
    ax.set_xlim(min(t_sound), right_lim)
    ax.set_ylim(-1-eps, 1+eps)

    ax = plt.subplot(2,2,2)
    plt.plot(t_rir, rir)
    plt.title('Room IRF')
    plt.xlabel('Time (s)')
    ax.set_xlim(min(t_rir), right_lim)
    ax.set_ylim(-1-eps, 1+eps)

    ax = plt.subplot(2,2,3)
    plt.plot(t_signal, signal)
    plt.title('Reverberant sound')
    plt.xlabel('Time (s)')
    ax.set_xlim(min(t_signal), right_lim)
    ax.set_ylim(-1-eps, 1+eps)

    ax = plt.subplot(2,2,4)
    plt.plot(times, alphas, '.')
    plt.title('Peaks data')
    plt.xlabel('Time (s)')
    ax.set_xlim(min(times)-0.002, right_lim+0.002)
   

    plt.figure()
    specshow(mfcc, sr=44100, x_axis='time')
    plt.title('MFCC spectrogram')
    plt.xlabel('Time (s)')
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1])
