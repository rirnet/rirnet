import rirnet.acoustic_utils as au
import numpy as np
import matplotlib.pyplot as plt
import rirnet.roomgen as rg




#irf = np.load('irf.npy')
#frf = np.fft.fft(irf)

wav, rate = au.read_wav('fuck.wav')

irf1, _ = au.read_wav('../../../Downloads/30x20y.wav', rate)
room = rg.generate(8, 8, 2.5, 2.5, 1, 44100, 10, 0.1, 0.2)
room.compute_rir()
irf2 = room.rir[0][0]

sound1 = au.convolve(wav, irf1)
sound2 = au.convolve(wav, irf2)

au.save_wav('sound1.wav', sound1, rate, norm=1)
au.save_wav('sound2.wav', sound2, rate, norm=1)

plt.subplot(2,1,1)
plt.plot(irf)

plt.subplot(2,1,2)
plt.plot(signal)


plt.show()



