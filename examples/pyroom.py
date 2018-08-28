import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra

floor = np.array([[0, 0, 6, 6, 3, 3], [0, 3, 3, 1.5, 1.5, 0]])
room = pra.Room.from_corners(floor, fs=16000, max_order=12, absorption=0.1)
room.extrude(2.4)
room.add_source([1.5, 1.2, 1.6])
R = np.array([[3.,4.2], [2.25, 2.1], [1.4, 1.4]])
bf = pra.MicrophoneArray(R, room.fs)
room.add_microphone_array(bf)
room.image_source_model()
room.plot(img_order=3, aspect='equal')
fig = plt.figure()
room.plot_rir()
plt.show()
