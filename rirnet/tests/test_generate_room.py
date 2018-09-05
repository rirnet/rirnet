import sys
#sys.path.append('../')
import rirnet.roomgen as rg
import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt

room = rg.generate(min_side=3, max_side=17, min_height=2, max_height=3, n_mics=5)
room.image_source_model(use_libroom=True)
room.plot(img_order=3, aspect='equal')
fig = plt.figure()
room.plot_rir()
plt.show()
