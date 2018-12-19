import rirnet.roomgen as rg
import numpy as np
import matplotlib.pyplot as plt

def main():
    room = rg.generate(4, 10, 2, 3, 10)
    room.plot(mic_marker_size=30)
    plt.show()

if __name__ == '__main__':
    main()
