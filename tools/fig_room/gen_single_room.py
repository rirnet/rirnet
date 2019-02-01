import rirnet.roomgen as rg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab

def plot(room, img_order=None, freq=None, figsize=None, no_axis=False, mic_marker_size=10, **kwargs):
    ''' Plots the room with its walls, microphones, sources and images '''

    import mpl_toolkits.mplot3d as a3
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import scipy as sp

    fig = plt.figure(figsize=figsize)
    ax = a3.Axes3D(fig, proj_type = 'ortho')
    #ax.set_aspect('equal')

    # plot the walls
    for w in room.walls:
        tri = a3.art3d.Poly3DCollection([w.corners.T])
        tri.set_facecolor((0,0,0,0))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)

    # define some markers for different sources and colormap for damping
    markers = ['o', 's', 'v', '.']
    cmap = plt.get_cmap('YlGnBu')
    # draw the scatter of images

    for i, source in enumerate(room.sources):
        # draw source
        ax.scatter(
            source.position[0],
            source.position[1],
            source.position[2],
            c='r',
            s=100)
        ax.plot([source.position[0], source.position[0]], [source.position[1], source.position[1]],[source.position[2], 0], '--r', linewidth=0.5)

    # draw the microphones
    if (room.mic_array is not None):
            for mic in room.mic_array.R.T:
                ax.scatter(mic[0], mic[1], mic[2],
                        marker='x', s=100, c='k')
                ax.plot([mic[0], mic[0]], [mic[1], mic[1]],[mic[2], 0], '--k', linewidth=0.5)

    X = w.corners[0]
    Y = w.corners[1]
    Z = w.corners[2]
    Z[0] = 0

    ax.set_xlim(min(X), (max(X)))
    ax.set_ylim(min(Y), (max(Y)))
    ax.set_zlim(min(Z), np.ceil(max(Z)))

    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 1))
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 1))
    start, end = ax.get_zlim()
    ax.zaxis.set_ticks(np.arange(start, end, 1))


    return fig, ax

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
              'lines.markersize' : 300,
              'legend.fontsize': 12,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'text.usetex': True,
              'figure.figsize': fig_size}
    pylab.rcParams.update(params)

    # Generate data
    room = rg.generate(3, 10, 2, 3, 10)
    fig, ax = plot(room)

    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.set_zlabel('$z$ (m)')
    ax.dist = 11
    plt.savefig('room.eps')


if __name__ == '__main__':
    main()
