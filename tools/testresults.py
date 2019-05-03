import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pylab

def main():
    cmap = colors.ListedColormap(['#FFA4A2', '#97EE9D'])
    bounds = [0, 5, 10]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    data = pd.read_csv("test1.csv", header=None)
    data = data.drop(data.index[0])
    data = data.astype('int')
    n_respondees = np.shape(data)[0]
    n_rooms = np.shape(data)[1]
    plt.xticks(range(1,n_rooms+1))
    plt.yticks(range(1,n_respondees+1))
    plt.xlabel('Case number')
    plt.ylabel('Respondee')
    plt.imshow(data, cmap=cmap, extent=[0.5,n_rooms+0.5,0.5,n_respondees+0.5], origin='lower')
    conf = round(np.sum(np.sum(data))/np.size(data)*100, 1)
    plt.title('Test 1, confusion = {} %'.format(conf))
    plt.savefig('test1.png')
    plt.savefig('test1.eps')

    data = pd.read_csv("test2.csv", header=None)
    data = data.drop(data.index[0])
    data = data.astype('int')
    n_respondees = np.shape(data)[0]
    n_rooms = np.shape(data)[1]
    plt.figure()
    plt.xticks(range(1,n_rooms+1))
    plt.yticks(range(1,n_respondees+1))
    plt.xlabel('Case number')
    plt.ylabel('Respondee')
    plt.imshow(data, cmap=cmap, extent=[0.5,n_rooms+0.5,0.5,n_respondees+0.5], origin='lower')
    conf = round(np.sum(np.sum(data))/np.size(data)*100, 1)
    plt.title('Test 2, confusion = {} %'.format(conf))
    plt.savefig('test2.png')
    plt.savefig('test2.eps')

    data = pd.read_csv("test3.csv", header=None)
    data = data.drop(data.index[0])
    data = data.astype('int')
    n_respondees = np.shape(data)[0]
    n_rooms = np.shape(data)[1]
    plt.figure()
    plt.xticks(range(1,n_rooms+1))
    plt.yticks(range(1,n_respondees+1))
    plt.xlabel('Case number')
    plt.ylabel('Respondee')
    plt.imshow(data, cmap=cmap, extent=[0.5,n_rooms+0.5,0.5,n_respondees+0.5], origin='lower')
    conf = round(np.sum(np.sum(data))/np.size(data)*100, 1)
    plt.title('Test 3, confusion = {} %'.format(conf))
    plt.savefig('test3.png')
    plt.savefig('test3.eps')

    data = pd.read_csv("test4.csv", header=None)
    data = data.drop(data.index[0])
    data = data.astype('int')
    n_respondees = np.shape(data)[0]
    n_rooms = np.shape(data)[1]
    plt.figure()
    plt.xticks(range(1,n_rooms+1))
    plt.yticks(range(1,n_respondees+1))
    plt.xlabel('Case number')
    plt.ylabel('Respondee')
    plt.imshow(data, extent=[0.5,n_rooms+0.5,0.5,n_respondees+0.5], origin='lower')
    conf = round(np.sum(np.sum(data))/np.size(data)*100, 1)
    plt.title('Test 4, confusion = {} %'.format(conf))
    plt.savefig('test4.png')
    plt.savefig('test4.eps')

    plt.show()

if __name__ == '__main__':
    fig_width_pt = 426.8                    # Get this from LaTeX using \showthe\columnwidth
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
              #'text.usetex': True,
              'figure.figsize': fig_size}
    pylab.rcParams.update(params)
    main()
