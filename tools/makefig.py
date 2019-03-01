import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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

# Generate data
x = np.arange(-2*np.pi,2*np.pi,0.01)
y1 = np.sin(x)
y2 = np.cos(x)
# Plot data

plt.figure()
plt.clf()
plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
plt.plot(x,y1,'g:',label='$\sin(x)$')
plt.plot(x,y2,'-b',label='$\cos(x)$')
plt.xlabel('$x$ (radians)')
plt.ylabel('$y$')
plt.legend()
plt.savefig('fig1.eps')
