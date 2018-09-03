import numpy as np
import shapely.geometry as geo
import matplotlib.pyplot as plt

min_side = 3
max_side = 10

maxx1, maxy1 = np.random.uniform(min_side, max_side, 2)
minx2 = np.random.uniform(0, maxx1)
miny2 = np.random.uniform(0, maxy1)
maxx2 = np.random.uniform(minx2, max_side)
maxy2 = np.random.uniform(miny2, max_side)

box1 = geo.box(0, 0, maxx1, maxy1)
box2 = geo.box(minx2, miny2, maxx2, maxy2)

vertices = box1.union(box2).exterior.coords

plt.plot(vertices.xy[0], vertices.xy[1])
plt.show()
