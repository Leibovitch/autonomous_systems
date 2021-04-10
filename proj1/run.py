
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from initialize import initialize
from mpl_toolkits.mplot3d import Axes3D


data_path = r'C:\\noy\\academics\\autonomous driving\\proj1\\2011_09_26_drive_0106_sync\\2011_09_26\\2011_09_26_drive_0106_sync'
data = initialize(data_path)
velo_points = list(data['velodyne_points'])
[xs, ys, zs] = velo_points[0][velo_points[0][:, 3] != 0][:, 0:3].T


# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(np.array(xs), np.array(ys), np.array(zs))
# plt.show()