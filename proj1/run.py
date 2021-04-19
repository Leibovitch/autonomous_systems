import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from initialize import initialize
from mpl_toolkits.mplot3d import Axes3D
from occupancy_grid import occupancy_map
import matplotlib.animation as animation 
from point_cloud import reduce_to_2d, movement_correct_velo


'''
parmaeters:
'''
res = 0.2
alpha = 1
locc = 0.85
lfree = -0.4
col_hits = 2
max_range = 30
min_height = -1.5
frees_threshold = 0.35
occupied_threshold = 0.8
prop_upper_saturation_threshold = 0.98
prop_lower_saturation_threshold = 0.02


data_path = r'C:\Users\noyl\workspace\autonomous_systems\proj1\2011_09_26_drive_0106_sync\2011_09_26\2011_09_26_drive_0106_sync'
data = initialize(data_path)
data = movement_correct_velo(data, min_height=min_height, col_hits=col_hits, res=res)

# no_filtered_data = movement_correct_velo(data, min_height=-2, col_hits=3, res=res)

update_grids, grids, occupancies = occupancy_map(data,
                  res=res,
                  alpha=alpha,
                  locc=locc,
                  lfree=lfree,
                  max_range=max_range,
                  prop_upper_saturation_threshold=prop_upper_saturation_threshold,
                  prop_lower_saturation_threshold=prop_lower_saturation_threshold,
                  occupied_threshold=occupied_threshold,
                  frees_threshold=frees_threshold
                  )

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax1.set_title('Scene Image')
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax2.set_title('instantenous point cloud')
ax3 = plt.subplot2grid((2, 2), (1, 1))
ax3.set_title('occupancy')

color_image_path = data_path + '\\image_02\\data'
axs = []
for i, (image_path, update_grid, occupancy) in enumerate(zip(os.listdir(color_image_path), update_grids, occupancies)):
    image = cv2.imread(color_image_path + '\\' + image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plot1 = ax1.imshow(image, animated=True)
    plot2 = ax2.scatter(data['East'].iloc[i], data['North'].iloc[i], s=.001, c='g')
    plot3 = ax3.imshow(cv2.flip(occupancy, 0), animated=True)
    axs.append([plot1, plot2, plot3])
    
# anim = animation.ArtistAnimation(fig, axs, interval=10, blit=True)

anim = animation.ArtistAnimation(fig, axs, interval=10, repeat=False)
writervideo = animation.FFMpegWriter(fps=10) 
anim.save("animation1.mp4" , writer=writervideo)
# ani.save('two images.gif',writer='imagemagick')