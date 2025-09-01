import roslibpy
import numpy as np
import time
import os
from datetime import datetime

import open3d as o3d  # Import Open3D for visualization
import matplotlib.pyplot as plt
from PIL import Image

print(os.path.exists("C:/Users/adamf/Codes/Mocap_process/raster.tif"))

height_matrix = np.genfromtxt("C:/Users/adamf/Codes/Mocap_process/raster_height.txt", filling_values=np.nan)
height_matrix = np.expand_dims(height_matrix, -1)
print(height_matrix.shape)
print(height_matrix)

im = plt.imread("C:/Users/adamf/Codes/Mocap_process/raster_rgb.tif")
print(im.shape)

rgb_d = np.concatenate((im, height_matrix), axis=-1)

print(rgb_d.shape)