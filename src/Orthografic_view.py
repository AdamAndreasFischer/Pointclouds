import open3d as o3d
import open3d.t as o3dt
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

"""
Read raster matrix and png image from cloudcompare in order to create RGB-D image. 
"""

pointcloud = o3d.io.read_point_cloud("C:/Users/adamf/Codes/Laptopcodes/Mocap_process/Alligned_clouds/ICP_reged.ply")
print(pointcloud)
print(o3d.__version__)

#height_matrix = np.loadtxt("/home/adamfi/Codes/Pointclouds/RGBD-data/raster_matrix.txt")
height_matrix = plt.imread("/home/adamfi/Codes/Pointclouds/RGBD-data/depth.png")


height_matrix = np.expand_dims(height_matrix, -1)
print(height_matrix.shape)
print(height_matrix)

im = plt.imread("/home/adamfi/Codes/Pointclouds/RGBD-data/rgb(1).png")
print(im.shape)

rgb_d = np.concatenate((im, height_matrix), axis=-1)

print(rgb_d.shape)
Height = im.shape[0]
Width = im.shape[1]
center = (Height//2,Width//2)

im[center[0], center[1],0] = 1 
im[center[0], center[1],1:2] = 0

plt.subplot(1, 2, 1)
plt.title('RGB image')
plt.imshow(im)
plt.subplot(1, 2, 2)
plt.title('Depth image')
plt.imshow(height_matrix, cmap='viridis')
plt.show()
