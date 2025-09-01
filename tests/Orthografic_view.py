import open3d as o3d
import open3d.t as o3dt
import matplotlib.pyplot as plt


pointcloud = o3d.io.read_point_cloud("/home/adamfi/Codes/Mocap_process/Alligned_clouds/ICP_reged.ply")
print(pointcloud)
print(o3d.__version__)
#tensor_cloud = o3dt.geometry.PointCloud.from_legacy(pointcloud)

#colour, depth = tensor_cloud.project_to_rgbd_image()

#plt.imshow(colour.numpy())
#plt.show()o3dt.Tensor(pointcloud.points)

