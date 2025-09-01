import open3d as o3d
import open3d.t as o3dt
import numpy as np
def display_inlier_outlier(cloud, ind):
    print("Displaying inlier outlier")
    inlier_cloud = cloud.select_by_index(ind)

    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud] )


def main():
    cloud = o3d.io.read_point_cloud("C:/Users/adamf/Codes/Mocap_process/Alligned_clouds/ICP_reged.ply")
    device =  "cpu"
    cloud_tens = o3dt.geometry.PointCloud.from_legacy(cloud)

    cloud, ind = cloud_tens.remove_statistical_outliers( nb_neighbors= 60, std_ratio = 0.8)
    
    index = ind.numpy()

    display_inlier_outlier(cloud, ind)


if __name__=="__main__":
    main()