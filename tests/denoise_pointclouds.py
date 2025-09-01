import open3d as o3d
import open3d.t as o3dt
import numpy as np
from pointcloud_utils import load_coord, pose_to_transform_matrix, read_clouds


def denoise_point_cloud(cloud, voxel_size, max_nn, std_ratio):
    """Denoise the point cloud using voxel filtering"""

    cloud_filtered,_ = cloud.remove_statistical_outlier(
            nb_neighbors=max_nn, std_ratio=std_ratio)
    
    cloud_filtered_2, ind_r = cloud_filtered.remove_radius_outlier(nb_points=max_nn//2, radius=voxel_size*3.0)

    return cloud_filtered_2


def main():
    clouds = read_clouds("C:/Users/adamf/Codes/Laptopcodes/Mocap_process/No_mocap_clouds/originals")

    save_dir = "C:/Users/adamf/Codes/Laptopcodes/Mocap_process/No_mocap_clouds"
    voxel_size = 40
    max_nn = 40
    std_ratio = 2.0
    
    for i,cloud in enumerate(clouds):
        denoised_cloud = denoise_point_cloud(cloud,voxel_size, max_nn, std_ratio)
        o3d.io.write_point_cloud(f"C:/Users/adamf/Codes/Laptopcodes/Mocap_process/No_mocap_clouds/Denoised_pose{i+1}.ply", denoised_cloud, write_ascii = True)
    


if __name__=="__main__":
    main()