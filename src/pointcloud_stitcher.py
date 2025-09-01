import open3d as o3d
import numpy as np
import os
from scipy.spatial.transform import Rotation as R# Example poses: replace with your real values
import copy


# Each row: [tx, ty, tz, qx, qy, qz, qw]

def load_coord(path):
    print("Coord func")
    print(os.listdir(path))
    paths = [ path for path in os.listdir(path) if path.endswith(".npy")]
    paths.sort()
    paths = sorted(paths, key=len)
    poses = []
    for pose_path in paths:
        poses.append(np.load(os.path.join(path,pose_path)))
    poses = np.array(poses)
    return poses
    

def load_clouds(path):
    pointclouds = [path for path in os.listdir(path) if path.endswith(".ply")]
    pointclouds.sort()
    pointclouds = sorted(pointclouds, key=len)
    return pointclouds

def stitch_clouds(path):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200)
    transform_coords = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    origin_pose = [0.008981803432106972, -0.010278492234647274, 0.0012946350034326315,
                  -0.0003873576642945409, -0.0008097215322777629, -0.0025566567201167345, 0.9999963045120239]
    origin_t = origin_pose[:3]
    origin_rot = origin_pose[3:]
    origin_rmat = R.from_quat(origin_rot).as_matrix()
    origin_T = np.eye(4)
    origin_T[:3, :3] = origin_rmat
    origin_T[:3, 3] = origin_t
    coordinate_frame.transform(origin_T)
    transformed_pcs = []
    camera_frames = []

    poses = load_coord(path)
    clouds = load_clouds(path)

    for pose, cloud_file in zip(poses, clouds):
        pcd = o3d.io.read_point_cloud(os.path.join(path, cloud_file))
        pcd.transform(transform_coords)

        t = pose[:3]
        q = pose[3:]
        R_mat = R.from_quat(q).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t * 1000  # Convert to millimeters

        pcd.transform(T)
        transformed_pcs.append(pcd)
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200)

        camera_frame.transform(T)
        camera_frames.append(camera_frame)
    o3d.visualization.draw_geometries([coordinate_frame] + transformed_pcs + camera_frames,
                                      window_name="All Point Clouds")

def preprocess_point_cloud(pcd, voxel_size):
    """Downsample and compute normals for better ICP performance"""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_down

def apply_icp(source, target, threshold=0.05, initial_transform=np.eye(4)):
    """Apply ICP to align source to target point cloud"""
    
    # Preprocess point clouds
    voxel_size = 5.0  # Adjust based on your point cloud scale
    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)
    
    print(f"Source points after downsampling: {len(source_down.points)}")
    print(f"Target points after downsampling: {len(target_down.points)}")
    
    # Point-to-plane ICP (generally more robust than point-to-point)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    print(f"ICP fitness: {reg_p2p.fitness:.4f}")
    print(f"ICP inlier RMSE: {reg_p2p.inlier_rmse:.4f}")
    
    return reg_p2p.transformation, reg_p2p.fitness

def stitch_clouds_with_icp(path):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200)
    transform_coords = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    origin_pose = [0.008981803432106972, -0.010278492234647274, 0.0012946350034326315,
                  -0.0003873576642945409, -0.0008097215322777629, -0.0025566567201167345, 0.9999963045120239]
    origin_t = origin_pose[:3]
    origin_rot = origin_pose[3:]
    origin_rmat = R.from_quat(origin_rot).as_matrix()
    origin_T = np.eye(4)
    origin_T[:3, :3] = origin_rmat
    origin_T[:3, 3] = origin_t
    coordinate_frame.transform(origin_T)
    
    poses = load_coord(path)
    clouds = load_clouds(path)
    
    # First, apply initial pose transformations
    initial_pcs = []
    camera_frames = []
    
    for pose, cloud_file in zip(poses, clouds):
        pcd = o3d.io.read_point_cloud(os.path.join(path, cloud_file))
        pcd.transform(transform_coords)

        t = pose[:3]
        q = pose[3:]
        R_mat = R.from_quat(q).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t * 1000  # Convert to millimeters

        pcd.transform(T)
        initial_pcs.append(pcd)
        
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200)
        camera_frame.transform(T)
        camera_frames.append(camera_frame)
        #o3d.visualization.draw_geometries([pcd], window_name = cloud_file)
    
    print(f"Loaded {len(initial_pcs)} point clouds")
    
    # Show initial alignment
    print("Showing initial alignment (before ICP)...")
    #o3d.visualization.draw_geometries([coordinate_frame] + initial_pcs + camera_frames,
    #                                  window_name="Initial Alignment")
    
    # Apply ICP refinement
    if len(initial_pcs) < 2:
        print("Need at least 2 point clouds for ICP")
        return
    
    refined_pcs = [copy.deepcopy(initial_pcs[0])]  # First cloud stays as reference
    
    # Sequentially align each cloud to the previous one
    for i in range(1, len(initial_pcs)):
        print(f"\nApplying ICP to align cloud {i+1} to cloud {i}")
        
        source = copy.deepcopy(initial_pcs[i])
        target = refined_pcs[-1]  # Previous aligned cloud
        
        # Apply ICP
        icp_transform, fitness = apply_icp(source, target, threshold=10)
        
        # Apply the refined transformation
        source.transform(icp_transform)
        refined_pcs.append(source)
        
        print(f"ICP completed with fitness: {fitness:.4f}")
    
    # Show refined alignment
    print("\nShowing refined alignment (after ICP)...")
    o3d.visualization.draw_geometries([coordinate_frame] + refined_pcs + camera_frames,
                                      window_name="ICP Refined Alignment")
    
    # Optionally, create a combined point cloud
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in refined_pcs:
        combined_pcd += pcd
    
    # Remove duplicates and outliers
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=2.0)
    combined_pcd, _ = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    print(f"\nCombined point cloud has {len(combined_pcd.points)} points")
    o3d.visualization.draw_geometries([combined_pcd], window_name="Combined Point Cloud")
    
    return refined_pcs, combined_pcd


def main():
    path = "C:/Users/adamf/Codes/Mocap_process/New_clouds"
    
    stitch_clouds_with_icp(path)


if __name__=="__main__":
    main()