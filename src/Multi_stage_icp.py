import open3d as o3d
from open3d.visualization import draw_geometries
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
os.environ["GDK_BACKEND"] = "x11"  # Force X11 backend


NUM_THREADS = max(1, multiprocessing.cpu_count())

def load_coord(path):

    print(os.listdir(path))
    paths = [path for path in os.listdir(path) if path.endswith(".npy") and not "transformed" in path]
    paths.sort()
    paths = sorted(paths, key=len)
    poses = []
    for pose_path in paths:
        poses.append(np.load(os.path.join(path, pose_path)))
    poses = np.array(poses)
    return poses

def read_clouds(path):
    paths = [path for path in os.listdir(path) if path.endswith(".ply")]
    paths.sort()
    paths = sorted(paths, key=len)
    clouds = []
    for ptc in paths:
        clouds.append(o3d.io.read_point_cloud(os.path.join(path, ptc)))

    return clouds

def read_multi_clouds(path):
    """Read one pointcloud from each folder containing multiple pointclouds"""
    folders = [f.path for f in os.scandir(path) if f.is_dir() and f.name.startswith("Cloud_pose")]
    folders.sort()
    folders = sorted(folders, key=len)
    cloud_list = []
    for folder in folders:
        print(folder)
        clouds = [cloud for cloud in os.listdir(os.path.join(path,folder)) if cloud.endswith(".ply")]
        print(clouds)
        pcd = o3d.io.read_point_cloud(os.path.join(folder, clouds[1]))
        cloud_list.append(pcd)
    return cloud_list

def filter_invalid_points(cloud, min_distance=150.0, max_distance=12000.0):
    """Remove invalid near-zero and very far points (distance in mm)."""
    points = np.asarray(cloud.points)
    if len(points) == 0:
        return cloud

    distances = np.linalg.norm(points, axis=1)
    valid_idx = np.where((distances >= min_distance) & (distances <= max_distance))[0]

    if len(valid_idx) == 0:
        return cloud

    return cloud.select_by_index(valid_idx)

def pose_to_transform_matrix(pose):
    """
    Convert pose [x, y, z, qx, qy, qz, qw] to 4x4 transformation matrix
    """
    translation = pose[:3]  # x, y, z
    quaternion = pose[3:]   # qx, qy, qz, qw
    
    # Create rotation matrix from quaternion
    rotation = R.from_quat(quaternion).as_matrix()
    
    # Create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation*1000
    
    return transform

def preprocess_for_registration(cloud, voxel_size, max_nn=30,std_ratio=2.0, remove_outliers=True):
    """Enhanced preprocessing for better registration results
    Args
    cloud: [pcd] Pointcloud to be preprocessed
    voxel_size: [float] size of voxel, i.e how fine the resolution for the allignment is
    max_nn: [float] The ammount of neighbouring voxels taken into account in denoising
    std_ration: [float] Deviation of points in denoising
    """

    cloud = filter_invalid_points(cloud)

    # Downsample
    cloud_down = cloud.voxel_down_sample(max(voxel_size / 2.0, 5.0))
    #cloud_down = cloud
    
    # Outlier removal
    if remove_outliers:
        cloud_filtered, ind = cloud_down.remove_statistical_outlier(
            nb_neighbors=max_nn, std_ratio=std_ratio)
        print(f"Statistical outlier removal: type(ind) = {type(ind)}, shape = {np.array(ind).shape}")
    else:
        cloud_filtered = cloud_down
        ind = np.arange(len(cloud_down.points))  # All points kept

    cloud_filtered_2, ind_r = cloud_filtered.remove_radius_outlier(
        nb_points=max(max_nn // 2, 12), radius=voxel_size * 2.5)
    print(f"Radius outlier removal: type(ind_r) = {type(ind_r)}, shape = {np.array(ind_r).shape}")
    
    # Combine the indices: first apply statistical outlier indices, then radius outlier indices
    if remove_outliers:
        # Convert to numpy arrays and ensure they are integer type
        ind = np.array(ind, dtype=int)
        ind_r = np.array(ind_r, dtype=int)
        
        # Check if ind_r is empty (no points survived radius outlier removal)
        if len(ind_r) == 0:
            print("Warning: No points survived radius outlier removal!")
            combined_indices = np.array([], dtype=int)
        else:
            combined_indices = ind[ind_r]  # Chain the indices properly
    else:
        ind_r = np.array(ind_r, dtype=int)
        combined_indices = ind_r
        
    print(f"Combined indices: type = {type(combined_indices)}, shape = {combined_indices.shape}, length = {len(combined_indices)}")
    
    # Estimate normals with consistent orientation
    radius_normal = voxel_size * 2
    cloud_filtered_2.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn))
   # cloud_filtered_2.orient_normals_towards_camera_location(np.array([0, 0, 0]))
    
    
    return cloud_filtered_2, combined_indices, cloud_down

def denoise_point_cloud(cloud, voxel_size, max_nn, std_ratio):
    """Denoise the point cloud using voxel filtering"""

    cloud_filtered,_ = cloud.remove_statistical_outlier(
            nb_neighbors=max_nn, std_ratio=std_ratio)
    
    cloud_filtered_2, ind_r = cloud_filtered.remove_radius_outlier(nb_points=max_nn//2, radius=voxel_size*3.0)

    return cloud_filtered_2

def multi_stage_registration(source, target, voxel_size, max_nn=30, std_ration = 2.0, initial_pose=None):
    """A multi-stage registration approach with progressively refined alignment
    Args:
    source: [pcd] the pointcloud that is supposed to be alligned with the target
    target: [pcd] Target for allignment of source
    voxel_size: [float] size of voxel, i.e how fine the resolution for the allignment is
    max_nn: [float] The ammount of neighbouring voxels taken into account in denoising
    std_ration: [float] Deviation of points in denoising
    initial_pose: [np.array] If the clouds are intially alligned, should be identity matrix. Otherwise a estimated transform from source to target. 
    """
    
    print("Initial allginment")
    #draw_geometries([source, target])
    originals = [copy.deepcopy(source), copy.deepcopy(target)]
    source_down = source
    target_down = target
    
    radius_normal = voxel_size * 2

    init_transform = np.eye(4) if initial_pose is None else initial_pose

    initial_evaluation = o3d.pipelines.registration.evaluate_registration(
        source_down, target_down, voxel_size*2, init_transform)
    print(f"Initial alignment fitness: {initial_evaluation.fitness:.4f}")
    
    # 4. Coarse alignment with larger threshold
    coarse_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, voxel_size *6,  init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

    coarse_result_fallback = o3d.pipelines.registration.registration_icp(
        source_down, target_down, voxel_size * 10, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))

    if coarse_result_fallback.fitness > coarse_result.fitness:
        coarse_result = coarse_result_fallback
    
    source_course = copy.deepcopy(source)
    #source_course.transform(coarse_result.transformation)
    #draw_geometries([source_course, target])
    if coarse_result.fitness <= initial_evaluation.fitness * 0.9:  # Only accept if significantly worse
        print("Warning: Coarse ICP made alignment worse, keeping initial transform")
        coarse_result.transformation = init_transform
        coarse_result.fitness = initial_evaluation.fitness


    print(f"Coarse alignment fitness: {coarse_result.fitness}")
    
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    source.orient_normals_towards_camera_location(np.array([0, 0, 0]))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target.orient_normals_towards_camera_location(np.array([0, 0, 0]))

    ## 5. Medium alignment 
    medium_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, voxel_size * 4, coarse_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    
    print(f"Medium alignment fitness: {medium_result.fitness}")
    #
    # 6. Fine alignment on original resolution
    fine_result = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size*1, medium_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    
    print(f"Fine alignment fitness: {fine_result.fitness}")

    result_colored = o3d.pipelines.registration.registration_colored_icp(
        source, target, voxel_size/2, fine_result.transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    
    print(f"Colored alignment fitness: {result_colored.fitness}")
    
    return fine_result, source_down


def main():
    """
    Multi stage ICP for local registration of pointclouds. In order to register pointclouds, save them in a folder corresponding to the pose it belongs to, e.g
    pointclouds -> Cloud_pose1 -> cloudX.ply
    and for each pose, save a numpy array 6 dof pose on form [x y z qx qy qz qw].

    Transform coords is specifically for Orbbec cameras to transform the pointcloud from the cameras coordinate system into the specified coordinate system from Motive i.e different depending on how you defined 
    the ridgid bodies. 
    """
    path = "/home/adamfi/codes/Pointclouds/pointclouds/test"
    poses = load_coord(path)
    voxel_size = 40
    max_nn = 40
    std_ratio = 1.5
    milimeters = True
    
    original_clouds = read_multi_clouds(path)
    #original_clouds = read_clouds(os.path.join(path, "Cloud_pose2"))
    print(f"Loaded {len(original_clouds)} point clouds and {len(poses)} poses")
  
    transform_coords = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]
    ])

    transform_2 = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    transform_coords = transform_coords @ transform_2
    print(transform_coords)
    #rotate_x_90 = np.eye(4)
    #rotate_x_90[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()

    #transform_coords = transform_coords @ rotate_x_90

    
    init_transforms = []
    initial_pcs = []
    pose_frames = []
    
    #initial allignment of pointclouds
    for pose, pcd_original in zip(poses, original_clouds):
        pcd = copy.deepcopy(pcd_original)
        
        t = pose[:3]
        q = pose[3:]

        q_norm = np.linalg.norm(q) # Normalize quaternion
        q = q/q_norm
        R_mat = R.from_quat(q).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t * 1000 if milimeters else t  # Convert pose to mm as pointclouds are measured in mm
        
        T_total = T@transform_coords
        
        
        init_transforms.append(T_total)
        pcd.transform(T_total)
        initial_pcs.append(pcd)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150.0)
        frame.transform(T_total)
        pose_frames.append(frame)
      
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=250.0)
    o3d.visualization.draw_geometries(initial_pcs + pose_frames + [world_frame])

    # Down sample and denoise clouds
    preprocessed_pcds = []
    for pcd in initial_pcs:
        pcd_down, _,_ = preprocess_for_registration(pcd, voxel_size, max_nn, std_ratio)
        preprocessed_pcds.append(pcd_down)

    refined_pcs = [copy.deepcopy(preprocessed_pcds[0])]
    resulting_transforms = [copy.deepcopy(init_transforms[0])]

    for i in range(1,len(initial_pcs)):
        print(f"Refining cloud {i+1}...")
        source = copy.deepcopy(preprocessed_pcds[i])
        target = o3d.geometry.PointCloud()
        for cloud in refined_pcs:
            target += cloud

        target = target.voxel_down_sample(max(voxel_size / 2.0, 5.0))

        result, source_down = multi_stage_registration(
            source, target, voxel_size, max_nn, std_ratio, initial_pose=np.eye(4)
        )
        resulting_transforms.append(result.transformation@init_transforms[i])

        source_down.transform(result.transformation)

        refined_pcs.append(source_down)

    # Save resulting pose
    for i, pose in enumerate(resulting_transforms):
        t = pose[:3,3]
        R_mat = pose[:3,:3]
        quat = R.from_matrix(R_mat).as_quat(scalar_first = False)
        
        coord = np.concatenate([t, quat])
        
        np.save(f"{path}/pose_{i+1}_transformed.npy", coord)
   
    # Save registered pointclouds
    final_cloud = o3d.geometry.PointCloud()
    for cloud in refined_pcs:
        final_cloud += cloud

    o3d.io.write_point_cloud("/home/adamfi/codes/Pointclouds/results/initial_test.ply",final_cloud, write_ascii = True )
    draw_geometries([final_cloud])


if __name__=="__main__":
   
    main()