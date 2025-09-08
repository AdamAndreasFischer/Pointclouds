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
    print("Coord func")
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
    """Enhanced preprocessing for better registration results"""
    
    # 1. Voxel downsampling
    cloud_down = cloud.voxel_down_sample(voxel_size//2)
    #cloud_down = cloud
    # 2. Outlier removal (statistical)
    if remove_outliers:
        cloud_filtered, ind = cloud_down.remove_statistical_outlier(
            nb_neighbors=max_nn, std_ratio=std_ratio)
        print(f"Statistical outlier removal: type(ind) = {type(ind)}, shape = {np.array(ind).shape}")
    else:
        cloud_filtered = cloud_down
        ind = np.arange(len(cloud_down.points))  # All points kept

    cloud_filtered_2, ind_r = cloud_filtered.remove_radius_outlier(nb_points=max_nn//2, radius=voxel_size*3.0)
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
    
    # 3. Estimate normals with consistent orientation
    radius_normal = voxel_size * 2
    cloud_filtered_2.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn))
   # cloud_filtered_2.orient_normals_towards_camera_location(np.array([0, 0, 0]))
    
    # 4. Optional: Smooth the point cloud
    # This uses a simple moving average approach
    # You might need to implement this as Open3D doesn't have it built-in
    
    return cloud_filtered_2, combined_indices, cloud_down

def denoise_point_cloud(cloud, voxel_size, max_nn, std_ratio):
    """Denoise the point cloud using voxel filtering"""

    cloud_filtered,_ = cloud.remove_statistical_outlier(
            nb_neighbors=max_nn, std_ratio=std_ratio)
    
    cloud_filtered_2, ind_r = cloud_filtered.remove_radius_outlier(nb_points=max_nn//2, radius=voxel_size*3.0)

    return cloud_filtered_2

def multi_stage_registration(source, target, voxel_size, max_nn=30, std_ration = 2.0, initial_pose=None, save=True):
    """A multi-stage registration approach with progressively refined alignment"""
    
    print("Initial allginment")
    #draw_geometries([source, target])
    originals = [copy.deepcopy(source), copy.deepcopy(target)]
    source_down,_,_ = preprocess_for_registration(source, voxel_size, max_nn=max_nn, std_ratio=std_ration)
    target_down,_,_ = preprocess_for_registration(target, voxel_size, max_nn=max_nn, std_ratio=std_ration)
    radius_normal = voxel_size * 2
    # 3. Initial alignment - use pose if available, otherwise identity
    init_transform = np.eye(4) if initial_pose is None else initial_pose

    initial_evaluation = o3d.pipelines.registration.evaluate_registration(
        source_down, target_down, voxel_size*2, init_transform)
    print(f"Initial alignment fitness: {initial_evaluation.fitness:.4f}")

    
    # 4. Coarse alignment with larger threshold
    coarse_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, voxel_size *6,  init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    
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

#
    #final_cloud = o3d.geometry.PointCloud()
#
    #final_cloud += source_copy
    #final_cloud += target_copy
    #if save:
    #    o3d.io.write_point_cloud("/home/adamfi/Codes/Mocap_process/Alligned_clouds/ICP_reged.ply",final_cloud, write_ascii = True )
    
    
    return fine_result, source_down


def main():

    path = "/home/adamfi/Codes/Pointclouds/pointclouds/Multi_cloud_more_poses"
    poses = load_coord(path)
    voxel_size = 40
    max_nn = 40
    std_ratio = 1.5
    original_clouds = read_multi_clouds(path)
    #original_clouds = read_clouds(os.path.join(path, "Cloud_pose2"))
    print(f"Loaded {len(original_clouds)} point clouds and {len(poses)} poses")
    #source = original_clouds[1]
    #target = original_clouds[0]
    #poses = np.load(os.path.join(path, "pose_2.npy"))
    #poses = np.repeat(poses[np.newaxis, ...], len(original_clouds), axis=0)
    transform_coords = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])    

    
    init_transforms = []
    initial_pcs = []
    i=0
    for pose, pcd_original in zip(poses, original_clouds):
        pcd = copy.deepcopy(pcd_original)

        t = pose[:3]
        q = pose[3:]

        q_norm = np.linalg.norm(q) #Normalize quaternion
        q = q/q_norm
        R_mat = R.from_quat(q).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t*1000  # Convert pose to mm as pointclouds are measured in mm
        #print(T)
        
        T_total = T@transform_coords
        init_transforms.append(T_total)
        pcd.transform(T_total)
        initial_pcs.append(pcd)
        #if i >0:
        #    print(f"Cloud {i} and {i+1}")
        #    o3d.visualization.draw_geometries([initial_pcs[i], initial_pcs[i-1]])
        #i+=1
    
    o3d.visualization.draw_geometries(initial_pcs)
    start_cloud = copy.deepcopy(original_clouds[0])

    
    refined_pcs = [copy.deepcopy(initial_pcs[0])]
    resulting_transforms = [copy.deepcopy(init_transforms[0])]
    for i in range(1,len(initial_pcs)):
        print(f"Refining cloud {i}...")
        source = copy.deepcopy(initial_pcs[i])
        target = copy.deepcopy(refined_pcs[-1])

        result, source_down = multi_stage_registration(source, target, voxel_size, max_nn, std_ratio)
        resulting_transforms.append(result.transformation@init_transforms[i])

        source_down.transform(result.transformation)
      

        refined_pcs.append(source_down)

    
    for i, pose in enumerate(resulting_transforms):
        
        t = pose[:3,3]
        
        R_mat = pose[:3,:3]
        quat = R.from_matrix(R_mat).as_quat(scalar_first = False)
        
        coord = np.concatenate([t, quat])
        
        np.save(f"{path}/pose_{i+1}_transformed.npy", coord)
    #o3d.visualization.draw_geometries(refined_pcs)
   
    #np.savez("home/adamfi/Codes/Pointclouds/pointclouds/Alligned_clouds/full_room3_transforms.npz", full_save = True,  transforms=resulting_transforms, init_transforms=init_transforms)


    final_cloud = o3d.geometry.PointCloud()
    for cloud in refined_pcs:
        final_cloud += cloud

    o3d.io.write_point_cloud("/home/adamfi/Codes/Pointclouds/pointclouds/Alligned_clouds/Scene_table_obstacle.ply",final_cloud, write_ascii = True )
    draw_geometries([final_cloud])


if __name__=="__main__":
   
    main()