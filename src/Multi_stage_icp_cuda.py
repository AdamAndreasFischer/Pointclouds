import open3d as o3d
from open3d.visualization import draw_geometries
import numpy as np
import os
import argparse
from scipy.spatial.transform import Rotation as R
import copy
import multiprocessing
import matplotlib.pyplot as plt
os.environ["GDK_BACKEND"] = "x11"  # Force X11 backend
os.environ["DISPLAY"] = ":1"
os.environ["GDK_BACKEND"] = "x11"
os.environ["PYOPENGL_PLATFORM"] = "glx"
os.environ["XDG_SESSION_TYPE"] = "x11"

NUM_THREADS = max(1, multiprocessing.cpu_count())
DEFAULT_ROOT_DIR = "/home/adamfi/codes/Pointclouds/pointclouds/new_calib_test_cam3"
vice = o3d.core.Device("CUDA:1")


def tensor_point_count(cloud):
    return int(cloud.point.positions.shape[0])


def ensure_tensor_cloud(cloud):
    if isinstance(cloud, o3d.t.geometry.PointCloud):
        return cloud.to(vice)
    return o3d.t.geometry.PointCloud.from_legacy(cloud, dtype=o3d.core.Dtype.Float32).to(vice)


def np_to_cuda_transform(transform):
    return o3d.core.Tensor(transform, dtype=o3d.core.Dtype.Float64, device=vice)


def cuda_transform_to_numpy(transform):
    if isinstance(transform, o3d.core.Tensor):
        return transform.cpu().numpy()
    return np.asarray(transform)


def concat_point_clouds(clouds):
    if not clouds:
        return o3d.t.geometry.PointCloud(vice)

    positions = [ensure_tensor_cloud(c).point.positions for c in clouds if tensor_point_count(ensure_tensor_cloud(c)) > 0]
    if not positions:
        return o3d.t.geometry.PointCloud(vice)

    merged = o3d.t.geometry.PointCloud(vice)
    merged.point.positions = o3d.core.concatenate(positions, axis=0)

    has_colors = all("colors" in ensure_tensor_cloud(c).point for c in clouds)
    if has_colors:
        colors = [ensure_tensor_cloud(c).point.colors for c in clouds if tensor_point_count(ensure_tensor_cloud(c)) > 0]
        merged.point.colors = o3d.core.concatenate(colors, axis=0)

    return merged


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-stage ICP for pose/cloud alignment")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help=f"Root directory containing pose_*.npy and Cloud_pose*/cloud*.ply files. Outputs are also saved here. If omitted, uses: {DEFAULT_ROOT_DIR}",
    )
    parser.add_argument(
        "--final_cloud_name",
        type=str,
        default="initial_test_5pcds.ply",
        help="Filename for the merged registered point cloud.",
    )
    parser.add_argument(
        "--pose_prefix",
        type=str,
        default="pose",
        help="Prefix used for saved transformed and delta pose files.",
    )
    parser.add_argument(
        "--cam-number",
        type= int,
        help = "The number on the camera. Used to load calibration matrix"
    )
    return parser.parse_args()

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
        clouds.append(o3d.t.io.read_point_cloud(os.path.join(path, ptc)).to(vice))

    return clouds

def read_multi_clouds(path, num_clouds):
    """Read one pointcloud from each folder containing multiple pointclouds"""
    folders = [f.path for f in os.scandir(path) if f.is_dir() and f.name.startswith("Cloud_pose")]
    folders.sort()
    folders = sorted(folders, key=len)
    cloud_list = []
    for folder in folders:
        print(folder)
        clouds = [cloud for cloud in os.listdir(folder) if cloud.endswith(".ply")]
        if len(clouds) < num_clouds:
            pass
        else: 
            clouds = clouds[:num_clouds]
        print(clouds)
        if not clouds:
            continue
        read_idx = min(1, len(clouds) - 1)
        pcd = o3d.t.io.read_point_cloud(os.path.join(folder, clouds[read_idx]))
        pcd.to(vice)
        cloud_list.append(pcd)
    return cloud_list

def filter_invalid_points(cloud, min_distance=150.0, max_distance=12000.0):
    """Remove invalid near-zero and very far points (distance in mm)."""
    cloud = ensure_tensor_cloud(cloud)
    if tensor_point_count(cloud) == 0:
        return cloud

    points = cloud.point.positions
    distances = (points * points).sum(dim=1).sqrt()
    valid_mask = (distances >= min_distance) & (distances <= max_distance)

    if int(valid_mask.sum().cpu().item()) == 0:
        return cloud

    return cloud.select_by_mask(valid_mask)

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

    cloud = ensure_tensor_cloud(cloud)
    cloud = filter_invalid_points(cloud)

    # Downsample
    cloud_down = cloud.voxel_down_sample(max(voxel_size / 2.0, 5.0))
    down_points = tensor_point_count(cloud_down)
    if down_points < 20:
        print(f"Warning: Too few points after downsampling ({down_points}). Skipping outlier removal.")
        return cloud_down, np.arange(down_points, dtype=int), cloud_down
    
    # Outlier removal
    if remove_outliers:
        try:
            cloud_filtered, mask_s = cloud_down.remove_statistical_outliers(
                nb_neighbors=max_nn, std_ratio=std_ratio
            )
            ind = np.where(mask_s.cpu().numpy())[0].astype(int)
        except Exception:
            cloud_filtered = cloud_down
            ind = np.arange(tensor_point_count(cloud_down), dtype=int)
        print(f"Statistical outlier removal kept {len(ind)} points")
    else:
        cloud_filtered = cloud_down
        ind = np.arange(tensor_point_count(cloud_down), dtype=int)  # All points kept

    try:
        cloud_filtered_2, mask_r = cloud_filtered.remove_radius_outliers(
            nb_points=max(max_nn // 2, 12), radius=voxel_size * 2.5
        )
        ind_r = np.where(mask_r.cpu().numpy())[0].astype(int)
    except Exception:
        cloud_filtered_2 = cloud_filtered
        ind_r = np.arange(tensor_point_count(cloud_filtered), dtype=int)
    print(f"Radius outlier removal kept {len(ind_r)} points")
    
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
    if tensor_point_count(cloud_filtered_2) > 3:
        cloud_filtered_2.estimate_normals(radius=radius_normal, max_nn=max_nn)
    else:
        print(f"Warning: Too few points for normal estimation ({tensor_point_count(cloud_filtered_2)}).")
    
    
    return cloud_filtered_2, combined_indices, cloud_down

def denoise_point_cloud(cloud, voxel_size, max_nn, std_ratio):
    """Denoise the point cloud using voxel filtering"""

    cloud = ensure_tensor_cloud(cloud)
    cloud_filtered, _ = cloud.remove_statistical_outliers(
        nb_neighbors=max_nn, std_ratio=std_ratio
    )

    cloud_filtered_2, _ = cloud_filtered.remove_radius_outliers(
        nb_points=max_nn // 2, radius=voxel_size * 3.0
    )

    return cloud_filtered_2

def compute_scene_leveling_transform(
    cloud,
    distance_threshold=20.0,
    ransac_n=3,
    num_iterations=2000,
    set_floor_to_z0=False,
):
    """Estimate one global transform that levels the dominant plane to world +Z."""
    cloud = ensure_tensor_cloud(cloud)
    if tensor_point_count(cloud) < ransac_n:
        print("Not enough points for plane fitting. Skipping leveling.")
        return np.eye(4), None, 0

    plane_model, inliers = cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        probability=0.999,
    )
    plane_model_np = plane_model.cpu().numpy().reshape(-1)
    a, b, c, d = plane_model_np
    normal = np.array([a, b, c], dtype=np.float64)
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-12:
        print("Degenerate floor normal. Skipping leveling.")
        return np.eye(4), plane_model_np, int(inliers.shape[0])

    normal /= n_norm
    if normal[2] < 0:
        normal = -normal

    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis = np.cross(normal, target)
    axis_norm = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(normal, target), -1.0, 1.0))

    if axis_norm < 1e-12:
        R_align = np.eye(3)
    else:
        axis = axis / axis_norm
        angle = np.arctan2(axis_norm, dot)
        R_align = R.from_rotvec(axis * angle).as_matrix()

    points = cloud.point.positions.cpu().numpy()
    centroid = points.mean(axis=0)

    T_to_origin = np.eye(4)
    T_to_origin[:3, 3] = -centroid
    T_back = np.eye(4)
    T_back[:3, 3] = centroid
    T_rot = np.eye(4)
    T_rot[:3, :3] = R_align

    leveling_transform = T_back @ T_rot @ T_to_origin

    inliers_np = inliers.cpu().numpy().astype(int)
    if set_floor_to_z0 and len(inliers_np) > 0:
        floor_pts = points[inliers_np]
        floor_pts_rot = (R_align @ (floor_pts - centroid).T).T + centroid
        mean_floor_z = float(np.mean(floor_pts_rot[:, 2]))
        Tz = np.eye(4)
        Tz[2, 3] = -mean_floor_z
        leveling_transform = Tz @ leveling_transform

    return leveling_transform, plane_model_np, len(inliers_np)


def run_icp_stage(source, target, max_corr_distance, init_transform, estimation, max_iteration):
    criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    return o3d.t.pipelines.registration.icp(
        source,
        target,
        max_corr_distance,
        init_transform,
        estimation,
        criteria,
    )

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
    source = ensure_tensor_cloud(source)
    target = ensure_tensor_cloud(target)

    if tensor_point_count(source) < 20 or tensor_point_count(target) < 20:
        print(
            f"Warning: Too few points for ICP (source={tensor_point_count(source)}, target={tensor_point_count(target)}). Returning identity delta."
        )
        class _Result:
            transformation = np_to_cuda_transform(np.eye(4))
            fitness = 0.0
        return _Result(), source

    source_down = source
    target_down = target
    
    radius_normal = voxel_size * 2

    init_transform = np_to_cuda_transform(np.eye(4) if initial_pose is None else initial_pose)
    
    source_down.estimate_normals(radius=radius_normal, max_nn=max_nn)
    target_down.estimate_normals(radius=radius_normal, max_nn=max_nn)

    # 4. Coarse alignment with larger threshold
    coarse_result = run_icp_stage(
        source_down,
        target_down,
        voxel_size * 6,
        init_transform,
        o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
        max_iteration=50,
    )

    coarse_result_fallback = run_icp_stage(
        source_down,
        target_down,
        voxel_size * 10,
        init_transform,
        o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
        max_iteration=80,
    )

    if coarse_result_fallback.fitness > coarse_result.fitness:
        coarse_result = coarse_result_fallback

    print(f"Coarse alignment fitness: {coarse_result.fitness}")

    ## 5. Medium alignment 
    medium_result = run_icp_stage(
        source_down,
        target_down,
        voxel_size * 4,
        coarse_result.transformation,
        o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
        max_iteration=100,
    )
    
    print(f"Medium alignment fitness: {medium_result.fitness}")

    # 6. Fine alignment on original resolution
    fine_result = run_icp_stage(
        source,
        target,
        voxel_size,
        medium_result.transformation,
        o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
        max_iteration=100,
    )
    
    print(f"Fine alignment fitness: {fine_result.fitness}")
    
    return fine_result, source_down


def main():
    """
    Multi stage ICP for local registration of pointclouds. In order to register pointclouds, save them in a folder corresponding to the pose it belongs to, e.g
    pointclouds -> Cloud_pose1 -> cloudX.ply
    and for each pose, save a numpy array 6 dof pose on form [x y z qx qy qz qw].

    Transform coords is specifically for Orbbec cameras to transform the pointcloud from the cameras coordinate system into the specified coordinate system from Motive i.e different depending on how you defined 
    the ridgid bodies. 
    """
    args = parse_args()
    path = args.root_dir if args.root_dir else DEFAULT_ROOT_DIR
    if not os.path.isdir(path):
        raise FileNotFoundError(f"root_dir does not exist: {path}")

    poses = load_coord(path)
    voxel_size = 40
    max_nn = 40
    std_ratio = 2.0
    post_level_scene = True         # Apply one global leveling transform after ICP
    set_floor_to_z0 = False         # Also shift floor vertically to z=0 after leveling
    floor_ransac_threshold = 20.0   # mm
    milimeters = True
    num_clouds = 5
    original_clouds = read_multi_clouds(path, num_clouds)
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

    T_mean = np.array([
    [0.99578279, -0.01414801, -0.09064469, 488.73271500],
    [0.01053994,  0.99913766, -0.04016036, 656.41666033],
    [0.09113471,  0.03903561,  0.99507321, -20.28939523],
    [0.0,         0.0,         0.0,          1.0       ]
    ], dtype=np.float64)
    transform_coords = transform_coords @ transform_2
    print(transform_coords)
    #rotate_x_90 = np.eye(4)
    #rotate_x_90[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()

    #transform_coords = transform_coords @ rotate_x_90


    T_rigid_body_to_orbbec2 = np.load(f"/home/adamfi/codes/Pointclouds/Orbbec_calibrations_mocaplab/orbbec{args.cam_number}.npy")

    # Keep extrinsic rotation unchanged. Only scale translation if pose/cloud pipeline is in mm.
    T_rigid_body_to_orbbec = T_rigid_body_to_orbbec2.copy()
    if milimeters:
        T_rigid_body_to_orbbec[:3, 3] *= 1000.0
    
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
        
        #T_total =T_mean@T@transform_coords
        T_total = T @ np.linalg.inv(T_rigid_body_to_orbbec)
        
        init_transforms.append(T_total)
        pcd.transform(np_to_cuda_transform(T_total))
        initial_pcs.append(pcd)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150.0)
        frame.transform(T_total)
        pose_frames.append(frame)
      
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=250.0)
    initial_vis = [pcd.to_legacy() for pcd in initial_pcs]
    o3d.visualization.draw_geometries(initial_vis + pose_frames + [world_frame])
    in_pcd = concat_point_clouds(initial_pcs)
    o3d.t.io.write_point_cloud(
        "/home/adamfi/codes/Pointclouds/pointclouds/new_calib_test_dist/inital_test_pcd.ply",
        in_pcd.to(o3d.core.Device("CPU:0")),
        write_ascii=True,
    )

    # Down sample and denoise clouds
    preprocessed_pcds = []
    for pcd in initial_pcs:
        pcd_down, _,_ = preprocess_for_registration(pcd, voxel_size, max_nn, std_ratio)
        preprocessed_pcds.append(pcd_down)

    refined_pcs = [copy.deepcopy(preprocessed_pcds[0])]
    resulting_transforms = [copy.deepcopy(init_transforms[0])]
    registration_deltas = [np.eye(4)]  # How much each cloud was moved by registration

    for i in range(1,len(initial_pcs)):
        print(f"Refining cloud {i+1}...")
        source = copy.deepcopy(preprocessed_pcds[i])
        target = concat_point_clouds(refined_pcs)  # Fit the next cloud to all previous clouds
        target = target.voxel_down_sample(max(voxel_size / 2.0, 5.0))

        result, source_down = multi_stage_registration(
            source, target, voxel_size, max_nn, std_ratio, initial_pose=np.eye(4)
        )
        result_tf_np = cuda_transform_to_numpy(result.transformation)
        resulting_transforms.append(result_tf_np @ init_transforms[i])
        registration_deltas.append(result_tf_np)

        source_down.transform(result.transformation)

        refined_pcs.append(source_down)

    # Optional post-leveling step: preserve relative geometry while making floor horizontal
    final_cloud = concat_point_clouds(refined_pcs)

    if post_level_scene:
        cloud_for_plane = final_cloud.voxel_down_sample(max(voxel_size, 20.0))
        leveling_transform, plane_model, num_inliers = compute_scene_leveling_transform(
            cloud_for_plane,
            distance_threshold=floor_ransac_threshold,
            ransac_n=3,
            num_iterations=2000,
            set_floor_to_z0=set_floor_to_z0,
        )
        if plane_model is not None:
            a, b, c, d = plane_model
            print(
                f"Leveling plane: {a:.5f}x + {b:.5f}y + {c:.5f}z + {d:.5f} = 0, "
                f"inliers={num_inliers}"
            )

        leveling_transform_cuda = np_to_cuda_transform(leveling_transform)
        for i in range(len(refined_pcs)):
            refined_pcs[i].transform(leveling_transform_cuda)
        for i in range(len(resulting_transforms)):
            resulting_transforms[i] = leveling_transform @ resulting_transforms[i]

    # Save resulting pose
    for i, pose in enumerate(resulting_transforms):
        t = pose[:3,3]
        R_mat = pose[:3,:3]
        quat = R.from_matrix(R_mat).as_quat(scalar_first = False)
        
        coord = np.concatenate([t, quat])
        
        np.save(os.path.join(path, f"{args.pose_prefix}_{i+1}_transformed.npy"), coord)
        if post_level_scene:
            np.save(os.path.join(path, f"{args.pose_prefix}_{i+1}_transformed_level.npy"), coord)

    # Compute + save per-cloud registration motion (delta) from initial mocap pose to final refined pose
    # delta_i maps: initial_aligned_cloud_i -> refined_cloud_i
    # i.e. refined_pose_i = delta_i @ initial_pose_i
    #registration_deltas_from_poses = []
    #for i, (init_T, refined_T) in enumerate(zip(init_transforms, resulting_transforms)):
    #    delta_T = refined_T @ np.linalg.inv(init_T)
    #    registration_deltas_from_poses.append(delta_T)
#
    #    # Save full 4x4 matrix
    #    np.save(os.path.join(path, f"{args.pose_prefix}_{i+1}_registration_delta.npy"), delta_T)
#
    #    # Save as [tx, ty, tz, qx, qy, qz, qw] for easy reuse
    #    delta_t = delta_T[:3, 3]
    #    delta_q = R.from_matrix(delta_T[:3, :3]).as_quat(scalar_first=False)
    #    delta_pose = np.concatenate([delta_t, delta_q])
    #    np.save(os.path.join(path, f"{args.pose_prefix}_{i+1}_registration_delta_pose.npy"), delta_pose)
#
    #    print(f"Cloud {i+1} registration delta:\n{delta_T}")
   
    # Save registered pointclouds
    final_cloud = concat_point_clouds(refined_pcs)

    final_cloud_path = os.path.join(path, args.final_cloud_name)
    o3d.t.io.write_point_cloud(final_cloud_path, final_cloud.to(o3d.core.Device("CPU:0")), write_ascii=True)
    print(f"Saved merged cloud to: {final_cloud_path}")

    # Visualize registered clouds and final pose frames
    vis_clouds = []
    cmap = plt.get_cmap("tab20")
    for i, cloud in enumerate(refined_pcs):
        c = cloud.to_legacy()
        c.paint_uniform_color(cmap(i % 20)[:3])
        vis_clouds.append(c)

    final_pose_frames = []
    for pose in resulting_transforms:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150.0)
        frame.transform(pose)
        final_pose_frames.append(frame)

    world_frame_final = o3d.geometry.TriangleMesh.create_coordinate_frame(size=250.0)
    draw_geometries(vis_clouds + final_pose_frames + [world_frame_final])


if __name__=="__main__":
   
    main()