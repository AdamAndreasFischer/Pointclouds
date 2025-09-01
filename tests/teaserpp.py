import teaserpp_python
import open3d as o3d
import numpy as np
import copy
import time
from scipy.spatial.transform import Rotation as R
import os

NOISE_BOUND = 0.05
N_OUTLIERS = 1700
OUTLIER_TRANSLATION_LB = 5
OUTLIER_TRANSLATION_UB = 10

def load_coord(path):
    print("Coord func")
    print(os.listdir(path))
    paths = [path for path in os.listdir(path) if path.endswith(".npy")]
    paths.sort()
    paths = sorted(paths, key=len)
    poses = []
    for pose_path in paths:
        poses.append(np.load(os.path.join(path, pose_path)))
    poses = np.array(poses)
    return poses

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
    transform[:3, 3] = translation
    
    return transform

def allign_clouds(source, target):
    print("==================================================")
    print("        TEASER++ Python registration example      ")
    print("==================================================")

    
    src = np.transpose(np.asarray(source.points))
    N = src.shape[1]

    dst = np.transpose(np.asarray(target.points))

    # Populating the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RotationEstimationAlgorithm.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    start = time.time()
    solver.solve(src, dst)
    end = time.time()

    solution = solver.getSolution()

    translation = solution.translation
    rotation = solution.rotation

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation

    return transformation_matrix




source = o3d.io.read_point_cloud("/home/adamfi/Codes/Laptopcodes/Mocap_process/No_mocap_clouds/RGBDPoints_pose2.ply")
target = o3d.io.read_point_cloud("/home/adamfi/Codes/Laptopcodes/Mocap_process/No_mocap_clouds/RGBDPoints_pose1.ply")

voxel_size = 20  # Adjust this value as needed
source = source.voxel_down_sample(voxel_size)
target = target.voxel_down_sample(voxel_size)

source_pose = np.load("/home/adamfi/Codes/Laptopcodes/Mocap_process/No_mocap_clouds/pose_2.npy")
target_pose = np.load("/home/adamfi/Codes/Laptopcodes/Mocap_process/No_mocap_clouds/pose_1.npy")

source_transform = pose_to_transform_matrix(source_pose)
target_transform = pose_to_transform_matrix(target_pose)

source.transform(source_transform)
target.transform(target_transform)

transformation_matrix = allign_clouds(source, target)

o3d.visualization.draw_geometries([source, target])

