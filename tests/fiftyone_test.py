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
from point_cloud_registration import NDT, PlaneICP
import matplotlib.pyplot as plt
#import fiftyone as fo
#import fiftyone.utils.utils3d as fou3d
#import fiftyone.core.collections as focc

NUM_THREADS = max(1, multiprocessing.cpu_count())



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

def read_clouds(path):
    paths = [path for path in os.listdir(path) if path.endswith(".ply")]
    paths.sort()
    paths = sorted(paths, key=len)
    clouds = []
    for ptc in paths:
        clouds.append(o3d.io.read_point_cloud(os.path.join(path, ptc)))

    return clouds

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


def get_transform_matrices(pose_source, pose_target):
    
    t_source = pose_source[:3]
    q_source = pose_source[3:]
    R_mat_source = R.from_quat(q_source).as_matrix()
    T_source = np.eye(4)
    T_source[:3, :3] = R_mat_source
    T_source[:3, 3] = t_source * 1000  # Convert to millimeters

    t_target = pose_target[:3]
    q_target = pose_target[3:]
    R_mat_target = R.from_quat(q_target).as_matrix()
    T_target = np.eye(4)
    T_target[:3, :3] = R_mat_target
    T_target[:3, 3] = t_target * 1000  # Convert to millimeters

    return T_source, T_target

def transform_between(pose1, pose2):
    """Function to find the transfrom from pose2 to pose1 in order to use as initial transform estimate
    """
    transform1 = pose_to_transform_matrix(pose1)
    transform2 = pose_to_transform_matrix(pose2)

    transfrom2to1 = transform1@np.linalg.inv(transform2)
    return transfrom2to1

def load_clouds(path):
    pointclouds = [path for path in os.listdir(path) if path.endswith(".ply")]
    pointclouds.sort()
    pointclouds = sorted(pointclouds, key=len)
    return pointclouds


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)

    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    
    draw_geometries([inlier_cloud, outlier_cloud])

def down_sample(cloud,voxel_size):

    cloud_down = cloud.voxel_down_sample(voxel_size= voxel_size)

    return cloud_down

def denoise_pointclouds(cloud, neighbours = 20, std_ratio = 2.0):
    
    denoised_cld, ind = cloud.remove_statistical_outlier(nb_neighbors=neighbours,
                                                    std_ratio=std_ratio)
    #denoised_cld, ind = cloud.remove_radius_outlier(neighbours, radius = 30)
   
    return denoised_cld, ind

def process_pointcloud(cloud,id, voxel_size, max_nn, std_ratio):
    #cloud = o3d.io.read_point_cloud(os.path.join(base_path, cloud_path))
    cloud_down = down_sample(cloud, voxel_size)
    #cloud_down = cloud
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    

    cloud_denoise,ind = denoise_pointclouds(cloud_down, neighbours=max_nn, std_ratio=std_ratio) # 50 0.5

    cloud_denoise.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn), 
        cloud_denoise.orient_normals_towards_camera_location(np.array([0,0,0])))
    
    radius_feature = voxel_size * 3
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        cloud_denoise,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn))
    
    return cloud_denoise, pcd_fpfh ,id

def process_clouds_multiprocess(clouds, voxel_size, max_nn, std_ratio):
    processed_clouds = len(clouds)*[None]
    fpfh_features = len(clouds)*[None]
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(process_pointcloud, clouds[id], id, voxel_size, max_nn, std_ratio) 
                  for id in range(len(clouds))]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                cloud_p,fpfh ,id = result
                processed_clouds[id] = cloud_p
                fpfh_features[id] = fpfh
    return processed_clouds, fpfh_features


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    try:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.999))
        print(":: Ransac done ::")
        return result
    except Exception as e:
        print(f"Error in RANSAC: {e}")

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 3.0
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 2
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    radius = voxel_size*2

    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30),
        source.orient_normals_towards_camera_location(np.array([0,0,0])))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30),
        target.orient_normals_towards_camera_location(np.array([0,0,0])))

    result = o3d.pipelines.registration.registration_colored_icp(
        source, target, distance_threshold, result_ransac,#result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        #o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def stich_pointclouds(original_clouds, poses, voxel_size=5.0, neighbours =60, std_ratio = 0.3 ):
    transform_coords = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])    

    initial_pcs = []
    camera_frames = []
    
    for pose, pcd_original in zip(poses, original_clouds):
        pcd = copy.deepcopy(pcd_original)

        t = pose[:3]
        q = pose[3:]

        q_norm = np.linalg.norm(q) #Normalize quaternion
        q = q/q_norm
        R_mat = R.from_quat(q).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t * 1000  # Convert pose to mm as pointclouds are measured in mm

        T_total = T@transform_coords
        pcd.transform(T_total)
        initial_pcs.append(pcd)
        
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200)
        camera_frame.transform(T)
        camera_frames.append(camera_frame)

    #draw_geometries(initial_pcs+ camera_frames)
    initial_pcs_processed, fpfh_features = process_clouds_multiprocess(initial_pcs, voxel_size,neighbours, std_ratio) #TODO: proccess clouds after initial allignment
    refined_pcs = [copy.deepcopy(initial_pcs_processed[0])]
    ransac_results= len(initial_pcs_processed)*[None]
    print("Starting Global registration")
    for i in tqdm(range(1, len(initial_pcs_processed))):

        source = copy.deepcopy(initial_pcs_processed[i])
        source_fpfh = fpfh_features[i]
        target_fpfh = fpfh_features[i-1]
        target = refined_pcs[-1]  # Previous aligned cloud

        #ransac_result = execute_global_registration(source, target, source_fpfh, target_fpfh, voxel_size)
        ransac_result = execute_fast_global_registration(source, target, source_fpfh, target_fpfh, voxel_size)
        ransac_results[i]=copy.deepcopy(ransac_result)
        print(ransac_result)
        source.transform(ransac_result.transformation)
        #original_clouds[i].transform(ransac_result.transformation)
        refined_pcs.append(source)

    print("Drawing ransac result")
    draw_geometries(refined_pcs+camera_frames)
    print("Starting local registration")

    icp_pieces = [copy.deepcopy(initial_pcs_processed[0])]
    fullscale_pcs = [copy.deepcopy(initial_pcs[0])]

    for i in tqdm(range(1, len(initial_pcs_processed))):
        source = copy.deepcopy(initial_pcs_processed[i])
        source_2 = copy.deepcopy(initial_pcs[i])
        target = icp_pieces[-1]
        s2c = transform_between(poses[i-1], poses[i])
        icp_result = refine_registration(source,target, ransac_results[i].transformation,voxel_size)

        print(icp_result)
        source.transform(icp_result.transformation)
        source_2.transform(icp_result.transformation)
        icp_pieces.append(source)
        fullscale_pcs.append(source_2)
    print("Drawing ICP result")
    draw_geometries(fullscale_pcs+ camera_frames)
    combined_old_pcd = o3d.geometry.PointCloud()
   
    for pcd in fullscale_pcs:
        combined_old_pcd += pcd
    combined_pcd = o3d.t.geometry.PointCloud.from_legacy(combined_old_pcd)
    return fullscale_pcs, camera_frames, combined_pcd

def test():
    path = "C:/Users/adamf/Codes/Mocap_process/No_mocap_clouds"
    poses = load_coord(path)

    voxel_size = 10
    max_nn = 60
    std_ratio = 0.3

    
    cloud_paths = load_clouds(path)
    
    original_clouds = read_clouds(path)
    print(original_clouds[0])
    #processed_clouds, fpfh_features = process_clouds_multiprocess(path, cloud_paths)
    
    full_scale, camera_frames, final_pcd = stich_pointclouds(original_clouds, poses,voxel_size=voxel_size, neighbours= max_nn, std_ratio = std_ratio)
    
    width, height = 640, 480
    intrinsic = o3d.core.Tensor([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
    extrinsic = o3d.core.Tensor(np.eye(4))

    # Create depth and color images
    #depth = final_pcd.project_to_depth_image(width, height, intrinsic, extrinsic, depth_scale=1000.0, depth_max=10000.0)
    #color = final_pcd.project_to_color_image(width, height, intrinsic, extrinsic)
    rgbd_image = final_pcd.project_to_rgbd_image(width, height, intrinsic, extrinsic)

    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()




if __name__=="__main__":
    test()