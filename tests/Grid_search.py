import open3d as o3d
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import time
from tqdm import tqdm
from Multi_stage_icp import multi_stage_registration
import json
import signal
import sys
import atexit
#import fiftyone as fo
#import fiftyone.utils.utils3d as fou3d
#import fiftyone.core.collections as focc

# Global variable to track the executor
executor = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nReceived interrupt signal (Ctrl+C)")
    print("Terminating all processes...")
    
    global executor
    if executor is not None:
        print("Shutting down ProcessPoolExecutor...")
        executor.shutdown(wait=False, cancel_futures=True)
        
    print("Exiting gracefully...")
    sys.exit(0)

def cleanup_on_exit():
    """Cleanup function called on exit"""
    global executor
    if executor is not None:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except:
            pass

# Register signal handlers and cleanup function
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
atexit.register(cleanup_on_exit)

def preprocess_for_registration(cloud, voxel_size, max_nn=30,std_ratio=2.0, remove_outliers=True):
    """Enhanced preprocessing for better registration results"""
    
    # 1. Voxel downsampling
    cloud_down = cloud.voxel_down_sample(voxel_size)
    
    # 2. Outlier removal (statistical)
    if remove_outliers:
        cloud_filtered, _ = cloud_down.remove_statistical_outlier(
            nb_neighbors=max_nn, std_ratio=std_ratio)
    else:
        cloud_filtered = cloud_down

    cloud_down.remove_radius_outlier(nb_points=max_nn, radius=std_ratio)

    # 3. Estimate normals with consistent orientation
    radius_normal = voxel_size * 2
    cloud_filtered.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn))
    cloud_filtered.orient_normals_towards_camera_location(np.array([0, 0, 0]))
    
    # 4. Optional: Smooth the point cloud
    # This uses a simple moving average approach
    # You might need to implement this as Open3D doesn't have it built-in
    
    return cloud_filtered

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

def evaluate_registration_params(params_tuple):
    """
    Worker function for multiprocessing grid search using multi_stage_registration
    Returns: (params_dict, fitness_score, transformation_matrix)
    """
    radius, max_nn, std_ratio, source_path, target_path, source_pose_path, target_pose_path = params_tuple
    
    try:
        # Load point clouds
        source = o3d.io.read_point_cloud(source_path)
        target = o3d.io.read_point_cloud(target_path)
        
        # Load poses
        pose_source = np.load(source_pose_path)
        pose_target = np.load(target_pose_path)
        
        # Transform coordinates
        transform_coords = np.array([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]) 
        
        t_target = pose_to_transform_matrix(pose_target)
        t_source = pose_to_transform_matrix(pose_source)
        
        source.transform(t_source @ transform_coords)
        target.transform(t_target @ transform_coords)
        
        # Use multi_stage_registration function with grid search parameters
        print("Registration...")

        source_down = preprocess_for_registration(source, voxel_size=radius, max_nn=max_nn, std_ratio=std_ratio)
        target_down = preprocess_for_registration(target, voxel_size=radius, max_nn=max_nn, std_ratio=std_ratio)
        
        coarse_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, radius * 5, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))

        finer_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, radius * 2, coarse_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))


        result = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius*0.5, finer_result.transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        #result = multi_stage_registration(
        #    source, target, 
        #    voxel_size=radius, 
        #    max_nn=max_nn, 
        #    std_ration=std_ratio,  # Note: using 'std_ration' to match the function signature
        #    initial_pose=None, 
        #    save=False
        #)

        print(f"Registration completed")

        params_dict = {
            "voxel_size": radius,
            "neighbours": max_nn,
            "std_ratio": std_ratio,
            "fitness": result.fitness,
            "inlier_rmse": result.inlier_rmse
        }
        
        return params_dict, result.fitness, result.transformation
        
    except Exception as e:
        print(f"Error with params {radius}, {max_nn}, {std_ratio}: {str(e)}")
        return {"voxel_size": radius, "neighbours": max_nn, "std_ratio": std_ratio, 
                "fitness": 0.0, "error": str(e)}, 0.0, np.eye(4)

def run_grid_search_multiprocessed():
    """
    Run grid search using all available CPU cores with multi_stage_registration
    """
    # File paths (updated for Linux)
    source_path = "/home/adamfi/Codes/Mocap_process/Clouds_thrusday/RGBDPoints_pose2.ply"
    target_path = "/home/adamfi/Codes/Mocap_process/Clouds_thrusday/RGBDPoints_pose1.ply"
    source_pose_path = "/home/adamfi/Codes/Mocap_process/Clouds_thrusday/pose_2.npy"
    target_pose_path = "/home/adamfi/Codes/Mocap_process/Clouds_thrusday/pose_1.npy"
    
    # Parameter grid
    voxel_radius = [100,80,60,40]
    neighbours = [80,60, 40]
    std_ratios = [2,2.5,3,3.5,4]
    
    # Create parameter combinations
    param_combinations = []
    for radius in voxel_radius:
        for max_nn in neighbours:
            for std_ratio in std_ratios:
                param_combinations.append((
                    radius, max_nn, std_ratio, 
                    source_path, target_path, 
                    source_pose_path, target_pose_path
                ))
    
    print(f"Running grid search with {len(param_combinations)} parameter combinations...")
    print(f"Using {multiprocessing.cpu_count()} CPU cores")
    print("Using multi_stage_registration function")
    multiprocessing.set_start_method('forkserver')
    # Results storage
    all_results = []
    best_fitness = 0.0
    best_params = None
    best_transformation = None
    NUM_WORKERS = 2
    # Run multiprocessed grid search
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all jobs
        future_to_params = {
            executor.submit(evaluate_registration_params, params): params 
            for params in param_combinations
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_params), 
                          total=len(param_combinations), 
                          desc="Grid Search Progress"):
            try:
                params_dict, fitness, transformation = future.result()
                all_results.append(params_dict)
                
                # Track best result
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = params_dict
                    best_transformation = transformation
                    
                print(f"Completed: voxel={params_dict['voxel_size']}, "
                      f"neighbours={params_dict['neighbours']}, "
                      f"std_ratio={params_dict['std_ratio']}, "
                      f"fitness={fitness:.6f}")
                      
            except Exception as e:
                print(f"Job failed: {e}")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save all results as JSON
    results_file = f"grid_search_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save best transformation as numpy array
    best_transform_file = f"best_transformation_{timestamp}.npy"
    np.save(best_transform_file, best_transformation)
    
    # Save best parameters separately
    best_params_file = f"best_params_{timestamp}.json"
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print("\n" + "="*60)
    print("GRID SEARCH COMPLETED!")
    print("="*60)
    print(f"Best fitness score: {best_fitness:.6f}")
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nResults saved to: {results_file}")
    print(f"Best transformation saved to: {best_transform_file}")
    print(f"Best parameters saved to: {best_params_file}")
    print("="*60)
    
    # Visualize best result
    print("\nDisplaying best registration result...")
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)
    
    # Apply pose transformations
    pose_source = np.load(source_pose_path)
    pose_target = np.load(target_pose_path)
    
    transform_coords = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]) 
    
    t_target = pose_to_transform_matrix(pose_target)
    t_source = pose_to_transform_matrix(pose_source)
    
    source.transform(t_source @ transform_coords)
    target.transform(t_target @ transform_coords)
    
    # Display result
    draw_registration_result_original_color(source, target, best_transformation)
    
    return best_params, best_fitness, best_transformation, all_results

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    #target.paint_uniform_color([1,0,0])
    o3d.visualization.draw_geometries([source_temp, target])

if __name__ == "__main__":
    # Run the multiprocessed grid search
    best_params, best_fitness, best_transformation, all_results = run_grid_search_multiprocessed()
    
    # Print summary statistics
    print(f"\nSummary of {len(all_results)} experiments:")
    fitness_scores = [r['fitness'] for r in all_results if 'fitness' in r and 'error' not in r]
    if fitness_scores:
        print(f"Mean fitness: {np.mean(fitness_scores):.6f}")
        print(f"Std fitness: {np.std(fitness_scores):.6f}")
        print(f"Min fitness: {np.min(fitness_scores):.6f}")
        print(f"Max fitness: {np.max(fitness_scores):.6f}")
        
        # Show top 5 results
        sorted_results = sorted([r for r in all_results if 'error' not in r], 
                               key=lambda x: x['fitness'], reverse=True)
        print(f"\nTop 5 parameter combinations:")
        for i, result in enumerate(sorted_results[:5]):
            print(f"  {i+1}. Fitness: {result['fitness']:.6f}, "
                  f"Voxel: {result['voxel_size']}, "
                  f"Neighbours: {result['neighbours']}, "
                  f"Std_ratio: {result['std_ratio']}")
    else:
        print("No successful registrations found.")