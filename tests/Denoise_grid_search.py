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
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def evaluate_denoising_quality(original_cloud, denoised_cloud, indices):
    """
    Evaluate the quality of denoising using multiple metrics
    Returns a dictionary with various quality metrics
    """
    metrics = {}
    
    # 1. Point retention ratio (how many points were kept)
    original_count = len(original_cloud.points)
    denoised_count = len(denoised_cloud.points)
    metrics['point_retention_ratio'] = denoised_count / original_count
    metrics['points_removed'] = original_count - denoised_count
    
    # 2. Density uniformity (coefficient of variation of local densities)
    if denoised_count > 10:  # Need enough points for meaningful calculation
        try:
            # Calculate local point density using k-nearest neighbors
            points = np.asarray(denoised_cloud.points)
            nbrs = NearestNeighbors(n_neighbors=min(10, denoised_count-1), algorithm='ball_tree').fit(points)
            distances, _ = nbrs.kneighbors(points)
            
            # Local density is inverse of average distance to k neighbors
            local_densities = 1 / (np.mean(distances[:, 1:], axis=1) + 1e-8)
            
            # Coefficient of variation (lower is better - more uniform)
            metrics['density_uniformity'] = np.std(local_densities) / np.mean(local_densities)
        except:
            metrics['density_uniformity'] = float('inf')
    else:
        metrics['density_uniformity'] = float('inf')
    
    # 3. Surface smoothness (using normal consistency)
    if denoised_cloud.has_normals():
        try:
            normals = np.asarray(denoised_cloud.normals)
            points = np.asarray(denoised_cloud.points)
            
            # Find neighboring points and calculate normal consistency
            nbrs = NearestNeighbors(n_neighbors=min(8, len(points)-1), algorithm='ball_tree').fit(points)
            _, neighbor_indices = nbrs.kneighbors(points)
            
            normal_consistencies = []
            for i, neighbors in enumerate(neighbor_indices):
                if len(neighbors) > 1:
                    current_normal = normals[i]
                    neighbor_normals = normals[neighbors[1:]]  # Exclude self
                    
                    # Calculate dot products (cosine similarity)
                    consistencies = np.abs(np.dot(neighbor_normals, current_normal))
                    normal_consistencies.append(np.mean(consistencies))
            
            metrics['surface_smoothness'] = np.mean(normal_consistencies) if normal_consistencies else 0
        except:
            metrics['surface_smoothness'] = 0
    else:
        metrics['surface_smoothness'] = 0
    
    # 4. Noise level estimation (using local variation)
    if denoised_count > 20:
        try:
            points = np.asarray(denoised_cloud.points)
            nbrs = NearestNeighbors(n_neighbors=min(5, denoised_count-1), algorithm='ball_tree').fit(points)
            distances, neighbor_indices = nbrs.kneighbors(points)
            
            # Calculate local variation
            local_variations = []
            for i, neighbors in enumerate(neighbor_indices):
                if len(neighbors) > 1:
                    center = points[i]
                    neighbor_points = points[neighbors[1:]]  # Exclude self
                    
                    # Calculate distances from center to neighbors
                    dists = np.linalg.norm(neighbor_points - center, axis=1)
                    local_variations.append(np.std(dists))
            
            metrics['noise_level'] = np.mean(local_variations) if local_variations else float('inf')
        except:
            metrics['noise_level'] = float('inf')
    else:
        metrics['noise_level'] = float('inf')
    
    # 5. Geometric feature preservation (using principal component analysis)
    if denoised_count > 10:
        try:
            points = np.asarray(denoised_cloud.points)
            
            # Calculate covariance matrix and eigenvalues
            centered_points = points - np.mean(points, axis=0)
            cov_matrix = np.cov(centered_points.T)
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
            
            # Feature preservation score based on eigenvalue ratios
            if eigenvalues[0] > 1e-8:
                metrics['feature_preservation'] = (eigenvalues[1] + eigenvalues[2]) / eigenvalues[0]
            else:
                metrics['feature_preservation'] = 0
        except:
            metrics['feature_preservation'] = 0
    else:
        metrics['feature_preservation'] = 0
    
    # 6. Overall quality score (weighted combination)
    # Lower is better for noise_level and density_uniformity
    # Higher is better for others
    weights = {
        'point_retention_ratio': 0.2,
        'density_uniformity': -0.25,  # Negative because lower is better
        'surface_smoothness': 0.25,
        'noise_level': -0.15,  # Negative because lower is better
        'feature_preservation': 0.15
    }
    
    quality_score = 0
    for metric, weight in weights.items():
        value = metrics[metric]
        if not np.isfinite(value):
            value = 0 if weight > 0 else -1  # Penalty for infinite values
        quality_score += weight * value
    
    metrics['overall_quality_score'] = quality_score
    
    return metrics

def display_inlier_outlier(cloud, ind):
    print(f"Index array length: {len(ind)}")
    print(f"Cloud points length: {len(cloud.points)}")
    print(f"Max index value: {np.max(ind) if len(ind) > 0 else 'N/A'}")
    print(f"Min index value: {np.min(ind) if len(ind) > 0 else 'N/A'}")
    
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    
    draw_geometries([inlier_cloud, outlier_cloud])

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


def main():
    # Load your point cloud
    cloud = o3d.io.read_point_cloud("/home/adamfi/Codes/Mocap_process/No_mocap_clouds/RGBDPoints_pose1.ply")

    # Store all results for comparison
    results = []
    
    # Preprocess the point cloud
    voxel_radia = [100,80,60,40]
    neighbours = [80,60, 40]
    std_ratios = [2,2.5,3,3.5,4]
    
    print("Running automated denoising evaluation...")
    print("="*60)
    
    for voxel_radius in voxel_radia:
        for max_nn in neighbours:
            for std_ratio in std_ratios:
                print(f"Testing: voxel={voxel_radius}, neighbors={max_nn}, std_ratio={std_ratio}")
                
                cloud_filtered, inlier_indices, cloud_downsampled = preprocess_for_registration(
                    cloud, voxel_radius, max_nn, std_ratio)
                
                # Evaluate denoising quality
                metrics = evaluate_denoising_quality(cloud_downsampled, cloud_filtered, inlier_indices)
                
                # Store parameters with results
                result = {
                    'voxel_radius': voxel_radius,
                    'max_neighbors': max_nn,
                    'std_ratio': std_ratio,
                    **metrics
                }
                results.append(result)
                
                print(f"  Quality Score: {metrics['overall_quality_score']:.4f}")
                print(f"  Points Retained: {metrics['point_retention_ratio']:.3f} ({metrics['points_removed']} removed)")
                print(f"  Density Uniformity: {metrics['density_uniformity']:.4f}")
                print(f"  Surface Smoothness: {metrics['surface_smoothness']:.4f}")
                print(f"  Noise Level: {metrics['noise_level']:.4f}")
                print("-" * 40)
    
    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(results)
    
    # Save results to CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"denoising_evaluation_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    
    # Find best configurations
    print("\n" + "="*60)
    print("TOP 5 DENOISING CONFIGURATIONS:")
    print("="*60)
    
    # Sort by overall quality score (higher is better)
    top_configs = df.nlargest(5, 'overall_quality_score')
    
    for i, (_, row) in enumerate(top_configs.iterrows(), 1):
        print(f"{i}. Voxel: {row['voxel_radius']}, Neighbors: {row['max_neighbors']}, "
              f"Std Ratio: {row['std_ratio']}")
        print(f"   Quality Score: {row['overall_quality_score']:.4f}")
        print(f"   Retention: {row['point_retention_ratio']:.3f}, "
              f"Smoothness: {row['surface_smoothness']:.3f}")
    
    # Create visualization plots
    create_evaluation_plots(df, timestamp)
    
    print(f"\nResults saved to: {csv_filename}")
    print(f"Evaluation plots saved with timestamp: {timestamp}")
    
    # Optionally show the best result
    best_config = top_configs.iloc[0]
    print(f"\nShowing best configuration:")
    print(f"Voxel: {best_config['voxel_radius']}, Neighbors: {best_config['max_neighbors']}, "
          f"Std Ratio: {best_config['std_ratio']}")
    
    # Recreate best result for visualization
    cloud_filtered, inlier_indices, cloud_downsampled = preprocess_for_registration(
        cloud, best_config['voxel_radius'], best_config['max_neighbors'], best_config['std_ratio'])
    
    # Uncomment the line below to show the best result
    # display_inlier_outlier(cloud_downsampled, inlier_indices)

def create_evaluation_plots(df, timestamp):
    """Create visualization plots for the evaluation results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Denoising Parameter Evaluation', fontsize=16)
    
    # 1. Overall Quality Score vs Parameters
    axes[0,0].scatter(df['std_ratio'], df['overall_quality_score'], 
                     c=df['voxel_radius'], cmap='viridis', alpha=0.7)
    axes[0,0].set_xlabel('Std Ratio')
    axes[0,0].set_ylabel('Overall Quality Score')
    axes[0,0].set_title('Quality Score vs Std Ratio')
    
    # 2. Point Retention vs Quality
    axes[0,1].scatter(df['point_retention_ratio'], df['overall_quality_score'],
                     c=df['max_neighbors'], cmap='plasma', alpha=0.7)
    axes[0,1].set_xlabel('Point Retention Ratio')
    axes[0,1].set_ylabel('Overall Quality Score')
    axes[0,1].set_title('Quality vs Retention')
    
    # 3. Surface Smoothness vs Noise Level
    axes[0,2].scatter(df['noise_level'], df['surface_smoothness'],
                     c=df['overall_quality_score'], cmap='coolwarm', alpha=0.7)
    axes[0,2].set_xlabel('Noise Level')
    axes[0,2].set_ylabel('Surface Smoothness')
    axes[0,2].set_title('Smoothness vs Noise')
    
    # 4. Parameter space heatmap - Quality Score
    pivot_quality = df.pivot_table(values='overall_quality_score', 
                                  index='max_neighbors', 
                                  columns='std_ratio', 
                                  aggfunc='mean')
    im1 = axes[1,0].imshow(pivot_quality.values, cmap='viridis', aspect='auto')
    axes[1,0].set_title('Quality Score Heatmap')
    axes[1,0].set_xlabel('Std Ratio Index')
    axes[1,0].set_ylabel('Max Neighbors Index')
    plt.colorbar(im1, ax=axes[1,0])
    
    # 5. Density Uniformity Heatmap
    pivot_density = df.pivot_table(values='density_uniformity', 
                                  index='max_neighbors', 
                                  columns='std_ratio', 
                                  aggfunc='mean')
    im2 = axes[1,1].imshow(pivot_density.values, cmap='viridis_r', aspect='auto')
    axes[1,1].set_title('Density Uniformity Heatmap')
    axes[1,1].set_xlabel('Std Ratio Index')
    axes[1,1].set_ylabel('Max Neighbors Index')
    plt.colorbar(im2, ax=axes[1,1])
    
    # 6. Box plot of quality scores by voxel radius
    voxel_groups = df.groupby('voxel_radius')['overall_quality_score'].apply(list)
    axes[1,2].boxplot(voxel_groups.values, labels=voxel_groups.index)
    axes[1,2].set_xlabel('Voxel Radius')
    axes[1,2].set_ylabel('Overall Quality Score')
    axes[1,2].set_title('Quality Distribution by Voxel Size')
    
    plt.tight_layout()
    plt.savefig(f'denoising_evaluation_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()


main()