import open3d as o3d
import numpy as np
import os
from scipy import stats

def analyze_point_cloud_colors(pcd_path):
    """Analyze color uniformity of a point cloud"""
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        
        if not pcd.has_colors():
            return None, "No colors"
        
        colors = np.asarray(pcd.colors)
        
        if len(colors) == 0:
            return None, "Empty colors"
        
        # Calculate color statistics
        mean_color = np.mean(colors, axis=0)
        std_color = np.std(colors, axis=0)
        
        # Calculate overall color variance (measure of uniformity)
        color_variance = np.var(colors)
        
        # Calculate dominant color (mode)
        # Discretize colors for mode calculation
        colors_discrete = (colors * 255).astype(int)
        
        # Check for brown-ish colors (rough heuristic)
        # Brown is typically RGB around (150, 75, 0) to (200, 100, 50)
        brown_threshold = np.array([0.6, 0.3, 0.0])  # In [0,1] range
        brown_upper = np.array([0.8, 0.5, 0.3])
        
        # Count brown-ish points
        is_brownish = np.all((colors >= brown_threshold) & (colors <= brown_upper), axis=1)
        brown_percentage = np.sum(is_brownish) / len(colors) * 100
        
        # Calculate color range
        color_range = np.max(colors, axis=0) - np.min(colors, axis=0)
        
        return {
            'num_points': len(colors),
            'mean_color': mean_color,
            'std_color': std_color,
            'color_variance': color_variance,
            'color_range': color_range,
            'brown_percentage': brown_percentage,
            'colors': colors
        }, None
        
    except Exception as e:
        return None, str(e)

def main():
    """
    Code to analyze and rank color diversity in pcds. Sometimes orbbec pcds only contain one brownish color, and this code analyzes the color to show which pcd has least diversity
    
    """
    cloud_dir = "/home/adamfi/Codes/Pointclouds/pointclouds/close_cube/Cloud_pose1"
    
    results = []
    
    print("Analyzing point cloud colors...")
    print("=" * 60)
    
    for i in range(1, 51):  # cloud1.ply to cloud50.ply
        cloud_path = os.path.join(cloud_dir, f"cloud{i}.ply")
        
        if os.path.exists(cloud_path):
            analysis, error = analyze_point_cloud_colors(cloud_path)
            
            if analysis is None:
                print(f"cloud{i}.ply: ERROR - {error}")
                continue
            
            results.append((i, analysis))
            
            # Print summary
            print(f"cloud{i}.ply:")
            print(f"  Points: {analysis['num_points']:,}")
            print(f"  Mean RGB: [{analysis['mean_color'][0]:.3f}, {analysis['mean_color'][1]:.3f}, {analysis['mean_color'][2]:.3f}]")
            print(f"  Color variance: {analysis['color_variance']:.6f}")
            print(f"  Brown percentage: {analysis['brown_percentage']:.1f}%")
            print(f"  Color range: [{analysis['color_range'][0]:.3f}, {analysis['color_range'][1]:.3f}, {analysis['color_range'][2]:.3f}]")
            print()
    
    if not results:
        print("No valid point clouds found!")
        return
    
    print("=" * 60)
    print("SUMMARY ANALYSIS:")
    print("=" * 60)
    
    # Find most uniform color (lowest variance)
    most_uniform = min(results, key=lambda x: x[1]['color_variance'])
    print(f"Most uniform color: cloud{most_uniform[0]}.ply")
    print(f"  Color variance: {most_uniform[1]['color_variance']:.6f}")
    print(f"  Mean color: [{most_uniform[1]['mean_color'][0]:.3f}, {most_uniform[1]['mean_color'][1]:.3f}, {most_uniform[1]['mean_color'][2]:.3f}]")
    print()
    
    # Find cloud with highest brown percentage
    most_brown = max(results, key=lambda x: x[1]['brown_percentage'])
    print(f"Most brown cloud: cloud{most_brown[0]}.ply")
    print(f"  Brown percentage: {most_brown[1]['brown_percentage']:.1f}%")
    print(f"  Mean color: [{most_brown[1]['mean_color'][0]:.3f}, {most_brown[1]['mean_color'][1]:.3f}, {most_brown[1]['mean_color'][2]:.3f}]")
    print()
    
    # Find clouds with very high brown percentage (>90%)
    very_brown_clouds = [r for r in results if r[1]['brown_percentage'] > 90]
    if very_brown_clouds:
        print("Clouds that are mostly brown (>90%):")
        for cloud_id, analysis in very_brown_clouds:
            print(f"  cloud{cloud_id}.ply: {analysis['brown_percentage']:.1f}% brown")
        print()
    
    # Sort by color variance to show uniformity ranking
    sorted_by_uniformity = sorted(results, key=lambda x: x[1]['color_variance'])
    print("Top 5 most uniform clouds (by color variance):")
    for i, (cloud_id, analysis) in enumerate(sorted_by_uniformity[:5]):
        print(f"  {i+1}. cloud{cloud_id}.ply - variance: {analysis['color_variance']:.6f}")
    print()
    
    print("Top 5 most variable clouds:")
    for i, (cloud_id, analysis) in enumerate(sorted_by_uniformity[-5:]):
        print(f"  {i+1}. cloud{cloud_id}.ply - variance: {analysis['color_variance']:.6f}")

if __name__ == "__main__":
    main()
