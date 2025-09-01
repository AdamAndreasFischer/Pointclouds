#!/usr/bin/env python3
"""
pc_sweep.py

Parameter sweep for preprocessing + registration on a pair of point clouds.
Assumes PLYs are in millimetres (mm).

Usage:
    python pc_sweep.py --src source.ply --tgt target.ply

Outputs:
    - sweep_results.csv (per-parameter combo metrics)
    - debug/ (optional saved pointclouds for best/worst combos)
"""

import os
import argparse
import copy
import csv
import itertools
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
# -------------------------
# Utility functions
# -------------------------
def read_ply(path):
    p = o3d.io.read_point_cloud(path)
    if len(p.points) == 0:
        raise RuntimeError(f"Empty cloud: {path}")
    return p

def write_ply(path, pcd):
    o3d.io.write_point_cloud(path, pcd)

def voxel_downsample_copy(pcd, voxel_size):
    return pcd.voxel_down_sample(voxel_size)

def denoise_radius_then_stat(pcd, radius, min_points=8, nb_neighbors=30, std_ratio=1.0):
    """
    radius: mm
    """
    p = copy.deepcopy(pcd)
    if radius is not None and radius > 0:
        p, ind_r = p.remove_radius_outlier(nb_points=min_points, radius=radius)
    p, ind_s = p.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return p

def estimate_normals(pcd, voxel_size, radius_multiplier=2.0, max_nn=50, orient_towards=None):
    radius = max(2.0 * voxel_size, 30.0) * radius_multiplier
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    # try to orient normals consistently
    try:
        if orient_towards is not None:
            pcd.orient_normals_towards_camera_location(orient_towards)
        else:
            pcd.orient_normals_consistent_tangent_plane(k=30)
    except Exception:
        pass
    return pcd

def compute_fpfh(pcd, voxel_size, feature_multiplier=4.0, max_nn=100):
    radius_feature = voxel_size * feature_multiplier
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn)
    )

def run_fgr(src_down, tgt_down, src_fpfh, tgt_fpfh, max_corr_dist):
    opt = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=max_corr_dist
    )
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, opt)
    return result

def run_gicp(src, tgt, init_T, max_corr_dist, max_iter=50):
    # Use Generalized ICP refinement
    result = o3d.pipelines.registration.registration_generalized_icp(
        src, tgt, max_corr_dist,
        init_T,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    return result

def compute_rmse_nn(source, target, transform):
    """
    Apply transform to source, compute RMSE to target via nearest neighbor (mm).
    Careful numeric computation as system guidelines advise.
    """
    s = copy.deepcopy(source)
    s.transform(transform)
    pts_s = np.asarray(s.points)
    pts_t = np.asarray(target.points)
    if pts_s.size == 0 or pts_t.size == 0:
        return float('inf')
    kdt = o3d.geometry.KDTreeFlann(target)
    d2 = []
    # compute squared distances one-by-one (digit-by-digit style)
    for p in pts_s:
        _, idx, dist2 = kdt.search_knn_vector_3d(p, 1)
        d2.append(dist2[0])
    d2 = np.array(d2, dtype=float)
    # rmse: sqrt(mean(d2))
    mean_d2 = float(np.mean(d2))
    rmse = float(np.sqrt(mean_d2))
    return rmse

# -------------------------
# Main pipeline for one param combo
# -------------------------
def evaluate_combo(src_orig, tgt_orig,
                   voxel_size,
                   nb_neighbors,
                   std_ratio,
                   radius_multiplier,
                   normal_radius_multiplier,
                   fpfh_multiplier,
                   voxel_fine,
                   debug_outdir=None):
    """
    All sizes in mm.
    radius_outlier = voxel_size * radius_multiplier
    """
    # copy inputs
    src = copy.deepcopy(src_orig)
    tgt = copy.deepcopy(tgt_orig)

    # 1) Downsample coarse
    src_down = voxel_downsample_copy(src, voxel_size)
    tgt_down = voxel_downsample_copy(tgt, voxel_size)

    # 2) Denoise radius + statistical on downsampled clouds
    radius_out = voxel_size * radius_multiplier
    src_den = denoise_radius_then_stat(src_down, radius=radius_out, min_points=8, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    tgt_den = denoise_radius_then_stat(tgt_down, radius=radius_out, min_points=8, nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # 3) Normals
    print("################ NORMALS ###########################")
    src_den = estimate_normals(src_den, voxel_size, radius_multiplier=normal_radius_multiplier, max_nn=50)
    tgt_den = estimate_normals(tgt_den, voxel_size, radius_multiplier=normal_radius_multiplier, max_nn=50)

    # 4) FPFH
    src_fpfh = compute_fpfh(src_den, voxel_size, feature_multiplier=fpfh_multiplier)
    tgt_fpfh = compute_fpfh(tgt_den, voxel_size, feature_multiplier=fpfh_multiplier)

    # 5) Fast Global Registration (initial)
    max_corr_fgr = voxel_size * 1.5
    fgr_res = run_fgr(src_den, tgt_den, src_fpfh, tgt_fpfh, max_corr_fgr)
    T_init = fgr_res.transformation

    # 6) Fine downsample for GICP
    src_fine = voxel_downsample_copy(src, voxel_fine)
    tgt_fine = voxel_downsample_copy(tgt, voxel_fine)
    src_fine = estimate_normals(src_fine, voxel_fine, radius_multiplier=normal_radius_multiplier, max_nn=50)
    tgt_fine = estimate_normals(tgt_fine, voxel_fine, radius_multiplier=normal_radius_multiplier, max_nn=50)

    max_corr_gicp = max(2.0 * voxel_fine, voxel_fine * 3.0)  # mm
    gicp_res = run_gicp(src_fine, tgt_fine, T_init, max_corr_gicp, max_iter=50)
    T_final = gicp_res.transformation

    # 7) Evaluate RMSE on original-scale clouds (not downsampled) to measure real residual
    rmse = compute_rmse_nn(src, tgt, T_final)

    # Optionally save debug clouds for best/worst analysis
    if debug_outdir is not None:
        os.makedirs(debug_outdir, exist_ok=True)
        # transformed source
        s_tr = copy.deepcopy(src)
        s_tr.transform(T_final)
        write_ply(os.path.join(debug_outdir, f"src_trans_voxel{int(voxel_size)}_rad{radius_multiplier}_std{std_ratio}_voxfine{int(voxel_fine)}.ply"), s_tr)
        write_ply(os.path.join(debug_outdir, f"tgt_voxel{int(voxel_size)}.ply"), tgt)

    return {
        "voxel_size": voxel_size,
        "nb_neighbors": nb_neighbors,
        "std_ratio": std_ratio,
        "radius_multiplier": radius_multiplier,
        "normal_radius_multiplier": normal_radius_multiplier,
        "fpfh_multiplier": fpfh_multiplier,
        "voxel_fine": voxel_fine,
        "fgr_fitness": float(getattr(fgr_res, "fitness", 0.0)),
        "gicp_fitness": float(getattr(gicp_res, "fitness", 0.0)),
        "rmse_mm": float(rmse)
    }

def load_coord(path):
    

    transform_coords = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    paths = [os.path.join(path, "pose_1.npy"), os.path.join(path, "pose_2.npy")]
    paths.sort()
    paths = sorted(paths, key=len)
    poses = []
    for pose_path in paths:
        poses.append(np.load(os.path.join(path, pose_path)))
    poses = np.array(poses)

    transforms = []
    for pose in poses:
        transform = pose_to_transform_matrix(pose)
        transforms.append(transform@transform_coords)


    return transforms

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


# -------------------------
# Sweep driver
# -------------------------
def run_sweep(src_ply, tgt_ply, out_csv="sweep_results.csv", debug_outdir="debug", max_cases=None):
    print("Loading clouds...")
    src = read_ply(src_ply)
    tgt = read_ply(tgt_ply)
    

    target_t, source_t = load_coord("C:/Users/adamf/Codes/Mocap_process/Roof_clouds")
    src.transform(source_t)
    tgt.transform(target_t)

    # -------------------------
    # Parameter grids (edit ranges here)
    # Units = mm for voxels
    # Keep grid small initially, e.g. 3x3x3 to avoid huge runtime
    voxel_sizes = [50.0, 20.0, 10.0]        # coarse downsample sizes to test (mm)
    nb_neighbors_list = [20, 30]           # for statistical outlier
    std_ratios = [0.8, 1.0, 1.5]           # statistical outlier std ratio
    radius_multipliers = [0.8, 1.2, 1.5]   # radius_outlier = voxel_size * multiplier
    normal_radius_multipliers = [1.0, 2.0] # multiplier for normal search radius
    fpfh_multipliers = [3.0, 4.0]          # multiplier for FPFH radius
    voxel_fine_list = [5.0, 10.0]          # final (fine) voxel size for GICP

    # generate grid
    grid = list(itertools.product(voxel_sizes, nb_neighbors_list, std_ratios,
                                  radius_multipliers, normal_radius_multipliers,
                                  fpfh_multipliers, voxel_fine_list))
    if max_cases is not None:
        grid = grid[:max_cases]
    print(f"Total parameter combinations to test: {len(grid)}")

    results = []
    # create debug dir for a few examples
    os.makedirs(debug_outdir, exist_ok=True)

    for combo in tqdm(grid):
        voxel_size, nb_neighbors, std_ratio, radius_mul, normal_rad_mul, fpfh_mul, voxel_fine = combo
        try:
            res = evaluate_combo(
                src, tgt,
                voxel_size=float(voxel_size),
                nb_neighbors=int(nb_neighbors),
                std_ratio=float(std_ratio),
                radius_multiplier=float(radius_mul),
                normal_radius_multiplier=float(normal_rad_mul),
                fpfh_multiplier=float(fpfh_mul),
                voxel_fine=float(voxel_fine),
                debug_outdir=None  # no per-case saves to keep disk small; set to debug_outdir for saving
            )
            results.append(res)
            # append to CSV progressively so progress is preserved
            write_header = not os.path.exists(out_csv)
            with open(out_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(res.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(res)
        except Exception as e:
            print(f"Combo {combo} failed: {e}")
            continue

    # sort by rmse ascending
    results_sorted = sorted(results, key=lambda r: r["rmse_mm"])
    if len(results_sorted) == 0:
        print("No successful runs.")
        return

    # Save top/bottom debug outputs for quick inspection
    best = results_sorted[0]
    worst = results_sorted[-1]
    print("Best combo:", best)
    print("Worst combo:", worst)

    # re-run best and worst with saving into debug dir
    _ = evaluate_combo(src, tgt,
                       voxel_size=best["voxel_size"],
                       nb_neighbors=best["nb_neighbors"],
                       std_ratio=best["std_ratio"],
                       radius_multiplier=best["radius_multiplier"],
                       normal_radius_multiplier=best["normal_radius_multiplier"],
                       fpfh_multiplier=best["fpfh_multiplier"],
                       voxel_fine=best["voxel_fine"],
                       debug_outdir=os.path.join(debug_outdir, "best"))
    _ = evaluate_combo(src, tgt,
                       voxel_size=worst["voxel_size"],
                       nb_neighbors=worst["nb_neighbors"],
                       std_ratio=worst["std_ratio"],
                       radius_multiplier=worst["radius_multiplier"],
                       normal_radius_multiplier=worst["normal_radius_multiplier"],
                       fpfh_multiplier=worst["fpfh_multiplier"],
                       voxel_fine=worst["voxel_fine"],
                       debug_outdir=os.path.join(debug_outdir, "worst"))

    print(f"Full results written to {out_csv}")
    print(f"Debug outputs saved to {debug_outdir}/best and {debug_outdir}/worst (transformed source + target)")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source PLY (already roughly aligned)")
    ap.add_argument("--tgt", required=True, help="target PLY (already roughly aligned)")
    ap.add_argument("--out", default="sweep_results.csv", help="output CSV")
    ap.add_argument("--debug", default="debug", help="folder for debug PLYs")
    ap.add_argument("--max_cases", default=None, type=int, help="limit number of combos (for speed)")
    args = ap.parse_args()

    run_sweep(args.src, args.tgt, out_csv=args.out, debug_outdir=args.debug, max_cases=args.max_cases)
