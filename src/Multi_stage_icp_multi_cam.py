import argparse
import ast
from pathlib import Path
import re
import sys

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


POINTCLOUDS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ROOT_DIR = POINTCLOUDS_DIR / "pointclouds" / "multi_cam_capture"
DEFAULT_CAMERA_DATA = POINTCLOUDS_DIR / "utils" / "camera_data.yml"
DEFAULT_CALIBRATION_DIR = POINTCLOUDS_DIR / "Orbbec_calibrations_mocaplab"
DEFAULT_CAMERA_IDS = "1,2,3,4"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calibration-only multi-camera alignment. This keeps the Multi_stage_icp name "
            "for compatibility, but intentionally skips ICP."
        )
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=str(DEFAULT_ROOT_DIR),
        help="Root directory containing pose_capture*_camera_*.npy and Cloud_pose*_camera_* folders.",
    )
    parser.add_argument(
        "--camera-data",
        type=str,
        default=str(DEFAULT_CAMERA_DATA),
        help="Path to camera_data.yml.",
    )
    parser.add_argument(
        "--camera-ids",
        type=str,
        default=DEFAULT_CAMERA_IDS,
        help="Comma-separated camera IDs to transform, e.g. 1,2,3,4.",
    )
    parser.add_argument(
        "--num-pcds",
        type=int,
        default=None,
        help="Expected point clouds per camera pose. If omitted, all PLYs in each folder are used.",
    )
    parser.add_argument(
        "--pose_prefix",
        type=str,
        default="pose",
        help="Prefix for compatibility pose_N_init_transformed.npy outputs.",
    )
    parser.add_argument(
        "--final_cloud_name",
        type=str,
        default="initial_multi_cam_cloud.ply",
        help="Preview merged cloud filename used with --write-preview-cloud.",
    )
    parser.add_argument(
        "--write-preview-cloud",
        action="store_true",
        help="Write a merged preview PLY using the first cloud in each camera group.",
    )
    parser.add_argument(
        "--no-strict-cloud-layout",
        action="store_true",
        help="Allow extra PLY files outside Cloud_pose*_camera_* folders.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=str(DEFAULT_CALIBRATION_DIR),
        help="Directory containing orbbec{N}.npy calibration files (one per camera ID).",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Only validate calibration files, then exit.",
    )
    return parser.parse_args()


def parse_camera_ids(value):
    camera_ids = []
    for part in value.split(","):
        part = part.strip()
        if part:
            camera_ids.append(int(part))
    if not camera_ids:
        raise ValueError("At least one camera ID is required.")
    return camera_ids


def is_missing(value):
    if value is None:
        return True
    text = str(value).strip().strip("\"'")
    return text == "" or text.upper() in {"TBD", "NONE", "NULL"}


def extract_scalar(block, name):
    match = re.search(rf"[\"']?{re.escape(name)}[\"']?\s*:\s*([^,\n}}]+)", block)
    if not match:
        return None
    return match.group(1).strip().strip("\"'")


def extract_matrix(block, name):
    match = re.search(rf"[\"']?{re.escape(name)}[\"']?\s*:", block)
    if not match:
        return None

    tail = block[match.end() :]
    if re.match(r"\s*TBD\b", tail, flags=re.IGNORECASE):
        return None

    start = tail.find("[")
    if start < 0:
        return None

    depth = 0
    end = None
    for index, char in enumerate(tail[start:], start=start):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end = index
                break

    if end is None:
        raise ValueError(f"Could not parse {name} matrix.")

    matrix_literal = tail[start : end + 1]
    matrix = np.asarray(ast.literal_eval(matrix_literal), dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got shape {matrix.shape}.")
    return matrix


def load_camera_data(path):
    text = Path(path).read_text(encoding="utf-8")
    cameras = {}
    pattern = re.compile(r"(?ms)^camera_(\d+)\s*:\s*\{(.*?)(?=^camera_\d+\s*:|\Z)")
    for match in pattern.finditer(text):
        camera_id = int(match.group(1))
        block = match.group(2)
        cameras[camera_id] = {
            "serial": extract_scalar(block, "serial"),
            "rigid_body_id": extract_scalar(block, "rigid_body_id"),
            "calibration": extract_matrix(block, "calibration"),
        }
    return cameras


def resolve_calibrations(camera_ids, calibration_dir):
    calibrations = {}
    for camera_id in camera_ids:
        path = Path(calibration_dir) / f"orbbec{camera_id}_RL.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Calibration file not found for camera_{camera_id}: {path}"
            )
        matrix = np.load(str(path))
        if matrix.shape != (4, 4):
            raise ValueError(
                f"Expected 4x4 matrix in {path.name}, got shape {matrix.shape}."
            )
        calibrations[camera_id] = matrix
    return calibrations


def natural_key(value):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value))]


def discover_capture_groups(root_dir, camera_ids, expected_num_pcds):
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"root_dir does not exist: {root}")

    camera_ids_set = set(camera_ids)
    pattern = re.compile(r"^Cloud_pose0*(\d+)_camera_0*(\d+)$")
    groups = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue
        capture_index = int(match.group(1))
        camera_id = int(match.group(2))
        if camera_id not in camera_ids_set:
            continue

        cloud_paths = sorted(child.glob("*.ply"), key=natural_key)
        if not cloud_paths:
            raise ValueError(f"{child} does not contain any PLY point clouds.")
        if expected_num_pcds is not None and len(cloud_paths) != expected_num_pcds:
            raise ValueError(
                f"{child} contains {len(cloud_paths)} PLY files, expected {expected_num_pcds}."
            )

        groups.append(
            {
                "capture_index": capture_index,
                "camera_id": camera_id,
                "cloud_dir": child,
                "cloud_paths": cloud_paths,
            }
        )

    groups.sort(key=lambda group: (group["capture_index"], group["camera_id"]))
    if not groups:
        raise ValueError("No Cloud_pose*_camera_* folders were found.")

    for camera_id in camera_ids:
        if not any(g["camera_id"] == camera_id for g in groups):
            raise ValueError(f"No captures found for camera_{camera_id}.")

    return groups


def reject_extra_plys(root_dir, groups):
    allowed_dirs = {group["cloud_dir"].resolve() for group in groups}
    extra = []
    for cloud_path in Path(root_dir).rglob("*.ply"):
        if cloud_path.parent.resolve() not in allowed_dirs:
            extra.append(cloud_path)
    if extra:
        example = "\n  ".join(str(path) for path in extra[:5])
        raise ValueError(
            "Found PLY files outside Cloud_pose*_camera_* folders. UFO will convert every "
            f"recursive PLY, so remove/move them or pass --no-strict-cloud-layout.\n  {example}"
        )


def raw_pose_path(root_dir, capture_index, camera_id):
    root = Path(root_dir)
    padded = root / f"pose_capture{capture_index:04d}_camera_{camera_id}.npy"
    if padded.exists():
        return padded
    unpadded = root / f"pose_capture{capture_index}_camera_{camera_id}.npy"
    if unpadded.exists():
        return unpadded
    raise FileNotFoundError(
        f"Missing NatNet pose for capture {capture_index}, camera {camera_id}: {padded}"
    )


def pose_to_matrix(pose, millimeters=True):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"Expected pose shape (7,), got {pose.shape}.")

    translation = pose[:3]
    quaternion = pose[3:]
    q_norm = np.linalg.norm(quaternion)
    if q_norm < 1e-12:
        raise ValueError("Pose quaternion has near-zero norm.")
    quaternion = quaternion / q_norm

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_quat(quaternion).as_matrix()
    transform[:3, 3] = translation * 1000.0 if millimeters else translation
    return transform


def matrix_to_pose(transform):
    translation = transform[:3, 3]
    quaternion = R.from_matrix(transform[:3, :3]).as_quat()
    return np.concatenate([translation, quaternion])


def calibrated_camera_pose(raw_pose, calibration):
    mocap_to_rigid_body = pose_to_matrix(raw_pose, millimeters=True)
    rigid_body_to_camera = np.array(calibration, dtype=np.float64, copy=True)
    rigid_body_to_camera[:3, 3] *= 1000.0
    return mocap_to_rigid_body @ np.linalg.inv(rigid_body_to_camera)


def pose_to_ufo_row(pose_scalar_last_mm):
    row = np.array(pose_scalar_last_mm, dtype=np.float64, copy=True)
    row[:3] /= 1000.0
    qx, qy, qz, qw = row[3:7]
    return [row[0], row[1], row[2], qw, qx, qy, qz]


def write_poses_tsv(path, rows):
    with Path(path).open("w", encoding="utf-8") as pose_file:
        for index, row in enumerate(rows):
            if index:
                pose_file.write("\n")
            pose_file.write("\t".join(f"{value:.8f}" for value in row))


def write_preview_cloud(root_dir, groups, sequence_poses, final_cloud_name):
    merged = o3d.geometry.PointCloud()
    for group, transform in zip(groups, sequence_poses):
        pcd = o3d.io.read_point_cloud(str(group["cloud_paths"][0]))
        pcd.transform(transform)
        merged += pcd

    output_path = Path(root_dir) / final_cloud_name
    o3d.io.write_point_cloud(str(output_path), merged, write_ascii=True)
    print(f"Saved merged preview cloud to: {output_path}")


def main():
    args = parse_args()
    camera_ids = parse_camera_ids(args.camera_ids)
    calibrations = resolve_calibrations(camera_ids, args.calibration_dir)

    print("Camera calibration mapping:")
    for camera_id in camera_ids:
        print(f"  camera_{camera_id}: calibration loaded")

    if args.validate_config:
        return 0

    groups = discover_capture_groups(args.root_dir, camera_ids, args.num_pcds)
    if not args.no_strict_cloud_layout:
        reject_extra_plys(args.root_dir, groups)

    pose_rows = []
    sequence_transforms = []
    for sequence_index, group in enumerate(groups, start=1):
        capture_index = group["capture_index"]
        camera_id = group["camera_id"]
        raw_pose_file = raw_pose_path(args.root_dir, capture_index, camera_id)
        raw_pose = np.load(raw_pose_file)
        camera_pose = calibrated_camera_pose(raw_pose, calibrations[camera_id])
        pose = matrix_to_pose(camera_pose)

        compatibility_pose_path = Path(args.root_dir) / f"{args.pose_prefix}_{sequence_index}_init_transformed.npy"
        np.save(compatibility_pose_path, pose)
        np.save(Path(args.root_dir) / f"{args.pose_prefix}_{sequence_index}_transformed.npy", pose)
        np.save(
            Path(args.root_dir) / f"pose_capture{capture_index:04d}_camera_{camera_id}_init_transformed.npy",
            pose,
        )

        sequence_transforms.append(camera_pose)
        pose_rows.extend([pose_to_ufo_row(pose)] * len(group["cloud_paths"]))
        print(
            f"Sequence {sequence_index}: capture {capture_index}, camera_{camera_id}, "
            f"{len(group['cloud_paths'])} clouds -> {compatibility_pose_path.name}"
        )

    poses_tsv_path = Path(args.root_dir) / "poses.tsv"
    write_poses_tsv(poses_tsv_path, pose_rows)
    print(f"Saved {len(pose_rows)} UFO pose rows to: {poses_tsv_path}")

    if args.write_preview_cloud:
        write_preview_cloud(args.root_dir, groups, sequence_transforms, args.final_cloud_name)

    print("Skipped ICP: calibrated NatNet poses are used directly.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
