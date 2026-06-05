from natnet import DataFrame, NatNetClient
from natnet.data_frame import RigidBody
import argparse
import os
from pathlib import Path
import re
import struct
import sys
import time

import numpy as np


POINTCLOUDS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ROOT_DIR = POINTCLOUDS_DIR / "pointclouds" / "multi_cam_capture"
DEFAULT_CAMERA_DATA = POINTCLOUDS_DIR / "utils" / "camera_data.yml"
DEFAULT_CAMERA_IDS = "1,2,3,4"

TARGET_RB_TO_CAM = {}
LATEST_POSES = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Listen for NatNet poses for multiple camera rigid bodies and save one pose per camera."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=str(DEFAULT_ROOT_DIR),
        help="Directory where pose_capture*_camera_*.npy files are saved.",
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
        help="Comma-separated camera IDs to capture, e.g. 1,2,3,4.",
    )
    parser.add_argument(
        "--rigid-body-ids",
        type=str,
        default="",
        help="Optional comma-separated overrides, e.g. 1:4,2:5,3:6,4:7.",
    )
    parser.add_argument(
        "--capture-index",
        type=int,
        default=None,
        help="Capture index to save. If omitted, the next pose_capture index is used.",
    )
    parser.add_argument(
        "--server-ip",
        type=str,
        default="192.168.1.155",
        help="NatNet/Motive server IP.",
    )
    parser.add_argument(
        "--client-ip",
        type=str,
        default="192.168.1.234",
        help="Local client IP.",
    )
    parser.add_argument(
        "--use-multicast",
        action="store_true",
        help="Enable multicast mode. Must match the Motive streaming setting.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=0.0,
        help="Seconds to wait for all poses. Use 0 to wait forever.",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Only validate camera_data.yml and rigid body mappings, then exit.",
    )
    return parser.parse_args()


def parse_camera_ids(value):
    camera_ids = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        camera_ids.append(int(part))
    if not camera_ids:
        raise ValueError("At least one camera ID is required.")
    return camera_ids


def parse_int_mapping(value):
    mapping = {}
    if not value:
        return mapping
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid mapping '{part}'. Expected camera:value.")
        camera_id, mapped_value = part.split(":", 1)
        mapping[int(camera_id.strip())] = int(mapped_value.strip())
    return mapping


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
        }
    return cameras


def resolve_rigid_body_ids(cameras, camera_ids, overrides):
    rb_by_camera = {}
    for camera_id in camera_ids:
        if camera_id not in cameras:
            raise ValueError(f"camera_{camera_id} is missing from camera_data.yml.")

        rb_id = overrides.get(camera_id, cameras[camera_id].get("rigid_body_id"))
        if is_missing(rb_id):
            raise ValueError(
                f"camera_{camera_id} has no rigid_body_id. Fill camera_data.yml "
                "or pass --rigid-body-ids 1:<id>,2:<id>,..."
            )
        rb_by_camera[camera_id] = int(rb_id)

    seen = {}
    for camera_id, rb_id in rb_by_camera.items():
        if rb_id in seen:
            raise ValueError(
                f"camera_{camera_id} and camera_{seen[rb_id]} both use rigid_body_id {rb_id}."
            )
        seen[rb_id] = camera_id
    return rb_by_camera


def _extract_rb_id(rb: RigidBody):
    for key in ("id_num", "id", "id_", "rigid_body_id"):
        if hasattr(rb, key):
            return int(getattr(rb, key))
    return None


def pose_callback(msg):
    if not isinstance(msg, DataFrame):
        return None

    rigid_bodies = getattr(msg, "rigid_bodies", None)
    if not rigid_bodies:
        return None

    for rb in rigid_bodies:
        rb_id = _extract_rb_id(rb)
        camera_id = TARGET_RB_TO_CAM.get(rb_id)
        if camera_id is None:
            continue

        pos_x, pos_y, pos_z = rb.pos
        orient_x, orient_y, orient_z, orient_w = rb.rot
        LATEST_POSES[camera_id] = np.array(
            [pos_x, pos_y, pos_z, orient_x, orient_y, orient_z, orient_w],
            dtype=np.float32,
        )
    return None


def next_capture_index(root_dir):
    root = Path(root_dir)
    if not root.exists():
        return 1

    max_index = 0
    pattern = re.compile(r"^pose_capture0*(\d+)_camera_\d+\.npy$")
    for entry in root.iterdir():
        match = pattern.match(entry.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def save_poses(root_dir, capture_index, camera_ids):
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    for camera_id in camera_ids:
        filename = root / f"pose_capture{capture_index:04d}_camera_{camera_id}.npy"
        np.save(filename, LATEST_POSES[camera_id])
        print(f"Saved camera {camera_id} pose to {filename}")


def main():
    args = parse_args()
    camera_ids = parse_camera_ids(args.camera_ids)
    rb_overrides = parse_int_mapping(args.rigid_body_ids)
    cameras = load_camera_data(args.camera_data)
    rb_by_camera = resolve_rigid_body_ids(cameras, camera_ids, rb_overrides)

    print("Camera rigid body mapping:")
    for camera_id in camera_ids:
        print(f"  camera_{camera_id}: rigid body {rb_by_camera[camera_id]}")

    if args.validate_config:
        return 0

    capture_index = args.capture_index or next_capture_index(args.root_dir)

    global TARGET_RB_TO_CAM
    TARGET_RB_TO_CAM = {rb_id: camera_id for camera_id, rb_id in rb_by_camera.items()}
    LATEST_POSES.clear()

    print("Connecting to NatNet server...")
    client = NatNetClient(
        server_ip_address=args.server_ip,
        local_ip_address=args.client_ip,
        use_multicast=args.use_multicast,
    )
    client.on_data_frame_received_event.handlers.append(pose_callback)

    print(
        "Waiting for poses from cameras: "
        + ", ".join(str(camera_id) for camera_id in camera_ids)
    )
    start_time = time.monotonic()
    with client:
        while len(LATEST_POSES) < len(camera_ids):
            try:
                client.update_sync()
            except (BlockingIOError, struct.error):
                time.sleep(0.01)
                continue

            missing = [camera_id for camera_id in camera_ids if camera_id not in LATEST_POSES]
            if args.timeout_sec > 0 and time.monotonic() - start_time > args.timeout_sec:
                raise TimeoutError(
                    "Timed out waiting for NatNet poses from cameras: "
                    + ", ".join(str(camera_id) for camera_id in missing)
                )
            time.sleep(0.01)

    save_poses(args.root_dir, capture_index, camera_ids)
    print(f"Saved all NatNet poses for capture {capture_index}.")
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
