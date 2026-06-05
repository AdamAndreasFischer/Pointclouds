from pyorbbecsdk import *
import argparse
from collections import defaultdict
from pathlib import Path
import re
import sys
import time

import numpy as np
import open3d as o3d


POINTCLOUDS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ROOT_DIR = POINTCLOUDS_DIR / "pointclouds" / "multi_cam_capture"
DEFAULT_CAMERA_DATA = POINTCLOUDS_DIR / "utils" / "camera_data.yml"
DEFAULT_CAMERA_IDS = "1,2,3,4"


def parse_args():
    parser = argparse.ArgumentParser(description="Capture point clouds from multiple Orbbec cameras.")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=str(DEFAULT_ROOT_DIR),
        help="Directory where Cloud_pose*_camera_* folders are saved.",
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
        "--serials",
        type=str,
        default="",
        help="Optional serial overrides, e.g. 1:SERIAL_A,2:SERIAL_B.",
    )
    parser.add_argument(
        "--device-indexes",
        type=str,
        default="",
        help="Optional 0-based device index overrides, e.g. 1:0,2:1,3:2,4:3.",
    )
    parser.add_argument(
        "--capture-index",
        type=int,
        default=None,
        help="Capture index for Cloud_poseXXXX_camera_N folders. If omitted, the next index is used.",
    )
    parser.add_argument(
        "--num_pcds",
        type=int,
        default=5,
        help="Number of point clouds to capture per camera.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=600000,
        help="Minimum number of points required before a cloud is accepted.",
    )
    parser.add_argument(
        "--frame-timeout-ms",
        type=int,
        default=100,
        help="Timeout for waiting on each camera frame.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help="Number of initial frames to skip per camera.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into existing Cloud_pose*_camera_* folders.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show captured clouds with Open3D after capture.",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Only validate camera_data.yml and serial mappings, then exit.",
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


def parse_str_mapping(value):
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
        mapping[int(camera_id.strip())] = mapped_value.strip()
    return mapping


def parse_int_mapping(value):
    return {camera_id: int(mapped_value) for camera_id, mapped_value in parse_str_mapping(value).items()}


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
        cameras[camera_id] = {"serial": extract_scalar(block, "serial")}
    return cameras


def resolve_serials(cameras, camera_ids, serial_overrides, device_index_overrides):
    serials = {}
    for camera_id in camera_ids:
        if camera_id not in cameras:
            raise ValueError(f"camera_{camera_id} is missing from camera_data.yml.")
        serial = serial_overrides.get(camera_id, cameras[camera_id].get("serial"))
        if is_missing(serial) and camera_id not in device_index_overrides:
            raise ValueError(
                f"camera_{camera_id} has no serial. Fill camera_data.yml, pass --serials, "
                "or pass --device-indexes."
            )
        serials[camera_id] = None if is_missing(serial) else str(serial)

    by_serial = defaultdict(list)
    for camera_id, serial in serials.items():
        if serial is not None:
            by_serial[serial].append(camera_id)

    for serial, duplicate_camera_ids in by_serial.items():
        if len(duplicate_camera_ids) <= 1:
            continue
        missing_index = [camera_id for camera_id in duplicate_camera_ids if camera_id not in device_index_overrides]
        if missing_index:
            raise ValueError(
                f"Duplicate serial '{serial}' is configured for cameras {duplicate_camera_ids}. "
                "Use --device-indexes for those cameras or fix camera_data.yml."
            )

    return serials


def available_devices(device_list):
    devices = []
    for index in range(device_list.get_count()):
        try:
            serial = device_list.get_device_serial_number_by_index(index)
        except Exception:
            serial = device_list.get_device_by_index(index).get_device_info().get_serial_number()
        devices.append((index, serial))
    return devices


def resolve_devices(camera_ids, serials, device_index_overrides):
    ctx = Context()
    device_list = ctx.query_devices()
    count = device_list.get_count()
    if count == 0:
        raise RuntimeError("No Orbbec devices are connected.")

    devices = available_devices(device_list)
    print("Connected Orbbec devices:")
    for index, serial in devices:
        print(f"  index {index}: {serial}")

    serial_to_index = {}
    for index, serial in devices:
        serial_to_index.setdefault(serial, []).append(index)

    camera_devices = {}
    used_indexes = {}
    for camera_id in camera_ids:
        if camera_id in device_index_overrides:
            index = device_index_overrides[camera_id]
            if index < 0 or index >= count:
                raise ValueError(f"camera_{camera_id} device index {index} is out of range.")
        else:
            serial = serials[camera_id]
            matches = serial_to_index.get(serial, [])
            if not matches:
                raise ValueError(
                    f"Could not find a connected Orbbec device with serial '{serial}' "
                    f"for camera_{camera_id}."
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Serial '{serial}' matched multiple connected devices. Use --device-indexes."
                )
            index = matches[0]

        if index in used_indexes:
            raise ValueError(
                f"camera_{camera_id} and camera_{used_indexes[index]} both map to device index {index}."
            )
        used_indexes[index] = camera_id
        camera_devices[camera_id] = device_list.get_device_by_index(index)

        expected_serial = serials.get(camera_id)
        actual_serial = devices[index][1]
        if expected_serial is not None and expected_serial != actual_serial:
            print(
                f"Warning: camera_{camera_id} uses device index {index} with serial {actual_serial}, "
                f"but camera_data.yml says {expected_serial}."
            )

    return camera_devices


def next_capture_index(root_dir):
    root = Path(root_dir)
    if not root.exists():
        return 1

    max_index = 0
    pattern = re.compile(r"^Cloud_pose0*(\d+)_camera_\d+$")
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def configure_pipeline(device):
    pipeline = Pipeline(device)
    config = Config()

    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        color_profile = profile_list.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
    except Exception:
        color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    try:
        depth_profile = profile_list.get_video_stream_profile(0, 0, OBFormat.Y16, 0)
    except Exception:
        depth_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    try:
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    except Exception as exc:
        print(f"Warning: could not set frame aggregate mode: {exc}")

    return pipeline, config


def point_cloud_from_points(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    colors = points[:, 3:6]
    if colors.size and np.max(colors) > 1.0:
        colors = colors / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def create_capture_states(root_dir, capture_index, camera_ids, camera_devices, overwrite):
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    states = []
    for camera_id in camera_ids:
        group_dir = root / f"Cloud_pose{capture_index:04d}_camera_{camera_id}"
        existing_clouds = list(group_dir.glob("*.ply")) if group_dir.exists() else []
        if existing_clouds and not overwrite:
            raise FileExistsError(
                f"{group_dir} already contains PLY files. Use a new capture index or --overwrite."
            )
        group_dir.mkdir(parents=True, exist_ok=True)

        pipeline, config = configure_pipeline(camera_devices[camera_id])
        point_cloud_filter = PointCloudFilter()
        point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)

        edge_noise_filter = NoiseRemovalFilter()
        try:
            edge_noise_filter.enable(True)
        except Exception:
            pass

        states.append(
            {
                "camera_id": camera_id,
                "group_dir": group_dir,
                "pipeline": pipeline,
                "config": config,
                "align_filter": AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM),
                "edge_noise_filter": edge_noise_filter,
                "point_cloud_filter": point_cloud_filter,
                "saved": 0,
                "warmup_remaining": 0,
            }
        )
    return states


def start_pipelines(states, warmup_frames):
    for state in states:
        camera_id = state["camera_id"]
        pipeline = state["pipeline"]
        print(f"Starting camera_{camera_id}")
        pipeline.enable_frame_sync()
        pipeline.start(state["config"])
        state["warmup_remaining"] = warmup_frames


def stop_pipelines(states):
    for state in states:
        try:
            state["pipeline"].stop()
        except Exception as exc:
            print(f"Warning: failed to stop camera_{state['camera_id']}: {exc}")


def capture_clouds(states, num_pcds, min_points, frame_timeout_ms, preview):
    preview_clouds = []
    last_status = time.monotonic()

    while any(state["saved"] < num_pcds for state in states):
        for state in states:
            if state["saved"] >= num_pcds:
                continue

            camera_id = state["camera_id"]
            frames = state["pipeline"].wait_for_frames(frame_timeout_ms)
            if frames is None:
                continue

            if state["warmup_remaining"] > 0:
                state["warmup_remaining"] -= 1
                continue

            align_frame = state["align_filter"].process(frames)
            if not align_frame:
                continue

            noise_removed = state["edge_noise_filter"].process(align_frame)
            point_cloud_frame = state["point_cloud_filter"].process(noise_removed)

            try:
                points = state["point_cloud_filter"].calculate(point_cloud_frame)
            except RuntimeWarning as warning:
                print(f"camera_{camera_id}: point cloud calculation warning: {warning}")
                continue

            if points.shape[0] < min_points:
                print(
                    f"camera_{camera_id}: rejected cloud with {points.shape[0]} points "
                    f"(minimum {min_points})"
                )
                continue

            pcd = point_cloud_from_points(points)
            next_cloud_number = state["saved"] + 1
            filename = state["group_dir"] / f"cloud{next_cloud_number}.ply"
            o3d.io.write_point_cloud(str(filename), pcd, write_ascii=True)
            state["saved"] = next_cloud_number
            print(f"camera_{camera_id}: saved {filename}")

            if preview:
                preview_clouds.append(pcd)

        if time.monotonic() - last_status > 5.0:
            status = ", ".join(
                f"camera_{state['camera_id']} {state['saved']}/{num_pcds}" for state in states
            )
            print(f"Capture status: {status}")
            last_status = time.monotonic()

    if preview and preview_clouds:
        o3d.visualization.draw_geometries(preview_clouds)


def main():
    args = parse_args()
    camera_ids = parse_camera_ids(args.camera_ids)
    serial_overrides = parse_str_mapping(args.serials)
    device_index_overrides = parse_int_mapping(args.device_indexes)
    cameras = load_camera_data(args.camera_data)
    serials = resolve_serials(cameras, camera_ids, serial_overrides, device_index_overrides)

    print("Camera serial mapping:")
    for camera_id in camera_ids:
        serial_text = serials[camera_id] if serials[camera_id] is not None else "<device index override>"
        print(f"  camera_{camera_id}: {serial_text}")

    if args.validate_config:
        return 0

    capture_index = args.capture_index or next_capture_index(args.root_dir)
    camera_devices = resolve_devices(camera_ids, serials, device_index_overrides)
    states = create_capture_states(
        args.root_dir,
        capture_index,
        camera_ids,
        camera_devices,
        args.overwrite,
    )

    try:
        start_pipelines(states, args.warmup_frames)
        capture_clouds(states, args.num_pcds, args.min_points, args.frame_timeout_ms, args.preview)
    finally:
        stop_pipelines(states)

    print(f"Captured {args.num_pcds} point clouds per camera for capture {capture_index}.")
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
