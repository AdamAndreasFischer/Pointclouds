from natnet import DataFrame, NatNetClient
from natnet.data_frame import RigidBody
import time
import numpy as np
from datetime import datetime
import os
import argparse
import struct


DEFAULT_ROOT_DIR = "/home/adamfi/codes/Pointclouds/pointclouds/new_calib_test"


def parse_args():
    parser = argparse.ArgumentParser(description="Listen for NatNet rigid body pose and save it to disk")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help=f"Directory where pose_*.npy is saved. If omitted, uses: {DEFAULT_ROOT_DIR}",
    )
    parser.add_argument(
        "--server-ip",
        type=str,
        default="192.168.1.155",
        help="NatNet/Motive server IP",
    )
    parser.add_argument(
        "--client-ip",
        type=str,
        default="192.168.1.234",
        help="Local client IP",
    )
    parser.add_argument(
        "--rigid-body-id",
        type=int,
        default=5,
        help="Optional rigid body ID to listen to (if omitted, first rigid body is used)",
    )
    parser.add_argument(
        "--use-multicast",
        action="store_true",
        help="Enable multicast mode (must match Motive streaming setting)",
    )
    return parser.parse_args()

# Global variable to store latest pose
latest_pose = None
pose_received = False

def _extract_rb_id(rb: RigidBody):
    for key in ("id_num", "id", "id_", "rigid_body_id"):
        if hasattr(rb, key):
            return int(getattr(rb, key))
    return None


def pose_callback(msg):
    """Process incoming NatNet data frame and extract position and orientation."""
    global latest_pose, pose_received

    if not isinstance(msg, DataFrame):
        return None

    rigid_bodies = getattr(msg, "rigid_bodies", None)
    if not rigid_bodies:
        return None

    if TARGET_RIGID_BODY_ID is None:
        rb = rigid_bodies[0]
    else:
        rb = None
        for candidate in rigid_bodies:
            if _extract_rb_id(candidate) == TARGET_RIGID_BODY_ID:
                rb = candidate
                break
        if rb is None:
            return None

    # Extract position (x, y, z)
    pos_x, pos_y, pos_z = rb.pos

    # Extract orientation (quaternion x, y, z, w)
    orient_x, orient_y, orient_z, orient_w = rb.rot

    # Create the combined pose list [x y z x y z w]
    latest_pose = np.array([pos_x, pos_y, pos_z, orient_x, orient_y, orient_z, orient_w], dtype=np.float32)

    #print(f"Received pose: {latest_pose}")
    pose_received = True
    return latest_pose

def save_pose_to_file(pose, filename):
    """Save pose data to a file."""
    # Make sure the directory exists
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Warning: Could not create directory {directory}: {e}")
    
    try:
        # Save the file
        np.save(filename, pose)
        print(f"Pose saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        
        # Try saving to current directory as fallback
        fallback_filename = f"pose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
        print(f"Attempting to save to current directory: {fallback_filename}")
        try:
            np.save(fallback_filename, pose)
            print(f"Pose saved to {fallback_filename}")
            return True
        except Exception as fallback_error:
            print(f"Fallback save failed: {fallback_error}")
            return False

TARGET_RIGID_BODY_ID = None


def main():
    """Listens for NatNet rigid body pose and saves the first received pose."""
    args = parse_args()
    global TARGET_RIGID_BODY_ID
    TARGET_RIGID_BODY_ID = args.rigid_body_id

    try:
        print("Connecting to NatNet server...")
        client = NatNetClient(
            server_ip_address=args.server_ip,
            local_ip_address=args.client_ip,
            use_multicast=args.use_multicast,
        )
        client.on_data_frame_received_event.handlers.append(pose_callback)

        print("Waiting for the first pose data...")

        with client:
            # Wait for the first pose to be received
            while not pose_received:
                try:
                    client.update_sync()
                except (BlockingIOError, struct.error):
                    # Ignore transient UDP / packet decode issues.
                    time.sleep(0.01)
                    continue
                time.sleep(0.01)
        
        # Save the pose automatically
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Resolve output directory (CLI arg overrides hardcoded default)
        dir_path = args.root_dir if args.root_dir else DEFAULT_ROOT_DIR

        files = os.listdir(dir_path)
        # Extract existing pose numbers from filenames
        existing_poses = []
        for file in files:
            if file.startswith("pose_") and file.endswith(".npy"):
                try:
                    # Extract number between "pose_" and ".npy"
                    number_str = file[len("pose_"):-4]  # Remove prefix and suffix
                    pose_num = int(number_str)
                    existing_poses.append(pose_num)
                except ValueError:
                    # Skip files that don't have valid numbers
                    continue
        
        # Find the next available pose number
        if existing_poses:
            n_files = max(existing_poses) + 1
        else:
            n_files = 1
        
        print(f"Existing poses: {sorted(existing_poses)}")
        print(f"Next pose number: {n_files}")
        

        filename = os.path.join(dir_path, f"pose_{n_files}.npy")
        
        # If saving fails, it will try saving to the current directory as a fallback
        save_successful = save_pose_to_file(latest_pose, filename)
        
        if save_successful:
            print("First pose saved. Exiting...")
        else:
            print("Could not save pose to any location.")
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # NatNet client is cleaned up by context manager.
        pass

if __name__ == "__main__":
    main()