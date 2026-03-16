import roslibpy
import time
import numpy as np
from datetime import datetime
import os
import argparse


DEFAULT_ROOT_DIR = "/home/adamfi/Codes/Pointclouds/pointclouds/Multi_cloud_more_poses"


def parse_args():
    parser = argparse.ArgumentParser(description="Listen for ROS pose and save it to disk")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help=f"Directory where pose_*.npy is saved. If omitted, uses: {DEFAULT_ROOT_DIR}",
    )
    return parser.parse_args()

# Global variable to store latest pose
latest_pose = None
pose_received = False

def pose_callback(msg):
    """Process incoming pose message and extract position and orientation."""
    global latest_pose, pose_received
    
    # Extract position (x, y, z)
    pos_x = msg['pose']['position']['x']
    pos_y = msg['pose']['position']['y']
    pos_z = msg['pose']['position']['z']
    
    # Extract orientation (quaternion x, y, z, w)
    orient_x = msg['pose']['orientation']['x']
    orient_y = msg['pose']['orientation']['y']
    orient_z = msg['pose']['orientation']['z']
    orient_w = msg['pose']['orientation']['w']
    
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

def main():
    """Listens for poses published to the topic specified. Needs rosbridge running on the rosmaster with port 9090"""
    args = parse_args()
    try:
        # ROS connection setup
        print("Connecting to ROS bridge server...")
        client = roslibpy.Ros(host='192.168.125.186', port=9090)
        client.run()
        print("Connected to ROS bridge server")
        
        # Subscribe to the pose topic
        topic = roslibpy.Topic(client, '/natnet_ros/Camera2/pose', 'geometry_msgs/PoseStamped')
        topic.subscribe(pose_callback)
        
        print("Waiting for the first pose data...")
        
        # Wait for the first pose to be received
        while not pose_received:
            time.sleep(0.1)
        
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
        # Clean up ROS connection
        if 'topic' in locals():
            topic.unsubscribe()
        if 'client' in locals() and client.is_connected:
            client.terminate()
            print("Disconnected from ROS bridge server")

if __name__ == "__main__":
    main()