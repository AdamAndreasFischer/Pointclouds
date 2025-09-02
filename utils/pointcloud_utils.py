import open3d as o3d
from open3d.visualization import draw_geometries
import numpy as np
import os


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
    transform[:3, 3] = translation*1000
    
    return transform

def re_save_pcd():
    path = "/home/adamfi/Codes/Pointclouds/pointclouds/full_room"
    clouds = read_clouds(path)
    for i, cloud in enumerate(clouds):
        o3d.io.write_point_cloud(os.path.join(path, f"Cloud_pose{i+1}.ply"), cloud, write_ascii=True)

def main():
    path = "/home/adamfi/Codes/Pointclouds/utils"

    poses = load_coord(path)

    with open("poses.txt", "w") as f:
        for i,pose in enumerate(poses):
            f.write(f"Corner {i}: " + " ".join(map(str, pose[:3])) + "\n")




if __name__ == "__main__":
    re_save_pcd()