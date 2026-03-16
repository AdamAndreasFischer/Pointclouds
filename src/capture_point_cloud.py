from pyorbbecsdk import * 
import os
import open3d as o3d
import copy
import numpy
import argparse


DEFAULT_ROOT_DIR = "/home/adamfi/Codes/Pointclouds/pointclouds/room_final2"

save_points_dir = os.path.join(os.getcwd(), "point_clouds")
if not os.path.exists(save_points_dir):
    os.mkdir(save_points_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Capture and save Orbbec point clouds")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help=f"Directory where Cloud_pose* folders are saved. If omitted, uses: {DEFAULT_ROOT_DIR}",
    )
    parser.add_argument("--num_pcds", 
                        type=int, 
                        default=5,
                        help=f"Number of pcd to capture from the pose. Default {5}")
    return parser.parse_args()


def save_point_cloud_to_ply(filename, point_cloud_frame, point_cloud_filter):
    try:
        assert point_cloud_frame.get_format() == OBFormat.RGB_POINT
    except Exception as e:
        print("Pointcloud is not xyzrgb")
        return False, None
    
    points = point_cloud_filter.calculate(point_cloud_frame)
    

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    colors = points[:, 3:6]
    if numpy.max(colors) > 1.0:
        colors = colors / 255.0

    
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
    return True, pcd


def main(dir_path, n_clouds, n_images, min_points):
    # 1.Create a pipeline with default device.
    pipeline = Pipeline()
    # 2.Create config.
    config = Config()

    device = pipeline.get_device()
    depth_sensor = device.get_sensor(OBSensorType.DEPTH_SENSOR)

    filter_list = depth_sensor.get_recommended_filters()

  
        
    # 3.Enable color profile
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
    config.enable_stream(color_profile)

    # 4.Enable depth profile
    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_video_stream_profile(0, 0, OBFormat.Y16, 0)

    config.enable_stream(depth_profile)

    # 5.Set the frame aggregate output mode to ensure all types of frames are included in the output frameset
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

    # 6.Start the stream
    pipeline.enable_frame_sync()
    pipeline.start(config)

    # 7.Create point cloud filter
    point_cloud_filter = PointCloudFilter()

    

    # 8.Create a filter to align depth frame to color frame
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    edge_noise_filter = NoiseRemovalFilter()
    edge_noise_filter.enable(True)
   

    point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)
    print("Capture pointcloud")
    i=1
    if not os.path.exists(os.path.join(dir_path, f"Cloud_pose{n_clouds}")):
        os.mkdir(os.path.join(dir_path, f"Cloud_pose{n_clouds}"))
    pcds = []
    while True:
        # 9.Wait for frames
        #Remove the first frame to reduce shutter opening artefacts
        for b in range(2):
            frames = pipeline.wait_for_frames(100)
        
        if frames is None:
            print("No frames received")
            continue
     
        # 10.Filter the data
        align_frame = align_filter.process(frames)
        if not align_frame:
            continue
       
        #print(dir(point_frame))
        noise_removed = edge_noise_filter.process(align_frame)

        #spatial_filtered = spatial_filter.process(noise_removed)

        # 11.Apply the point cloud filter
        point_cloud_frame = point_cloud_filter.process(noise_removed)
        try:
            points = point_cloud_filter.calculate(point_cloud_frame)
        except RuntimeWarning as w:
            print("Pointcloud calculation warning:", w)
            i-=1
            continue

        num_points = points.shape[0]
        if num_points< min_points:#600000: 
            i=-1
            continue
        
        # 12.save point cloud
        
        
        success, pcd = save_point_cloud_to_ply(os.path.join(dir_path, f"Cloud_pose{n_clouds}/cloud{i}.ply"), point_cloud_frame, point_cloud_filter)

        if success:
            print(f"Successfully saved {os.path.join(dir_path, f"Cloud_pose{n_clouds}/cloud{i}.ply")}")
        else:
            print("Failed to save pointcloud")
            i -=1
            continue

        pcds.append(pcd)

        if i==n_images:
            break
        i+=1
      
    o3d.visualization.draw_geometries(pcds)
    # 13.Stop the pipeline
    pipeline.stop()


if __name__ == "__main__":
    """
    Script to capture and save pointclouds using Orbbec cameras. n_images decides how many pcds should be saved each time. Filters out pcds with to few points
    and saves them as a PLY in specified directory
    """
    args = parse_args()
    dir_path = args.root_dir if args.root_dir else DEFAULT_ROOT_DIR
    os.makedirs(dir_path, exist_ok=True)
    files = os.listdir(dir_path)
    n_images = args.num_pcds
    min_points=600000
    
    # Extract existing pose numbers from filenames
    existing_poses = []
    for file in files:
        if file.startswith("Cloud_pose"):
            try:
                # Extract number between "Cloud_pose" and ".ply"
                number_str = file[len("Cloud_pose")]  # Remove prefix and suffix
                pose_num = int(number_str)
                existing_poses.append(pose_num)
            except ValueError:
                # Skip files that don't have valid numbers
                continue
    
    # Find the next available pose number
    if existing_poses:
        n_clouds = max(existing_poses) + 1
    else:
        n_clouds = 1
    
    print(f"Existing poses: {sorted(existing_poses)}")
    print(f"Next pose number: {n_clouds}")
    
    main(dir_path, n_clouds, n_images, min_points)