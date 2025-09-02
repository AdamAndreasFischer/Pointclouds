from pyorbbecsdk import * 
import os
import open3d as o3d
import copy

save_points_dir = os.path.join(os.getcwd(), "point_clouds")
if not os.path.exists(save_points_dir):
    os.mkdir(save_points_dir)

def main(dir_path, n_clouds):
    # 1.Create a pipeline with default device.
    pipeline = Pipeline()
    # 2.Create config.
    config = Config()

    device = pipeline.get_device()
    depth_sensor = device.get_sensor(OBSensorType.DEPTH_SENSOR)

    filter_list = depth_sensor.get_recommended_filters()

    for i in range(len(filter_list)):
        filter = filter_list[i]
        if filter:
            print(f"filter name {filter.get_name()}")
  
        
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
   

    print(dir(edge_noise_filter))
    print(edge_noise_filter.get_filter_params())

    point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)
    print("Capture pointcloud")
    while True:
        # 9.Wait for frames
        frames = pipeline.wait_for_frames(100)
        

        if frames is None:
            continue
        
        
     
        # 10.Filter the data
        align_frame = align_filter.process(frames)
        if not align_frame:
            continue
        
        noise_removed = edge_noise_filter.process(align_frame)

        #spatial_filtered = spatial_filter.process(noise_removed)

        # 11.Apply the point cloud filter
        point_cloud_frame = point_cloud_filter.process(noise_removed)

        # 12.save point cloud
        
        print("Saving pointcloud...")
        save_point_cloud_to_ply(os.path.join(dir_path, f"Cloud_pose{n_clouds}.ply"), point_cloud_frame)

        print(f"Saving {os.path.join(dir_path, f"Cloud_pose{n_clouds}.ply")}")

        pcd = o3d.io.read_point_cloud(os.path.join(dir_path, f"Cloud_pose{n_clouds}.ply"))
        o3d.io.write_point_cloud(os.path.join(dir_path, f"Cloud_pose{n_clouds}.ply"), pcd, write_ascii = True )
        print(pcd)
        o3d.visualization.draw_geometries([pcd])

        break

    # 13.Stop the pipeline
    pipeline.stop()


if __name__ == "__main__":
    dir_path = "/home/adamfi/Codes/Pointclouds/pointclouds/full_room3"
    files = os.listdir(dir_path)
    
    # Extract existing pose numbers from filenames
    existing_poses = []
    for file in files:
        if file.startswith("Cloud_pose") and file.endswith(".ply"):
            try:
                # Extract number between "Cloud_pose" and ".ply"
                number_str = file[len("Cloud_pose"):-4]  # Remove prefix and suffix
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
    
    main(dir_path, n_clouds)