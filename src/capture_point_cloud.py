from pyorbbecsdk import * 
import os
import open3d as o3d
import copy
import numpy

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
   

    point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)
    print("Capture pointcloud")
    i=1
    if not os.path.exists(os.path.join(dir_path, f"Cloud_pose{n_clouds}")):
        os.mkdir(os.path.join(dir_path, f"Cloud_pose{n_clouds}"))
    pcds = []
    while True:
        # 9.Wait for frames
        frames = pipeline.wait_for_frames(100)
        
        if frames is None:
            continue
     
        # 10.Filter the data
        align_frame = align_filter.process(frames)
        if not align_frame:
            continue
        print("Depth frame functions")
        depth_data = frames.get_depth_frame()
        print(dir(depth_data))
        print(depth_data.__getstate__())
        print(dir(frames))
        #print(dir(point_frame))

     
        #noise_removed = edge_noise_filter.process(align_frame)

        #spatial_filtered = spatial_filter.process(noise_removed)

        # 11.Apply the point cloud filter
        point_cloud_frame = point_cloud_filter.process(align_frame)
        points = point_cloud_filter.calculate(point_cloud_frame)
        num_points = points.shape[0]
        if num_points< 600000: 
            i=-1
            continue
        print(f"Cloud contains {num_points} points")
    
        # 12.save point cloud
        
        print("Saving pointcloud...")
        save_point_cloud_to_ply(os.path.join(dir_path, f"Cloud_pose{n_clouds}/cloud{i}.ply"), point_cloud_frame, )

        print(f"Saving {os.path.join(dir_path, f"Cloud_pose{n_clouds}/cloud{i}.ply")}")

        pcd = o3d.io.read_point_cloud(os.path.join(dir_path, f"Cloud_pose{n_clouds}/cloud{i}.ply"))
        o3d.io.write_point_cloud(os.path.join(dir_path, f"Cloud_pose{n_clouds}/cloud{i}.ply"), pcd, write_ascii = True )
        pcd_pts = numpy.array(pcd.points)
        if pcd_pts.shape[0] <600000:
            i-=1
        else:
            pcds.append(pcd)

        if i==100:
            break
        i+=1
    o3d.visualization.draw_geometries(pcds)
    # 13.Stop the pipeline
    pipeline.stop()


if __name__ == "__main__":
    dir_path = "/home/adamfi/Codes/Pointclouds/pointclouds/multi_cloud_test"
    files = os.listdir(dir_path)
    
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
    
    main(dir_path, n_clouds)