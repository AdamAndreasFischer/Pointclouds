# Point cloud capture and registration

## Requirements 

This code is built for orbbec cameras, but can be changed to other cameras. If orbbec is used, the pyorbbecsdk if required. Install guide and package can be found at: https://github.com/orbbec/pyorbbecsdk

other packages
```
open3d
scipy
tqdm
natnet
opencv-python
roslibpy
```


## Capture pointclouds and poses

### Calibrate camera to mocap system

To get a good calibration between the cameras internal frame, and the mocaps rigid body coodinate system, this repo contains a script called `calibrate_camera_to_mocap.py`. This script relies on two other functionalities: A natnet pose listner, and a charuco board estimator. In this repo there are two natnet listners, one for the ROS client, and one for the Python client depending on if natnet is used with ROS or not. Currently the `calibrate_camera_to_mocap.py` is build around the natnet Python package, but it can be changed with little effort by looking at the two script which use the separate packages. 

The charuco board pose estimator in this repo, called `calibrate_orbbec.py` is built for orbbecs gemini 2L, but the camera model can easily be changed to any other camera. Remember to import the intrinsics for the camera in use. 

To calibrate the camera, setup a rigid body for it in motive, and get a large charuco board, preferably A3 format. Start the script and move the camera SLOWLY infront of the charuco board. It needs to move away and towards, as well as rotate in relation to the board. Keep the board still. 

If the script detects unreasonable distances in calibration, it will tell you and you should rerun the script. Move slower if this happens. 

The calibration transform T_rigid_body_to_camera is then printed in the terminal. To use it, multiply it with the pose captured from natnet as follows: 
`Original_transform @ np.linalg.inv(T_rigid_body_to_camera)`

When calling the script, it expects atleast arguments for the server ip (motive stream) and client ip (reciveing pc). Notice that these IPs must be on the same subnet, i.e the IPs should coindice in the first three numbers. For example 192.168.111.10 and 192.168.111.20. 

### Capture Point clouds
In order to capture pointclouds, the `capture_point_clouds.py` script must be modified to fit the camera in use. The script captures as many clouds as specified and saves them in a Cloud_poseX folder in the specified directory. The folders number is based on the amount of folders already in the directory. The code filters out clouds with points less than a set threshold, depends on your camera.

Before using, create a folder for the pointclouds, otherwise it will flood the main directory with clouds.

### Capture poses
In order to capture poses with `ros_pose_listener.py`, a rosmaster with `natnet_for_ros` and `rosbridge` must be setup to recieve and publish poses of the cameras from the motive software. The pose is then save in the specified directory named pose_x.npy based on the ammount of poses already in the directory


Before registering the pointclouds with the captured poses, it is important to compare the internal coordinate system of the camera to the coordinate system of camera pose in Rviz as the xyz directions can differ and will result in no alligment. Modify the transfrom_coords in `Multi_stage_icp.py` to represent the coordinate system transform from the internal coordinate system to the Rviz coordinate system. 

### Register pointclouds

`Multi_stage_icp.py` is used to register the pointclouds from the initial positions captured by the mocap system. The script takes the clouds and poses and registers them with increasingly finer precision. When the registration is done, a preview of the registered clouds is shown and the registered poses and the alligned pointclouds are saved. **Note** that the current script expects the pointclouds to be in milimeters instead of meters. Check the camera for information on this.

