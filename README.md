# Point cloud capture and registration

## Requirements 

This code is built for orbbec cameras, but can be changed to other cameras. If orbbec is used, the pyorbbecsdk if required. Install guide and package can be found at: https://github.com/orbbec/pyorbbecsdk

other packages
```
open3d
scipy
tqdm
```


## Capture pointclouds and poses

### Capture Point clouds
In order to capture pointclouds, the `capture_point_clouds.py` script must be modified to fit the camera in use. The script captures as many clouds as specified and saves them in a Cloud_poseX folder in the specified directory. The folders number is based on the amount of folders already in the directory. The code filters out clouds with points less than a set threshold, depends on your camera.

Before using, create a folder for the pointclouds, otherwise it will flood the main directory with clouds.

### Capture poses
In order to capture poses with `ros_pose_listener.py`, a rosmaster with `natnet_for_ros` and `rosbridge` must be setup to recieve and publish poses of the cameras from the motive software. The pose is then save in the specified directory named pose_x.npy based on the ammount of poses already in the directory

Before registering the pointclouds with the captured poses, it is important to compare the internal coordinate system of the camera to the coordinate system of camera pose in Rviz as the xyz directions can differ and will result in no alligment. Modify the transfrom_coords in `Multi_stage_icp.py` to represent the coordinate system transform from the internal coordinate system to the Rviz coordinate system. 

### Register pointclouds

`Multi_stage_icp.py` is used to register the pointclouds from the initial positions captured by the mocap system. The script takes the clouds and poses and registers them with increasingly finer precision. When the registration is done, a preview of the registered clouds is shown and the registered poses and the alligned pointclouds are saved. **Note** that the current script expects the pointclouds to be in milimeters instead of meters. Check the camera for information on this.

