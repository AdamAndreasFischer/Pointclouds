# Point cloud capture and registration

Capture point clouds from Orbbec cameras, calibrate each camera to a Motive/NatNet
mocap rigid body, and merge per-pose clouds into a single registered point cloud.

## Requirements

This repo is designed **only with Orbbec cameras in mind** (Gemini 2L) via
[`pyorbbecsdk`](https://github.com/orbbec/pyorbbecsdk). It can be adapted to other cameras,
but the components that touch the hardware must be changed accordingly — capture and
pipeline setup (`pyorbbecsdk`), the calibration intrinsics, and the stream profiles.

Python packages:

```
open3d
numpy
scipy
opencv-python
tqdm
natnet        # NatNet streaming client
roslibpy      # only for the deprecated ROS listener

```

## Pipeline overview

1. **Calibrate** each camera to its mocap rigid body (`T_rigid_body_to_camera`).
2. **Capture** point clouds and the camera pose for each viewpoint.
3. **Register / merge** the per-pose clouds into one cloud using the captured poses.

The repo is organised around a **multi-camera** workflow; older single-camera scripts
are kept for reference but are deprecated (see below).

## 0. Directory layout for it to work together with UFOMap
First, download and install https://github.com/AdamAndreasFischer/Manipulation_ufomap 
The three components which needs to work together are 
1.  The Pointcloud repo code
2.  The shell script called `capture_cloud_and_pose_multi_cam.sh`
3.  The UFOMap branch from the repo above

The codes in Pointclouds are built after this directory structure
```
root dir 
      |-Pointclouds 
      |-capture_cloud_and_pose_multi_cam.sh
      |-UFOMap 
   ```   
I.e the three components must share the same root directory to avoid having to change the pathing in the files. 
The shell script requires changes to IP's of the PC used and mocap system, as well as paths to store captured clouds and poses. 

## 1. Calibration

- **`src/calibrate_orbbec.py`** — Charuco board pose estimator (`Estimate_charuco_pose`),
  built for the Gemini 2L. Update the intrinsics if you use another camera.
- **`src/calibrate_camera_to_mocap_static.py`** *(recommended)* — static, tripod-based
  hand-eye calibration. Capture 15–25 diverse poses; raw frames are averaged per pose
  and a variance gate rejects motion, giving a much cleaner result than the dynamic version.
- **`src/calibrate_camera_to_mocap.py`** — dynamic version: move the camera slowly in
  front of a (still) A3 Charuco board. Noisier; prefer the static script.

Both calibration scripts use `cv2.calibrateHandEye` and need the NatNet **server IP**
(Motive stream) and **client IP** (receiving PC) on the same `/24` subnet
(e.g. `192.168.111.10` / `192.168.111.20`). The result `T_rigid_body_to_camera` is
applied to a mocap pose as:

```
camera_world = Original_transform @ np.linalg.inv(T_rigid_body_to_camera)
```

Saved calibrations live in `Orbbec_calibrations_mocaplab/` (one `.npy` per camera).

## 2. Capture
# The code automatically sees which camera is being used from the serial number. If you number your cameras, make sure that the serial number is updated in the camera info yaml file together with the calibration

- **`src/capture_point_cloud_multi_cam.py`** *(current)* — captures clouds from several
  Orbbec cameras at once into `Cloud_pose*_camera_*` folders under a root dir. Cameras are
  selected via `--camera-ids` and mapped to serials through `utils/camera_data.yml`
  (or `--serials`). Clouds below a point-count threshold are discarded.

> **`camera_data*.yml`** holds, per camera, its **serial number**, mocap **`rigid_body_id`**,
> and the **`calibration`** matrix (`T_rigid_body_to_camera`, from step 1). The multi-cam
> scripts key off this file, so it must be filled in for your cameras — and the serial/rigid-body
> fields are Orbbec/NatNet specific, so adapting to another camera model means appropriating
> them (and the scripts that read them) accordingly.
- **`src/natnet_pose_listener_multi_cam.py`** *(current)* — listens to NatNet and saves one
  pose per camera rigid body (`pose_capture*_camera_*.npy`), using the same `camera_data.yml`.
- **`src/natnet_pose_listener.py`** — single-camera NatNet listener; saves `pose_*.npy`.

Create the output directory before capturing so clouds don't flood the working dir.

> **Coordinate frames:** the camera's internal axes may differ from the mocap/RViz pose
> axes. Before registering, compare the two and set the `transform_coords` matrix in the
> registration script accordingly, or alignment will fail.

## 3. Register / merge

- **`src/Multi_stage_icp_multi_cam.py`** *(current)* — calibration-only multi-camera
  alignment. Keeps the `Multi_stage_icp` name for compatibility but **intentionally skips
  ICP**, relying on the per-camera calibration to place each cloud.
- **`src/Multi_stage_icp_cuda.py`** — CUDA/tensor Open3D variant of multi-stage ICP for
  larger clouds (uses an Open3D `CUDA` device).
- **`src/pointcloud_stitcher.py`** — simple pose-based stitcher: loads poses + clouds from a
  folder and transforms them into a common frame.

> Several scripts expect point clouds in **millimetres**, not metres — check your camera.

## Layout

```
src/      capture, calibration, pose-listening and registration scripts
utils/    Orbbec helpers, point cloud / pose I/O, camera_data*.yml, misc tools
tests/    experiments: denoising, grid/parameter sweeps, loop closure, TEASER++
stubs/    pyorbbecsdk type stubs
Orbbec_calibrations_mocaplab/   saved per-camera calibration matrices (.npy)
pointclouds/, results/          captured data and outputs
```

Useful `utils/`: `orbbec_utils.py` (frame → BGR conversion), `pointcloud_utils.py`
(load poses/clouds, pose → 4×4 transform), `get_serial.py` (list connected camera serials),
`detect_floor_plane.py` in `src/` (largest-plane / floor detection).

## Deprecated

These predate the multi-camera workflow and are kept only for reference:

- **`src/ros_pose_listener.py`** — pose capture over `rosbridge`/`roslibpy`, requiring a
  ROS master running `natnet_for_ros`. Superseded by the direct NatNet listeners above.
- **`src/Multi_stage_icp.py`** — single-camera multi-stage ICP registration.
- **`src/capture_point_cloud.py`** — single-camera capture.
