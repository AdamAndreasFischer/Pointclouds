from pyorbbecsdk import *
import os 
import cv2
import numpy as np
import json
import open3d as o3d
from scipy.spatial.transform import Rotation as R, Slerp
os.environ["DISPLAY"] = ":1"
os.environ["GDK_BACKEND"] = "x11"
os.environ["PYOPENGL_PLATFORM"] = "glx"
os.environ["XDG_SESSION_TYPE"] = "x11"
import time

class Estimate_charuco_pose:
    def __init__(self):
        
        with open("/home/adamfi/codes/Camera_calibration/orbbec_params.json", "r") as f:
            matrices = json.load(f)

        self.color_intrinsics = matrices["color"]
        self.depth_intrincics = matrices["depth"]

        x_markers = 10
        y_markers = 7
        square_length = 0.037
        marker_lenght = 0.027


        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.board = cv2.aruco.CharucoBoard(size= (x_markers, y_markers),
                                            squareLength=square_length,
                                            markerLength=marker_lenght,
                                            dictionary=self.aruco_dict)
        
        self.board.setLegacyPattern(True)

        self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
        
        # Start camera stream
        #self.pipeline = Pipeline()
        max_retries = 100
        retry_delay = 0.1
        self.pipeline = None
        for attempt in range(1, max_retries + 1):
            try:
                ctx = Context()
                device_list = ctx.query_devices()
                if device_list.get_count() == 0:
                    raise RuntimeError("No Orbbec device found")
                print(f"Orbbec device found (attempt {attempt})")
                self.pipeline = Pipeline()
                break
            except Exception as e:
                print(f"[{attempt}/{max_retries}] Waiting for Orbbec device: {e}")
                time.sleep(retry_delay)
        
        if self.pipeline is None:
            raise RuntimeError(f"Failed to find Orbbec device after {max_retries} attempts")
        
        self.config = Config()
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
        #print(color_profile)
        self.config.enable_stream(color_profile)
        #print(dir(self.config))

        #Initialization for solvePnP ransac
        self.rvec_old = np.zeros((3,1))
        self.tvec_old = np.zeros((3,1))
    
        # Start the stream
        self.pipeline.start(self.config)


    def detect_markers(self,in_img):

        gray = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids = (self.charuco_detector.detectBoard(gray))

        if marker_ids is None or len(marker_ids) == 0:
            print("No detection")
            return None
        
        cv2.aruco.drawDetectedCornersCharuco(in_img, charucoCorners=charuco_corners, charucoIds=charuco_ids, cornerColor=(0,255,0))

        cv2.imshow("Markers on board", in_img)
        key = cv2.waitKey(1) & 0xFF
        return key
    
    def smooth_pose(self, prev_rvec, prev_tvec, rvec, tvec):
        alpha_t = 0.1
        alpha_r = 0.1

        # Fail-safe: if incoming pose is invalid, keep previous estimate
        if rvec is None or tvec is None:
            return prev_rvec, prev_tvec
        if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
            return prev_rvec, prev_tvec
        if np.linalg.norm(rvec) < 1e-12:
            return prev_rvec, prev_tvec

        # If previous pose is invalid, initialize with current pose
        if prev_rvec is None or prev_tvec is None:
            return rvec, tvec
        if not (np.isfinite(prev_rvec).all() and np.isfinite(prev_tvec).all()):
            return rvec, tvec

        # translation EMA
        tvec_s = (1 - alpha_t) * prev_tvec + alpha_t * tvec

        # rotation slerp between prev and current
        try:
            r_prev = R.from_rotvec(prev_rvec.reshape(3))
            r_curr = R.from_rotvec(rvec.reshape(3))
            slerp = Slerp([0, 1], R.concatenate([r_prev, r_curr]))
            r_s = slerp(alpha_r)
            rvec_s = r_s.as_rotvec().reshape(3, 1)
        except Exception:
            # Fail-safe if scipy slerp fails (e.g. zero-norm quaternion internally)
            rvec_s = rvec

        return rvec_s, tvec_s

    def detect_board(self, in_img, debug=False):
        """Return classic board->camera pose (target2cam) from solvePnPRansac."""
        gray = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)

        if charuco_corners is None or charuco_ids is None or len(charuco_ids) == 0:
            return None, None

        obj_pts, im_pts = cv2.aruco.Board.matchImagePoints(
            self.board,
            detectedCorners=charuco_corners,
            detectedIds=charuco_ids,
        )
        if obj_pts.shape[0] < 4 or im_pts.shape[0] < 4:
            return None, None

        if not (np.isfinite(obj_pts).all() and np.isfinite(im_pts).all()):
            return None, None

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj_pts,
            imagePoints=im_pts,
            cameraMatrix=np.array(self.color_intrinsics["camera_matrix"]),
            distCoeffs=np.array(self.color_intrinsics["distortion_coefficients"]),
            rvec=self.rvec_old,
            tvec=self.tvec_old,
            useExtrinsicGuess=True,
        )

        if not retval:
            return None, None

        if rvec is None or tvec is None:
            return None, None
        if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
            return None, None

        rvec, tvec = self.smooth_pose(
            prev_rvec=self.rvec_old,
            prev_tvec=self.tvec_old,
            rvec=rvec,
            tvec=tvec,
        )
        self.rvec_old = rvec
        self.tvec_old = tvec

        R_b2c, _ = cv2.Rodrigues(rvec)

        if debug:
            cv2.drawFrameAxes(
                in_img,
                np.array(self.color_intrinsics["camera_matrix"]),
                np.array(self.color_intrinsics["distortion_coefficients"]),
                rvec,
                tvec,
                0.15,
            )
            cv2.imshow("image with coordinates", in_img)
            key = cv2.waitKey(1) & 0xFF
            return key, R_b2c, tvec

        return R_b2c, tvec

    def detect_board_old(self, in_img, debug=False):
        gray = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids = (self.charuco_detector.detectBoard(gray))

        # Requires camera_matrix/dist_coeffs from calibration
        #pose_ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        #    charuco_corners, charuco_ids, self.board, self.color_intrinsics["camera_matrix"], self.color_intrinsics["distortion_coefficients"], None, None
        #)
        if charuco_corners is None or len(charuco_ids)==0:
            return None, None
        
        obj_pts, im_pts = cv2.aruco.Board.matchImagePoints(self.board, detectedCorners=charuco_corners, detectedIds=charuco_ids)
        if obj_pts.shape[0] <4 or im_pts.shape[0]<4:
            return None, None
        
        if not (np.isfinite(obj_pts).all() and np.isfinite(im_pts).all()):
            return None, None

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=obj_pts, imagePoints=im_pts, cameraMatrix=np.array(self.color_intrinsics["camera_matrix"]), distCoeffs=np.array(self.color_intrinsics["distortion_coefficients"]),
                                                rvec=self.rvec_old,
                                                 tvec=self.tvec_old,
                                                  useExtrinsicGuess=True )
        rvec, tvec = self.smooth_pose(prev_rvec = self.rvec_old,prev_tvec=self.tvec_old, rvec=rvec, tvec=tvec)
        
        self.rvec_old = rvec
        self.tvec_old = tvec

        if retval:
            # Find marker-1 corner in board coordinates
            board_ids = self.board.getIds()
            board_obj = self.board.getObjPoints()  # list of (4,3) corners per marker
            marker1_idx = list(board_ids).index(0)  # marker id = 1
            marker1_corners = board_obj[marker1_idx]
            origin_corner = marker1_corners[0]  # pick corner 0 as origin

            # Shift origin from board corner to marker-1 corner
            theta = -np.pi / 2.0  # clockwise 90 deg
            Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta),  np.cos(theta), 0],
               [0,              0,             1]], dtype=np.float64)

            R, _ = cv2.Rodrigues(rvec)
            Rx = np.array([[1, 0, 0],
               [0,-1, 0],
               [0, 0,-1]], dtype=np.float64)
            R_flipped = R@Rx @ Rz
            rvec_flipped,_ = cv2.Rodrigues(R_flipped)
            tvec_shifted = tvec + R_flipped @ origin_corner.reshape(3, 1)

            if debug:
                # Draw axes at marker-1 corner
                cv2.drawFrameAxes(in_img, np.array(self.color_intrinsics["camera_matrix"]), np.array(self.color_intrinsics["distortion_coefficients"]), rvec_flipped, tvec_shifted, 0.15)

                cv2.imshow("image with coordinates",in_img)
                key= cv2.waitKey(1) & 0xFF
                return key, R_flipped, tvec_shifted
            else: 
                return R_flipped, tvec_shifted
        else:
            return None, None

    def get_camera_stream(self):

        if self.pipeline is not None:
            frames = self.pipeline.wait_for_frames(100)

            if frames== None:
                print("No frame from Orbbec")
                return None
            time_stamp = time.time()
            color_frame = frames.get_color_frame()
            if color_frame is None:
                print("No color frame from Orbbec")
                return None
                
            # Convert to numpy array
            width = color_frame.get_width()
            height = color_frame.get_height()
            
            # Get frame data
            frame_data = color_frame.get_data()
            
            # Convert to OpenCV format (BGR)
            color_image = np.frombuffer(frame_data, dtype=np.uint8)
            color_image = color_image.reshape((height, width, 3))
            
            # Make array C-contiguous before OpenCV operations
            color_image = np.ascontiguousarray(color_image)
            
            # Convert RGB to BGR for OpenCV
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            return color_image, time_stamp
    
    def get_camera_pose(self, frames = None, max_attempts=150, timeout_s=6.0):
        """Returns T_b2cam (charuco board pose in the camera frame)."""
        
        batch = 3
        R_cb_batch = np.zeros((batch,3,3))
        t_cb_batch = np.zeros((batch,3,1))
        idx = 0
        done = False
        attempts = 0
        t_start = time.time()
        while not done:
            attempts += 1
            if attempts > max_attempts or (time.time() - t_start) > timeout_s:
                # Fail-safe: no stable board detection within limits
                return None, None

            frame, time_stamp = self.get_camera_stream()

            if frame is None:
                continue
            R_cb, t_cb = self.detect_board(frame)
            if R_cb is None or t_cb is None:
                continue

            R_cb_batch[idx] = R_cb
            t_cb_batch[idx] = t_cb
            idx+=1
            if idx == batch:
                done = True
        
        # SVD mean of rotation
        M = np.mean(R_cb_batch, axis=0)
        U, _, Vt = np.linalg.svd(M)
        R_mean = U @ Vt
        if np.linalg.det(R_mean) < 0:
            U[:, -1] *= -1
            R_mean = U @ Vt
     
        mean_t_cb = np.mean(t_cb_batch, axis=0)

        T_b2cam = np.zeros((4,4))
        T_b2cam[:3,:3] = R_mean
        T_b2cam[:3,3] = mean_t_cb.squeeze(-1)
        T_b2cam[-1,-1] = 1


        return T_b2cam, time_stamp

    def get_camera_pose_one_frame(self, frame):

        R_cb, t_cb = self.detect_board(frame)
        if R_cb is None or t_cb is None:
            return None

        T_b2cam = np.zeros((4,4))
        T_b2cam[:3,:3] = R_cb
        T_b2cam[:3,3] = t_cb.squeeze(-1)
        T_b2cam[-1,-1] = 1

        return T_b2cam




def make_frame(size=0.1):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

def main():

    board_estimator = Estimate_charuco_pose()

    
    print(board_estimator.get_camera_pose())

   
    visualize = True
    if visualize: 
        vis = o3d.visualization.Visualizer()
        vis.create_window("Board/Camera", width=1920, height=1080)

        board_frame = make_frame(0.1)   # board at origin
        cam_frame = make_frame(0.1)     # camera frame

        vis.add_geometry(board_frame)
        vis.add_geometry(cam_frame)
        T_old = None
        while True:
            # get R_cam, t_cam from your pipeline
            # R_cam: (3,3), t_cam: (3,1) in board frame



            T,_= board_estimator.get_camera_pose()
            T_cam_in_board = np.linalg.inv(T)
            if not T_old is None:
                cam_frame.transform(np.linalg.inv(T_old))  # reset
            cam_frame.transform(T_cam_in_board)
            #cam_frame = make_frame(0.1)
            #cam_frame.transform(T)  
            vis.update_geometry(cam_frame)
            vis.poll_events()
            vis.update_renderer()
            T_old = T_cam_in_board
        
        

if __name__ == "__main__":
    main()



"""
To use this code for mocap finetuning, we can run the mocap to get the transfrom of the camera in the world coordinate
T_w2cam. As the transform T_w2chaboard also exists, we can calculate the T_w2cam as well using this script and find the error in the rigid bodys position and orientation. 
This error can then be applied to all registrations of pointcloud where the cameras pose captured from the mocap is used, i.e ever time we use T_w2cam from mocap

i.e T_w2cam = T_w2chaboard @ T_chaboard2cam

In the current setup Y is down in camera frame and Z is forward. This needs to be taken into account during calculations so that it is correct
i.e Upward = -y forward = y
"""