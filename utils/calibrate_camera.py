import os
import numpy as np
import cv2
import roslibpy
import time
from scipy.spatial.transform import Rotation as Rot

# Try to import pyorbbecsdk
try:
    from pyorbbecsdk import *
    PYORBEC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pyorbbecsdk not available: {e}")
    print("Falling back to OpenCV camera...")
    PYORBEC_AVAILABLE = False

# Camera calibration parameters
k_mat = np.array([
    [610.051,   0,     632.486],
    [  0,     609.762, 393.775],
    [  0,       0,        1   ]
])

color_dist = np.array([-0.98427, 0.280092, 0.0578961, -0.964254, 0.2516, 0.0690387, 0.000159038, -0.000424764])

# ChArUco board parameters
ARUCO_DICT = cv2.aruco.DICT_4X4_100
SQUARES_VERTICALLY = 8
SQUARES_HORIZONTALLY = 11
SQUARE_LENGTH = 0.023
MARKER_LENGTH = 0.017

class CharucoPoseEstimator:
    def __init__(self, ros_host='192.168.125.186', ros_port=9090):
        # ChArUco board parameters
        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, self.dictionary)
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
        
        # Camera parameters
        self.camera_matrix = np.zeros((3, 3))
        self.dist_coeffs = np.zeros((8,))
        
        # ROS connection
        self.ros_client = None
        self.tf_publisher = None
        self.ros_host = ros_host
        self.ros_port = ros_port
        
        # Orbbec camera pipeline
        self.pipeline = None
        self.config = None
        
    def setup_orbbec_camera(self):
        """Setup Orbbec camera pipeline"""
        if not PYORBEC_AVAILABLE:
            print("pyorbbecsdk not available, cannot setup Orbbec camera")
            return False
            
        try:
            self.pipeline = Pipeline()
            self.config = Config()

            # Enable color profile
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = profile_list.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
            self.config.enable_stream(color_profile)

            # Start the stream
            self.pipeline.start(self.config)
            
            print("Orbbec camera initialized successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to initialize Orbbec camera: {e}")
            return False
    
    def get_orbbec_frame(self):
        """Get color frame from Orbbec camera or fallback to OpenCV"""
        if PYORBEC_AVAILABLE and self.pipeline is not None:
            try:
                # Wait for frames
                frames = self.pipeline.wait_for_frames(100)
                
                if frames is None:
                    return None
                    
                # Get color frame
                color_frame = frames.get_color_frame()
                if color_frame is None:
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
                
                return color_image
                
            except Exception as e:
                print(f"Error getting Orbbec frame: {e}")
                return None
        else:
            # Fallback to OpenCV camera
            if not hasattr(self, 'opencv_cap'):
                self.opencv_cap = cv2.VideoCapture(0)
                if not self.opencv_cap.isOpened():
                    print("Could not open OpenCV camera")
                    return None
            
            ret, frame = self.opencv_cap.read()
            if ret:
                return frame
            else:
                print("Failed to capture OpenCV frame")
                return None
    
    def connect_to_ros(self):
        """Connect to ROS bridge and create TF publisher"""
        try:
            print(f"Connecting to ROS bridge at {self.ros_host}:{self.ros_port}...")
            self.ros_client = roslibpy.Ros(host=self.ros_host, port=self.ros_port)
            self.ros_client.run()
            
            # Create TF publisher
            self.tf_publisher = roslibpy.Topic(
                self.ros_client, 
                '/tf', 
                'tf2_msgs/TFMessage'
            )
            
            print("Connected to ROS bridge successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to connect to ROS: {e}")
            return False
    
    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion [x, y, z, w]"""
        
        quat = Rot.from_matrix(R).as_quat()  # [x, y, z, w]

        return quat

    def publish_tf(self, rvec, tvec, timestamp=None):
        """Publish TF transform from charuco_board to camera_link"""
        if self.tf_publisher is None:
            print("TF publisher not initialized!")
            return
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Convert rotation matrix to quaternion
        quat = self.rotation_matrix_to_quaternion(R)
        
        # Create timestamp
        if timestamp is None:
            now = time.time()
            secs = int(now)
            nsecs = int((now - secs) * 1e9)
        else:
            secs = int(timestamp)
            nsecs = int((timestamp - secs) * 1e9)
        
        # Create TF message
        # Transform: charuco_board -> camera_link
        # This means the board is the parent frame, camera is the child
        tf_msg = {
            'transforms': [{
                'header': {
                    'stamp': {
                        'sec': secs,
                        'nanosec': nsecs
                    },
                    'frame_id': 'charuco_board'  # Parent frame
                },
                'child_frame_id': 'camera_link',  # Child frame
                'transform': {
                    'translation': {
                        'x': float(tvec[0]),
                        'y': float(tvec[1]),
                        'z': float(tvec[2])
                    },
                    'rotation': {
                        'x': float(quat[0]),
                        'y': float(quat[1]),
                        'z': float(quat[2]),
                        'w': float(quat[3])
                    }
                }
            }]
        }
        
        # Publish the transform
        self.tf_publisher.publish(roslibpy.Message(tf_msg))
    
    def draw_coordinate_system_info(self, frame, rvec, tvec):
        """Draw additional coordinate system information on the frame"""
        # Project origin and axis endpoints to image plane
        axis_points = np.array([
            [0, 0, 0],      # Origin
            [0.1, 0, 0],    # X-axis endpoint (Red)
            [0, 0.1, 0],    # Y-axis endpoint (Green) 
            [0, 0, 0.1]     # Z-axis endpoint (Blue)
        ], dtype=np.float32)
        
        # Project 3D points to 2D image plane
        projected_points, _ = cv2.projectPoints(
            axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        
        # Convert to integer pixel coordinates
        origin = tuple(projected_points[0].ravel().astype(int))
        x_end = tuple(projected_points[1].ravel().astype(int))
        y_end = tuple(projected_points[2].ravel().astype(int))
        z_end = tuple(projected_points[3].ravel().astype(int))
        
        # Draw thicker axis lines with labels
        line_thickness = 3
        
        # X-axis (Red)
        cv2.arrowedLine(frame, origin, x_end, (0, 0, 255), line_thickness, tipLength=0.3)
        cv2.putText(frame, 'X', (x_end[0] + 5, x_end[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Y-axis (Green) 
        cv2.arrowedLine(frame, origin, y_end, (0, 255, 0), line_thickness, tipLength=0.3)
        cv2.putText(frame, 'Y', (y_end[0] + 5, y_end[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Z-axis (Blue)
        cv2.arrowedLine(frame, origin, z_end, (255, 0, 0), line_thickness, tipLength=0.3)
        cv2.putText(frame, 'Z', (z_end[0] + 5, z_end[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw origin point
        cv2.circle(frame, origin, 5, (255, 255, 255), -1)
        cv2.circle(frame, origin, 5, (0, 0, 0), 2)
        
        # Add coordinate system legend
        legend_y_start = 100
        cv2.putText(frame, 'Coordinate System:', (frame.shape[1] - 200, legend_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, 'X: Red', (frame.shape[1] - 200, legend_y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, 'Y: Green', (frame.shape[1] - 200, legend_y_start + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, 'Z: Blue', (frame.shape[1] - 200, legend_y_start + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    def detect_and_estimate_pose(self, frame):
        """Detect ChArUco board and estimate its pose"""
        # Convert to grayscale for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use the CharucoDetector API for OpenCV 4.11+
        charuco_corners, charuco_ids, aruco_corners, aruco_ids = self.charuco_detector.detectBoard(gray)
        
        if charuco_corners is not None and len(charuco_corners) > 3:
            # Create object points from charuco corners
            objPoints = []
            imgPoints = []
            
            # Match detected corners with their 3D positions on the board
            for i in range(len(charuco_ids)):
                corner_id = charuco_ids[i][0]
                # Get 3D position of this corner on the board
                objPt = self.board.getChessboardCorners()[corner_id]
                objPoints.append(objPt)
                # Get 2D position in the image
                imgPoints.append(charuco_corners[i][0])
            
            # Convert to numpy arrays
            objPoints = np.array(objPoints, dtype=np.float32)
            imgPoints = np.array(imgPoints, dtype=np.float32)
            
            # Only proceed if we have enough points
            if len(objPoints) >= 4:
                # Estimate pose using solvePnP
                retval, rvec, tvec = cv2.solvePnP(
                    objPoints, imgPoints, 
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if retval:
                    # Draw detected markers
                    if aruco_corners is not None and aruco_ids is not None:
                        cv2.aruco.drawDetectedMarkers(frame, aruco_corners, aruco_ids)
                    
                    # Draw detected charuco corners
                    if charuco_corners is not None and charuco_ids is not None:
                        cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
                    
                    # Draw coordinate axes (X=Red, Y=Green, Z=Blue)
                    axis_length = 0.1  # 10cm axes
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                     rvec, tvec, axis_length)
                    
                    # Draw additional coordinate system information
                    self.draw_coordinate_system_info(frame, rvec, tvec)
                    
                    # Publish TF
                    self.publish_tf(rvec, tvec)
                    
                    return True, rvec, tvec, frame
    
        return False, None, None, frame
    
    def run_camera_calibration(self):
        """Run the main camera calibration loop using Orbbec camera"""
        if not self.connect_to_ros():
            return
            
        if not self.setup_orbbec_camera():
            return
        
        print("Starting ChArUco pose estimation with Orbbec camera...")
        print("Position the ChArUco board in front of the camera")
        print("Press 'q' to quit")
        
        frame_count = 0
        try:
            while True:
                # Get frame from Orbbec camera
                frame = self.get_orbbec_frame()
                if frame is None:
                    print(f"Failed to capture frame {frame_count}")
                    continue
                
                frame_count += 1
                
                # Detect and estimate pose
                detected, rvec, tvec, processed_frame = self.detect_and_estimate_pose(frame)
                
                #if detected:
                #    # Display pose information
                #    cv2.putText(processed_frame, f"ChArUco Board Detected!", (10, 30),
                #              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #    
                #    # Display translation
                #    cv2.putText(processed_frame, f"Translation (m):", (10, 60),
                #              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                #    cv2.putText(processed_frame, f"  X: {tvec[0]:.3f}", (10, 80),
                #              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                #    cv2.putText(processed_frame, f"  Y: {tvec[1]:.3f}", (10, 100),
                #              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                #    cv2.putText(processed_frame, f"  Z: {tvec[2]:.3f}", (10, 120),
                #              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                #    
                #    # Display rotation (in degrees)
                #    angles = rvec.flatten() * 180.0 / np.pi
                #    cv2.putText(processed_frame, f"Rotation (deg):", (10, 150),
                #              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                #    cv2.putText(processed_frame, f"  RX: {angles[0]:.1f}", (10, 170),
                #              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                #    cv2.putText(processed_frame, f"  RY: {angles[1]:.1f}", (10, 190),
                #              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                #    cv2.putText(processed_frame, f"  RZ: {angles[2]:.1f}", (10, 210),
                #              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                #else:
                #    cv2.putText(processed_frame, "No ChArUco board detected", (10, 30),
                #              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                #    cv2.putText(processed_frame, "Position board in camera view", (10, 60),
                #              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                #
                ## Add frame counter
                #cv2.putText(processed_frame, f"Frame: {frame_count}", (10, processed_frame.shape[0] - 10),
                #          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                camera_type = "Orbbec Camera" if PYORBEC_AVAILABLE else "OpenCV Camera"
                cv2.imshow(f'ChArUco Pose Estimation - {camera_type}', processed_frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            # Stop the pipeline
            if PYORBEC_AVAILABLE and self.pipeline:
                self.pipeline.stop()
            
            # Release OpenCV camera if used
            if hasattr(self, 'opencv_cap'):
                self.opencv_cap.release()
                
            cv2.destroyAllWindows()
            if self.ros_client:
                self.ros_client.terminate()

def main():
    print(cv2.__version__)
    print("ChArUco Camera Calibration Tool")
    print("Real-time ChArUco pose estimation")
    print("Board will be published as parent frame to camera_link")
    
    if PYORBEC_AVAILABLE:
        print("Using Orbbec camera (pyorbbecsdk)")
    else:
        print("Using OpenCV camera (fallback)")
    
    estimator = CharucoPoseEstimator()

    estimator.run_camera_calibration()

if __name__ == "__main__":
    main()