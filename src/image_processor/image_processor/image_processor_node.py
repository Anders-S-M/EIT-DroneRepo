import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
from .MarkerTracker import MarkerTracker

# Camera intrinsic parameters (code calibrated)
#camera_matrix = np.array([[3960, 0, 1784],
#                          [0, 3959, 1361],
#                          [0, 0, 1]], dtype=np.float32)
#dist_coeffs = np.array([-0.18, 0.07, -0.22, 0.21], dtype=np.float32)

# Camera intrinsic parameters (self calibrated)
camera_matrix = np.array([[1000, 0, 340],
                          [0, 1000, 260],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([0, 0, 0, 0], dtype=np.float32)

# Marker size in meters (known size of ArUco marker)
marker_size = 0.105 # 17.5 cm


class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.bridge = CvBridge()

        # Subscribe to the raw image topic
        self.subscription = self.create_subscription(
            Image,
            '/image',
            self.image_callback,
            1
        )

        # Publisher for simple marker xy coordinates both Aruco and Secchi
        self.xy_publisher = self.create_publisher(
            Point, 
            '/marker_xy', 
            1
        )

        # Publisher for vector to Aruco center
        self.aruco_vec_publisher = self.create_publisher(
            Point, 
            '/aruco_vec', 
            1
        )

        # Publisher for Aruco marker yaw
        self.aruco_yaw_publisher = self.create_publisher(
            Float32, 
            '/aruco_yaw', 
            1
        )
        ###################################################################
        ############ Debug and QOS related params #########################
        ###################################################################

        self.lowqual = True # make all BW to lower bandwidth when publishing and processing
        self.debug = False # publish markers to logger 
        self.visualize = False # Publish the processed image, for rosbags or sim

        # Publisher for processed images
        if self.visualize:
            self.publisher = self.create_publisher(
                Image,
                '/processed_image',
                10
            )

        # Marker tracker initialization
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.tracker = MarkerTracker(
            order=2,
            kernel_size=25,
            scale_factor=0.1
        )
        self.tracker.track_marker_with_missing_black_leg = False
        self.secchi_threshold = 0.5

    def publish_xy(self, x, y):
        """Publish the xy marker coordinates as a ROS Point message."""
        point_msg = Point()
        point_msg.x = float(x)
        point_msg.y = float(y)
        point_msg.z = 0.0  # Z is unused as secchi doesnt have depth
        self.xy_publisher.publish(point_msg)

    def publish_aruco_vec(self, x, y, z):
        """Publish the xy marker coordinates as a ROS Point message."""
        point_msg = Point()
        point_msg.x = float(x)
        point_msg.y = float(y)
        point_msg.z = float(z) 
        self.aruco_vec_publisher.publish(point_msg)

    def publish_aruco_yaw(self, yaw):
        """Publish the xy marker coordinates as a ROS Point message."""
        msg = Float32()
        msg.data = float(yaw)
        self.aruco_yaw_publisher.publish(msg)

    def calculate_yaw(self, rvec):
        # Get yaw from rotation vector

        # Convert the rotation vector to a rotation matrix using Rodrigues' formula
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Extract yaw from the rotation matrix
        # Yaw = atan2(R[1, 0], R[0, 0])
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        # Convert yaw from radians to degrees
        yaw_degrees = math.degrees(yaw)
        return yaw_degrees

    def calculate_distance(self, tvec):
        """Calculate the distance to the marker."""
        return np.linalg.norm(tvec)

    def draw_axis(self, frame, rvec, tvec, length=0.1):
        """Draw 3D axis on the frame."""
        axis_points = np.float32([
            [0, 0, 0],
            [length, 0, 0],
            [0, length, 0],
            [0, 0, length]
        ]).reshape(-1, 3)

        img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
        img_points = np.int32(img_points).reshape(-1, 2)

        origin = tuple(img_points[0])
        cv2.line(frame, origin, tuple(img_points[1]), (0, 0, 255), 2)  # X-axis (red)
        cv2.line(frame, origin, tuple(img_points[2]), (0, 255, 0), 2)  # Y-axis (green)
        cv2.line(frame, origin, tuple(img_points[3]), (255, 0, 0), 2)  # Z-axis (blue)

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        try:
            frame = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        # Make everything BW if needed for bandwidth
        if self.lowqual:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_height, frame_width = frame.shape[:2]
        origin_x = frame_width // 2
        origin_y = frame_height // 2

        # Aruco marker detection and distance calculation
        aruco_found = False
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            frame, self.dictionary, parameters=self.parameters)

        if marker_ids is not None:
            aruco_found = True
            for i, corners in enumerate(marker_corners):
                # Estimate pose of each marker
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
                marker_id = marker_ids[i][0]

                # Convert to new coordinate system with origin at the frame center
                center_x = int(np.mean(corners[0][:, 0]))
                center_y = int(np.mean(corners[0][:, 1]))
                aruco_x_centered = center_x - origin_x
                aruco_y_centered = origin_y - center_y  # Flip Y-axis for conventional Cartesian coordinates

                # Calc yaw
                yaw = self.calculate_yaw(rvec[0][0])

                # Publish marker center coordinates and aruco vector and yaw
                self.publish_xy(aruco_x_centered, aruco_y_centered)
                self.publish_aruco_vec(tvec[0][0][0],tvec[0][0][1],tvec[0][0][2])
                self.publish_aruco_yaw(yaw)
                
                # Calculate distance for log or visualizer
                if self.debug or self.visualize:
                    distance = self.calculate_distance(tvec[0])

                # If debugging publish coordinates to logger
                if self.debug:
                    self.get_logger().info(f"Aruco Marker ID: {marker_id}, Distance: {distance:.2f}m, Center: ({aruco_x_centered}, {aruco_y_centered}, Yaw: {yaw})")
                    self.get_logger().info(f"Aruco Vector, x: {tvec[0][0][0]} y: {tvec[0][0][1]} z: {tvec[0][0][2]}")

                # Visualize to topic if needed
                if self.visualize:
                    # Draw marker and axis
                    cv2.aruco.drawDetectedMarkers(frame, marker_corners)
                    # Display
                    self.draw_axis(frame, rvec[0], tvec[0], 0.1)
                    # Display distance on the frame
                    cv2.line(frame, (origin_x, origin_y), (center_x, center_y), (0,240,0), 2)
                    cv2.putText(frame, f"ID: {marker_id} Dist: {distance:.2f}m", 
                                (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"({aruco_x_centered}, {aruco_y_centered})", 
                                (center_x, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (255, 255, 0), 2)

        # Secchi marker detection
        if not aruco_found:
            
            if self.lowqual:
                pose = self.tracker.locate_marker(frame)
            else:
                grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                pose = self.tracker.locate_marker(grayscale_image)

            if pose.quality > self.secchi_threshold:
                secchi_x_centered = pose.x - origin_x
                secchi_y_centered = origin_y - pose.y
                
                # Publish marker center coordinates
                self.publish_xy(secchi_x_centered, secchi_y_centered)
                
                if self.debug:
                    # publish if good Secchi found
                    self.get_logger().info(f"Secchi Marker Center: ({secchi_x_centered}, {secchi_y_centered}), Quality: {pose.quality:.2f}")

                if self.visualize:
                    color = (0, int(255 * pose.quality), 255 - int(255 * pose.quality))
                    cv2.line(frame, (origin_x, origin_y), (pose.x, pose.y), color, 2)
                    cv2.putText(frame, f"Secchi: ({secchi_x_centered}, {secchi_y_centered}) Q: {pose.quality:.2f}",
                                (pose.x, pose.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, color, 2)

        if self.visualize:
            # Publish the processed image
            processed_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher.publish(processed_image_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

