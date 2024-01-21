# Student name: Darian Irani

import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseStamped, TransformStamped
from std_msgs.msg import String, Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
import matplotlib.pyplot as plt
import time
from tf2_msgs.msg import TFMessage
from copy import copy
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration
from sm_algo import split_and_merge

# Further info:
# On markers: http://wiki.ros.org/rviz/DisplayTypes/Marker
# Laser Scan message: http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/LaserScan.html

class CodingExercise3(Node):
    
    def __init__(self):
        super().__init__('CodingExercise3')

        self.ranges = [] # lidar measurements
        
        self.point_list = [] # A list of points to draw lines
        self.line = Marker()
        self.line_marker_init(self.line)

        self.threshold_d = 9.5
        self.threshold_a = 0.75
        self.marker = 0


        # Ros subscribers and publishers
        self.subscription_ekf = self.create_subscription(Odometry, 'terrasentia/ekf', self.callback_ekf, 10)
        self.subscription_scan = self.create_subscription(LaserScan, 'terrasentia/scan', self.callback_scan, 10)
        self.pub_lines = self.create_publisher(Marker, 'lines', 10)
        
        
        self.line = Marker()
        self.line_marker_init(self.line)

    
    def callback_ekf(self, msg):
        # You will need this function to read the translation and rotation of the robot with respect to the odometry frame
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.current_pose = msg.pose.pose
        pass

    def callback_scan(self, msg):
        # # LiDAR data to Cartesian
        # lidar_angle = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        # x, y = self.polar_to_cartesian(np.array(msg.ranges), lidar_angle)
        # line_seg = self.split_and_merge(x, y)
        # data_test = np.hstack((x, y))

    
        self.ranges = np.array([list(msg.ranges)]).T # Lidar measurements
        rho_test = self.ranges
        n = 1081
        # print("n", n)
        print("some-ranges:", self.ranges)
        # print("Number of ranges:", len(self.ranges))
        theta_test = (math.pi / 180) * np.linspace(-135, 135, n).reshape(-1, 1)
        # print("theta test", theta_test)

        # Convert polar coordinates to Cartesian
        x_test = rho_test * np.cos(theta_test)
        y_test = rho_test * np.sin(theta_test)

        # Combine x and y into a single array
        data_test = np.hstack((x_test, y_test))

        # Apply Split-and-Merge algorithm
        threshold_test = 0.5
        self.result_points = split_and_merge(data_test, threshold_test)
        self.filtered_points = self.result_points[(self.result_points[:, 0] <= 30) & (self.result_points[:, 1] <= 30)&(self.result_points[:, 0] >= -30) & (self.result_points[:, 1] >= -30)]

        plt.figure(figsize=(8, 8))
        plt.scatter(data_test[:, 0], data_test[:, 1], label='Original Data', color='blue')
        plt.plot(self.filtered_points[:, 0], self.filtered_points[:, 1], label='Detected Line', color='red')
        plt.title('Split-and-Merge Line Fitting')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()    
   
    def polar_to_cartesian(self, rho, theta):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y

    
    def transform_point(self, x, y):    # q4: mapping
        # Transform a point from the LIDAR frame to the odometry frame using the current pose
        # Quaternion to Euler conversion
        orientation_q = self.current_pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        yaw = self.euler_from_quaternion(orientation_list)

        # Rotation matrix around Z axis
        rot_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])

        # Rotate the point
        point = np.array([x, y])
        rotated_point = rot_matrix.dot(point)

        # Translate the point
        x_transformed = rotated_point[0] + self.current_pose.position.x
        y_transformed = rotated_point[1] + self.current_pose.position.y

        return Point(x=x_transformed, y=y_transformed, z=0.0)
        
    def euler_from_quaternion(self, orientation_list):
        # Converts quaternion (x, y, z, w) to euler roll, pitch, yaw
        x, y, z, w = orientation_list
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw


    def line_marker_init(self, line):
        line.header.frame_id="/odom"
        line.header.stamp=self.get_clock().now().to_msg()

        line.ns = "markers"
        line.id = 0

        line.type=Marker.LINE_LIST
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0

        line.scale.x = 0.05
        line.scale.y= 0.05
        
        line.color.r = 1.0
        line.color.a = 1.0
        #line.lifetime = 0
        

def main(args=None):
    rclpy.init(args=args)

    cod3_node = CodingExercise3()
    
    rclpy.spin(cod3_node)

    cod3_node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()
