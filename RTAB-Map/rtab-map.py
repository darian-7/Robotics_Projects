# Implement a ROS node to get the coordinates (x,y,z) of the estimated trajectories by RTAB map (/rtabmap/odom) 
# and the ground truth trajectories (/terrasentia/ekf). Store the data in .txt files. Add the 3D plots of these trajectories to your report.

import numpy as np
import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from utils import quaternion_from_euler, lonlat2xyz #edit according to your package's name


class RTAB_Map(Node):

    def __init__(self):
        super().__init__('RTAB_Map')

        self.ground_truth_data = [0.0, 0.0, 0.0]
        self.estimated_data = [0.0, 0.0, 0.0]

        self.odom_x = []
        self.odom_y = []
        self.odom_z = []

        self.ekf_x = []
        self.ekf_y = []
        self.ekf_z = []

        # Open files in write mode
        self.file1 = open("Odometry.txt", "w")
        self.file2 = open("EKF.txt", "w")

        # Subscribers
        self.subscription_rtabmap_odom = self.create_subscription(Odometry, '/rtabmap/odom', self.odom_callback, 10)
        self.subscription_terrasentia_ekf = self.create_subscription(Odometry, '/terrasentia/ekf', self.ekf_callback, 10)

        self.timer_ekf = self.create_timer(0.1, self.callback)
        self.timer_plot = self.create_timer(1, self.data)

    def ekf_callback(self, msg):
        self.ground_truth_data[0] = msg.pose.pose.position.x
        self.ground_truth_data[1] = msg.pose.pose.position.y
        self.ground_truth_data[2] = msg.pose.pose.position.z

    def odom_callback(self, msg):
        self.estimated_data[0] = msg.pose.pose.position.x
        self.estimated_data[1] = msg.pose.pose.position.y
        self.estimated_data[2] = msg.pose.pose.position.z

    def callback(self):
        self.data()

    def data(self):
        file1 = open("Odometry.txt", "a")
        file2 = open("EKF.txt", "a")

        print("ground truth", self.ground_truth_data)
        print("estimate", self.estimated_data)
        self.ekf_x.append(self.ground_truth_data[0])
        self.ekf_y.append(self.ground_truth_data[1])
        self.ekf_z.append(self.ground_truth_data[2])

        self.odom_x.append(self.estimated_data[0])
        self.odom_y.append(self.estimated_data[1])
        self.odom_z.append(self.estimated_data[2])

        self.file1.write(f"{self.estimated_data[0]}, {self.estimated_data[1]}, {self.estimated_data[2]}\n")
        self.file2.write(f"{self.ground_truth_data[0]}, {self.ground_truth_data[1]}, {self.ground_truth_data[2]}\n")

    def plot():
        ekf_file_path = 'ekf.txt'
        ekf_data = np.loadtxt(ekf_file_path, delimiter=',')

        x_ekf = ekf_data[:, 0]
        y_ekf = ekf_data[:, 1]
        z_ekf = ekf_data[:, 2]

        odom_file_path = 'Odom.txt'
        odom_data = np.loadtxt(odom_file_path, delimiter=',')

        x_odom = odom_data[:, 0]
        y_odom = odom_data[:, 1]
        z_odom = odom_data[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ekf_plot = ax.plot(x_ekf, y_ekf, z_ekf, color='b', label='EKF')
        odom_plot = ax.plot(x_odom, y_odom, z_odom, color='r', label='Odometry')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.legend()
        plt.show()

    def destroy_node(self):
        # Close the files when destroying the node
        self.file1.close()
        self.file2.close()
        super().destroy_node()
    
    def compute_rmse(ekf_rmse, odom_rmse):
        ekf_rmse_path = 'ekf_rmse.txt'
        ekf_rmse = np.loadtxt(ekf_rmse_path, delimiter=',')
        odom_rmse_path = 'odom_rmse.txt'
        odom_rmse = np.loadtxt(odom_rmse_path, delimiter=',')
        N = 197
        rmse = np.sqrt(np.mean(((ekf_rmse - odom_rmse) ** 2)/N))
        print(rmse)
        return rmse
        
        
def main(args=None):
    rclpy.init(args=args)
    rtab_node = RTAB_Map()
    rclpy.spin(rtab_node)
    rtab_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

