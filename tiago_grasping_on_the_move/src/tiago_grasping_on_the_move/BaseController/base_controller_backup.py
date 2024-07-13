#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import rospy
import time 

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist 
from .plots import *
from .quaternion import *
from .tangent_point import *

'''
        eps_c: closest approach pose (computed as an offline pose as soon as the robot spawns)
        eps_t: target pose
        r_c: closest approach radius
        ro: distance between eps_c and eps_b
        eps_b: current base pose
        alpha: angle between ro and the robot's current forward direction (the latter identified by theta)
        theta: base oreintation wrt the x-axis of the world RF
        d: distance between eps_b and eps_t
        gamma: theta + alpha (computed as an offline angle as soon as the robot spawns)
'''

class BaseController:
    def __init__(self):
        # Node settings
        rospy.init_node('cust_base_vel_controller', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)
        self.base_odom_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, self.base_odom_callback)
        self.target_odom_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, self.target_odom_callback)
        self.rate = rospy.Rate(10)  # 10 Hz

        # Goal thresolds
        self.is_desired_pose_reached = False
        self.is_desired_orientation_updated = False
        self.threshold = 0.40 # Grasping from the front
        # self.threshold = 0.57 # Grasping from the top
        
        # Closest approach circle
        self.r_c = 0.6
        self.d = 1.0

        # TIAgo initial configuration
        self.eps_b_x = 0.0
        self.eps_b_y = 0.0
        self.theta = 0.0 
        self.orientation_quaternion = Quaternion()

        # Target position
        self.eps_t_x = 0.0
        self.eps_t_y = 0.0

        # Closest approach position
        self.eps_c_x = 0
        self.eps_c_y = 0

        # Errors
        self.alpha = 1.0
        self.beta = 0.0
        self.rho = 1.0
        self.history = {
            "v" : [],
            "w" : [],
            "alpha" : [],
            "rho" : [],
            "time" : 0.0
        }

        # Controller
        self.v_b = 0.5
        self.k_alpha = 4
        self.k_beta = -1.5

    def base_odom_callback(self, msg):
        self.eps_b_x = msg.pose[3].position.x
        self.eps_b_y = msg.pose[3].position.y
        self.orientation_quaternion.set_coeffs(
            w = msg.pose[3].orientation.w,
            x = msg.pose[3].orientation.x, 
            y = msg.pose[3].orientation.y,
            z = msg.pose[3].orientation.z
        )
        rad, _ = self.orientation_quaternion.to_euler_angles()
        self.theta = rad[2]

        self.d = math.sqrt((self.eps_t_x - self.eps_b_x)**2 + (self.eps_t_y - self.eps_b_y)**2)
        self.rho = 0 if self.r_c > self.d else math.sqrt(self.d**2 - self.r_c**2)
        self.alpha = math.atan2(self.eps_b_y - self.eps_c_y, self.eps_b_x - self.eps_c_x) + math.pi - self.theta
        
        self.history["alpha"].append(self.alpha)
        self.history["rho"].append(self.rho)
        
        self.eps_c_x,self.eps_c_y = tangent_point(self.eps_b_x,self.eps_b_y,self.eps_t_x,self.eps_t_y,self.r_c)

    def target_odom_callback(self, msg):
        self.eps_t_x = msg.pose[2].position.x
        self.eps_t_y = msg.pose[2].position.y

    def move_base_to_desired_orientation(self):
        start_time = time.time()

        while not rospy.is_shutdown() and self.is_desired_pose_reached == False:
            if self.rho <= self.threshold:
                self.is_desired_pose_reached = True
                end_time = time.time()
                self.history["time"] = round(end_time - start_time,1)
                self.final_errors(self.history)
            else:
                vel_msg = Twist()
                vel_msg.linear.x = self.v_b
                vel_msg.angular.z = (self.k_alpha * self.alpha) * (self.v_b / self.rho)
                self.velocity_publisher.publish(vel_msg)
                self.rate.sleep()

                self.history["v"].append(self.v_b)
                self.history["w"].append(vel_msg.angular.z)
    
    def final_errors(self,history):
        plot_errors(history)

# if __name__ == '__main__':
#     try:
#         controller = BaseController()
#         controller.move_base_to_desired_orientation()
#     except rospy.ROSInterruptException:
#         pass


def main():
    controller = BaseController()
    controller.move_base_to_desired_orientation()
