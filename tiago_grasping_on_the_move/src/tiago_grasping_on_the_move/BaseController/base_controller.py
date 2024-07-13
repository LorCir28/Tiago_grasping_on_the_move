#!/usr/bin/env python

import math
import os
import pickle as pkl
import rospy 
import time 

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist 
from .plots import *
from .quaternion import *
from .tangent_point import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

'''
        eps_c: closest approach pose (computed as an offline pose as soon as the robot spawns)
        eps_t: target pose
        r_c: closest approach radius
        rho: distance between eps_c and eps_b
        eps_b: current base pose
        alpha: angle between rho and the robot's current forward direction (the latter identified by theta)
        theta: base oreintation wrt the x-axis of the world RF
        d: distance between eps_b and eps_t
        gamma: theta + alpha (computed as an offline angle as soon as the robot spawns)
        eps_c_n: closest approach pose to the next target (computed as an offline pose as soon as the robot spawns)
        eps_n: next target pose (e.g., box)
        rho_n: distance between eps_c_n and eps_b
        d_n: distance between eps_b and eps_n
        beta: angle between rho_n and the robot's current forward direction (the latter identified by theta)
'''



class BaseController:
    def __init__(self):
        # Node settings
        rospy.init_node('cust_base_vel_controller', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)
        self.velocity_subscriber = rospy.Subscriber('/mobile_base_controller/cmd_vel', Twist, self.base_vel_callback)
        self.base_odom_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, self.base_odom_callback)
        self.target_odom_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, self.target_odom_callback)
        self.rate = rospy.Rate(10)  # 10 Hz

        # Goal thresolds
        self.is_desired_pose_reached = False
        self.is_desired_orientation_updated = False
        self.second_phase = False
        self.set_init_rho = False
        # self.gripper_threshold = 1.625 # RRTstarkConfigDefault
        self.gripper_threshold = 1.65 # PRMstarkConfigDefault
        # self.gripper_threshold = 1.765 # SBLkConfigDefault
        # self.threshold = 1.85 # Grasping from the front
        self.threshold = 1.75 # Grasping from the front
        # self.threshold = 0.57 # Grasping from the top
        self.gripper_threshold_n = 2.30
        self.threshold_n = 2.20 # Leave the ball from the frontS
        
        # Closest approach circle
        self.r_c = 0.6
        self.d = 1.0
        self.d_n = 1.0

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
        self.eps_c_n_x = 0
        self.eps_c_n_y = 0

        # Next target position
        self.eps_n_x = 0.0
        self.eps_n_y = 0.0

        # Errors
        self.alpha = 1.0
        self.beta = 1.0
        self.rho = 1.0
        self.rho_n = 1.0
        self.history = {
            "v" : [],
            "w" : [],
            "alpha" : [],
            "rho" : [],
            "beta": [],
            "rho_n": [],
            "time" : 0.0,
            "x" : [],
            "y" : []
        }

        # Controller
        self.v_b = 0.3
        self.k_alpha = 4
        self.k_beta = 2.2

    def base_odom_callback(self, msg):
        for i in range(len(msg.name)):
            if msg.name[i] == 'tiago':    
                self.eps_b_x = msg.pose[i].position.x
                self.eps_b_y = msg.pose[i].position.y
                self.orientation_quaternion.set_coeffs(
                    w = msg.pose[i].orientation.w,
                    x = msg.pose[i].orientation.x, 
                    y = msg.pose[i].orientation.y,
                    z = msg.pose[i].orientation.z
                )
                rad, _ = self.orientation_quaternion.to_euler_angles()
                self.theta = rad[2]

                self.d = math.sqrt((self.eps_t_x - self.eps_b_x)**2 + (self.eps_t_y - self.eps_b_y)**2)
                self.d_n = math.sqrt((self.eps_n_x - self.eps_b_x)**2 + (self.eps_n_y - self.eps_b_y)**2)

                # if self.second_phase == False:
                self.rho = 0 if self.r_c > self.d else math.sqrt(self.d**2 - self.r_c**2)
                    # if self.set_init_rho == False and self.rho != 0:
                        # print('------------------------------------base rho init:', self.rho)
                        # self.set_init_rho = True
                self.alpha = math.atan2(self.eps_b_y - self.eps_c_y, self.eps_b_x - self.eps_c_x) + math.pi - self.theta

                self.beta = math.atan2(self.eps_b_y - self.eps_c_n_y, self.eps_b_x - self.eps_c_n_x) + math.pi - self.theta
                self.rho_n = 0 if self.r_c > self.d_n else math.sqrt(self.d_n**2 - self.r_c**2)
                
                self.history["alpha"].append(self.alpha)
                self.history["beta"].append(self.beta)
                self.history["rho"].append(self.rho)
                self.history["rho_n"].append(self.rho_n)
                self.history["x"].append(self.eps_b_x)
                self.history["y"].append(self.eps_b_y)
                
                self.eps_c_x,self.eps_c_y = tangent_point(self.eps_b_x,self.eps_b_y,self.eps_t_x,self.eps_t_y,self.r_c)
                self.eps_c_n_x,self.eps_c_n_y = tangent_point(self.eps_b_x,self.eps_b_y,self.eps_n_x,self.eps_n_y,self.r_c)

    def close_gripper(self):
        # publish gripper status on joint trajectory when TIAGo close gripper 
        pub_gripper_controller = rospy.Publisher(
            '/gripper_controller/command', JointTrajectory, queue_size=1)

        # loop continues until the grippers close well
        # for _ in range(10):
        trajectory = JointTrajectory()
        # call joint group for take object 
        trajectory.joint_names = ['gripper_left_finger_joint', 'gripper_right_finger_joint']

        trajectory_points = JointTrajectoryPoint()
        # define the distance to the right & left of the two gripper w.r.t the object
        trajectory_points.positions = [0.000, 0.000]
        # time action  
        trajectory_points.time_from_start = rospy.Duration(1.0)

        trajectory.points.append(trajectory_points)

        pub_gripper_controller.publish(trajectory)
            # interval to start next movement
            # rospy.sleep(0.1)  

    def open_gripper(self):
        """
            function for open gripper of end effector when TIAGo made mission  
        """
        # publish gripper status on joint trajectory when TIAGo open gripper
        pub_gripper_controller = rospy.Publisher(
            '/gripper_controller/command', JointTrajectory, queue_size=1)

        # loop continues until the grippers open &  object is released
        # for _ in range(10):
        trajectory = JointTrajectory()
        trajectory.joint_names = ['gripper_left_finger_joint', 'gripper_right_finger_joint']

        trajectory_points = JointTrajectoryPoint()
        # define the distance to the right & left of the two gripper for opening
        trajectory_points.positions = [0.044, 0.044]
        # time action  
        trajectory_points.time_from_start = rospy.Duration(1.0)

        trajectory.points.append(trajectory_points)

        pub_gripper_controller.publish(trajectory)
        # interval to start next movement
        # rospy.sleep(0.1)

    def target_odom_callback(self, msg):
        for i in range(len(msg.name)):
            if msg.name[i] == 'unit_sphere':    
                self.eps_t_x = msg.pose[i].position.x
                self.eps_t_y = msg.pose[i].position.y
            if msg.name[i] == 'unit_box_1':    
                self.eps_n_x = msg.pose[i].position.x
                self.eps_n_y = msg.pose[i].position.y

    def base_vel_callback(self, msg):
        self.history["v"].append(msg.linear.x)
        self.history["w"].append(msg.angular.z)

    def final_errors(self,history):
        plot_errors(history)

    def move_base_to_desired_orientation(self):
        rospy.sleep(2.5)
        start_time = time.time()

        while not rospy.is_shutdown() and self.is_desired_pose_reached == False:
            if self.rho <= self.gripper_threshold:
                self.close_gripper()
            
            if self.rho > self.threshold:
                vel_msg = Twist()
                vel_msg.linear.x = self.v_b
                vel_msg.angular.z = (self.k_alpha * self.alpha) * (self.v_b / self.rho)
                self.velocity_publisher.publish(vel_msg)
                self.rate.sleep()

                # self.history["v"].append(self.v_b)
                # self.history["w"].append(vel_msg.angular.z)
            else:
                # self.is_desired_pose_reached = True
                # end_time = time.time()
                # self.history["time"] = round(end_time - start_time,1)

                # CODICE PER ARRIVARE AL NEXT TARGET (e.g, box)
                # self.second_phase = True # to not upload rho and alpha during this second phase

                if self.rho_n <= self.gripper_threshold_n:
                    self.open_gripper()

                if self.rho_n > self.threshold_n:
                    vel_msg = Twist()
                    vel_msg.linear.x = self.v_b
                    vel_msg.angular.z = (self.k_beta * self.beta) * (self.v_b / self.rho_n)
                    self.velocity_publisher.publish(vel_msg)
                    self.rate.sleep()

                    # self.history["v"].append(self.v_b)
                    # self.history["w"].append(vel_msg.angular.z)
                else:
                    self.is_desired_pose_reached = True
                    end_time = time.time()
                    print("********************", round(end_time - start_time,1))
                    vel_msg = Twist()
                    vel_msg.linear.x = 0.0
                    vel_msg.angular.z = 0.0
                    self.velocity_publisher.publish(vel_msg)
                    self.rate.sleep()
                    self.is_desired_pose_reached = True

        # self.final_errors(self.history)
    
        return self.history


def main():
    controller = BaseController()
    history = controller.move_base_to_desired_orientation()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    savefile_path = "results/go_qunitic_base_controller.pkl"
    output_path = os.path.join(script_dir, savefile_path)
    
    with open(output_path, "wb") as pkl_file:
        pkl.dump(history,pkl_file,protocol=pkl.HIGHEST_PROTOCOL)