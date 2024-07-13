#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
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
        self.pub_gripper_controller = rospy.Publisher('/gripper_controller/command', JointTrajectory, queue_size=1)     
        self.rate = rospy.Rate(10)  # 10 Hz

        # Goal flags
        self.is_desired_pose_reached = False
        self.is_desired_orientation_updated = False
        self.second_phase = False
        self.set_init_rho = False
        
        # Grasping threshold -> SBLK = 1.76 | RRT = 1.62 | PRM = 1.59
        self.gripper_threshold = 1.76
        
        # Controller switch threshold
        self.threshold = 2.21 
        
        # Dropping threshold
        self.gripper_threshold_n = 2.42

        # Stopping threshold
        self.threshold_n = 2.60 # Leave the ball from the frontS
        
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
        
        # Controller
        self.v_b = 0.3
        self.k_alpha = 4
        self.k_beta = 2.2

        # On the stop variables
        self.wait = True        # If TIAGo has to be stopped in the second phase
        self.stopped = False    # If TIAGo is stopped

        # Workspace save
        self.history = {
            # TIAGo configuration
            "x" : [],
            "y" : [],
            "theta": [],

            # TIAGo velocities
            "v" : [],
            "w" : [],

            # Base controller
            "alpha" : [],
            "rho" : [],
            "beta" : [],
            "rho_n" : [],
            "v_b" : 0.0,
            "k_alpha" : 0.0,
            "k_beta" : 0.0,

            # Environment
            "xi_target" : np.array([0,0]),
            "xi_target_closest" : np.array([0,0]),
            "xi_next" : np.array([0,0]),
            "xi_next_closest" : np.array([0,0]),
            "r_c" : 0.0,
            "time" : 0.0,
        }

    def base_odom_callback(self, msg):
        for i in range(len(msg.name)):
            if msg.name[i] == 'tiago':
                # TIAGo mobile base configuration
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

                # Targets distances
                self.d = math.sqrt((self.eps_t_x - self.eps_b_x)**2 + (self.eps_t_y - self.eps_b_y)**2)
                self.d_n = math.sqrt((self.eps_n_x - self.eps_b_x)**2 + (self.eps_n_y - self.eps_b_y)**2)
                
                # Grasping target errors
                self.rho = 0 if self.r_c > self.d else math.sqrt(self.d**2 - self.r_c**2)    
                self.alpha = math.atan2(self.eps_b_y - self.eps_c_y, self.eps_b_x - self.eps_c_x) + math.pi - self.theta

                # Next target errors
                self.beta = math.atan2(self.eps_b_y - self.eps_c_n_y, self.eps_b_x - self.eps_c_n_x) + math.pi - self.theta
                self.rho_n = 0 if self.r_c > self.d_n else math.sqrt(self.d_n**2 - self.r_c**2)
                
                # Closest approach points
                self.eps_c_x,self.eps_c_y = tangent_point(self.eps_b_x,self.eps_b_y,self.eps_t_x,self.eps_t_y,self.r_c)
                self.eps_c_n_x,self.eps_c_n_y = tangent_point(self.eps_b_x,self.eps_b_y,self.eps_n_x,self.eps_n_y,self.r_c)
                
                # Workspace update
                self.history["alpha"].append(self.alpha)
                self.history["beta"].append(self.beta)
                self.history["rho"].append(self.rho)
                self.history["rho_n"].append(self.rho_n)
                self.history["x"].append(msg.pose[i].position.x)
                self.history["y"].append(msg.pose[i].position.y)
                self.history["theta"].append(self.theta)    
                # if self.stopped:
                #     self.history["v"].append(0.0)
                #     self.history["w"].append(0.0)
                
    def close_gripper(self):
        # Build gripper message
        trajectory = JointTrajectory()
        trajectory.joint_names = ['gripper_left_finger_joint', 'gripper_right_finger_joint']
        trajectory_points = JointTrajectoryPoint()
        trajectory_points.positions = [0.000, 0.000]
        trajectory_points.time_from_start = rospy.Duration(0.5)
        trajectory.points.append(trajectory_points)

        # Send gripper message
        self.pub_gripper_controller.publish(trajectory)
            
    def open_gripper(self):
        # Build gripper message
        trajectory = JointTrajectory()
        trajectory.joint_names = ['gripper_left_finger_joint', 'gripper_right_finger_joint']
        trajectory_points = JointTrajectoryPoint()
        trajectory_points.positions = [0.044, 0.044]
        trajectory_points.time_from_start = rospy.Duration(0.5)
        trajectory.points.append(trajectory_points)
        
        # Send gripper message
        self.pub_gripper_controller.publish(trajectory)
        
    def final_errors(self,history):
        plot_errors(history)

    def target_odom_callback(self, msg):
        for i in range(len(msg.name)):
            if msg.name[i] == 'unit_sphere':    
                # Grasping target position
                self.eps_t_x = msg.pose[i].position.x
                self.eps_t_y = msg.pose[i].position.y
            if msg.name[i] == 'unit_box_1':
                # Next target position:    
                self.eps_n_x = msg.pose[i].position.x
                self.eps_n_y = msg.pose[i].position.y

    def base_vel_callback(self, msg):
        # Current TIAGo velcities
        if self.stopped:
            self.history["v"].append(0.0)
            self.history["w"].append(0.0)
        else:
            self.history["v"].append(msg.linear.x)
            self.history["w"].append(msg.angular.z)

    def wait_for_arm(self):
        while(self.wait):
            # Stop TIAGo to wait the grasping
            self.stopped = True
            vel_msg = Twist()
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()

            # Inform the user when restart TIAGo
            stop_message = "Wait until arm is place, than write \"go\" to restart TIAGo base trajectory: "
            release_signal = input(stop_message)

            if release_signal.lower() == "go":
                # Restart TIAGo motion when the grasping is secured
                self.wait = False
                self.stopped = False
        # tempo_iniziale = time.time()
        # while self.wait:
        #     # Ottieni il tempo corrente
        #     tempo_corrente = time.time()
            
        #     # Calcola la differenza di tempo
        #     differenza_tempo = tempo_corrente - tempo_iniziale
        #     vel_msg = Twist()
        #     vel_msg.linear.x = 0.0
        #     vel_msg.angular.z = 0.0
        #     self.velocity_publisher.publish(vel_msg)
        #     self.rate.sleep()
        
        #     # Verifica se sono trascorsi 24 secondi
        #     if differenza_tempo >= 24:
        #         self.wait = False
        #         self.stopped = False
        #         break
                

    def move_base_to_desired_orientation(self):
        # Wait to fill the values from callbacks
        rospy.sleep(5.0)
        # Start to measure the simulation time
        start_time = time.time()
    
        while not rospy.is_shutdown() and self.is_desired_pose_reached == False:
            if self.rho > self.threshold:
                # First controller to reach grasping target
                vel_msg = Twist()
                vel_msg.linear.x = self.v_b
                vel_msg.angular.z = (self.k_alpha * self.alpha) * (self.v_b / self.rho)
                self.velocity_publisher.publish(vel_msg)
                self.rate.sleep()
            else:
                # Stop TIAGo before the second controller starts, to grasp the target
                self.wait_for_arm()                
                
                # When the grasping is secured, take the target
                self.close_gripper()

                # Second controller to reach next target
                self.second_phase = True
                if self.rho_n > self.threshold_n:
                    vel_msg = Twist()
                    vel_msg.linear.x = self.v_b
                    vel_msg.angular.z = (self.k_beta * self.beta) * (self.v_b / self.rho_n)
                    self.velocity_publisher.publish(vel_msg)
                    self.rate.sleep()
                else:
                    # Simulation is ended
                    self.is_desired_pose_reached = True
                    
                    # Stop TIAGo definitively
                    vel_msg = Twist()
                    vel_msg.linear.x = 0.0
                    vel_msg.angular.z = 0.0
                    self.velocity_publisher.publish(vel_msg)
                    self.rate.sleep()

                    # Release the grasped target
                    self.open_gripper()

                    # Workspace update
                    # self.history["v"].append(self.v_b)
                    # self.history["w"].append(vel_msg.angular.z)
                    self.history["time"] = round(time.time() - start_time,2)
                    # self.history["v_b"] = self.v_b
                    self.history["k_alpha"] = self.k_alpha
                    self.history["k_beta"] = self.k_beta
                    self.history["xi_target"] = np.array([round(self.eps_t_x,2),round(self.eps_t_y,2)])
                    self.history["xi_target_closest"] = np.array([round(self.eps_c_x,2),round(self.eps_c_y,2)])
                    self.history["xi_next"] = np.array([round(self.eps_n_x,2),round(self.eps_n_y,2)])
                    self.history["xi_next_closest"] = np.array([round(self.eps_c_n_x,2),round(self.eps_c_n_y,2)])
                    self.history["r_c"] = round(self.r_c,2)
                    
        # self.final_errors(self.history)
        return self.history


def main():
    controller = BaseController()
    history = controller.move_base_to_desired_orientation()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    savefile_path = "results/ost_vel.pkl"
    output_path = os.path.join(script_dir, savefile_path)
    
    with open(output_path, "wb") as pkl_file:
        pkl.dump(history,pkl_file,protocol=pkl.HIGHEST_PROTOCOL)