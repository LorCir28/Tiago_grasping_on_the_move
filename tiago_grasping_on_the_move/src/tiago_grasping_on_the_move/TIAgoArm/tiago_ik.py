#!/usr/bin/env python3

import math
import moveit_commander 
import numpy as np
import os
import pickle as pkl
import rospy
import sys
import tf.transformations
import tf2_geometry_msgs
import tf2_ros

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from tiago_grasping_on_the_move.BaseController.tangent_point import tangent_point
from tiago_grasping_on_the_move.BaseController.quaternion import *


class ArmController:
    def __init__(self):
        # Node settings
        rospy.init_node('plan_arm_torso_ik', anonymous=True)
        moveit_commander.roscpp_initialize(sys.argv)
        self.group_arm_torso = moveit_commander.MoveGroupCommander("arm_torso")
        self.base_odom_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, self.base_odom_callback)
        self.target_odom_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, self.target_odom_callback)
        self.rate = rospy.Rate(10) # 10 Hz

        # delay
        self.init_d = 1.0
        self.init_rho = 1.0
        self.set_init_rho = False

        # Target position
        self.eps_t_x = 0.0
        self.eps_t_y = 0.0
        self.eps_t_z = 0.0

        # Base position
        self.eps_b_x = 0.0
        self.eps_b_y = 0.0
        self.theta = 0.0

        # Closest approach point
        self.r_c = 0.6
        self.eps_c_x = 0.0
        self.eps_c_y = 0.0
        self.d = 0.0
        self.rho = 0.0
        self.alpha = 0.0
        self.orientation_quaternion = Quaternion()

        # Grasping goal
        self.from_frame = "odom"
        self.to_frame = "base_footprint"
        self.ee_orientation_r = -1.5
        self.ee_orientation_p = 0.5
        self.ee_orientation_y = -0.0

        # Trajectory planner
        # self.planner = "SBLkConfigDefault"
        # self.planner = "RRTstarkConfigDefault"
        self.planner = "PRMstarkConfigDefault"

        self.history = {
            "ee_pos_x" : [],
            "ee_pos_y" : [],
            "ee_pos_z" : [],
            "ee_truth_x" : [],
            "ee_truth_y" : [],
            "ee_truth_z" : []
        }

         # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        #     "q1" : [],
        #     "q2" : [],
        #     "q3" : [],
        #     "q4" : [],
        #     "q5" : [],
        #     "q6" : [],
        #     "q7" : []
        # }
    
    def get_end_effector_position(self):
        current_pose = self.group_arm_torso.get_current_pose()
        return current_pose.pose.position.x,current_pose.pose.position.y,current_pose.pose.position.z

    def transform_point_ee(self,point, from_frame, to_frame):
        # Crea un buffer per tf2
        # tf_buffer = tf2_ros.Buffer()
        # listener = tf2_ros.TransformListener(tf_buffer)
        
        # Definisci il punto da trasformare
        point_stamped = PointStamped()
        point_stamped.header.stamp = rospy.Time.now()
        point_stamped.header.frame_id = from_frame
        point_stamped.point.x = point[0]
        point_stamped.point.y = point[1]
        point_stamped.point.z = point[2]
        
        try:
            # Ottieni la trasformazione dal frame di origine al frame di destinazione
            transform = self.tf_buffer.lookup_transform(to_frame, from_frame, rospy.Time(0), rospy.Duration(1.0))
            
            # Applica la trasformazione al punto
            transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            
            return transformed_point.point.x, transformed_point.point.y, transformed_point.point.z
        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Transform error: %s", str(e))
            return None

    def transform_point(self,point, from_frame, to_frame):
        # Crea un buffer per tf2
        # tf_buffer = tf2_ros.Buffer()
        # listener = tf2_ros.TransformListener(tf_buffer)
        
        # Definisci il punto da trasformare
        point_stamped = PointStamped()
        point_stamped.header.stamp = rospy.Time.now()
        point_stamped.header.frame_id = from_frame
        point_stamped.point.x = point[0]
        point_stamped.point.y = point[1]
        point_stamped.point.z = point[2]
        
        try:
            # Ottieni la trasformazione dal frame di origine al frame di destinazione
            transform = self.tf_buffer.lookup_transform(to_frame, from_frame, rospy.Time(0), rospy.Duration(1.0))
            
            # Applica la trasformazione al punto
            transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            
            return self.project_into_workspace(transformed_point.point.x, transformed_point.point.y, transformed_point.point.z)
        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Transform error: %s", str(e))
            return None
    
    def project_into_workspace(self,x,y,z):
        self.eps_c_x,self.eps_c_y = tangent_point(self.eps_b_x,self.eps_b_y,self.eps_t_x,self.eps_t_y,self.r_c)
        self.d = math.sqrt((self.eps_t_x - self.eps_b_x)**2 + (self.eps_t_y - self.eps_b_y)**2)
        self.rho = math.sqrt(self.d**2 - self.r_c**2)
        self.alpha = math.atan2(self.eps_b_y - self.eps_c_y, self.eps_b_x - self.eps_c_x) + math.pi - self.theta

        x_c = self.rho * math.cos(self.alpha)
        y_c = self.rho * math.sin(self.alpha)

        x_out = x - x_c + self.r_c
        # y_out = y - y_c + 0.15 # SBLK
        y_out = y - y_c - 0.023 # RRT
        # y_out = y - y_c - 0.015 # PRM
        # z_out = z + 0.11 # SBLK
        z_out = z + 0.13 # RRT
        # z_out = z + 0.12 # RRT, PRM
        
        return x_out,y_out,z_out

    def target_odom_callback(self, msg):
        for i in range(len(msg.name)):
            if msg.name[i] == 'unit_sphere':    
                self.eps_t_x = msg.pose[i].position.x
                self.eps_t_y = msg.pose[i].position.y            
                self.eps_t_z = msg.pose[i].position.z          
        x_ee, y_ee, z_ee = self.get_end_effector_position()
        
        
        transformed_coordinates = self.transform_point_ee(
            np.array([x_ee,y_ee,z_ee]),  
            from_frame = "base_footprint", 
            to_frame = "odom"
        )
        
        self.history["ee_pos_x"].append(transformed_coordinates[0])  
        self.history["ee_pos_y"].append(transformed_coordinates[1])  
        self.history["ee_pos_z"].append(transformed_coordinates[2])
                
        self.history["ee_truth_x"].append(self.eps_t_x)  
        self.history["ee_truth_y"].append(self.eps_t_y)  
        self.history["ee_truth_z"].append(self.eps_t_z)  

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

            
                # self.history["q1"].append(msg.pose[i].position.x)
                # self.history["q2"].append(msg.pose[i].position.y)
                # self.history["q3"].append(msg.pose[i].position.z)
                # self.history["q4"].append(msg.pose[i].orientation.x)
                # self.history["q5"].append(msg.pose[i].orientation.y)
                # self.history["q6"].append(msg.pose[i].orientation.z)
                # self.history["q7"].append(msg.pose[i].orientation.w)

        if self.set_init_rho == False:
            self.init_d = math.sqrt((self.eps_t_x - self.eps_b_x)**2 + (self.eps_t_y - self.eps_b_y)**2)
            self.init_rho = 0 if self.r_c > self.d else math.sqrt(self.d**2 - self.r_c**2)
        
        if self.init_rho != 0:
            self.set_init_rho = True

    def move_arm(self):
        """
            Grasping from the front: [x,y,z] = [0.35,0.60,0.78], [R,P,Y] = [-1.5,0.5,-0.0]
            Grasping from the top:   [x,y,z] = [0.35,0.62,0.85], [R,P,Y] = [-1.5,1.5,-0.0]
        """
        rospy.sleep(1.6)
        print("*************************** ", np.array([self.eps_t_x,self.eps_t_y,self.eps_t_z]))

        transformed_coordinates = self.transform_point(
            np.array([self.eps_t_x,self.eps_t_y,self.eps_t_z]),  
            from_frame = self.from_frame, 
            to_frame = self.to_frame
        )
        
        rospy.loginfo(transformed_coordinates)

        quaternion = tf.transformations.quaternion_from_euler(
            self.ee_orientation_r,
            self.ee_orientation_p,
            self.ee_orientation_y
        ) 

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.to_frame
        goal_pose.pose.position.x = 0.58 # transformed_coordinates[0]
        goal_pose.pose.position.y = 0.40 # transformed_coordinates[1] 
        goal_pose.pose.position.z = 0.74 # transformed_coordinates[2] 
        goal_pose.pose.orientation.x = quaternion[0]
        goal_pose.pose.orientation.y = quaternion[1]
        goal_pose.pose.orientation.z = quaternion[2]
        goal_pose.pose.orientation.w = quaternion[3]

        self.group_arm_torso.set_planner_id(self.planner)
        self.group_arm_torso.allow_replanning(False)
        self.group_arm_torso.set_pose_reference_frame(self.from_frame)
        self.group_arm_torso.set_pose_target(goal_pose)
        self.group_arm_torso.set_start_state_to_current_state()
        self.group_arm_torso.set_max_velocity_scaling_factor(1.0)

        rospy.loginfo(
            "Planning to move %s to a target pose expressed in %s",
            self.group_arm_torso.get_end_effector_link(), 
            self.group_arm_torso.get_planning_frame()
        )
    
        grasping_plan = self.group_arm_torso.plan()
        
        if not grasping_plan:
            raise RuntimeError("No plan found")
        
        try:
            rospy.loginfo(
                "Plan found in %s seconds", 
                grasping_plan[1].joint_trajectory.points[-1].time_from_start.to_sec()
            )
        except:
            rospy.loginfo("Plan not found")
            return

        # IDEA TEMPO: calcolare stima tempo impiegato dalla base per arrivare alla palla (tb=ro/vf). calcolare tempo della traiettoria del braccio ta. 
        # far partire controllore braccio a tb-ta    
        arm_duration = 4 # SBLK
        # arm_duration = 10 # Others
        safe_seconds = 3
        rospy.sleep((self.init_rho/0.3) - arm_duration - safe_seconds) # to move the arm after a certain delay
        
        start_time = rospy.Time.now()
        print("********************", len(self.history["ee_pos_y"]))
        self.group_arm_torso.go(wait=True)
        rospy.loginfo("Motion duration: %s", (rospy.Time.now() - start_time).to_sec())

        return self.history

def main():
    controller = ArmController()
    history = controller.move_arm()
    moveit_commander.roscpp_shutdown()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    savefile_path = "results/stop_prm_ee_2.pkl"
    output_path = os.path.join(script_dir, savefile_path)
    
    with open(output_path, "wb") as pkl_file:
        pkl.dump(history,pkl_file,protocol=pkl.HIGHEST_PROTOCOL)
    # import matplotlib.pyplot as plt

    # # Numero di subplot, uno per ogni chiave nel dizionario
    # num_plots = len(history)

    # # Creazione della figura e degli assi
    # fig, axs = plt.subplots(num_plots, 1, figsize=(10, num_plots * 3))

    # # Se c'è solo un grafico, axs non sarà una lista, quindi lo convertiamo in lista
    # if num_plots == 1:
    #     axs = [axs]

    # # Iterazione attraverso il dizionario per plottare i dati
    # for ax, (key, values) in zip(axs, history.items()):
    #     ax.plot(values, label=key)
    #     ax.set_title(f'Data for {key}')
    #     ax.set_xlabel('X Axis Label')
    #     ax.set_ylabel('Y Axis Label')
    #     ax.legend()

    # # Layout ottimale per evitare sovrapposizioni
    # plt.tight_layout()

    # # Mostrare il grafico
    # plt.show()
    # print(controller.history["q1"][0])
    # print(controller.history["q1"][-1])
    # print("***************")
    # print(controller.history["q2"][0])
    # print(controller.history["q2"][-1])
    # print("***************")
    # print(controller.history["q3"][0])
    # print(controller.history["q3"][-1])
    # print("***************")
    # print(controller.history["q4"][0])
    # print(controller.history["q4"][-1])
    # print("***************")
    # print(controller.history["q5"][0])
    # print(controller.history["q5"][-1])
    # print("***************")
    # print(controller.history["q6"][0])
    # print(controller.history["q6"][-1])
    # print("***************")
    # print(controller.history["q7"][0])
    # print(controller.history["q7"][-1])
    # print("***************")
    
if __name__ == "__main__":
    main()