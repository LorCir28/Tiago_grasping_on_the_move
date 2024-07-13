import pinocchio
import numpy as np
import sys
import os
from os.path import dirname, join, abspath 

# Goal: Build a reduced model from an existing URDF model by fixing the desired joints at a specified position.
 
# Load UR robot arm
# This path refers to Pinocchio source code but you can define your own directory here.
#pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))),"models")

def getReducedModel(model_xml):

  

    #pinocchio_model_dir = os.path.join(os.getcwd(), "models")

    
    #mesh_dir = pinocchio_model_dir
    #urdf_filename = "tiago.urdf"
    
    #urdf_model_path = os.path.dirname(os.path.abspath(__file__)) + '/models/example-robot-data/robots/tiago_description/robots/' + urdf_filename

    #print(model_path)
    #print(mesh_dir)


    #model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(urdf_model_path, mesh_dir)

    model = pinocchio.buildModelFromXML(model_xml, pinocchio.JointModelFreeFlyer())
    # model = pinocchio.buildModelFromXML(model_xml)

    #print('standard model: dim=' + str(len(model.joints)))
    #for jn in model.joints:
    #    print(jn)
    #print('-' * 30)
    #print('\n \n')

    #print('Joints Total Model:')
    #for i in model.names: 
    #  print(i)

    # Create a list of joints to take
    jointsToLock = [
        'caster_back_left_1_joint',
        'caster_back_left_2_joint',
        'caster_back_right_1_joint',
        'caster_back_right_2_joint',
        'caster_front_left_1_joint',
        'caster_front_left_2_joint',
        'caster_front_right_1_joint',
        'caster_front_right_2_joint',
        'suspension_left_joint',
        'wheel_left_joint',
        'suspension_right_joint',
        'wheel_right_joint',
        'torso_lift_joint',
        'head_1_joint',
        'head_2_joint',
        'gripper_left_joint',
        'gripper_right_joint'
    ]
    #   TODO: rimuovere giunti gripper

    jointsToLockIDs = []
    for jn in jointsToLock:
        if model.existJointName(jn):
            jointsToLockIDs.append(model.getJointId(jn))
        else:
            print('Warning: joint ' + str(jn) + ' does not belong to the model!')

    #print(jointsToLockIDs)

    # Set initial position of both fixed and revolute joints
    initialJointConfig = pinocchio.neutral(model)                                
 
    # Option 1: Only build the reduced model in case no display needed:
    model_reduced = pinocchio.buildReducedModel(model, jointsToLockIDs, initialJointConfig)
    
    return model_reduced
