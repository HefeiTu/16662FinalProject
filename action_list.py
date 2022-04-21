"""
action_list.py
====================================
List of actions that franka take to pour drinks.

Last edited: April 9th, 2022
By: Di Hu, Hefei Tu, Simi Asher

Class: 


Functions: 
    grab_bottle(): grab bottle from the robot home pose 
    goto_cup(): move the bottle to near the cup, ready to pour 
    pour_drink(): pour the liquid out of the bottle into the cup
    return_bottle(): return the bottle from near the cup to the robot home pose 

Constants:
    home_pose: fa.reset_joints()
    near_cup_pose: the destination pose for goto_cup()
    pouring_angle: rotation angle for end-effector to pour liquid from bottle into cup
    pouring_time: duration of pouring (s)

Parameters: 
    bottle_pose: the pose of the bottle to be grabbed or returned, 
                 passed in by the controller module 
    cup_pose: the pose of the cup to be poured drinks in 

"""
import numpy as np
from autolab_core import RigidTransform

class ActionList:
    def __init__(self, fa):
        self.fa = fa
        # Initialize the configuration
        # TODO: change the configuration to what they should be
        # self.cup_pose = cup_pose
        self.lift_bottle_pose = None
        self.near_cup_pose = None
        self.bottle_pose = None
        self.near_bottle_pose = None

    def set_near_bottle_pose(self, bottle_pose):
        near_bottle_pose = bottle_pose.copy()
        near_bottle_pose.translation[1] -= 0.1
        self.near_bottle_pose = near_bottle_pose
        return near_bottle_pose

    def set_near_cup_pose(self, cup_pose):
        near_cup_pose = self.fa.get_pose()
        near_cup_pose.translation[:2] = [cup_pose[0] - 0.1, cup_pose[1]]
        self.near_cup_pose = near_cup_pose.copy()
       
        return near_cup_pose

    def set_lift_bottle_pose(self):
        lift_bottle_pose = self.fa.get_pose()
        lift_bottle_pose.translation[2] += 0.2
        self.lift_bottle_pose = lift_bottle_pose.copy()
        return
        
    def grab_bottle(self, bottle_pose):
        near_bottle_pose = self.set_near_bottle_pose(bottle_pose)
        print("Going to intermediate bottle pose")
        self.fa.goto_pose(near_bottle_pose, 5)
        
        self.bottle_pose = bottle_pose.copy()
        print("Going to bottle pose")
        self.fa.goto_pose(bottle_pose, 5, force_thresholds=[10, 10, 10, 10, 10, 10])
        self.fa.publish_joints()
        # close gripper
        self.fa.goto_gripper(0.045, grasp=True, force=100.0)
        # self.fa.close_gripper()

        return True
        
    def goto_cup(self, cup_pose):
        self.set_lift_bottle_pose()
        print('Lifting bottle to cup')
        self.fa.goto_pose(self.lift_bottle_pose)

        self.set_near_cup_pose(cup_pose)
        print('Going to cup')
        self.fa.goto_pose(self.near_cup_pose)
        return True

    def pour_drink(self, cup_pose):
        print("Pouring drink")
        pouring_angle = 120 # deg
        pouring_time = 8 #s

        pour_pose = self.fa.get_pose()
        pour_pose.translation[:2] = cup_pose[:2]
        R = RigidTransform(rotation=RigidTransform.z_axis_rotation(np.deg2rad(pouring_angle)), from_frame='franka_tool', to_frame='franka_tool')
        pour_pose = pour_pose * R
        self.fa.goto_pose(pour_pose, pouring_time, force_thresholds=[100, 100, 100, 100, 100, 100], buffer_time=5)
        return True

    def return_bottle(self, bottle_pose):
        # Return the bottle to the initial pose
        self.fa.goto_pose(self.near_cup_pose, 4, force_thresholds=[10, 10, 20, 10, 10, 10])
        self.fa.goto_pose(self.lift_bottle_pose, 4, force_thresholds=[10, 10, 10, 10, 10, 10])
        self.fa.goto_pose(self.bottle_pose, 5, force_thresholds=[10, 10, 20, 10, 10, 10])
        self.fa.open_gripper()
        self.fa.goto_pose(self.near_bottle_pose, 3, force_thresholds=[10, 10, 10, 10, 10, 10])
        return True

    def grab_mixer(self, relative_mixer_pose):
        mixer_pose = self.fa.get_pose()
        mixer_pose.translation = relative_mixer_pose
        self.mixer_pose = mixer_pose.copy()
        self.fa.goto_pose(mixer_pose, 5, force_thresholds=[10, 10, 10, 10, 10, 10])
        self.fa.publish_joints()
        self.fa.goto_gripper(0.04, grasp=True, force=5.0)
        
    def grab_mixer(self, relative_mixer_pose):
        mixer_pose = self.fa.get_pose()
        mixer_pose.translation = relative_mixer_pose
        self.mixer_pose = mixer_pose.copy()
        self.fa.goto_pose(mixer_pose, 5, force_thresholds=[10, 10, 10, 10, 10, 10])
        self.fa.publish_joints()
        self.fa.goto_gripper(0.04, grasp=True, force=5.0)

    def goto_cup_mixing(self, relative_mixer_pose, cup_pose):
        mixer_pose = self.fa.get_pose()
        mixer_pose.translation = [relative_mixer_pose[0], relative_mixer_pose[1], 0.5]
        self.fa.goto_pose(mixer_pose, 5, force_thresholds=[10, 10, 10, 10, 10, 10])

        cup_mixing_pose = self.fa.get_pose()
        cup_mixing_pose.translation = [cup_pose[0] + 0.05, cup_pose[1], 0.3]
        self.fa.goto_pose(cup_mixing_pose, 5)

        cup_mixing_pose = self.fa.get_pose()
        cup_mixing_pose.translation = [cup_pose[0] + 0.05, cup_pose[1], 0.12]
        self.fa.goto_pose(cup_mixing_pose, 5)

        cup_mixing_pose = self.fa.get_pose()
        cup_mixing_pose.translation = [cup_pose[0] + 0.05, cup_pose[1], 0.12]
        R = RigidTransform(rotation=RigidTransform.z_axis_rotation(np.deg2rad(179)), from_frame='franka_tool', to_frame='franka_tool')
        cup_mixing_pose = cup_mixing_pose * R
        self.fa.goto_pose(cup_mixing_pose, 2)

        cup_mixing_pose = self.fa.get_pose()
        cup_mixing_pose.translation = [cup_pose[0] + 0.05, cup_pose[1], 0.12]
        R = RigidTransform(rotation=RigidTransform.z_axis_rotation(np.deg2rad(-179)), from_frame='franka_tool', to_frame='franka_tool')
        cup_mixing_pose = cup_mixing_pose * R
        self.fa.goto_pose(cup_mixing_pose, 2)

        self.fa.open_gripper()
        cup_mixing_pose = self.fa.get_pose()
        cup_mixing_pose.translation = [cup_pose[0] + 0.05, cup_pose[1], 0.3]
        self.fa.goto_pose(cup_mixing_pose, 5)