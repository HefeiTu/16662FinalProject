# /usr/bin/env python3

import numpy as np

from frankapy import FrankaArm

from action_list import ActionList

import cv2
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point
from perception import CameraIntrinsics
from utils import *
from RobotUtil import *

class Detection:
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.azure_kinect_intrinsics = CameraIntrinsics.load("/home/student/team6/16662FinalProject/calib/azure_kinect.intr")
        self.azure_kinect_to_world_transform = RigidTransform.load("/home/student/team6/16662FinalProject/calib/azure_kinect_overhead/azure_kinect_overhead_to_world.tf")    

    def fetch_rgb_image(self):
        print("Fetching rgb image")
        azure_kinect_rgb_image = get_azure_kinect_rgb_image(self.cv_bridge)

        # cv2.imwrite('rgb.png', azure_kinect_rgb_image)
        # cv2.imwrite('depth.png', azure_kinect_depth_image)

        # border = define_borders(azure_kinect_rgb_image)
        # print(border)

        border = [[366, 140], [1704, 922]]
        mask = np.zeros(azure_kinect_rgb_image.shape[:2], np.uint8)
        mask[border[0][1]:border[1][1], border[0][0]:border[1][0]] = 255
        rgb_image = cv2.bitwise_and(azure_kinect_rgb_image, azure_kinect_rgb_image, mask=mask)

        return rgb_image
    
    def fetch_depth_image(self):
        print("Fetching depth image")
        return get_azure_kinect_depth_image(self.cv_bridge)

    def get_bottle_center(self, rgb_image, lower, upper):
        print("Finding bottle center")
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)
        mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1)
        mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1)
        mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)
        cnts, hie = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        maxc_i = np.argmax([cv2.contourArea(c) for c in cnts])
        c = cnts[maxc_i]
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(rgb_image, (x-30,y-60), (x+w+30,y+h), (0,0,255), 2)

        # Create a bounding box which only contains the upper half of the bottle
        # HSV thresholding the black nozzle
        nozzle_mask = np.zeros(rgb_image.shape[:2], np.uint8)
        nozzle_mask[y-60:y+h, x-30:x+w+30] = 255
        nozzle_mask = cv2.bitwise_and(rgb_image, rgb_image, mask=nozzle_mask)
        nozzle_hsv = cv2.cvtColor(nozzle_mask, cv2.COLOR_BGR2HSV)
        nozzle_mask = cv2.inRange(nozzle_hsv, np.array([0,1,0]), np.array([179,255,50]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        nozzle_mask = cv2.morphologyEx(nozzle_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
        nozzle_mask = cv2.morphologyEx(nozzle_mask, cv2.MORPH_OPEN, kernel, iterations=5)
        nozzle_cnts, _ = cv2.findContours(nozzle_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        nozzle_c = max(nozzle_cnts, key=cv2.contourArea)
        (nozzle_x, nozzle_y), radius = cv2.minEnclosingCircle(nozzle_c)
        center = [int(nozzle_x), int(nozzle_y)]
        radius = int(radius)
        cv2.circle(rgb_image, center, radius, (255,0,0), 2)

        cv2.imshow('img', cv2.resize(rgb_image, (1024,768)))
        cv2.waitKey(0)
        
        return center

    def transform_pose_to_world(self, center, current_pose):
        print("Transforming bottle detection to world pose")
        object_center_point_in_world = get_object_center_point_in_world(center[0],
                                                                        center[1],
                                                                        self.fetch_depth_image(), 
                                                                        self.azure_kinect_intrinsics,
                                                                        self.azure_kinect_to_world_transform)
        object_center_pose = current_pose
        object_center_pose.rotation = object_center_pose.rotation @ np.array([[0,1,0], [0,0,-1], [-1,0,0]])
        hardcoded_offset = 0.01
        object_z_height = 0.2
        object_center_pose.translation = [object_center_point_in_world[0], object_center_point_in_world[1] + hardcoded_offset, object_z_height]
        return object_center_pose
        
    def get_bottle_pose(self, current_pose):
        rgb_image = self.fetch_rgb_image()
        gLower = np.array([70,69,20])
        gUpper = np.array([83,255,255])
        bottle_center = self.get_bottle_center(rgb_image, gLower, gUpper)
        return self.transform_pose_to_world(bottle_center, current_pose)

class ActionManager:
    def __init__(self):
        print('Starting robot')
        self.fa = FrankaArm()

        self.action_list = ActionList(self.fa)
        self.detection = Detection()
        self.states = ["grab_bottle", "goto_cup", "pour_drink", "return_bottle"]
        self.current_state_idx = 0

        # State
        self.bottles = []
        self.cup_pose = [0.4, 0, 0]

    def reset(self):
        print("Resetting!")
        self.fa.open_gripper()
        self.fa.reset_joints()
        self.fa.open_gripper()

    def update_bottle_poses(self):
        self.bottles.append(self.detection.get_bottle_pose(self.fa.get_pose()))

    def dispense_drink(self, bottle_idx):
        bottle_pose = self.bottles[bottle_idx]

        self.action_list.grab_bottle(bottle_pose)
        self.action_list.goto_cup(self.cup_pose)
        self.action_list.pour_drink(self.cup_pose)
        self.action_list.return_bottle(bottle_pose)

    def run(self):
        self.reset()
        self.update_bottle_poses()
        self.dispense_drink(0)
        
if __name__ == "__main__":
    manager = ActionManager()
    manager.run()
    manager.reset()

    
    # reset franka to its home joints
    # fa.reset_joints()

    # # read functions
    # T_ee_world = fa.get_pose()
    # print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))

    # joints = fa.get_joints()
    # print('Joints: {}'.format(joints))

    # gripper_width = fa.get_gripper_width()
    # print('Gripper width: {}'.format(gripper_width))

    # # gripper controls
    # print('Closing gripper')
    # fa.close_gripper()

    # print('Opening gripper to a specified position')
    # fa.goto_gripper(0.02)

    # print('Opening gripper all the way')
    # fa.open_gripper()

    # # joint controls
    # print('Rotating last joint')
    # joints = fa.get_joints()
    # joints[6] += np.deg2rad(45)
    # fa.goto_joints(joints)
    # joints[6] -= np.deg2rad(45)
    # fa.goto_joints(joints)

    # # end-effector pose control
    # print('Translation')
    # T_ee_world = fa.get_pose()
    # T_ee_world.translation += [0.1, 0, 0.1]
    # fa.goto_pose(T_ee_world)
    # T_ee_world.translation -= [0.1, 0, 0.1]
    # fa.goto_pose(T_ee_world)

    # print('Rotation in end-effector frame')
    # T_ee_rot = RigidTransform(
    #     rotation=RigidTransform.x_axis_rotation(np.deg2rad(45)),
    #     from_frame='franka_tool', to_frame='franka_tool'
    # )
    # T_ee_world_target = T_ee_world * T_ee_rot
    # fa.goto_pose(T_ee_world_target)
    # fa.goto_pose(T_ee_world)

    # # reset franka back to home
    # fa.reset_joints()