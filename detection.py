from frankapy import FrankaArm
import numpy as np
import argparse
import cv2
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point
from perception import CameraIntrinsics
from utils import *
from RobotUtil import *

AZURE_KINECT_INTRINSICS = '/home/student/team6/16662FinalProject/calib/azure_kinect.intr'
AZURE_KINECT_EXTRINSICS = '/home/student/team6/16662FinalProject/calib/azure_kinect_overhead/azure_kinect_overhead_to_world.tf'

cup_world = [0.4, 0, 0]  # TODO: record a fixed position

rLower = np.array([160,59,20])
rUpper = np.array([179,255,255])

gLower = np.array([70,69,20])
gUpper = np.array([83,255,255])

bLower = np.array([102,7,28])
bUpper = np.array([111,196,255])

def define_borders(img):
    border = []
    
    def drawBorder(action, x, y, flags, param):
        if action == cv2.EVENT_LBUTTONDOWN:
           param.append([x,y])
        elif action == cv2.EVENT_LBUTTONUP:
            param.append([x,y])
        
    cv2.namedWindow('image')
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', drawBorder, border)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return border

def find_drink(rgb_image, lower, upper):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics_file_path', type=str, default=AZURE_KINECT_INTRINSICS)
    parser.add_argument('--extrinsics_file_path', type=str, default=AZURE_KINECT_EXTRINSICS) 
    args = parser.parse_args()
    
    print('Starting robot')
    fa = FrankaArm()    

    print('Opening Grippers')
    #Open Gripper
    fa.open_gripper()

    #Reset Pose
    fa.reset_pose() 
    #Reset Joints
    fa.reset_joints()

    cv_bridge = CvBridge()
    azure_kinect_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
    azure_kinect_to_world_transform = RigidTransform.load(args.extrinsics_file_path)    

    azure_kinect_rgb_image = get_azure_kinect_rgb_image(cv_bridge)
    azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)

    cv2.imwrite('rgb.png', azure_kinect_rgb_image)
    cv2.imwrite('depth.png', azure_kinect_depth_image)

    # border = define_borders(azure_kinect_rgb_image)
    # print(border)
    border = [[366, 140], [1704, 922]]
    mask = np.zeros(azure_kinect_rgb_image.shape[:2], np.uint8)
    mask[border[0][1]:border[1][1], border[0][0]:border[1][0]] = 255
    rgb_image = cv2.bitwise_and(azure_kinect_rgb_image, azure_kinect_rgb_image, mask=mask)

    center = find_drink(rgb_image, gLower, gUpper)

    object_z_height = 0.2
    # intermediate_pose_z_height = 0.4

    object_center_point_in_world = get_object_center_point_in_world(center[0],
                                                                    center[1],
                                                                    azure_kinect_depth_image, azure_kinect_intrinsics,
                                                                    azure_kinect_to_world_transform)

    object_center_pose = fa.get_pose()
    object_center_pose.rotation = object_center_pose.rotation @ np.array([[0,1,0], [0,0,-1], [-1,0,0]])
    # TODO: hardcoded_offset
    hardcoded_offset = 0.01
    object_center_pose.translation = [object_center_point_in_world[0], object_center_point_in_world[1] + hardcoded_offset, object_z_height]

    #####
    intermediate_robot_pose = object_center_pose.copy()
    intermediate_robot_pose.translation = [object_center_point_in_world[0], object_center_point_in_world[1] - 0.1, object_z_height]
    fa.goto_pose(intermediate_robot_pose, 5)

    #Bottle
    fa.goto_pose(object_center_pose, 5, force_thresholds=[10, 10, 10, 10, 10, 10])

    fa.publish_joints()

    #Close Gripper
    fa.goto_gripper(0.045, grasp=True, force=10.0)

    #Lift
    lift_pose = fa.get_pose()
    lift_pose.translation[2] += 0.2
    fa.goto_pose(lift_pose)

    #Move to cup intermediate
    cup_intermediate_pose = fa.get_pose()
    cup_intermediate_pose.translation[:2] = [cup_world[0] - 0.1, cup_world[1]]
    fa.goto_pose(cup_intermediate_pose)

    #Move to cup
    cup_pose = cup_intermediate_pose.copy()
    cup_pose.translation[:2] = cup_world[:2]
    # TODO: rotate around nozzle
    R = RigidTransform(rotation=RigidTransform.z_axis_rotation(np.deg2rad(179)), from_frame='franka_tool', to_frame='franka_tool')
    cup_pose = cup_pose * R
    fa.goto_pose(cup_pose, 5, force_thresholds=[10, 10, 20, 10, 10, 10], buffer_time=5)

    #Return
    fa.goto_pose(cup_intermediate_pose)
    fa.goto_pose(lift_pose)
    fa.goto_pose(object_center_pose, 5, force_thresholds=[10, 10, 20, 10, 10, 10])

    print('Opening Grippers')
    #Open Gripper
    fa.open_gripper()

    fa.goto_pose(intermediate_robot_pose)

    #Reset Pose
    fa.reset_pose() 
    #Reset Joints
    fa.reset_joints()
