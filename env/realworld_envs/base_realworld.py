import os
import sapien
from franka_impedance import RealworldEnv
from utils.transform import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import time as Time

class BaseRealworldEnv(RealworldEnv) :

    def __init__(self) :

        root = os.path.dirname(os.path.abspath(__file__))
        super().__init__(os.path.join(root, "panda_rs_handeyecalibration_eye_on_hand.yaml"))
        sam = sam_model_registry["vit_h"](checkpoint="env/realworld_envs/sam_ckpt/sam_vit_h_4b8939.pth")
        self.sam_predictor = SamPredictor(sam)

    def get_observation(self, gt=False) :

        # TODO
        assert(gt==False)
        # See sapien_envs.base_manipulation for instructions
        hand_pose = self.get_hand_pose()
        hand_xyz = hand_pose[:3]
        target_xyz = None  # ???
        total_move_distance = np.linalg.norm(target_xyz - hand_xyz)
        success = 0
        return total_move_distance, success

    # show results of sam masks
    def _show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)

    def get_image(self, mask="handle") :

        print("Getting Image from Realsense")
        # TODO
        assert(mask in ["handle", ])
        # See sapien_envs.base_manipulation for instructions
        # Color, Mask, Intrinsic, Extrinsic
        cam_name = "camera0"
        cam_pose = self.camera_pose()
        cam_pos = cam_pose[:3]
        cam_rot = cam_pose[3:]

        
        color_image, z_depth = self.camera.take_picture()
        self.sam_predictor.set_image(color_image)
        intrinsic = None 
        extrinsic = cam_pose 
        #camera_intrinsics: [near, far, fovy, width, height]
        #camera_extrinsics: pose of the camera (sapien.Pose), or transformation of the camera (numpy).

        
        if mask == "handle":
            prompt = "handle on cabinet"
        
        masks, _, _ = self.sam_predictor.predict(prompt)


        images = {}

                
        images[cam_name] = {
            'Color': color_image,
            'Position': cam_pos,
            'Depth': z_depth,
            'Norm': cam_rot,
            'Mask': masks,
            'Intrinsic': intrinsic,
            'Extrinsic': extrinsic
        }

        return color_image, masks, intrinsic, extrinsic

    def camera_pose(self) :

        return super().get_cam_pose()
    
    def hand_pose(self) :

        return super().get_hand_pose()
    
    def gripper_move_to(self, pose, time=2, wait=1, planner="ik", robot_frame=False, skip_move=False, no_collision_with_front=True) :
        '''
        Move the gripper to a target pose
        '''
        print("gripper_move_to", pose)
        open_dir = quat_to_axis(pose[3:], 2) * 0.105
        new_pose = sapien.Pose(p=pose[:3]-open_dir, q=pose[3:])
        res = super().hand_move_to(new_pose)
        Time.sleep(time)
        print("done")
        return res
    
    def cam_move_to(self, pose, time=2, wait=1, planner="ik", robot_frame=False, skip_move=False, no_collision_with_front=True) :
        '''
        Move the camera to a target pose
        '''
        
        print("cam_move_to", pose)
        res = super().cam_move_to(pose)
        Time.sleep(time)
        print("done")
        return res

        # open_dir = quat_to_axis(pose[3:], 2) * 0.105
        # new_pose = sapien.Pose(p=pose[:3]-open_dir, q=pose[3:])
        # return super().hand_move_to(new_pose)

    def hand_move_to(self, pose, time=2, wait=1, planner="ik", robot_frame=False, skip_move=False, no_collision_with_front=True) :
        '''
        Move the camera to a target pose
        '''

        print("hand_move_to", pose)
        res = super().hand_move_to(pose)
        Time.sleep(time)
        print("done")
        return res