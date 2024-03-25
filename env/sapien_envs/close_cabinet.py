from typing import Union
from ..base_sapien_env import BaseEnv
from sapien.core import renderer as R
import sapien.core as sapien
import numpy as np
from PIL import Image, ImageColor
import mplib
import os
from utils.transform import *
from utils.tools import *
from utils.sapien_utils import *
from gym.spaces import Space, Box
from env.base_viewer import BaseViewer
from env.sapien_envs.osc_planner import OSCPlanner
from env.sapien_envs.open_cabinet import OpenCabinetEnv
import utils.logger
import random
import ujson as json
from tqdm import tqdm

CAMERA_INTRINSIC = [0.05, 100, 1, 640, 480]

class CloseCabinetEnv(OpenCabinetEnv):

    def get_observation(self, gt=False) :
        '''
        Get the observation of the environment as a dict
        If gt=True, return ground truth bbox
        '''

        raw_observation = super().get_observation()

        # if self.renderer_type == 'sapien' :
        #     self.scene.update_render()
        #     for cam in self.registered_cameras :
        #         cam.take_picture()
        # elif self.renderer_type == 'client' :
        #     self.scene._update_render_and_take_pictures(self.registered_cameras)

        if gt :
            raw_observation["handle_bbox"] = self._get_handle_bbox_and_link_transformation(link_name=self.active_link)[0][0]
            raw_observation["success"] = (self.obj_dof() < self.obj_success_dof) * 1.0
        else :
            raw_observation["success"] = (self.obj_dof() < self.obj_success_dof) * 1.0

        # if self.renderer_type == 'client' :
        #     # provide np.array observation
        #     observation = regularize_dict(regularize_dict)

        return raw_observation

    def get_reward(self, action) :

        close_reward = -self.obj_dof()[0]
        end_eff_p = self.gripper_pose().p
        handle_bbox = self._get_handle_bbox_and_link_transformation(link_name=self.active_link)[0][0]
        handle_p = (handle_bbox[0] + handle_bbox[6]) / 2
        dist = np.linalg.norm(end_eff_p - handle_p)
        near_reward = 1.0 / (1.0 + dist**2) + (dist < 0.1)

        eff_x = quat_to_axis(self.gripper_pose().q, 0)
        eff_y = quat_to_axis(self.gripper_pose().q, 1)
        eff_z = quat_to_axis(self.gripper_pose().q, 2)
        handle_x = quat_to_axis(self.handle_pose().q, 0)
        handle_y = quat_to_axis(self.handle_pose().q, 1)
        handle_z = quat_to_axis(self.handle_pose().q, 2)

        dir_reward = (
            (eff_x*handle_z).sum() + (eff_z * (-handle_x)).sum() # 1.0 / (1.0 + np.linalg.norm(eff_y - handle_y)**2)
        ) * 0.1

        # print(near_reward, dir_reward)
        robot_qpos = self.robot.get_qpos()
        # action_reward = - np.linalg.norm(robot_qpos[:7] - action[:7]) * 0.01 - np.linalg.norm(robot_qpos[7:14]) * 0.01
        action_reward = 0

        # print(near_reward, dir_reward, open_reward * (dist < 0.1))

        return near_reward + dir_reward + close_reward * (dist < 0.1)
        # return near_reward + open_reward + dir_reward + action_reward
