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
from env.sapien_envs.base_manipulation import BaseManipulationEnv
import utils.logger
import random
import ujson as json
from tqdm import tqdm

CAMERA_INTRINSIC = [0.05, 100, 1, 640, 480]

class OpenCabinetEnv(BaseManipulationEnv):

    def _generate_object_config(self) :

        '''
        Generate object pose, dof and path from randomization params
        '''

        def randomize_pose(ang_low, ang_high, rot_low, rot_high, dis_low, dis_high, height_low, height_high) :

            ang = np.random.uniform(ang_low, ang_high)
            rot = np.random.uniform(rot_low, rot_high)
            dis = np.random.uniform(dis_low, dis_high)
            height = np.random.uniform(height_low, height_high)

            p0 = sapien.Pose(p=[dis, 0, height])
            r0 = sapien.Pose(q=axis_angle_to_quat([0,0,1], ang))
            r1 = sapien.Pose(q=axis_angle_to_quat([0,0,1], rot))

            p1 = r0 * p0 * r1

            return p1
    
        def randomize_dof(dof_low, dof_high) :
            if dof_low == 'None' or dof_high == 'None' :
                return None
            return np.random.uniform(dof_low, dof_high)
        
        path = random.choice(list(self.obj_path_list.values()))
        # for a in list(self.obj_path_list.values()):
        #     if "46033_link_0" in a:
        #         path = a
        #         break

        data_dir = os.path.dirname(path)
        bbox_path = os.path.join(data_dir, "bounding_box.json")
        with open(bbox_path, "r") as f:
            bbox = json.load(f)
        
        pose = randomize_pose(
            self.obj_init_pos_angle_low,
            self.obj_init_pos_angle_high,
            self.obj_init_rot_low,
            self.obj_init_rot_high,
            self.obj_init_dis_low - bbox["min"][2]*0.75,
            self.obj_init_dis_high - bbox["min"][2]*0.75,
            self.obj_init_height_low - bbox["min"][1]*0.75,
            self.obj_init_height_high - bbox["min"][1]*0.75
        )
        
        dof = randomize_dof(
            self.obj_init_dof_low,
            self.obj_init_dof_high
        )
        
        self.current_obj_config = {
            "path": path,
            "name": os.path.split(os.path.split(path)[0])[1],
            "dof": dof,
            "pose_mat": pose.to_transformation_matrix(),
            "pose_7d": pose.p.tolist() + pose.q.tolist()
        }
        return path, dof, pose
    
    def _prepare_data(self, obj_cfg : dict, task_cfg : dict) :

        '''
        Preload dataset and randomization params from config file
        '''

        self.obj_cfg = obj_cfg
        self.task_cfg = task_cfg

        if obj_cfg["name"] == "partnet_mobility" :

            self.dataset_root = obj_cfg["dataset_root"]
            self.obj_path_list = {}
            for k,v in obj_cfg["objects"].items() :
                self.obj_path_list[k] = os.path.join(self.dataset_root, v["path"])
            
        else :

            raise NotImplementedError
        
        self.robot_path = os.path.join(task_cfg["robot_root"], task_cfg["robot_name"])
    
        self.obj_init_dof_low = task_cfg["object_conf"]["randomization"]["dof"]["low"]
        self.obj_init_dof_high = task_cfg["object_conf"]["randomization"]["dof"]["high"]
        self.obj_init_rot_low = task_cfg["object_conf"]["randomization"]["rot"]["low"]
        self.obj_init_rot_high = task_cfg["object_conf"]["randomization"]["rot"]["high"]
        self.obj_init_pos_angle_low = task_cfg["object_conf"]["randomization"]["pos_angle"]["low"]
        self.obj_init_pos_angle_high = task_cfg["object_conf"]["randomization"]["pos_angle"]["high"]
        self.obj_init_dis_low = task_cfg["object_conf"]["randomization"]["dis"]["low"]
        self.obj_init_dis_high = task_cfg["object_conf"]["randomization"]["dis"]["high"]
        self.obj_init_height_low = task_cfg["object_conf"]["randomization"]["height"]["low"]
        self.obj_init_height_high = task_cfg["object_conf"]["randomization"]["height"]["high"]

        self.robot_init_xyz_low = task_cfg["robot_conf"]["randomization"]["pose"]["xyz"]["low"]
        self.robot_init_xyz_high = task_cfg["robot_conf"]["randomization"]["pose"]["xyz"]["high"]
        self.robot_init_rot = task_cfg["robot_conf"]["init_pose"]["rot"]
        self.robot_init_rot_low = task_cfg["robot_conf"]["randomization"]["pose"]["rot"]["low"]
        self.robot_init_rot_high = task_cfg["robot_conf"]["randomization"]["pose"]["rot"]["high"]
        self.robot_init_dof_low = task_cfg["robot_conf"]["randomization"]["dof"]["low"]
        self.robot_init_dof_high = task_cfg["robot_conf"]["randomization"]["dof"]["high"]

        self.obj_success_dof = task_cfg["object_conf"]["success_dof"]

    def _set_part_mask(self, active_link) :
        '''
        Set the mask of critical part of the object to 255
        '''

        # set id for getting mask
        for link in self.obj.get_links():
            if active_link == link.get_name():
                for s in link.get_visual_bodies():
                    if "handle" in s.get_name() :
                        s.set_visual_id(129)
                    else :
                        s.set_visual_id(128)
            else :
                for s in link.get_visual_bodies():
                    s.set_visual_id(0)

    def handle_pose(self) :
        '''
        Get the pose of target handle
        '''
        if self.calculated_handle_pose != None :
            return self.calculated_handle_pose

        handle_bbox = self._get_handle_bbox_and_link_transformation(link_name=self.active_link)[0][0]

        handle_p = (handle_bbox[0] + handle_bbox[6]) / 2
        handle_x = (handle_bbox[1] - handle_bbox[0])
        handle_y = (handle_bbox[0] - handle_bbox[2])
        handle_z = (handle_bbox[4] - handle_bbox[0])
        handle_x /= np.linalg.norm(handle_x)
        handle_y /= np.linalg.norm(handle_y)
        handle_z /= np.linalg.norm(handle_z)

        handle_q = get_quaternion(
            [
                np.asarray([1,0,0]),
                np.asarray([0,1,0]),
                np.asarray([0,0,1]),
            ],
            [
                handle_x,
                handle_y,
                handle_z
            ]
        )

        self.calculated_handle_pose = sapien.Pose(p=handle_p, q=handle_q)

        return self.calculated_handle_pose

    def obj_dof(self) :
        '''
        Get the dof of the object
        '''

        return self.obj.get_qpos()
    
    def get_success(self) :

        return (self.obj_dof() > self.obj_success_dof)

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
        raw_observation["success"] = self.get_success() * 1.0
        raw_observation["object_dof"] = self.obj_dof()
        # if self.renderer_type == 'client' :
        #     # provide np.array observation
        #     observation = regularize_dict(regularize_dict)

        return raw_observation
    
    def get_state(self) :
        '''
        Get the state of the environment as a dict
        '''
        state = self.get_observation()
        state["obj_qpos"] = self.obj.get_qpos()
        return state
    
    def get_reward(self, action) :

        open_reward = self.obj_dof()[0]
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

        return near_reward + dir_reward + open_reward * (dist < 0.1)
        # return near_reward + open_reward + dir_reward + action_reward
    
    def _get_handle_bbox_and_link_transformation(self, link_name=None):
        '''
        Obtain the bounding box of all handles of the object.
        '''

        handle_bbox = []
        part_vs = []
        global_part_vs = []
        part_fs = []
        link_transmats = []
        links = self.obj.get_links()

        # traverse all handles of all links
        if link_name is None:
            for link in links:
                part_v, global_part_v, part_f, transmat = get_part_mesh_and_pose(link)
                for k in part_v.keys():
                    if 'handle' in k:
                        xmin, ymin, zmin = np.min(part_v[k], axis=0)
                        xmax, ymax, zmax = np.max(part_v[k], axis=0)

                        box_corners = []
                        box_corners.append([xmin, ymin, zmax]) #4
                        box_corners.append([xmin, ymin, zmin]) #0
                        box_corners.append([xmax, ymin, zmax]) #7
                        box_corners.append([xmax, ymin, zmin]) #3

                        box_corners.append([xmin, ymax, zmax]) #5
                        box_corners.append([xmin, ymax, zmin]) #1
                        box_corners.append([xmax, ymax, zmax]) #6
                        box_corners.append([xmax, ymax, zmin]) #2


                        box_corners = np.array(box_corners)
                        ones = np.ones((box_corners.shape[0], 1), dtype=np.float32)
                        bbox_ones = np.concatenate([box_corners, ones], axis=1)

                        global_bbox = (bbox_ones @ transmat.T)[:, :3]
                        handle_bbox.append(global_bbox)

                        part_vs.append(part_v[k])
                        part_fs.append(part_f[k])
                        global_part_vs.append(global_part_v[k])
                        link_transmats.append(transmat)
            return handle_bbox, part_vs, global_part_vs, part_fs, link_transmats
        else:
            for link in links:
                if link_name == link.get_name():
                    part_v, global_part_v, part_f, transmat = get_part_mesh_and_pose(link)
                    for k in part_v.keys():
                        if 'handle' in k:
                            xmin, ymin, zmin = np.min(part_v[k], axis=0)
                            xmax, ymax, zmax = np.max(part_v[k], axis=0)

                            box_corners = []
                            box_corners.append([xmin, ymin, zmax]) #4
                            box_corners.append([xmin, ymin, zmin]) #0
                            box_corners.append([xmax, ymin, zmax]) #7
                            box_corners.append([xmax, ymin, zmin]) #3
                            
                            box_corners.append([xmin, ymax, zmax]) #5
                            box_corners.append([xmin, ymax, zmin]) #1
                            box_corners.append([xmax, ymax, zmax]) #6
                            box_corners.append([xmax, ymax, zmin]) #2

                            box_corners = np.array(box_corners)
                            ones = np.ones((box_corners.shape[0], 1), dtype=np.float32)
                            bbox_ones = np.concatenate([box_corners, ones], axis=1)

                            global_bbox = (bbox_ones @ transmat.T)[:, :3]
                            handle_bbox.append(global_bbox)

                            part_vs.append(part_v[k])
                            part_fs.append(part_f[k])
                            global_part_vs.append(global_part_v[k])
                            link_transmats.append(transmat)
                    return handle_bbox, part_vs, global_part_vs, part_fs, link_transmats
                
            return None, None, None, None, None
