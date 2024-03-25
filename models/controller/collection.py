from models.controller.base_controller import BaseController
from models.manipulation.open_cabinet  import OpenCabinetManipulation
from models.pose_estimator.base_estimator import BasePoseEstimator
from models.pose_estimator.groundtruth_estimator import GroundTruthPoseEstimator
from env.sapien_envs.open_cabinet import OpenCabinetEnv
from env.sapien_envs.open_cabinet import CAMERA_INTRINSIC
import sapien.core as sapien
from utils.transform import lookat_quat, axis_angle_to_quat, quat_mul
from utils.tools import split_obs, numpy_dict_to_list_dict, show_image
from logging import Logger
import numpy as np
import ujson
import pickle
import os

GLOBAL_COUNTER = 0

def print_keys_recursively(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key))
        if isinstance(value, dict):
            print_keys_recursively(value, indent + 1)

class CollectionController(BaseController) :

    def __init__(self, env : OpenCabinetEnv, pose_estimator : BasePoseEstimator, manipulation : OpenCabinetManipulation, cfg : dict, logger : Logger):
        super().__init__(env, pose_estimator, manipulation, cfg, logger)

    def run(self, eval=False) :
        '''
        Run the controller.
        Picture-taking not implemented yet!
        '''

        global GLOBAL_COUNTER

        if self.cfg["target"] == "pose_estimator" :

            pose_min = np.asarray(self.cfg["pose_estimator"]["pose_min"])
            pose_min = np.repeat(pose_min[np.newaxis, :], self.env.num_envs, axis=0)
            pose_max = np.asarray(self.cfg["pose_estimator"]["pose_max"])
            pose_max = np.repeat(pose_max[np.newaxis, :], self.env.num_envs, axis=0)
            obs_all = self.env.get_observation(gt=True)
            obj_pose_per_env = self.env.get_attr("current_obj_config")
            handle_center = (obs_all["handle_bbox"][:, 0] + obs_all["handle_bbox"][:, 7])/2

            x_axis = np.zeros((self.env.num_envs, 3))
            x_axis[:, 0] = 1

            # Picture of view 1
            while True :
                pose1 = np.random.uniform(pose_min, pose_max)
                target1 = handle_center + np.random.uniform(-0.2, 0.2, size=(self.env.num_envs, 3))
                rand_rot1 =  axis_angle_to_quat(x_axis, np.random.uniform(-np.pi/8, np.pi/8, size=(self.env.num_envs,)))
                picture_pose1 = np.zeros((self.env.num_envs, 7))
                picture_pose1[:, :3] = pose1
                picture_pose1[:, 3:] = quat_mul(lookat_quat(target1-pose1), rand_rot1)
                # picture_pose1 = [sapien.Pose(p=pose, q=lookat_quat(target-pose)) for pose,target in zip(pose1, target1)]

                self.env.cam_move_to(
                    pose = picture_pose1,
                    time = 2,
                    wait = 1,
                    planner = "path",
                    robot_frame = True,
                    no_collision_with_front = True
                )

                pic_1 = self.env.get_image(mask="handle")
                # show_image(pic_1["camera0"]["Mask"][0]*255)
                # obs1_per_env = self.env.get_observation()
                cam1_pose_per_env = self.env.camera_pose()

                p_env, p_x, p_y = np.nonzero(pic_1["camera0"]["Mask"])

                x_min = CAMERA_INTRINSIC[-1]*2
                x_max = 0
                y_min = CAMERA_INTRINSIC[-2]*2
                y_max = 0
                for i in range(self.env.num_envs) :
                    if p_env.shape[0] :
                        x_min = min(np.min(np.where(p_env == i, p_x, CAMERA_INTRINSIC[-1]*2)), x_min)
                        x_max = max(np.max(np.where(p_env == i, p_x, 0)), x_max)
                        y_min = min(np.min(np.where(p_env == i, p_y, CAMERA_INTRINSIC[-2]*2)), y_min)
                        y_max = max(np.max(np.where(p_env == i, p_y, 0)), y_max)

                if x_min > 0 and y_min > 0 and x_max < CAMERA_INTRINSIC[-1]-1 and y_max < CAMERA_INTRINSIC[-2]-1 and x_min<x_max:
                    break

            # Picture of view 2
            while True :
                pose2 = np.random.uniform(pose_min, pose_max)
                target2 = handle_center + np.random.uniform(-0.2, 0.2, size=(self.env.num_envs, 3))
                rand_rot2 =  axis_angle_to_quat(x_axis, np.random.uniform(-np.pi/8, np.pi/8, size=(self.env.num_envs,)))
                picture_pose2 = np.zeros((self.env.num_envs, 7))
                picture_pose2[:, :3] = pose2
                picture_pose2[:, 3:] = quat_mul(lookat_quat(target2-pose2), rand_rot2)

                self.env.cam_move_to(
                    pose = picture_pose2,
                    time = 2,
                    wait = 1,
                    planner = "path",
                    robot_frame = True,
                    no_collision_with_front = True
                )

                pic_2 = self.env.get_image(mask="handle")
                # obs2_per_env = self.env.get_observation()
                cam2_pose_per_env = self.env.camera_pose()

                p_env, p_x, p_y = np.nonzero(pic_2["camera0"]["Mask"])

                x_min = CAMERA_INTRINSIC[-1]*2
                x_max = 0
                y_min = CAMERA_INTRINSIC[-2]*2
                y_max = 0
                for i in range(self.env.num_envs) :
                    if p_env.shape[0] :
                        x_min = min(np.min(np.where(p_env == i, p_x, CAMERA_INTRINSIC[-1]*2)), x_min)
                        x_max = max(np.max(np.where(p_env == i, p_x, 0)), x_max)
                        y_min = min(np.min(np.where(p_env == i, p_y, CAMERA_INTRINSIC[-2]*2)), y_min)
                        y_max = max(np.max(np.where(p_env == i, p_y, 0)), y_max)

                if x_min > 0 and y_min > 0 and x_max < CAMERA_INTRINSIC[-1]-1 and y_max < CAMERA_INTRINSIC[-2]-1 and x_min<x_max :
                    break

            for cam1, cam2, obj in zip(
                cam1_pose_per_env,
                cam2_pose_per_env,
                obj_pose_per_env) :

                result = {
                    "obj": obj,
                    "view1": {
                        "cam_pose": cam1
                    },
                    "view2": {
                        "cam_pose": cam2
                    }
                }

                save_dir = os.path.join(self.cfg["learn"]["save_dir"], self.cfg["exp_name"])

                if not os.path.exists(save_dir) :
                    os.makedirs(save_dir)

                save_path = os.path.join(save_dir, "data{0}.pickle".format(GLOBAL_COUNTER))

                GLOBAL_COUNTER += 1

                with open(save_path, "wb") as f:
                    self.logger.info("Saving data to {0}".format(save_path))
                    pickle.dump(result, f)
                    self.logger.info("Done.")

        else :

            pose_min = np.asarray(self.cfg["pose_estimator"]["pose_min"])
            pose_min = np.repeat(pose_min[np.newaxis, :], self.env.num_envs, axis=0)
            pose_max = np.asarray(self.cfg["pose_estimator"]["pose_max"])
            pose_max = np.repeat(pose_max[np.newaxis, :], self.env.num_envs, axis=0)
            obs_all = self.env.get_observation(gt=True)
            obj_pose_per_env = self.env.get_attr("current_obj_config")
            handle_center = (obs_all["handle_bbox"][:, 0] + obs_all["handle_bbox"][:, 7])/2

            x_axis = np.zeros((self.env.num_envs, 3))
            x_axis[:, 0] = 1

            # Picture of view 1
            while True :
                pose1 = np.random.uniform(pose_min, pose_max)
                target1 = handle_center + np.random.uniform(-0.2, 0.2, size=(self.env.num_envs, 3))
                rand_rot1 =  axis_angle_to_quat(x_axis, np.random.uniform(-np.pi/8, np.pi/8, size=(self.env.num_envs,)))
                picture_pose1 = np.zeros((self.env.num_envs, 7))
                picture_pose1[:, :3] = pose1
                picture_pose1[:, 3:] = quat_mul(lookat_quat(target1-pose1), rand_rot1)
                # picture_pose1 = [sapien.Pose(p=pose, q=lookat_quat(target-pose)) for pose,target in zip(pose1, target1)]

                self.env.cam_move_to(
                    pose = picture_pose1,
                    time = 2,
                    wait = 1,
                    planner = "path",
                    robot_frame = True,
                    no_collision_with_front = True
                )

                pic_1 = self.env.get_image(mask="handle")
                # show_image(pic_1["camera0"]["Mask"][0]*255)
                # obs1_per_env = self.env.get_observation()
                cam1_pose_per_env = self.env.camera_pose()

                p_env, p_x, p_y = np.nonzero(pic_1["camera0"]["Mask"])

                x_min = CAMERA_INTRINSIC[-1]*2
                x_max = 0
                y_min = CAMERA_INTRINSIC[-2]*2
                y_max = 0
                for i in range(self.env.num_envs) :
                    if p_env.shape[0] :
                        x_min = min(np.min(np.where(p_env == i, p_x, CAMERA_INTRINSIC[-1]*2)), x_min)
                        x_max = max(np.max(np.where(p_env == i, p_x, 0)), x_max)
                        y_min = min(np.min(np.where(p_env == i, p_y, CAMERA_INTRINSIC[-2]*2)), y_min)
                        y_max = max(np.max(np.where(p_env == i, p_y, 0)), y_max)

                if x_min > 0 and y_min > 0 and x_max < CAMERA_INTRINSIC[-1]-1 and y_max < CAMERA_INTRINSIC[-2]-1 and x_min<x_max:
                    break

            obs_all = self.env.get_observation()
            pic_all = self.env.get_image(mask="link")
            robot_conf_per_env = self.env.get_attr("current_robot_config")
            obj_conf_per_env = self.env.get_attr("current_obj_config")

            observation_all = {
                "obs": obs_all,
                "pic": pic_all
            }

            observation_per_env = split_obs(observation_all, self.env.num_envs)

            for observation, robot_conf, obj_conf in zip(observation_per_env, robot_conf_per_env, obj_conf_per_env) :

                result = {
                    "observation": observation,
                    "robot_config": robot_conf,
                    "obj_config": obj_conf
                }

                pc = result["observation"]["pic"]["camera0"]["Position"]
                index = np.random.choice(pc.shape[0] * pc.shape[1], 10000, replace=False)
                pc_downsample = pc.reshape(-1, 3)[index]
                result["observation"]["pic"]["camera0"]["Position10000"] = pc_downsample

                save_dir = os.path.join(self.cfg["learn"]["save_dir"], self.cfg["exp_name"])

                if not os.path.exists(save_dir) :
                    os.makedirs(save_dir)

                save_path = os.path.join(save_dir, "data{0}.pickle".format(GLOBAL_COUNTER))

                GLOBAL_COUNTER += 1

                with open(save_path, "wb") as f:
                    self.logger.info("Saving data to {0}".format(save_path))
                    pickle.dump(result, f)
                    self.logger.info("Done.")
