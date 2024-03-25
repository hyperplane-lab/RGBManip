from models.manipulation.base_manipulation import BaseManipulation
from env.base_sapien_env import BaseEnv
from env.sapien_envs.open_cabinet import OpenCabinetEnv
from env.my_vec_env import MultiVecEnv
from utils.transform import *
from logging import Logger

class OpenPotManipulation(BaseManipulation) :

    def __init__(self, env : MultiVecEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)

    def plan_pathway(self, center, axis, eval=False) :

        batch_size = center.shape[0]

        x_ = np.array([[1, 0, 0]] * batch_size)
        y_ = np.array([[0, 1, 0]] * batch_size)
        z_ = np.array([[0, 0, 1]] * batch_size)

        pre_grasp_axis = -z_
        pre_grasp_p = center - pre_grasp_axis * 0.08
        pre_grasp_y = np.cross(pre_grasp_axis, axis[:, 1])
        pre_grasp_y /= np.linalg.norm(pre_grasp_y, axis=-1, keepdims=True)
        pre_grasp_x = -np.cross(pre_grasp_axis, pre_grasp_y)
        pre_grasp_x /= np.linalg.norm(pre_grasp_y, axis=-1, keepdims=True)
        pre_grasp_z = pre_grasp_axis
        axis_from = np.concatenate([
            x_[:, np.newaxis, :],
            y_[:, np.newaxis, :],
            z_[:, np.newaxis, :]
        ], axis=1)
        axis_to = np.concatenate([
            pre_grasp_x[:, np.newaxis, :],
            pre_grasp_y[:, np.newaxis, :],
            pre_grasp_z[:, np.newaxis, :]
        ], axis=1)
        pre_grasp_q = batch_get_quaternion(axis_from, axis_to)
        pre_grasp_pose = np.concatenate([pre_grasp_p, pre_grasp_q], axis=-1)

        grasp_p = center + pre_grasp_axis * 0.03
        grasp_gripper_p = center
        grasp_q = pre_grasp_q
        grasp_pose = np.concatenate([grasp_p, grasp_q], axis=-1)

        self.env.class_method("toggle_gripper", open=True)

        res = self.env.gripper_move_to(pre_grasp_pose, time=2, wait=1, planner="path")

        self.env.class_method("toggle_gripper", open=True)

        res = self.env.gripper_move_to(grasp_pose, time=2, wait=1, planner="ik")

        self.env.class_method("toggle_gripper", open=False)

        gripper_p = [grasp_gripper_p + pre_grasp_axis*0.1, grasp_gripper_p]
        prev_dir = -pre_grasp_axis

        last_dir = prev_dir

        for step_size in self.cfg["step_sizes"] :


            next_p = gripper_p[-1] + last_dir / (np.linalg.norm(last_dir, axis=-1, keepdims=True)+1e-4) * step_size

            next_q = pre_grasp_q

            next_pose = np.concatenate([next_p, next_q], axis=-1)
            res = self.env.gripper_move_to(next_pose, time=2, wait=1, planner="ik")

            gripper_p.append(self.env.gripper_pose()[:, :3])
