from models.manipulation.base_manipulation import BaseManipulation
from env.base_sapien_env import BaseEnv
from env.sapien_envs.open_cabinet import OpenCabinetEnv
from env.my_vec_env import MultiVecEnv
from utils.transform import *
from logging import Logger

class CloseCabinetManipulation(BaseManipulation) :

    def __init__(self, env : MultiVecEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)

    def plan_pathway(self, center, axis, eval=False) :

        batch_size = center.shape[0]

        x_ = np.array([[1, 0, 0]] * batch_size)
        y_ = np.array([[0, 1, 0]] * batch_size)
        z_ = np.array([[0, 0, 1]] * batch_size)

        # Computing Pre-grasp Pose
        pre_grasp_axis = axis[:, 0]
        pre_grasp_axis -= z_ * (pre_grasp_axis * z_).sum(axis=-1, keepdims=True)
        norm = np.linalg.norm(pre_grasp_axis, axis=-1, keepdims=True)
        pre_grasp_axis = np.where(norm<1e-8, y_, pre_grasp_axis / (norm+1e-8))
        pre_grasp_p = center - pre_grasp_axis * 0.2
        pre_grasp_x = -z_
        pre_grasp_z = pre_grasp_axis
        pre_grasp_y = np.cross(pre_grasp_z, pre_grasp_x)
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

        # Performing Grasp
        self.env.class_method("toggle_gripper", open=True)
        res = self.env.hand_move_to(pre_grasp_pose, time=2, wait=2, planner="path", no_collision_with_front=True)

        grasp_p = pre_grasp_p

        # Computing Grasp Pose
        grasp_p = grasp_p + pre_grasp_axis * 0.18
        grasp_q = pre_grasp_q
        grasp_pose = np.concatenate([grasp_p, grasp_q], axis=-1)

        res = self.env.hand_move_to(grasp_pose, time=2, wait=1, planner="ik")

        self.env.class_method("toggle_gripper", open=False)

        cur_dir = pre_grasp_axis

        for step_size in self.cfg["step_sizes"] :

            cur_p = self.env.gripper_pose()[:, :3]
            pred_p = cur_p + cur_dir * step_size

            next_x = -z_
            next_z = -cur_dir
            next_y = np.cross(next_z, next_x)
            axis_from = np.concatenate([
                x_[:, np.newaxis, :],
                y_[:, np.newaxis, :],
                z_[:, np.newaxis, :]
            ], axis=1)
            axis_to = np.concatenate([
                next_x[:, np.newaxis, :],
                next_y[:, np.newaxis, :],
                next_z[:, np.newaxis, :]
            ], axis=1)
            pred_q = batch_get_quaternion(axis_from, axis_to)

            pred_pose = np.concatenate([pred_p, pred_q], axis=-1)

            self.env.gripper_move_to(pred_pose, time=step_size*10, wait=step_size*5)

            new_p = self.env.gripper_pose()[:, :3]
            new_dir = new_p - cur_p
            new_dir[:, 2] = 0       # project to xy-plane
            new_dir = normalize(new_dir)

            delta = new_dir - cur_dir

            dot = (new_dir * cur_dir).sum(axis=-1, keepdims=True)
            dot = np.clip(dot, -1, 1)

            cur_dir = normalize(cur_dir + 2*delta*dot)
