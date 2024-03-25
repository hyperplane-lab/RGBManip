import sapien.core as sapien
import numpy as np
from models.controller.base_controller import BaseController
from models.manipulation.open_cabinet  import OpenCabinetManipulation
from models.pose_estimator.base_estimator import BasePoseEstimator
from models.pose_estimator.groundtruth_estimator import GroundTruthPoseEstimator
from models.pose_estimator.AdaPose.interface import AdaPoseEstimator
from models.pose_estimator.AdaPose.interface_v2 import AdaPoseEstimator_v2
from models.pose_estimator.AdaPose.interface_v3 import AdaPoseEstimator_v3
from models.pose_estimator.AdaPose.interface_v4 import AdaPoseEstimator_v4
from models.pose_estimator.AdaPose.interface_v5 import AdaPoseEstimator_v5
from env.sapien_envs.open_cabinet import OpenCabinetEnv
from env.my_vec_env import MultiVecEnv
from logging import Logger
from utils.tools import show_image
from utils.transform import lookat_quat
import cv2
import _pickle as cPickle

class HeuristicPoseController(BaseController) :

    def __init__(self, env : MultiVecEnv, pose_estimator : BasePoseEstimator, manipulation : OpenCabinetManipulation, cfg : dict, logger : Logger):
        super().__init__(env, pose_estimator, manipulation, cfg, logger)

    def run(self, eval=False) :
        '''
        Run the controller.
        '''

        p1 = np.asarray([-0.1, 0.0, 0.8])
        p2 = np.asarray([-0.0, 0.5, 0.7])
        target = np.asarray([0.5, 0.0, 0.5])
        q1 = lookat_quat(target-p1)
        q2 = lookat_quat(target-p2)
        picture_pose1 = sapien.Pose(p=p1, q=q1)
        picture_pose2 = sapien.Pose(p=p2, q=q2)

        self.env.cam_move_to(pose = picture_pose1, time=2, wait=1, planner="path", robot_frame=True, no_collision_with_front=False)
        img_1 = self.env.get_image()
        self.env.cam_move_to(pose = picture_pose2, time=2, wait=1, planner="path", robot_frame=True, no_collision_with_front=False)
        img_2 = self.env.get_image()

        mask_1 = img_1["camera0"]["Mask"]
        mask_2 = img_2["camera0"]["Mask"]

        if np.sum(mask_1) == 0 or np.sum(mask_2) == 0 :
            self.logger.info("No mask detected")
            return

        if isinstance(self.pose_estimator, GroundTruthPoseEstimator) :
            original_bbox = self.pose_estimator.estimate()
        elif isinstance(self.pose_estimator, AdaPoseEstimator)\
            or isinstance(self.pose_estimator, AdaPoseEstimator_v2)\
            or isinstance(self.pose_estimator, AdaPoseEstimator_v3)\
            or isinstance(self.pose_estimator, AdaPoseEstimator_v4)\
            or isinstance(self.pose_estimator, AdaPoseEstimator_v5) :
            original_bbox = self.pose_estimator.estimate(
                img_1["camera0"]["Intrinsic"],
                img_1["camera0"]["Color"],
                img_1["camera0"]["Mask"],
                img_1["camera0"]["Extrinsic"],
                img_2["camera0"]["Color"],
                img_2["camera0"]["Mask"],
                img_2["camera0"]["Extrinsic"]
            )
        else :
            raise NotImplementedError

        center = (original_bbox[:, 1] + original_bbox[:, 7]) / 2
        direction = np.zeros((original_bbox.shape[0], 3, 3))
        direction[:, 0] = original_bbox[:, 1] - original_bbox[:, 0]
        direction[:, 1] = original_bbox[:, 0] - original_bbox[:, 2]
        direction[:, 2] = original_bbox[:, 4] - original_bbox[:, 0]
        frame = np.zeros((original_bbox.shape[0], 3, 3))
        frame[:, 0, 0] = 1
        frame[:, 1, 1] = 1
        frame[:, 2, 2] = 1
        d_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        direction = direction / (d_norm + 1e-8)
        direction = np.where(d_norm > 1e-8, direction, frame)
        self.manipulation.plan_pathway(center, direction, eval)
