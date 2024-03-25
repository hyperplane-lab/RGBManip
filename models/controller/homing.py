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

class HomingController(BaseController) :

    def __init__(self, env : MultiVecEnv, pose_estimator : BasePoseEstimator, manipulation : OpenCabinetManipulation, cfg : dict, logger : Logger):
        super().__init__(env, pose_estimator, manipulation, cfg, logger)

    def run(self, eval=False) :
        '''
        Run the controller.
        '''

        p1 = np.asarray([0.53, 0.0, 0.40])
        target = np.asarray([0.68, 0.0, 0.40])
        q1 = lookat_quat(target-p1)
        picture_pose1 = sapien.Pose(p=p1, q=q1)

        self.env.hand_move_to(pose = picture_pose1, time=2, wait=1, planner="path", robot_frame=True, no_collision_with_front=False)
