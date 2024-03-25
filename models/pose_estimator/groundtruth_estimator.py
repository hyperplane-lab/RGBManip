from models.pose_estimator.base_estimator import BasePoseEstimator
from env.base_sapien_env import BaseEnv
from env.sapien_envs.open_cabinet import OpenCabinetEnv
from logging import Logger

class GroundTruthPoseEstimator(BasePoseEstimator) :

    def __init__(self, env : OpenCabinetEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)

    def append_picture(self, pic, pose) :

        pass

    def estimate(self) :

        return self.env.get_observation(gt=True)["handle_bbox"]