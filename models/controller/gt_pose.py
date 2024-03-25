from models.controller.base_controller import BaseController
from models.manipulation.open_cabinet  import OpenCabinetManipulation
from models.pose_estimator.base_estimator import BasePoseEstimator
from models.pose_estimator.groundtruth_estimator import GroundTruthPoseEstimator
from env.sapien_envs.open_cabinet import OpenCabinetEnv
from logging import Logger
import numpy as np

class GtPoseController(BaseController) :

    def __init__(self, env : OpenCabinetEnv, pose_estimator : BasePoseEstimator, manipulation : OpenCabinetManipulation, cfg : dict, logger : Logger):
        super().__init__(env, pose_estimator, manipulation, cfg, logger)

    def run(self, eval=False) :
        '''
        Run the controller.
        Picture-taking not implemented yet!
        '''

        # for i in range(1000) :
        #     self.env.step(np.asarray([[0]*self.env.action_space.shape[0]]))

        if isinstance(self.pose_estimator, GroundTruthPoseEstimator) :
            original_bbox = self.pose_estimator.estimate()
        else :
            raise NotImplementedError

        center = (original_bbox[:, 0]+original_bbox[:, 7]) / 2
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
