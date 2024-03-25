from abc import abstractclassmethod
from env.base_sapien_env import BaseEnv
from logging import Logger

class BasePoseEstimator :

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        self.env : BaseEnv = env
        self.cfg = cfg
        self.logger = logger
    
    @abstractclassmethod
    def append_picture(self, pic, pose) :

        pass

    @abstractclassmethod
    def estimate(self) :

        pass