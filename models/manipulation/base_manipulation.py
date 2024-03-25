from abc import abstractclassmethod
from env.base_sapien_env import BaseEnv
from env.my_vec_env import MultiVecEnv
from logging import Logger

class BaseManipulation :

    def __init__(self, env : MultiVecEnv, cfg : dict, logger : Logger) :

        self.env = env
        self.cfg = cfg
        self.logger = logger

    @abstractclassmethod
    def plan_pathway(self, obs, eval=False) :

        pass