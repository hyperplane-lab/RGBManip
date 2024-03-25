from models.manipulation.base_manipulation import BaseManipulation
from env.base_sapien_env import BaseEnv
from env.sapien_envs.open_cabinet import OpenCabinetEnv
from env.my_vec_env import MultiVecEnv
from utils.transform import *
from utils.tools import *
from logging import Logger
from gym.spaces import Space, Box
from algo.ppo.ppo import PPO
import torch

class RLManipulation(BaseManipulation) :

    def __init__(self, vec_env : MultiVecEnv, cfg : dict, logger : Logger) :

        super().__init__(vec_env, cfg, logger)

        self.agent = PPO(vec_env, cfg)

    def learn(self, steps, log_interval=1, save_interval=1) :

        self.agent.run(steps, log_interval, save_interval)

    def plan_pathway(self, obs, eval=False) :

        self.agent.play()
