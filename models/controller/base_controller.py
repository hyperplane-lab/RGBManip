from abc import abstractclassmethod
from models.manipulation.base_manipulation  import BaseManipulation
from models.pose_estimator.base_estimator import BasePoseEstimator
from env.base_sapien_env import BaseEnv
from env.my_vec_env import MultiVecEnv
from logging import Logger

class BaseController :

    def __init__(
            self,
            env : MultiVecEnv,
            pose_estimator : BasePoseEstimator,
            manipulation : BaseManipulation,
            cfg : dict,
            logger : Logger
        ) :

        self.env = env
        self.pose_estimator = pose_estimator
        self.manipulation = manipulation
        self.controller = None
        self.cfg = cfg
        self.logger = logger
    
    def take_picture(pose) :
        '''
        Take a picture of the environment with the camera at the given pose.
        Not implemented yet!
        '''

        raise NotImplementedError
    
    @abstractclassmethod
    def run(self) :

        pass
    
    def train_controller(self, steps, log_interval=1, save_interval=1) :
        '''
        Train controller model only
        '''

        self.logger.info("Training controller model...")
        self.controller.learn(
            steps=steps,
            log_interval=log_interval,
            save_interval=save_interval
        )
    
    def train_manipulation(self, steps, log_interval=1, save_interval=1) :
        '''
        Train manipulation model only
        '''

        self.logger.info("Training manipulation model...")
        self.manipulation.learn(
            steps=steps,
            log_interval=log_interval,
            save_interval=save_interval
        )