from functools import partial
import numpy as np
import sapien.core as sapien

from utils.sapien_utils import *
from utils.tools import *
from utils.transform import *

from dm_env import specs

from env.sapien_envs.base_manipulation import BaseManipulationEnv, CAMERA_INTRINSIC

IMAGE_SIZE = 84

class GymManipulationEnv():

    def __init__(self, env : BaseManipulationEnv, max_step = 4):

        if isinstance(env, partial) :
            env = env()
        self.env = env
        self.max_step = max_step

        obs = self.reset()
        self.observation_space = convert_observation_to_space(obs)
        self.state_space = convert_observation_to_space(obs)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.last_image = None
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.step_num = 0
        img = self.env.get_image()
        self.last_image = img["camera0"]
        img["camera1"] = img["camera0"]
        return img
    
    def step(self, actions, **kwargs) :

        self.env.current_driving_target[-1] = actions[-1]
        self.env.gripper_move_to(actions[:7])

        obs = self.env.get_image()
        obs["camera1"] = self.last_image
        self.last_image = obs["camera0"]
        rew = self.env.get_reward(actions)
        suc = self.env.get_success()
        done = False
        self.step_num += 1
        if self.step_num == self.max_step :
            done = True

        return obs, rew, done, {'is_success': self.env.get_success().any()}
    
    def close(self) :

        self.env.close()

class DMCManipulationEnv() :

    def __init__(self, env : BaseManipulationEnv, max_step = 4):

        if isinstance(env, partial) :
            env = env()
        self.env = env
        self.max_step = max_step
        self.step_num = 0

        # obs = self.reset()
        # self.observation_space = convert_observation_to_space(obs)
        # self.state_space = convert_observation_to_space(obs)
        # self.action_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

    def observation_spec(self) :

        return specs.Array((10, IMAGE_SIZE, IMAGE_SIZE), np.float32, 'observation')
    
    def action_spec(self) :

        return specs.Array((8,), np.float32, 'action')
    
    def step(self, actions, **kwargs) :

        self.env.current_driving_target[-1] = actions[-1]
        self.env.gripper_move_to(actions[:7])
        self.step_num += 1

        obs = self._get_obsevation()

        return obs
    
    def reset(self) :

        self.env.reset()
        self.step_num = 0

        obs = self._get_obsevation()

        return obs

    def get_done(self) :

        return self.step_num >= self.max_step
    
    def _get_obsevation(self):

        original_obs = self.env.get_observation()
        original_img = self.env.get_image()
        color = cv2.resize(original_img['camera0']['Color'], dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
        hand_pose = original_obs['hand_pose']
        hand_pose = hand_pose[np.newaxis, np.newaxis, :] * np.ones((IMAGE_SIZE, IMAGE_SIZE, 1))
        # new_obs = np.concatenate((original_obs['cam1_rgb'], original_obs['cam2_rgb'], original_obs['cam1_depth'][..., np.newaxis], original_obs['cam2_depth'][..., np.newaxis]), axis=-1).transpose((2, 0, 1))
        new_obs = np.concatenate((hand_pose, color), axis=-1).transpose((2, 0, 1))
        reward = self.env.get_reward(None)
        succes = original_obs["success"]
        done = self.get_done()
        new_obs = new_obs.astype('float32')
        action = self.env.last_action.astype('float32')
        discount = np.array((1.0,), dtype='float32')

        class TMP:

            def __getitem__(self, key):

                if key == "observation" :

                    return new_obs

                elif key == "reward" :

                    return reward

                elif key == "action" :

                    return action
            
                elif key == "discount" :

                    return discount

                else :

                    raise KeyError(key)
            
            @property
            def success(self) :

                return succes
            
            @property
            def observation(self) :

                return new_obs

            @property
            def reward(self) :

                return reward

            @property
            def action(self) :

                return action

            def last(self) :

                return done

        return TMP()