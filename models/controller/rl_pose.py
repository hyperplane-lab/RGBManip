from models.pose_estimator.base_estimator import BasePoseEstimator
from models.manipulation.open_cabinet import OpenCabinetManipulation
from models.controller.base_controller import BaseController
from env.sapien_envs.open_cabinet import CAMERA_INTRINSIC
from env.my_vec_env import MultiVecEnv
from utils.transform import *
from utils.tools import *
from logging import Logger
from gym.spaces import Space, Box
from algo.ppo.ppo import PPO
import os
import torch

class ControlInterface():

    def __init__(self,
                vec_env : MultiVecEnv,
                pose_estimator : BasePoseEstimator,
                manipulation : OpenCabinetManipulation,
                cfg : dict
        ) :

        self.env = vec_env
        self.estimator = pose_estimator
        self.manipulation = manipulation
        self.num_envs = self.env.num_envs
        self.max_steps = cfg["controller"]["max_steps"] + 1
        self.action_type = cfg["controller"]["action_type"]
        if self.action_type == "pose" :
            self.pose_min = np.asarray(cfg["controller"]["pose_min"])
            self.pose_max = np.asarray(cfg["controller"]["pose_max"])
            self.pose_mid = (self.pose_min + self.pose_max) / 2
        self.cfg = cfg

        self.action_space = Box(low=-1.5, high=1.5, shape=(7+self.max_steps,)) # xyz and lookat
        self.state_space = Box(low=-1.5, high=1.5, shape=(self.max_steps*15,))
        self.observation_space = Box(low=-1.5, high=1.5, shape=(self.max_steps*12,))

        self.last_pose_target = None
        self.accumulate_steps = 0

        self.reset_queue()

        self.proper_pos = np.asarray([[0.0, 0.0, 0.9]])
        self.proper_ori = np.asarray([[1.0, 0.0, -0.2]])

        self.last_done = np.zeros((self.num_envs,))

        self.obj_saved_num = {}
        self.save_path = "saves/third_stage"
        if not os.path.exists(self.save_path) :
            os.makedirs(self.save_path)

        self.reset_robot()

    def _save_data(self) :

        current_obj_config = self.env.get_attr("current_obj_config")
        first_view_idx = np.clip(self.available_num - 1, 0, None)
        second_view_idx = np.clip(self.available_num - 2, 0, None)

        for id, obj_config in enumerate(current_obj_config) :
            obj = obj_config["name"]
            if obj not in self.obj_saved_num :
                self.obj_saved_num[obj] = 0
            self.obj_saved_num[obj] += 1

            idf = self.obj_saved_num[obj]
            id1 = first_view_idx[id]
            id2 = second_view_idx[id]

            root = os.path.join(self.save_path, obj, str(idf))
            if not os.path.exists(root) :
                os.makedirs(root)

            np.savez_compressed(os.path.join(root, "camera_intrinsic.npy"), self.intrinsic_queue[id1])
            np.savez_compressed(os.path.join(root, "rgb1.npy"), self.image_queue[id1])
            np.savez_compressed(os.path.join(root, "rgb2.npy"), self.image_queue[id2])
            np.savez_compressed(os.path.join(root, "view1_mask.npy"), self.mask_queue[id1])
            np.savez_compressed(os.path.join(root, "view2_mask.npy"), self.mask_queue[id2])
            np.savez_compressed(os.path.join(root, "view1_extrinsic.npy"), self.extrinsic_queue[id1])
            np.savez_compressed(os.path.join(root, "view2_extrinsic.npy"), self.extrinsic_queue[id2])
            np.savez_compressed(os.path.join(root, "ground_truth.npy"), self.gt_bbox[-1])

    def reset_queue(self) :

        self.image_queue = np.zeros((self.max_steps, self.num_envs, CAMERA_INTRINSIC[-1], CAMERA_INTRINSIC[-2], 3))
        self.mask_queue = np.zeros((self.max_steps, self.num_envs, CAMERA_INTRINSIC[-1], CAMERA_INTRINSIC[-2]))
        self.bbox_queue = np.zeros((self.max_steps, self.num_envs, 4))
        self.pose_queue = np.zeros((self.max_steps, self.num_envs, 7))
        self.intrinsic_queue = np.zeros((self.max_steps, self.num_envs, 3, 3))
        self.extrinsic_queue = np.zeros((self.max_steps, self.num_envs, 4, 4))
        self.available = np.zeros((self.max_steps, self.num_envs))
        self.pred_bbox = np.zeros((self.max_steps, self.num_envs, 8, 3))
        self.gt_bbox = np.zeros((self.max_steps, self.num_envs, 8, 3))
        self.available_num = np.zeros((self.num_envs,), dtype=np.int32)
        self.accumulate_steps = 0

    def reset_robot(self) :

        # gt_bbox = self.env.get_observation(gt=True)["handle_bbox"]
        # gt_center = (gt_bbox[:, 0] + gt_bbox[:, 6]) / 2
        # pos = np.broadcast_to(self.proper_pos[:, :3], (self.num_envs, 3))
        # ori = lookat_quat((gt_center-self.env.robot_pose()[:, :3])-pos)
        # pose = np.concatenate((pos, ori), axis=-1)
        # self.env.cam_move_to(pose, time=2, wait=1, planner="path", robot_frame=True, skip_move=True)
        pos = np.zeros((3, ))
        pos[0] = self.pose_min[0]
        pos[1] = 0
        pos[2] = (self.pose_min[2] + self.pose_max[2]) / 2
        ori = lookat_quat(self.proper_ori[0])
        pose = np.concatenate((pos, ori), axis=-1)
        self.env.cam_move_to(pose, time=2, wait=1, planner="path", robot_frame=True, skip_move=True)
        image = self.env.get_image()
        self.add_view(image, self.env.camera_pose(robot_frame=True))
        self.accumulate_steps += 1

    def add_view(self, image, cam_pose) :

        insert_id = self.accumulate_steps % self.max_steps

        self.image_queue[insert_id] = image["camera0"]["Color"]
        self.mask_queue[insert_id] = image["camera0"]["Mask"]
        self.pose_queue[insert_id] = cam_pose
        self.intrinsic_queue[insert_id] = image["camera0"]["Intrinsic"]
        self.extrinsic_queue[insert_id] = image["camera0"]["Extrinsic"]

        # show_image(image["camera0"]["Mask"][0]*255)

        p_env, p_x, p_y = np.nonzero(image["camera0"]["Mask"])
        for i in range(self.num_envs) :
            if p_env.shape[0] :
                x_min = np.min(np.where(p_env == i, p_x, CAMERA_INTRINSIC[-1]*2))
                x_max = np.max(np.where(p_env == i, p_x, 0))
                y_min = np.min(np.where(p_env == i, p_y, CAMERA_INTRINSIC[-2]*2))
                y_max = np.max(np.where(p_env == i, p_y, 0))
                self.available[insert_id, i] = 1
                self.available_num[i] += 1
            else :
                x_min = CAMERA_INTRINSIC[-1]*2
                x_max = 0
                y_min = CAMERA_INTRINSIC[-2]*2
                y_max = 0
                self.available[insert_id, i] = 0
            self.bbox_queue[insert_id, i] = np.asarray([
                x_min/CAMERA_INTRINSIC[-1],
                y_min/CAMERA_INTRINSIC[-2],
                x_max/CAMERA_INTRINSIC[-1],
                y_max/CAMERA_INTRINSIC[-2]
            ])

    def add_bbox(self, pred_bbox, gt_bbox) :

        insert_id = self.accumulate_steps % self.max_steps
        self.pred_bbox[insert_id] = pred_bbox
        self.gt_bbox[insert_id] = gt_bbox

    def get_state(self) :

        cur_pose_state = torch.tensor(self.pose_queue).float()
        cur_bbox_state = torch.tensor(self.bbox_queue).float()
        cur_handle_pos = torch.tensor((self.gt_bbox[:, :, 0] + self.gt_bbox[:, :, 6])/2).float()
        one_cur_time = torch.nn.functional.one_hot(
            torch.tensor(self.accumulate_steps-1),
            num_classes=self.max_steps
        )
        cur_time = torch.broadcast_to(one_cur_time.view(1, -1), (self.num_envs, self.max_steps))
        cur_state = torch.cat((cur_pose_state, cur_bbox_state, cur_handle_pos), dim=-1)
        ret = cur_state.permute(1, 0, 2).reshape(self.num_envs, -1)

        return torch.cat((ret, cur_time), dim=-1)

    def get_observation(self) :

        cur_pose_state = torch.tensor(self.pose_queue).float()
        cur_bbox_state = torch.tensor(self.bbox_queue).float()
        # cur_handle_pos = torch.tensor((self.pred_bbox[:, :, 0] + self.pred_bbox[:, :, 7])/2).float()
        # cur_state = torch.cat((cur_pose_state, cur_bbox_state, cur_handle_pos), dim=-1)
        one_cur_time = torch.nn.functional.one_hot(
            torch.tensor(self.accumulate_steps-1),
            num_classes=self.max_steps
        )
        cur_time = torch.broadcast_to(one_cur_time.view(1, -1), (self.num_envs, self.max_steps))
        cur_state = torch.cat((cur_pose_state, cur_bbox_state), dim=-1)
        ret = cur_state.permute(1, 0, 2).reshape(self.num_envs, -1)

        return torch.cat((ret, cur_time), dim=-1)

    def get_estimation(self) :
        '''
        Call the pose estimator to get the current pose estimation
        '''

        camera_intrinsic_batch = np.zeros((2, self.num_envs, 3, 3))
        camera_extrinsic_batch = np.zeros((2, self.num_envs, 4, 4))
        rgb_batch = np.zeros((2, self.num_envs, CAMERA_INTRINSIC[-1], CAMERA_INTRINSIC[-2], 3))
        mask_batch = np.zeros((2, self.num_envs, CAMERA_INTRINSIC[-1], CAMERA_INTRINSIC[-2]))
        used_idx = np.zeros((self.num_envs,), dtype=np.int32)


        for i in range(self.max_steps) :
            for j in range(self.num_envs) :
                if self.available[i, j] :
                    camera_intrinsic_batch[used_idx[j]%2, j] = self.intrinsic_queue[i, j]
                    camera_extrinsic_batch[used_idx[j]%2, j] = self.extrinsic_queue[i, j]
                    rgb_batch[used_idx[j]%2, j] = self.image_queue[i, j]
                    mask_batch[used_idx[j]%2, j] = self.mask_queue[i, j]
                    used_idx[j] += 1

        bbox = self.estimator.estimate(
            camera_intrinsic_batch[0],
            rgb_batch[0],
            mask_batch[0],
            camera_extrinsic_batch[0],
            rgb_batch[1],
            mask_batch[1],
            camera_extrinsic_batch[1]
        )

        if self.estimator.cfg["task_name"] == "mugs" :
            bbox = bbox[:, [0, 2, 4, 6, 1, 3, 5, 7]]

        return bbox

    def get_reward(self, action, move_res, view_weight, success) :

        view_norm = np.linalg.norm(view_weight, axis=-1, keepdims=True)
        view_weight = view_weight / (view_norm + 1e-9)
        view_norm_penalty = np.clip((view_norm[:, 0]-1)**2, -1, 1)

        cam_pose = self.env.camera_pose(robot_frame=True)

        ori = quat_to_axis(cam_pose[:, 3:], 0)

        move_success, move_period = move_res
        move_success = move_success.astype(np.float32)
        move_period = move_period.astype(np.float32)
        move_period = np.clip(move_period, 0, 1024)

        if self.action_type == "pose" :
            diff = np.clip(np.linalg.norm((cam_pose - self.last_pose_target), axis=-1), -2, 2)
        elif self.action_type == "joint" :
            diff = np.zeros((self.num_envs))
        else :
            raise TypeError
        far_diff = np.clip(np.linalg.norm(cam_pose[:, :3] - self.proper_pos, axis=-1), -2, 2)
        far_rew = far_diff
        last_bbox = self.bbox_queue[self.accumulate_steps % self.max_steps]
        bbox_dist = np.linalg.norm((last_bbox[:, :2] + last_bbox[:, 2:]) / 2 - np.array([[0.5, 0.5]]), axis=-1)
        bbox_dist = bbox_dist * self.available[self.accumulate_steps % self.max_steps]
        bbox_penalty = np.clip(bbox_dist, -1, 1)
        bbox_boundary_penalty = ((last_bbox[:, 0]<=1e-9) + (last_bbox[:, 1]<=1e-9) + (last_bbox[:, 2]>=1-(1e-9)) + (last_bbox[:, 3]>=1-(1e-9)) > 0).astype(np.float32)
        have_bbox_rew = self.available[self.accumulate_steps % self.max_steps]

        gt_center = (self.gt_bbox[self.accumulate_steps, :, 0] + self.gt_bbox[self.accumulate_steps, :, 6]) / 2
        gt_open_dir = self.gt_bbox[self.accumulate_steps, :, 0] - self.gt_bbox[self.accumulate_steps, :, 4]
        gt_open_dir = gt_open_dir / (np.linalg.norm(gt_open_dir, axis=-1, keepdims=True) + 1e-9)

        pred_center = (self.pred_bbox[self.accumulate_steps, :, 0] + self.pred_bbox[self.accumulate_steps, :, 7]) / 2
        pred_open_dir = self.pred_bbox[self.accumulate_steps, :, 1] - self.pred_bbox[self.accumulate_steps, :, 0]
        pred_open_dir = pred_open_dir / (np.linalg.norm(pred_open_dir, axis=-1, keepdims=True) + 1e-9)

        if self.estimator.cfg["task_name"] == "pots" :

            # pots require higher resolution for xy plane
            center_diff = pred_center - gt_center
            center_diff[:, :2] *= 3
            center_diff = np.clip(np.linalg.norm(center_diff, axis=-1), -20.0, 20.0)
            open_diff = np.clip(np.linalg.norm(pred_open_dir - gt_open_dir, axis=-1)*2, -20.0, 20.0)

        else :


            center_diff = np.clip(np.linalg.norm(pred_center - gt_center, axis=-1), -20.0, 20.0)
            open_diff = np.clip(np.linalg.norm(pred_open_dir - gt_open_dir, axis=-1)*2, -20.0, 20.0)

        if self.estimator.cfg["task_name"] == "mugs":

            # Mugs are small, requires better precision
            precision = 0.1

        else :

            precision = 0.2

        center_rew =  precision**2 / (precision**2 + center_diff ** 2)
        open_rew =  1 / (1 + open_diff ** 2)

        robot_root = self.env.robot_pose()[:, :3]
        tar_ori = gt_center - (robot_root + self.pose_queue[self.accumulate_steps, :, 0:3])
        tar_ori = tar_ori / (np.linalg.norm(tar_ori, axis=-1, keepdims=True) + 1e-9)
        ori_rew = (ori * tar_ori).sum(axis=-1)

        if self.action_type == "pose" :
            xyz_lookat = np.clip((np.linalg.norm(action[:, 3:6] - action[:, :3], axis=-1) - 1)**2, -2, 2)
        elif self.action_type == "joint" :
            xyz_lookat = np.zeros((self.num_envs))
        else :
            raise TypeError

        last_view_dir = self.pose_queue[self.accumulate_steps-1, :, :3] - (gt_center-robot_root)
        last_view_dir = last_view_dir / (np.linalg.norm(last_view_dir, axis=-1, keepdims=True) + 1e-9)
        this_view_dir = self.pose_queue[self.accumulate_steps, :, :3] - (gt_center-robot_root)
        this_view_dir = this_view_dir / (np.linalg.norm(this_view_dir, axis=-1, keepdims=True) + 1e-9)

        move_period = np.linalg.norm(
            self.pose_queue[self.accumulate_steps-1, :, :3] - self.pose_queue[self.accumulate_steps, :, :3],
            axis=-1
        )

        view_rew = np.zeros((self.num_envs,))
        if self.accumulate_steps > 0 :
            view_rew = np.arccos(np.sum(last_view_dir * this_view_dir, axis=-1))
            view_rew = np.where(view_rew > 0.3, 1.0, 0.0)
        else :
            center_rew *= 0
            open_rew *= 0

        diff *= self.cfg["reward"]["diff_coef"]
        move_success *= self.cfg["reward"]["move_success_coef"]
        move_period *= self.cfg["reward"]["move_period_coef"]
        far_rew *= self.cfg["reward"]["far_coef"]
        ori_rew *= self.cfg["reward"]["ori_coef"]
        xyz_lookat *= self.cfg["reward"]["xyz_lookat_coef"]
        bbox_penalty *= self.cfg["reward"]["bbox_coef"]
        bbox_boundary_penalty *= self.cfg["reward"]["bbox_boundary_coef"]
        have_bbox_rew *= self.cfg["reward"]["have_bbox_coef"]
        center_rew *= self.cfg["reward"]["center_coef"]
        open_rew *= self.cfg["reward"]["open_coef"]
        view_rew *= self.cfg["reward"]["view_coef"]
        view_norm_penalty *= self.cfg["reward"]["view_norm_coef"]
        success_reward = success * self.cfg["reward"]["success_coef"]

        reward = diff + move_success + move_period + far_rew + ori_rew + xyz_lookat + bbox_penalty + bbox_boundary_penalty + have_bbox_rew + center_rew + open_rew + view_rew + view_norm_penalty + success_reward

        info = {
            "REW:diff": torch.from_numpy(diff),
            "REW:move_success": torch.from_numpy(move_success),
            "REW:move_period": torch.from_numpy(move_period),
            "REW:far": torch.from_numpy(far_rew),
            "REW:ori_rew": torch.from_numpy(ori_rew),
            "REW:xyz_lookat": torch.from_numpy(xyz_lookat),
            "REW:bbox_penalty": torch.from_numpy(bbox_penalty),
            "REW:bbox_boundary_penalty": torch.from_numpy(bbox_boundary_penalty),
            "REW:have_bbox": torch.from_numpy(have_bbox_rew),
            "REW:center_rew": torch.from_numpy(center_rew),
            "REW:open_rew": torch.from_numpy(open_rew),
            "REW:view_rew": torch.from_numpy(view_rew),
            "REW:view_norm_penalty": torch.from_numpy(view_norm_penalty),
            "REW:success": torch.from_numpy(success_reward),
            "LOSS:center_diff": torch.from_numpy(center_diff),
            "LOSS:open_diff": torch.from_numpy(open_diff),
            "LOSS:far": torch.from_numpy(far_diff),
        }

        # print(view_weight, view_norm_penalty, open_rew, center_rew)

        return torch.tensor(reward), info

    def get_done(self) :

        return torch.ones((self.num_envs,)).bool() * (self.max_steps <= self.accumulate_steps)

    def call_manipulation(self, estimation, eval) :

        center = (estimation[:, 0] + estimation[:, 7]) / 2
        direction = np.zeros((estimation.shape[0], 3, 3))
        direction[:, 0] = estimation[:, 1] - estimation[:, 0]
        direction[:, 1] = estimation[:, 0] - estimation[:, 2]
        direction[:, 2] = estimation[:, 4] - estimation[:, 0]
        frame = np.zeros((estimation.shape[0], 3, 3))
        frame[:, 0, 0] = 1
        frame[:, 1, 1] = 1
        frame[:, 2, 2] = 1
        d_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        direction = direction / (d_norm + 1e-8)
        direction = np.where(d_norm > 1e-8, direction, frame)
        self.manipulation.plan_pathway(center, direction, eval)

    def step(self, action, eval=False) :
        '''
        Translate control signals to environment actions
        '''

        if self.last_done.any() :
            self.reset()

        action = action.cpu().numpy()

        weight = action[:, 6:6+self.max_steps]

        # move the camera in environment and gather observations
        if self.action_type == "pose" :
            xyz = action[:, :3]
            heading = np.zeros((self.num_envs, 3))
            dy = action[:, 3]
            dz = action[:, 4]
            # lookat = action[:, 3:6]
            z_ = np.zeros((self.num_envs, 3))
            z_[:, 2] = 1
            heading[:, 0] = 1
            lookat_len = np.linalg.norm(heading, axis=-1, keepdims=True)
            lookat_norm = heading / (lookat_len + 1e-9)
            lookat_y = np.cross(z_, lookat_norm)
            ori = lookat_quat(lookat_norm + lookat_y * dy[:, None] + z_ * dz[:, None])
            xyz = np.clip(xyz + self.pose_mid, self.pose_min, self.pose_max)
            env_action = np.concatenate([xyz, ori], axis=1)
            self.last_pose_target = env_action
            no_collision = False
            if self.cfg["task"]["name"] in ["cabinet", "drawer"] :
                no_collision = True
            move_res = self.env.cam_move_to(
                env_action,
                time=2,
                wait=0.5,
                planner="path",
                robot_frame=True,
                skip_move=not eval,
                no_collision_with_front=no_collision
            )
        elif self.action_type == "joint" :
            low = self.env.action_space.low[None, :7]
            high = self.env.action_space.high[None, :7]
            env_action = action[:, :7] * (high-low) * 0.5 + (low+high) * 0.5
            for i in range(1024) :
                self.env.step(env_action[:, :7], drive_mode="pos", quite=True)
            qpos = self.env.robot_qpos()
            err = np.linalg.norm(qpos[:, :7] - env_action, axis=-1)
            move_res = np.where(err<0.1, 1.0, 0.0), np.ones((self.num_envs,))
        image = self.env.get_image()
        self.add_view(image, self.env.camera_pose(robot_frame=True))

        pred_bbox = self.get_estimation()
        gt_bbox = self.env.get_observation(gt=True)["handle_bbox"]

        self.add_bbox(pred_bbox, gt_bbox)
        obs = self.get_observation()
        success = np.zeros((self.num_envs,))
        if self.accumulate_steps == self.max_steps - 1 and self.cfg["reward"]["success_coef"] > 1e-9 and not eval:
            self.call_manipulation(pred_bbox, eval=True)
            success = self.env.get_observation(gt=True)["success"][:, 0]
        reward, info = self.get_reward(action, move_res, weight, success)

        self.accumulate_steps += 1

        if self.accumulate_steps == self.max_steps - 1 and eval:
            self._save_data()

        done = self.get_done()

        self.last_done = done

        return obs, reward, done, info

    def reset(self, indicies=None, reset_env=True) :

        if reset_env :
            obs = self.env.reset(indicies)
        self.reset_queue()
        self.reset_robot()

        return self.get_observation()

class RLPoseController(BaseController) :

    def __init__(self,
                 vec_env : MultiVecEnv,
                 pose_estimator : BasePoseEstimator,
                 manipulation : OpenCabinetManipulation,
                 cfg : dict,
                 logger : Logger
                ):

        super().__init__(vec_env, pose_estimator, manipulation, cfg, logger)

        self.control_interface = ControlInterface(vec_env, pose_estimator, manipulation, cfg)

        self.controller = PPO(self.control_interface, cfg)

    def train_controller(self, steps, log_interval=1, save_interval=1) :
        '''
        Train controller model only
        '''

        self.logger.info("Training controller model...")
        self.controller.run(
            steps,
            log_interval,
            save_interval
        )

    def run(self, eval=False) :

        from algo.ppo.ppo import prepare_obs

        current_obs, _ = prepare_obs(self.control_interface.reset(reset_env = False))
        current_obs = current_obs.to(self.controller.device)
        weight = None
        cur_step = 0
        max_step = self.cfg["controller"]["early_stop"]
        while True :
            cur_step += 1
            actions = self.controller.actor_critic.act_inference(current_obs)
            weight = actions[:, 6:6+self.control_interface.max_steps].cpu().numpy()
            next_obs, rews, dones, infos = self.control_interface.step(actions, eval=True)
            next_obs, _ = prepare_obs(next_obs)
            next_obs = next_obs.to(self.controller.device)
            current_obs.copy_(next_obs)

            if dones.any() or cur_step >= max_step:
                break

        estimation = self.control_interface.pred_bbox[cur_step]

        self.control_interface.call_manipulation(estimation, eval)
        return
