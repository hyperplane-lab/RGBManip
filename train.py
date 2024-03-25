import os
import time
from functools import partial

import hydra
import numpy as np
import sapien.core as sapien
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
import pickle
import yaml

import utils.logger as logger
from env.base_sapien_env import BaseEnv
from env.my_vec_env import MultiVecEnv
from env.sapien_envs.open_cabinet import OpenCabinetEnv
from env.sapien_envs.open_pot import OpenPotEnv
from models.controller.base_controller import BaseController
from models.controller.collection import CollectionController
from models.controller.gt_pose import GtPoseController
from models.controller.heuristic_pose import HeuristicPoseController
from models.controller.rl_pose import RLPoseController
from models.controller.homing import HomingController
from models.controller.baseline import BaselineController
from models.manipulation.base_manipulation import BaseManipulation
from models.manipulation.open_cabinet import OpenCabinetManipulation
from models.manipulation.open_drawer import OpenDrawerManipulation
from models.manipulation.open_pot import OpenPotManipulation
from models.manipulation.pick_mug import PickMugManipulation
from models.manipulation.rl import RLManipulation
from models.pose_estimator.AdaPose.interface import AdaPoseEstimator
from models.pose_estimator.AdaPose.interface_v2 import AdaPoseEstimator_v2
from models.pose_estimator.AdaPose.interface_v3 import AdaPoseEstimator_v3
from models.pose_estimator.AdaPose.interface_v4 import AdaPoseEstimator_v4
from models.pose_estimator.AdaPose.interface_v5 import AdaPoseEstimator_v5
from models.pose_estimator.AdaPose.interface_baseline import AdaPoseEstimator_baseline
from models.pose_estimator.AdaPose.interface_realworld import AdaPoseEstimator_realworld
from models.pose_estimator.base_estimator import BasePoseEstimator
from models.pose_estimator.groundtruth_estimator import \
    GroundTruthPoseEstimator
from utils.transform import *

def prepare_env(task_cfg, data_cfg, headless, viewerless, log) -> BaseEnv :

    if task_cfg["name"] == "open_cabinet" :

        env_list = []
        for i in range(task_cfg["num_envs"]) :
            env_list.append(
                partial(
                    OpenCabinetEnv,
                    data_cfg,
                    task_cfg,
                    headless = headless,
                    viewerless = viewerless,
                    logger = log,
                    renderer = 'sapien'
                )
            )

        env = MultiVecEnv(env_list)

        return env

    if task_cfg["name"] == "open_drawer" :

        env_list = []
        for i in range(task_cfg["num_envs"]) :
            env_list.append(
                partial(
                    OpenCabinetEnv,
                    data_cfg,
                    task_cfg,
                    headless = headless,
                    viewerless = viewerless,
                    logger = log,
                    renderer = 'sapien'
                )
            )

        env = MultiVecEnv(env_list)

        return env

    elif task_cfg["name"] == "open_cabinet_visualize" :

        env = partial(
            OpenCabinetEnv,
            data_cfg,
            task_cfg,
            headless,
            viewerless,
            logger = log
        )

        return MultiVecEnv([env])

    elif task_cfg["name"] == "open_pot" :

        env_list = []
        for i in range(task_cfg["num_envs"]) :
            env_list.append(
                partial(
                    OpenPotEnv,
                    data_cfg,
                    task_cfg,
                    headless = headless,
                    viewerless = viewerless,
                    logger = log,
                    renderer = 'sapien'
                )
            )

        env = MultiVecEnv(env_list)

        return env

    elif task_cfg["name"] == "pick_mug" :

        env_list = []
        for i in range(task_cfg["num_envs"]) :
            env_list.append(
                partial(
                    OpenPotEnv,
                    data_cfg,
                    task_cfg,
                    headless = headless,
                    viewerless = viewerless,
                    logger = log,
                    renderer = 'sapien'
                )
            )

        env = MultiVecEnv(env_list)

        return env

    elif task_cfg["name"] == "real_world" :

        from env.realworld_envs.base_realworld import BaseRealworldEnv
        env = BaseRealworldEnv()

        return env

    else :

        raise NotImplementedError

def prepare_manipulation(manipulation_cfg, env, log, log_dir, save_dir) :

    if manipulation_cfg["name"] == "open_cabinet" :

        return OpenCabinetManipulation(env, manipulation_cfg, logger = log)

    elif manipulation_cfg["name"] == "open_drawer" :

        return OpenDrawerManipulation(env, manipulation_cfg, logger = log)

    elif manipulation_cfg["name"] == "open_pot" :

        return OpenPotManipulation(env, manipulation_cfg, logger = log)

    elif manipulation_cfg["name"] == "pick_mug" :

        return PickMugManipulation(env, manipulation_cfg, logger = log)

    elif manipulation_cfg["name"] == "rl" :

        manipulation_cfg["learn"]["log_dir"] = log_dir
        manipulation_cfg["learn"]["save_dir"] = save_dir

        return RLManipulation(env, manipulation_cfg, logger = log)

    else :

        raise NotImplementedError

def prepare_controller(controller_cfg, env, pose_estimator, manipulation, log, log_dir, save_dir) :

    if controller_cfg["name"] == "gt_pose" :

        return GtPoseController(env, pose_estimator, manipulation, controller_cfg, logger = log)

    elif controller_cfg["name"] == "heuristic_pose" :

        return HeuristicPoseController(env, pose_estimator, manipulation, controller_cfg, logger = log)

    elif controller_cfg["name"] == "rl" :

        controller_cfg["learn"]["log_dir"] = log_dir
        controller_cfg["learn"]["save_dir"] = save_dir

        return RLPoseController(env, pose_estimator, manipulation, controller_cfg, logger = log)

    elif controller_cfg["name"] == "collection" :

        controller_cfg["learn"]["log_dir"] = log_dir
        controller_cfg["learn"]["save_dir"] = save_dir

        return CollectionController(env, pose_estimator, manipulation, controller_cfg, logger = log)

    elif controller_cfg["name"] == "homing" :

        return HomingController(env, pose_estimator, manipulation, controller_cfg, logger = log)

    elif controller_cfg["name"] == "baseline" :

        return BaselineController(env, pose_estimator, manipulation, controller_cfg, logger = log)

    else :

        raise NotImplementedError

def prepare_pose_estimator(pose_estimator_cfg, env, log) :

    if pose_estimator_cfg["name"] == "ground_truth" :

        return GroundTruthPoseEstimator(env, pose_estimator_cfg, logger = log)

    elif pose_estimator_cfg["name"] == "adapose" :

        return AdaPoseEstimator(env, pose_estimator_cfg, logger = log)

    elif pose_estimator_cfg["name"] == "adapose_v2" :

        return AdaPoseEstimator_v2(env, pose_estimator_cfg, logger = log)

    elif pose_estimator_cfg["name"] == "adapose_v3" :

        return AdaPoseEstimator_v3(env, pose_estimator_cfg, logger = log)

    elif pose_estimator_cfg["name"] == "adapose_v4" :

        return AdaPoseEstimator_v4(env, pose_estimator_cfg, logger = log)

    elif pose_estimator_cfg["name"] == "adapose_v5" :

        return AdaPoseEstimator_v5(env, pose_estimator_cfg, logger = log)

    elif pose_estimator_cfg["name"] == "adapose_baseline" :

        return AdaPoseEstimator_baseline(env, pose_estimator_cfg, logger = log)

    elif pose_estimator_cfg["name"] == "adapose_realworld" :

        return AdaPoseEstimator_realworld(env, pose_estimator_cfg, logger = log)

    # elif pose_estimator_cfg["name"] == "adapose_handle" :

    #     return AdaPoseEstimator_v3(env, pose_estimator_cfg, logger = log)

    # elif pose_estimator_cfg["name"] == "adapose_pot" :

    #     return AdaPoseEstimator_v3(env, pose_estimator_cfg, logger = log)

    else :

        raise NotImplementedError

def test(env : MultiVecEnv, controller : BaseController, cfg : dict) :

    success = 0
    move_distance = 0
    total_num_traj = 0
    total_round = cfg["train"]["total_round"]

    for i in range(total_round) :

        logger.log.info("Test episode: %d" % i)
        controller.run()
        obs = env.get_observation()
        move_distance += np.sum(obs["total_move_distance"])
        success += np.sum(obs["success"])
        print(obs["success"][:, 0], obs["object_dof"][:, 0])
        total_num_traj += obs["success"].shape[0]
        env.reset()

    env.close()

    logger.log.info("Total round: %d" % total_num_traj)
    logger.log.info("Success round: %d" % success)
    logger.log.info("Success rate: %f" % (success/total_num_traj))
    logger.log.info("Average distance: %f" % (move_distance/total_num_traj))

def test_baseline(env : MultiVecEnv, controller : BaselineController, cfg : dict) :

    import open3d as o3d

    success = 0
    move_distance = 0
    total_num_traj = 0

    logger.log.info("Testing baseline controller.")
    task_setting_root = cfg["train"]["task_setting_root"]
    task_settings = {}
    for (root, dirs, file) in os.walk(task_setting_root):
        for f in file:
            if '.pickle' in f:
                task_setting = pickle.load(open(os.path.join(root, f), 'rb'))
                task_settings[f] = task_setting
                # print(f)
                # break
    action_path = cfg["train"]["action_path"]
    i = 0
    with open(action_path, 'r') as f:
        for line in f.readlines():
            # processing input file
            if "_w2a_report" in action_path :
                block = line.split(' ')
                block = [a for a in block if a != '' and a != '[' and a != ']']
                file_name = block[0]
                if ".pickle" not in file_name :
                    file_name += ".pickle"
                task_setting = task_settings[file_name]
                cx = int(block[1].split('(')[1].split(',')[0])
                cy = int(block[2].split(')')[0])
                # print(block)
                px = task_setting["observation"]["pic"]["camera0"]["Position"][cx][cy][0]
                py = task_setting["observation"]["pic"]["camera0"]["Position"][cx][cy][1]
                pz = task_setting["observation"]["pic"]["camera0"]["Position"][cx][cy][2]

                print(block)

                x_dx = float(block[4].split('[')[-1])
                x_dy = float(block[5])
                x_dz = float(block[6])
                y_dx = float(block[6])
                y_dy = float(block[7])
                y_dz = float(block[8].split(']')[0])

                x = np.array([x_dx, x_dy, x_dz])
                y = np.array([y_dx, y_dy, y_dz])
                z = np.cross(x, y)
                dx = x_dx
                dy = x_dy
                dz = x_dz

            else :
                block = line.split(', ')
                file_name = block[0]
                if ".pickle" not in file_name :
                    file_name += ".pickle"
                task_setting = task_settings[file_name]
                if ']' not in block[2] :
                    px = float(block[1].split('[')[1])
                    py = float(block[2])
                    pz = float(block[3].split(']')[0])
                    dir = block[4].split(' ')
                    dir = [a for a in dir if a != '' and a != '[']
                    dx = float(dir[0].split('[')[-1])
                    dy = float(dir[1])
                    dz = float(dir[2].split(']')[0])
                else :
                    cx = int(block[1].split('[')[1])
                    cy = int(block[2].split(']')[0])
                    px = task_setting["observation"]["pic"]["camera0"]["Position"][cx][cy][0]
                    py = task_setting["observation"]["pic"]["camera0"]["Position"][cx][cy][1]
                    pz = task_setting["observation"]["pic"]["camera0"]["Position"][cx][cy][2]
                    block = [a for a in block if a != '']
                    dx = float(block[3].split('[')[1])
                    dy = float(block[4])
                    dz = float(block[5].split(']')[0])
            action = np.array([px, py, pz, dx, dy, dz])
            logger.log.info("Test episode: %d" % i)

            setting = task_settings[file_name]
            controller.run(setting, action)
            obs = env.get_observation()
            move_distance += np.sum(obs["total_move_distance"])
            success += np.sum(obs["success"])
            print(env.get_observation()["success"])
            total_num_traj += env.get_observation()["success"].shape[0]
            i += 1

    env.close()

    logger.log.info("Total round: %d" % total_num_traj)
    logger.log.info("Success round: %d" % success)
    logger.log.info("Success rate: %f" % (success/total_num_traj))
    logger.log.info("Average distance: %f" % (move_distance/total_num_traj))

def collect(env : MultiVecEnv, controller : BaseController, cfg : dict) :

    total_round = cfg["train"]["total_round"]

    for i in range(total_round) :

        logger.log.info("Collect episode: %d" % i)
        controller.run()
        env.reset()

    env.close()

def train(env : BaseEnv, controller : BaseController, cfg : dict) :

    if "train_manipulation" in cfg["train"] and cfg["train"]["train_manipulation"] :
        controller.train_manipulation(
            cfg["train"]["iterations_per_epoch"],
            log_interval=cfg["train"]["log_interval"],
            save_interval=cfg["train"]["save_interval"]
        )

    elif "train_controller" in cfg["train"] and cfg["train"]["train_controller"] :
        controller.train_controller(
            cfg["train"]["iterations_per_epoch"],
            log_interval=cfg["train"]["log_interval"],
            save_interval=cfg["train"]["save_interval"]
        )

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def my_app(cfg) :
    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg_yaml = OmegaConf.to_yaml(cfg)

    # refresh log dir
    exp_name = cfg["exp_name"]
    cfg["controller"]["exp_name"] = exp_name
    cfg["controller"]["task"] = cfg["task"]
    global graph, start_time
    start_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    graph_path = os.path.join(cfg["train"]["log_dir"], cfg["train"]["name"], start_time + exp_name)
    if not logger.dry_run :
        graph = SummaryWriter(graph_path)
    save_path = os.path.join(cfg["train"]["save_dir"], cfg["train"]["name"], start_time + exp_name)
    env = prepare_env(cfg["task"], cfg["dataset"], cfg["headless"], cfg["viewerless"], logger.log)
    manipulation = prepare_manipulation(cfg["manipulation"], env, logger.log, log_dir=graph_path, save_dir=save_path)
    pose_estimator = prepare_pose_estimator(cfg["pose_estimator"], env, logger.log)
    controller = prepare_controller(cfg["controller"], env, pose_estimator, manipulation, logger.log, log_dir=graph_path, save_dir=save_path)

    if not os.path.exists(save_path) :
        os.makedirs(save_path)
    yaml.dump(cfg, open(os.path.join(save_path, "config.yaml"), "w"))

    logger.log.info("Loaded config.")
    logger.log.info("Graph save into {}.".format(graph_path))
    logger.log.info("Checkpoints save into {}.".format(save_path))
    logger.log.info("Env:{}".format(env))
    logger.log.info("Manipulation:{}".format(manipulation))
    logger.log.info("Pose Estimator:{}".format(pose_estimator))
    logger.log.info("Controller:{}".format(controller))

    logger.log.info("Start {}, experiment name {}.".format(cfg["train"]["name"], exp_name))

    if cfg["train"]["name"] == "test" :

        test(env, controller, cfg)

    elif cfg["train"]["name"] == "collect":

        collect(env, controller, cfg)

    elif cfg["train"]["name"] == "train" :

        train(env, controller, cfg)

    elif cfg["train"]["name"] == "test_baseline" :

        test_baseline(env, controller, cfg)

    else :

        raise NotImplementedError

    logger.log.info("Controller returned")
    logger.log.info("{} finished".format(exp_name))

if __name__ == "__main__" :

    my_app()
    pass
