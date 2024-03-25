from abc import abstractclassmethod
from typing import Union

import mplib
import numpy as np
import sapien.core as sapien
from gym.spaces import Box, Space
from sapien.core import renderer as R

import utils.logger
from env.sapien_envs.impedance_control import ImpedanceController
from env.sapien_envs.osc_planner import OSCPlanner
from utils.sapien_utils import *
from utils.tools import *
from utils.transform import *

from ..base_sapien_env import BaseEnv


CAMERA_INTRINSIC = [0.05, 100, 1, 640, 480]

def randomize_pose(xyz_low, xyz_high, rot, rot_low, rot_high) :
        xyz = np.random.uniform(xyz_low, xyz_high)
        rot = quat_mul(rot, axis_angle_to_quat([0,0,1], np.random.uniform(rot_low, rot_high)))
        return sapien.Pose(p=xyz, q=rot)

def randomize_dof(dof_low, dof_high) :
    if dof_low == 'None' or dof_high == 'None' :
        return None
    return np.random.uniform(dof_low, dof_high)

class BaseManipulationEnv(BaseEnv):
    def __init__(self,
                 obj_cfg,
                 task_cfg,
                 headless=False,
                 viewerless=False,
                 logger=None,
                 renderer: str = "sapien",
                 renderer_kwargs: dict = {}):
        self.total_move_distance = 0

        super().__init__(
            headless = headless,
            viewerless = viewerless,
            logger = logger,
            time_step = 1 / 360,
            renderer = renderer,
            renderer_kwargs = renderer_kwargs
        )


        # URDF loader
        self.loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        self.loader.fix_root_link = True
        # load objects
        # self.object_path = object_path
        self.step_count = 0
        self._prepare_data(obj_cfg, task_cfg)
        self._add_object(*self._generate_object_config())
        self._add_robot(*self._generate_robot_config())
        self._setup_planner()
        # Add some lights so that you can observe the scene
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        self.calculated_gripper_pose = {True: None, False: None}
        self.calculated_handle_pose = None
        self.calculated_camera_pose = {True: None, False: None}
        self.calculated_hand_pose = {True: None, False: None}

        # prepare buffer
        self.action_dof = 8
        self.last_action = np.zeros((self.action_dof,))

        # prepare obsservation & action spaces
        obs = regularize_dict(self.get_observation())
        state = regularize_dict(self.get_state())
        self.observation_space = convert_observation_to_space(obs)
        self.state_space = convert_observation_to_space(state)
        self.action_space = Box(
            low=np.asarray([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,  0.,      0.    ]),
            high=np.asarray([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,  0.04,    0.04  ])
        )

        self.draw_camera_frame = False

        if self.draw_camera_frame :

            self.cam_o = self._draw_point([0,0,0], color=[1, 0, 0], size=0.03)
            self.cam_x_axis = self._draw_point([0,0,0], color=[0, 1, 0], size=0.015)
            self.cam_y_axis = self._draw_point([0,0,0], color=[0, 0, 1], size=0.015)
            self.cam_z_axis = self._draw_point([0,0,0], color=[1, 0, 1], size=0.015)
        
        # self.handle_o = self._draw_point([0,0,0], color=[1, 0, 0], size=0.03)
        # self.handle_x_axis = self._draw_point([0,0,0], color=[0, 1, 0], size=0.015)
        # self.handle_y_axis = self._draw_point([0,0,0], color=[0, 0, 1], size=0.015)
        # self.handle_z_axis = self._draw_point([0,0,0], color=[1, 0, 1], size=0.015)

        # self.camera_o = self._draw_point([0,0,0], color=[1, 0, 0], size=0.03)
        # self.camera_x = self._draw_point([0,0,0], color=[0, 1, 0], size=0.015)

        self.reset()

    @abstractclassmethod
    def _generate_object_config(self) :

        '''
        Generate object pose, dof and path from randomization params
        '''

        pass

    def _load_object_config(self, cfg) :

        '''
        Load object pose, dof and path from config file
        '''
        
        path = cfg["path"]
        dof = cfg["dof"]
        pose = sapien.Pose(
            p=cfg["pose_7d"][:3],
            q=cfg["pose_7d"][3:]
        )
        return path, dof, pose


    def _generate_robot_config(self) :

        '''
        Generate robot pose, dof and path from randomization params
        '''

        pose = randomize_pose(
            self.robot_init_xyz_low,
            self.robot_init_xyz_high,
            self.robot_init_rot,
            self.robot_init_rot_low,
            self.robot_init_rot_high
        )
        dof = randomize_dof(
            self.robot_init_dof_low,
            self.robot_init_dof_high
        )
        path = self.robot_path

        self.current_robot_config = {
            "path": path,
            "dof": dof,
            "pose": pose.to_transformation_matrix()
        }

        return path, dof, pose

    def _load_robot_config(self, cfg) :

        '''
        Load robot pose, dof and path from rndoming config
        '''

        path = cfg["path"]
        dof = cfg["dof"]
        pose = sapien.Pose.from_transformation_matrix(cfg["pose"])

        return path, dof, pose

    @abstractclassmethod
    def _prepare_data(self, obj_cfg : dict, task_cfg : dict) :

        '''
        Preload dataset and randomization params from config file
        '''

        pass
        
    def _setup_planner(self) :

        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        self.pinocchio = self.robot.create_pinocchio_model()

        self.path_planner = mplib.Planner(
            urdf = self.robot_path,
            srdf = self.robot_path.replace(".urdf", ".srdf"),
            user_link_names = link_names,
            user_joint_names = joint_names,
            move_group="panda_hand",
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7)
        )

        self.osc_planner = OSCPlanner(
            self.pinocchio,
            9,
            damping = 0.05,
            qmask = np.asarray([1, 1, 1, 1, 1, 1, 1, 0, 0]),
            dt = 0.1
        )

        # self.impedance_controller = ImpedanceController(
        #     self.pinocchio,
        #     9,
        #     damping = 0.05,
        #     qmask = np.asarray([1, 1, 1, 1, 1, 1, 1, 0, 0]),
        #     dt = 0.1
        # )
    
    def _change_object(self, config=None):
        '''
        Change the object in the env
        '''

        if self.renderer_type == 'sapien' :
            self.scene.remove_articulation(self.obj)
            if config is None :
                self._add_object(*self._generate_object_config())
            else :
                self._add_object(*self._load_object_config(config))
        elif self.renderer_type == 'client' :
            # remove_articulation not supported in client
            # So only change the randomization params
            path, dof, pose = self._generate_object_config()
            self.obj.set_qpos(dof)
            self.obj.set_root_pose(pose)
            self.obj_root_pose = pose
            self.obj_init_dof = dof
        pass
    
    def _change_robot(self, config=None):
        '''
        Change the robot pose in the env
        '''

        if config is None :
            path, dof, pose = self._generate_robot_config()
        else :
            path, dof, pose = self._load_robot_config(config)

        if pose == None :
            pose = sapien.Pose([0, 0, 0.5], [1, 0, 0, 0])
        elif isinstance(pose, list) :
            pose = sapien.Pose(p=pose[:3], q=pose[3:])
        if dof is None :
            dof = (self.arm_q_higher + self.arm_q_lower) / 2

        self.current_driving_target = dof

        self.robot.set_root_pose(pose)
        self.robot.set_qpos(dof)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        self.robot.set_qf(np.zeros(self.robot.dof))
        self.robot.set_qacc(np.zeros(self.robot.dof))
        self.robot_root_pose = pose
        self.robot_init_dof = dof
    
    def _set_part_mask(self, active_link) :
        '''
        Set the mask of critical part of the object to 255
        '''

        pass

    def _add_object(self, object_path, dof_value : Union[float, list], pose : Union[sapien.Pose, list]=None):
        '''
        Add an object to the scene
        '''

        self.active_link = "link_" + object_path.split("/")[-2].split("_")[2]

        # prepare dof
        if dof_value == None :
            dof_value = 0
        elif isinstance(dof_value, list):
            dof_value = np.array(dof_value)
        else :
            dof_value = np.array([dof_value])
        
        # prepare pose
        if pose == None :
            pose = sapien.Pose([0, 0, 0.5], [1, 0, 0, 0])
        elif isinstance(pose, list) :
            pose = sapien.Pose(p=pose[:3], q=pose[3:])
        
        urdf_config = {
            "_materials": {
                "gripper" : {
                    "static_friction": 2.0,
                    "dynamic_friction": 2.0,
                    "restitution": 0.0
                }
            },
            "link": {
                self.active_link: {
                    "material": "gripper",
                    "density": 1.0,
                }
            }
        }
        config = parse_urdf_config(urdf_config, self.scene)
        check_urdf_config(config)
        
        self.obj_id = object_path.split("/")[-2]
        self.logger.info("Object ID: {}".format(self.obj_id))
        self.obj: sapien.Articulation = self.loader.load(object_path, config)
        assert(self.obj)
        for joint in self.obj.get_active_joints():
            joint.set_drive_property(stiffness=0, damping=0.01)
        self.obj.set_root_pose(pose)
        self.obj.set_qpos(dof_value)
        self.obj.set_qvel(np.zeros_like(dof_value))

        self.obj_root_pose = pose
        self.obj_init_dof = dof_value

        self._set_part_mask(self.active_link)

    def _add_robot(self, robot_path, dof_value : list,  pose : sapien.Pose=None):
        '''
        Add a robot to the scene
        '''

        urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
            ),
            link=dict(
                panda_leftfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                panda_rightfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
            ),
        )
        config = parse_urdf_config(urdf_config, self.scene)
        check_urdf_config(config)
        self.robot: sapien.Articulation = self.loader.load(robot_path, config)
        self.arm_qlimit = self.robot.get_qlimits()
        self.arm_q_lower = self.arm_qlimit[:, 0]
        self.arm_q_higher = self.arm_qlimit[:, 1]

        init_qpos = dof_value
        if dof_value is None :
            init_qpos = (self.arm_q_higher + self.arm_q_lower) / 2
        if pose == None :
            pose = sapien.Pose([-1.1, 0, 0.05], [0, 0, 0, 1])
        elif isinstance(pose, list) :
            pose = sapien.Pose(p=pose[:3], q=pose[3:])
        
        # Setup control properties
        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints[:4]:
            joint.set_drive_property(stiffness=160, damping=40, force_limit=10)    # original: 200
        for joint in self.active_joints[4:-2]:
            joint.set_drive_property(stiffness=160, damping=40, force_limit=5)    # original: 200
        for joint in self.active_joints[-2:]:
            joint.set_drive_property(stiffness=4000, damping=10)
        
        self.current_driving_target = init_qpos

        self.robot.set_root_pose(pose)
        self.robot.set_qpos(init_qpos)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        self.robot.set_qf(np.zeros(self.robot.dof))
        self.robot.set_qacc(np.zeros(self.robot.dof))
        # self.robot.set_qacc(np.zeros(self.robot.dof))
        # self.robot.set_qf(np.zeros(self.robot.dof))
        self.robot_root_pose = pose
        self.robot_init_dof = init_qpos

        self.hand_actor = get_entity_by_name(self.robot.get_links(), "panda_hand")
        hand_cam_pose = self.task_cfg["robot_conf"]["hand_cam_pose"]
        rotation = sapien.Pose(p=[0, 0, 0], q=axis_angle_to_quat([0, 1, 0], np.pi/2))
        self.hand_cam_pose = sapien.Pose(p=hand_cam_pose["xyz"], q=hand_cam_pose["rot"])
        self.user_hand_cam_pose = self.hand_cam_pose #rotation * 

        if not self.viewerless :
            self.camera_1, self.camera_1_intrinsic, self.camera_1_extrinsic = self.get_viewer(
                CAMERA_INTRINSIC,
                self.hand_cam_pose,
                mount=self.hand_actor,
            )
        
        # set id for getting mask
        for link in self.robot.get_links():
            for s in link.get_visual_bodies():
                s.set_visual_id(0)
    
    def _release_target(self) :

        for i in range(7) :
            self.current_driving_target[i] = self.robot.get_qpos()[i]
    
    def _move_to(self, pose, time=2, wait=1, planner="ik", robot_frame=False, skip_move=False, no_collision_with_front=True) :
        '''
        Move the end effector to a target pose
        '''

        if isinstance(pose, list) or isinstance(pose, np.ndarray):
            pose = sapien.Pose(p=pose[:3], q=pose[3:])
        elif isinstance(pose, sapien.Pose) :
            pass
        else :
            raise ValueError("Pose type not supported")

        # target_point = self._draw_point(pose.p, size=0.05, name="target")

        if not robot_frame :
            target_to_robot = self.robot_root_pose.inv() * pose
        else :
            target_to_robot = pose

        run_step = int(time / self.time_step)
        wait_step = int(wait / self.time_step)

        if not hasattr(self, "last_action_pose") :
            self.last_action_pose = pose
        self.total_move_distance += np.linalg.norm(self.last_action_pose.p - pose.p)

        if skip_move :

            if planner == "ik" :

                self.logger.error("IK cannot be skipped")
                assert(0)

            elif planner == "path" :

                result = self.path_planner.plan(
                    np.concatenate((target_to_robot.p, target_to_robot.q)),
                    self.robot.get_qpos(),
                    time_step=self.time_step,
                    use_point_cloud=False
                )

                if result['status'] != "Success":
                    
                    self.logger.warning("Path planner failed, use ik planner instead")
                    self._move_to(pose, time=time, wait=wait, planner="ik", robot_frame=robot_frame)

                    return False, run_step + wait_step
                
                self.logger.info("Skipped execution")

                new_qpos = np.zeros(self.robot.dof)
                new_qpos[:7] = result['position'][-1][:7]
                new_qpos[7:] = self.current_driving_target[-1]
                self.robot.set_qpos(new_qpos)
                self.robot.set_qvel(np.zeros(self.robot.dof))
                self.robot.set_qacc(np.zeros(self.robot.dof))
                action = np.zeros(self.action_dof)
                action[:7] = result['position'][-1][:7]
                action[7] = self.current_driving_target[7]
                self.step(action, drive_mode="pos", quite=True)

                for i in range(wait_step) :
                    self.step(action, drive_mode="pos", quite=True)

                return True, result['position'].shape[0] + wait_step

        else :

            if planner == "ik" :

                for i in range(run_step) :

                    if i%10 == 0 :
                        result, success, error = self.osc_planner.control_ik(
                            target_to_robot,
                            self.robot.get_qpos()
                        )
                    action = np.zeros(self.action_dof)
                    action[:7] = (result[:7] - self.current_driving_target[:7]) / (run_step-i)
                    action[-1] = self.current_driving_target[-1]
                    self.step(action, drive_mode="delta", quite=True)
                
                for i in range(wait_step) :

                    action = np.zeros(self.action_dof)
                    action[:7] = result[:7]
                    action[-1] = self.current_driving_target[-1]
                    self.step(action, drive_mode="pos", quite=True)

                return True, run_step + wait_step

            elif planner == "path" :

                import trimesh
                box = trimesh.creation.box([0.01, 1.6, 1.6])
                points, _ = trimesh.sample.sample_surface(box, 4096)

                handle_pose = self.handle_pose()

                mat = (self.robot_root_pose.inv() * self.handle_pose()).to_transformation_matrix()
                # mat = self.handle_pose().to_transformation_matrix()
                points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
                points = np.matmul(mat, points.T).T[:, :3] + quat_to_axis(handle_pose.q, 2) * 0.17

                # for point in points :
                #     self._draw_point(point, size=0.01, name="target")
                self.path_planner.update_point_cloud(points)
                result = self.path_planner.plan(
                    np.concatenate((target_to_robot.p, target_to_robot.q)),
                    self.robot.get_qpos(),
                    time_step=self.time_step,
                    use_point_cloud=no_collision_with_front
                )

                if result['status'] != "Success":
                    
                    self.logger.warning("Path planner failed, use ik planner instead")
                    self._move_to(pose, time=time, wait=wait, planner="ik", robot_frame=robot_frame)

                    return False, run_step + wait_step

                for i in range(result['position'].shape[0]):  
                    action = np.zeros(self.action_dof)
                    action[:7] = result['position'][i][:7]
                    action[-1] = self.current_driving_target[-1]
                    self.step(action, drive_mode="pos", quite=True)
                
                for i in range(wait_step) :

                    action = np.zeros(self.action_dof)
                    action[:7] = result['position'][-1][:7]
                    action[-1] = self.current_driving_target[-1]
                    self.step(action, drive_mode="pos", quite=True)
                
                return True, result['position'].shape[0] + wait_step

            else :

                raise ValueError("Planner type [{}] not supported".format(planner))

    def cam_move_to(self, pose, time=1, wait=2, planner="ik", robot_frame=False, skip_move=False, no_collision_with_front=True) :
        '''
        Move the camera on hand to a target pose
        '''

        if isinstance(pose, np.ndarray) :
            pose = sapien.Pose(p=pose[:3], q=pose[3:])

        hand_pose = pose * self.user_hand_cam_pose.inv()
        # self.camera_to = self._draw_point(pose.p, color=[0, 0, 1], size=0.03)

        return self._move_to(
            hand_pose,
            time=time,
            wait=wait,
            planner=planner,
            robot_frame=robot_frame,
            skip_move=skip_move,
            no_collision_with_front=no_collision_with_front
        )
    
    def hand_move_to(self, pose, time=2, wait=1, planner="ik", robot_frame=False, skip_move=False, no_collision_with_front=True) :
        '''
        Move the hand to a target pose
        '''

        return self._move_to(
            pose,
            time=time,
            wait=wait,
            planner=planner,
            robot_frame=robot_frame,
            skip_move=skip_move,
            no_collision_with_front=no_collision_with_front
        )
    
    def gripper_move_to(self, pose, time=2, wait=1, planner="ik", robot_frame=False, skip_move=False, no_collision_with_front=True) :
        '''
        Move the gripper to a target pose
        '''

        open_dir = quat_to_axis(pose[3:], 2) * 0.105
        new_pose = sapien.Pose(p=pose[:3]-open_dir, q=pose[3:])
        return self.hand_move_to(
            new_pose,
            time,
            wait,
            planner,
            robot_frame,
            skip_move,
            no_collision_with_front
        )

    @abstractclassmethod
    def handle_pose(self) :
        '''
        Get the pose of handle
        '''

        pass

    def hand_pose(self, robot_frame=False) :
        '''
        Get the pose of end effector
        '''

        if self.calculated_hand_pose[robot_frame] != None :
            return self.calculated_hand_pose[robot_frame]

        pose = self.hand_actor.get_pose()

        if robot_frame :
            pose = self.robot_root_pose.inv() * pose

        self.calculated_hand_pose[robot_frame] = pose
        return pose

    def camera_pose(self, robot_frame=False) :

        if self.calculated_camera_pose[robot_frame] != None :
            return self.calculated_camera_pose[robot_frame]

        hand_pose = self.hand_pose(robot_frame=robot_frame)
        pose = hand_pose * self.user_hand_cam_pose

        self.calculated_camera_pose[robot_frame] = pose
        return pose

    def gripper_pose(self, robot_frame=False) :
        '''
        Get the pose of gripper, which is 10cm away from the end effector
        '''
        if self.calculated_gripper_pose[robot_frame] != None :
            return self.calculated_gripper_pose[robot_frame]

        pose = self.hand_pose(robot_frame=robot_frame)
        open_dir = quat_to_axis(pose.q, 2) * 0.105
        self.calculated_gripper_pose[robot_frame] = sapien.Pose(p=pose.p+open_dir, q=pose.q)

        return self.calculated_gripper_pose[robot_frame]

    def robot_qpos(self) :

        return self.robot.get_qpos()
    
    def get_success(self) :

        return False

    def get_image(self, mask="handle") :
        '''
        Take picture from robot hand
        '''

        if self.renderer_type == 'sapien' :
            self.scene.update_render()
            for cam in self.registered_cameras :
                cam.take_picture()
        elif self.renderer_type == 'client' :
            self.scene._update_render_and_take_pictures(self.registered_cameras)

        images = {}
        if self.renderer_type == 'sapien' :
            for cam in self.registered_viewers:
                intrinsic, extrinsic = cam.get_param()
                segmentation = cam.get_segmentation()[:, :, 0]
                if mask == "handle" :
                    seg = (segmentation == 129)
                else :
                    seg = np.logical_or(segmentation == 128, segmentation == 129)
                images[cam.name] = {
                    'Color': cam.get_rgba()[:, :, :3],
                    'Position': cam.get_pos(),
                    'Depth': cam.get_depth(),
                    'Norm': cam.get_norm(),
                    'Mask': seg,
                    'Intrinsic': intrinsic,
                    'Extrinsic': extrinsic
                }

        elif self.renderer_type == 'client' :
            raise NotImplementedError("Client renderer not implemented yet")

        return images

    def get_observation(self, gt=False) :
        '''
        Get the observation of the environment as a dict
        If gt=True, return ground truth bbox
        '''

        if gt :
            observation = {
                "robot_qpos": self.robot.get_qpos(),
                "hand_pose": pose_to_array(self.hand_pose()),
                "gripper_pose": pose_to_array(self.gripper_pose()),
                "pose_difference": pose_to_array(self.gripper_pose().inv() * self.handle_pose()),
                "last_action": self.last_action,
                "total_move_distance": np.asarray(self.total_move_distance, dtype=np.float32),
            }
        else :
            observation = {
                "robot_qpos": self.robot.get_qpos(),
                "hand_pose": pose_to_array(self.hand_pose()),
                "gripper_pose": pose_to_array(self.gripper_pose()),
                "pose_difference": pose_to_array(self.gripper_pose().inv() * self.handle_pose()),
                "last_action": self.last_action,
                "total_move_distance": np.asarray(self.total_move_distance, dtype=np.float32),
            }
        
        # if self.renderer_type == 'client' :
        #     # provide np.array observation
        #     observation = regularize_dict(regularize_dict)

        return observation
    
    def get_state(self) :
        '''
        Get the state of the environment as a dict
        '''
        state = self.get_observation()
        return state
    
    def get_reward(self, action) :

        return 0

    def get_done(self) :

        return self.step_count >= self.task_cfg["max_step"]
    
    def step(self, action, gt=False, drive_mode="delta", quite=False) :
        '''
        Step the environment with actions
        '''

        action = np.asarray(action)

        qf = self.robot.compute_passive_force(
            gravity=True, 
            coriolis_and_centrifugal=True,
            external=False
        )
        self.robot.set_qf(qf)
        if drive_mode == "delta" :
            self.current_driving_target[:7] += action[:7]
        elif drive_mode == "pos" :
            self.current_driving_target[:7] = action[:7]
        else :
            raise ValueError("drive_mode should be either delta or pos")
        self.current_driving_target[:7] = np.clip(self.current_driving_target[:7], self.arm_q_lower[:7], self.arm_q_higher[:7])
        action[-1] = np.clip(action[-1], self.arm_q_lower[-1], self.arm_q_higher[-1])
        self.current_driving_target[-1] = action[-1]
        self.current_driving_target[-2] = action[-1]
        # action[:7] = (self.arm_q_lower[:7] + self.arm_q_higher[:7])/2 + action[:7]
        qf = self.robot.compute_passive_force(
            gravity=True, 
            coriolis_and_centrifugal=True,
            external=False
        )
        self.robot.set_qf(qf)
        for j in range(9):
            self.active_joints[j].set_drive_target(self.current_driving_target[j])
            # self.active_joints[j].set_drive_velocity_target(action[j+7])

        # self.camera_o.set_pose(self.camera_pose())
        # self.camera_x.set_pose(sapien.Pose(p=0.1*quat_to_axis(self.camera_pose().q, 0)) * self.camera_pose())

        if self.draw_camera_frame :

            self.cam_o.set_pose(self.camera_pose())
            self.cam_x_axis.set_pose(sapien.Pose(p=0.1*quat_to_axis(self.camera_pose().q, 0)) * self.camera_pose())
            self.cam_y_axis.set_pose(sapien.Pose(p=0.1*quat_to_axis(self.camera_pose().q, 1)) * self.camera_pose())
            self.cam_z_axis.set_pose(sapien.Pose(p=0.1*quat_to_axis(self.camera_pose().q, 2)) * self.camera_pose())
        
        self._step_simulation()

        obs = None
        if not quite :
            obs = self.get_observation(gt=gt)

        done = self.get_done()

        self.last_action = action

        rew = None
        if not quite :
            rew = self.get_reward(action)

        # self.eff_x_axis.set_pose(sapien.Pose(p=self.gripper_pose().p + quat_to_axis(self.gripper_pose().q, 0)*0.1))
        # self.eff_y_axis.set_pose(sapien.Pose(p=self.gripper_pose().p + quat_to_axis(self.gripper_pose().q, 1)*0.1))
        # self.eff_z_axis.set_pose(sapien.Pose(p=self.gripper_pose().p + quat_to_axis(self.gripper_pose().q, 2)*0.1))
        # self.eff_o.set_pose(self.gripper_pose())

        # self.handle_x_axis.set_pose(sapien.Pose(p=self.handle_pose().p + quat_to_axis(self.handle_pose().q, 0)*0.1))
        # self.handle_y_axis.set_pose(sapien.Pose(p=self.handle_pose().p + quat_to_axis(self.handle_pose().q, 1)*0.1))
        # self.handle_z_axis.set_pose(sapien.Pose(p=self.handle_pose().p + quat_to_axis(self.handle_pose().q, 2)*0.1))
        # self.handle_o.set_pose(self.handle_pose())

        return obs, rew, done, {}

    def _step_simulation(self, force_refresh=False) :

        self.scene.step()
        self.calculated_gripper_pose = {True: None, False: None}
        self.calculated_handle_pose = None
        self.calculated_camera_pose = {True: None, False: None}
        self.calculated_hand_pose = {True: None, False: None}
        self.step_count += 1
        if (self.step_count % 8 == 0 or force_refresh) and not self.headless:
            self.scene.update_render()
            self.main_viewer.render()
    
    def toggle_gripper(self, open=True) :
        '''
        Open or close the gripper
        '''
        
        for i in range(40): 
            action = np.zeros(self.action_dof)
            if open :
                action[-1] = 0.04
            else :
                action[-1] = 0.0
            self.step(action)
    
    def reset(self, gt=False) :

        self._change_robot()
        self._change_object()
        self._step_simulation(force_refresh=True)
        self.step_count = 0
        self.last_action = np.zeros((self.action_dof,))
        self.total_move_distance = 0
        
        return self.get_observation(gt=gt)
    
    def load(self, cfg) :

        self._change_robot(cfg["robot_config"])
        self._change_object(cfg["obj_config"])
        self._step_simulation(force_refresh=True)
        self.step_count = 0
        self.last_action = np.zeros((self.action_dof,))
        self.total_move_distance = 0
