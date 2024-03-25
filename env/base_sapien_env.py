from abc import abstractclassmethod
from logging import Logger
from typing import Union

import numpy as np
import sapien.core as sapien
from sapien.core import renderer as R
from sapien.utils import Viewer

from env.base_viewer import BaseViewer


class BaseEnv:
    def __init__(self,
                 headless,
                 viewerless,
                 logger : Logger,
                 time_step = 1/360,
                 renderer: str = "sapien",
                 renderer_kwargs: dict = {}):

        self.logger = logger
        
        self.time_step = time_step
        
        self.engine = sapien.Engine()  # Create a physical simulation engine
        self.renderer_type = renderer

        scene_config = sapien.SceneConfig()
        scene_config.default_dynamic_friction = 1.0
        scene_config.default_static_friction = 1.0
        scene_config.default_restitution = 0.0
        scene_config.contact_offset = 0.02
        scene_config.enable_pcm = False
        scene_config.solver_iterations = 25
        scene_config.solver_velocity_iterations = 1

        if self.renderer_type == "sapien" :
            self.renderer = sapien.SapienRenderer(**renderer_kwargs)  # Create a Vulkan renderer
            self.renderer_context: R.Context = self.renderer._internal_context
        elif self.renderer_type == "client" :
            scene_config.disable_collision_visual = True
            self.renderer = sapien.RenderClient(**renderer_kwargs)  # Create a clinet rendere in Sapien2.2
        else :
            raise NotImplementedError
        self.engine.set_renderer(self.renderer)  # Bind the renderer and the engine

        self.scene = self.engine.create_scene(scene_config)  # Create an instance of simulation world (aka scene)
        self.scene.set_timestep(self.time_step)  # Set the simulation frequency

        # NOTE: How to build actors (rigid bodies) is elaborated in create_actors.py
        self.scene.add_ground(altitude=0)  # Add a ground

        self.headless = headless
        self.viewerless = viewerless
        if not headless :
            self.main_viewer = Viewer(self.renderer)  # Create a viewer (window)
            self.main_viewer.set_scene(self.scene)  # Bind the viewer and the scene
            # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
            # The principle axis of the camera is the x-axis
            self.main_viewer.set_camera_xyz(x=-1, y=1, z=2)
            # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
            # The camera now looks at the origin
            self.main_viewer.set_camera_rpy(r=0, p=-np.arctan2(1, 1), y=np.arctan2(1, 1))
            self.main_viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

            # do not display axes of the world frame
            self.main_viewer.toggle_axes(False)
        
        # list of user-generated viewers
        self.registered_viewers = []
        self.registered_cameras = []
        self.viewer_less = viewerless

        self.observation_space = None
        self.action_space = None
        self.state_space = None

        self.debug_points = []
    
    def get_viewer(self, camera_intrinsics, camera_extrinsics : Union[sapien.Pose, np.ndarray], mount=None):
        '''
        create a new viewer and return it.
        camera_intrinsics: [near, far, fovy, width, height]
        camera_extrinsics: pose of the camera (sapien.Pose), or transformation of the camera (numpy).
        mount: actor or None.
        '''

        near, far = camera_intrinsics[0], camera_intrinsics[1]
        width, height = camera_intrinsics[3], camera_intrinsics[4]
        name = "camera"+str(len(self.registered_viewers))
        if mount is None :
            camera = self.scene.add_camera(
                name=name,
                width=width,
                height=height,
                fovy=camera_intrinsics[2],
                near=near,
                far=far
            )
        else :
            camera = self.scene.add_mounted_camera(
                name=name,
                actor=mount,
                pose=camera_extrinsics,
                width=width,
                height=height,
                fovy=camera_intrinsics[2],
                near=near,
                far=far
            )
        if isinstance(camera_extrinsics, sapien.Pose):
            pose = camera_extrinsics
        elif isinstance(camera_extrinsics, np.ndarray):
            pose = sapien.Pose.from_transformation_matrix(camera_extrinsics)
        else :
            raise ValueError('camera_extrinsics must be either a sapien.Pose or a numpy.ndarray')
        camera.set_local_pose(pose)

        # print('Intrinsic matrix\n', camera.get_intrinsic_matrix())
        # print('Extrinsic matrix\n', camera.get_extrinsic_matrix())
        self.registered_cameras.append(camera)

        # assign methods to the viewer
        def refresh() :
            self.scene.step()  # Simulate the world
            self.scene.update_render()  # Update the world to the renderer
            camera.take_picture()
        def get_rgba_method() :
            # refresh()
            rgba = camera.get_float_texture('Color')
            return rgba
        def get_depth_method() :
            # refresh()
            depth = camera.get_float_texture('Position')
            return -depth[..., 2]
        def get_norm_method() :
            # refresh()
            norm = camera.get_float_texture('Normal')
            new_norm = np.array(norm, dtype=np.float32)
            new_norm[:, :, 0] = -norm[:, :, 2]
            new_norm[:, :, 1] = -norm[:, :, 0]
            new_norm[:, :, 2] = norm[:, :, 1]
            return new_norm
        def get_pos_method():
            # refresh()
            pos = camera.get_float_texture('Position')
            model_matrix = camera.get_model_matrix()
            pos = pos[..., :3] @ model_matrix[:3, :3].T + model_matrix[:3, 3]
            return pos
        def get_segmentation_method() :
            # refresh()
            seg_labels = camera.get_uint32_texture('Segmentation')
            return seg_labels[..., :2]
        def update_param_method(new_camera_extrinsics) :
            camera.set_local_pose(sapien.Pose.from_transformation_matrix(new_camera_extrinsics))
        def get_param_method() :
            return camera.get_intrinsic_matrix(), camera.get_extrinsic_matrix()

        cur_viewer = BaseViewer(
            name,
            get_rgba_method,
            get_depth_method,
            get_pos_method,
            get_segmentation_method,
            update_param_method,
            get_param_method,
            get_norm_method
        )
        self.registered_viewers.append(cur_viewer)
        
        return cur_viewer, camera.get_intrinsic_matrix(), camera.get_extrinsic_matrix()
    
    @abstractclassmethod
    def step(self, action):
        '''
        Step the environment
        '''
        
        pass

    @abstractclassmethod
    def reset(self):
        '''
        Reset the environment
        '''

        pass

    def close(self) :
        '''
        Close the environment
        '''

        if not self.headless :
            self.main_viewer.close()
            self.main_viewer = None
        self.scene = None

        pass

    @abstractclassmethod
    def get_observation(self) :
        '''
        Get the observation of the environment as a dict
        '''

        observation = None

        return observation
    
    @abstractclassmethod
    def get_state(self) :
        '''
        Get the state of the environment as a dict
        '''

        state = None

        return state

    def _run(self):
        '''
        main_viewer render the scene
        '''
        if self.headless :
            self.logger.info('Environment running in headless mode')
            while True :
                self.scene.step()
                # self.scene.update_render()
        else :
            while not self.headless and not self.main_viewer.closed:  # Press key q to quit
                self.scene.step()  # Simulate the world
                self.scene.update_render()  # Update the world to the renderer
                # self.main_viewer.create_my_coordinate_axes([1, 0, 0])
                self.main_viewer.render()

    def _draw_point(self, xyz, size=0.01, color=[1, 0, 0], name="debug_point", ret=True):
        '''
        draw a debug point
        '''

        actor_builder = self.scene.create_actor_builder()
        # actor_builder.add_box_collision(half_size=[0.5, 0.5, 0.5])
        actor_builder.add_sphere_visual(radius=size, color=color)
        sphere = actor_builder.build(name=name)  # Add a box
        sphere.set_pose(sapien.Pose(p=xyz))
        sphere.lock_motion()
        self.debug_points.append(sphere)
        if ret :
            return sphere
        else :
            return None
    
    def _clear_point(self, actor=None):
        '''
        clear a debug point
        '''

        if actor is None :
            for actor in self.debug_points :
                self.scene.remove_actor(actor)
            self.debug_points = []
        else :
            self.scene.remove_actor(actor)