import os, time
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bullet_client
import gymnasium as gym
from gymnasium import spaces
from robot import UR5Robotiq85, Gripper
from camera import Camera
import transformations as tf
import utils
from utils import Pybullet_Utils
from uuid import uuid4
import open3d as o3d



class Environment(gym.Env):

    def __init__(self, vis=False):

        self.debug_frame = dict()
        self.workspace_limits = np.asarray([[0.304, 0.752], [-0.02, 0.428], [0, 0.4]])
        anchor_offset_limits = self.workspace_limits * 100
        
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3, 5000), dtype=np.float16)
        # action_params: [:3] anchor m; [3:6] anchor offset cm; [6:9] axis_y; [9] approach angle;
        action_down = np.array([-np.inf, -np.inf, -np.inf, anchor_offset_limits[0,0], anchor_offset_limits[1,0], anchor_offset_limits[2,0], 0, 0, 0, -np.pi/2])
        action_up = np.array([np.inf, np.inf, np.inf, anchor_offset_limits[0,1], anchor_offset_limits[1,1], anchor_offset_limits[2,1], 1, 1, 1, np.pi/2])
        self.action_space = spaces.Box(action_down, action_up, shape=(10,), dtype=np.float16)

        # Connect to simulator
        self._vis = vis
        conn_mode = p.GUI if vis else p.DIRECT
        self.sim = bullet_client.BulletClient(connection_mode=conn_mode)
        self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.sim.setGravity(0, 0, -10)
        self.pb_utils = Pybullet_Utils(simulator=self.sim)
        self.mesh_to_urdf = {}

        self.env_body = dict()
        plane_id = self.sim.loadURDF("plane100.urdf", useMaximalCoordinates=True)
        self.env_body['plane'] = plane_id
        work_floor_id = self.sim.loadURDF('resources/objects/work_floor.urdf', basePosition=[0.528, 0.204, 0], useMaximalCoordinates=True)
        self.env_body['work_floor'] = work_floor_id

        # self.robot = UR5Robotiq85(sim=self.sim)
        # self.robot.load()
        # self.robot.step_simulation = self.step_simulation

        # load gripper
        self.gripper = Gripper(simulator=self.sim, pybullet_utils=self.pb_utils)
        self.gripper.step_simulation = self._step_simulation
        self.env_body['gripper'] = self.gripper.id
        self._draw_body_pose(body_id=self.gripper.id, link_id=3, line_width=5)
        # self._add_debug_frame('grasp', np.eye(4))

        # Setup virtual camera in simulation
        K = np.array([2257.7500557850776, 0,                  1032,
                        0,                  2257.4882391629421, 772,
                        0,                  0,                  1]).reshape(3,3)
        H = 1544   # image height
        W = 2064   # image width
        self.camera = Camera(K, H, W)
        self.camera.id = self._load_mesh(f'resources/camera/kinectsensor.obj', 
                                         ob_in_world=self.camera.view_matrix, mass=0, 
                                         has_collision=False)
        self.camera.getCameraImage = self._getCameraImage
        self.env_body['camera'] = self.camera.id
        
        # add object
        self.obj_dir = 'resources/objects/blocks'
        self.num_obj = 1
        self.mesh_list = os.listdir(self.obj_dir)
        self.obj_mesh_ind = np.random.randint(0, 8, self.num_obj)
        self.obj_ids = []
        

    #---------------------------------------------------------------------------
    # Standard Gym Functions
    #---------------------------------------------------------------------------
    def seed(self, seed=None):
        
        self._random = np.random.RandomState(seed)
        return seed


    def reset(self, seed = None, options = None):
        
        super().reset(seed=seed, options=options)
        
        self.gripper.reset()
        
        self._clean_objects()
        drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
        drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
        self.object_in_world = np.eye(4)
        self.object_in_world[:3,3] = [drop_x, drop_y, 0.20]
        add_success = self._add_objects()
        while add_success == False:
            add_success = self._add_objects()

        observation = self._get_obs()
        
        return observation, {}


    def step(self, action_param):
        
        self.grasp_matrix_in_camera = self._param_to_grasp(grasp_param=action_param)
        grasp_in_world = self.camera.view_matrix @ self.grasp_matrix_in_camera
        gripper_in_world = grasp_in_world @ self.gripper.gripper_in_grasp

        # if self.debug_frame['grasp'] != None:
        #     self.debug_frame['grasp']['pose_in_world'] = grasp_in_world
        # self._update_debug_frame()

        self.gripper.open()
        self.pb_utils.set_body_pose_in_world(self.gripper.id, gripper_in_world)
        contact_pts_gripper_obj = self.sim.getContactPoints(bodyA=self.gripper.id, bodyB=self.obj_ids[0])
        contact_pts_gripper_plane = self.sim.getContactPoints(bodyA=self.gripper.id, bodyB=self.env_body['plane'])
        if len(contact_pts_gripper_obj) != 0 or len(contact_pts_gripper_plane) != 0: # gripper invoke target object
            terminated = False
        else:
            self.gripper.close()
            terminated = self._is_grasp_success()

        if terminated == False:
            self.pb_utils.set_body_pose_in_world(self.obj_ids[0], self.object_in_world)
            self.pb_utils.set_body_pose_in_world(self.gripper.id, np.eye(4))

        observation = self._get_obs()
        info = {}
        reward = int(terminated)

        return observation, reward, terminated, False, info


    def close(self):
        
        self.sim.disconnect()


    #---------------------------------------------------------------------------
    # Environment Control Functions
    #---------------------------------------------------------------------------
    def _step_simulation(self, delay=None):
       
        self.sim.stepSimulation()
        if self._vis and delay != None:
            time.sleep(delay)


    def _check_out_workspace(self, obj_id):
        
        ob_in_world = self.pb_utils.get_ob_pose_in_world(obj_id) 
        position_x = ob_in_world[0,3]
        position_y = ob_in_world[1,3]
        position_z = ob_in_world[2,3]
        if  position_x > self.workspace_limits[0,1] or position_x < self.workspace_limits[0,0] or\
            position_y > self.workspace_limits[1,1] or position_y > self.workspace_limits[1,1] or\
            position_z > self.workspace_limits[2,1] or position_z > self.workspace_limits[2,1]:
            return True
        else:
            return False

        
    def _simulation_until_stable(self):

        while 1:
            last_poses = {}
            accum_motions = {}
        
            for body_id in self.obj_ids:
                if self._check_out_workspace(body_id):
                    self._remove_object(body_id)
                    continue
                last_poses[body_id] = self.pb_utils.get_ob_pose_in_world(body_id) 
                accum_motions[body_id] = 0

            stabled = True
            for _ in range(50):
                p.stepSimulation()
                for body_id in self.obj_ids:
                    cur_pose = self.pb_utils.get_ob_pose_in_world(body_id)
                    motion = np.linalg.norm(cur_pose[:3,3] - last_poses[body_id][:3,3])
                    accum_motions[body_id] += motion
                    last_poses[body_id] = cur_pose.copy()
                    if accum_motions[body_id]>=0.001:
                        stabled = False
                        break
                if stabled==False:
                    break
        
            if stabled:
                for body_id in self.obj_ids:
                    self.sim.resetBaseVelocity(body_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0])
                break


    def _add_objects(self):
        
        # for object_idx in self.obj_mesh_ind:
        curr_mesh_file = f'resources/objects/blocks/{4}.obj'
        
        # drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
        # drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
        # R = tf.random_rotation_matrix(np.random.rand(3))
        # object_in_world = np.eye(4)
        # object_in_world[:3,3] = [drop_x, drop_y, 0.15]
        # object_in_world[:3,:3] = R[:3,:3]
        obj_id = self._load_mesh(obj_file=curr_mesh_file, mass=0.1, ob_in_world=self.object_in_world)
        self.sim.changeDynamics(obj_id, -1, lateralFriction=0.7, spinningFriction=0.7, collisionMargin=0.0001)
        self.obj_ids.append(obj_id)
        # self._simulation_until_stable()
        
        if len(self.obj_ids) < self.num_obj:
            return False            
        else:
            return True


    def _remove_object(self, obj_id):
        
        self.sim.removeBody(obj_id)
        self.obj_ids.remove(obj_id)


    def _get_obs(self):
            
        rgb, depth, seg = self.camera.shot()
        bg_mask = depth<0.1
        for name, id in self.env_body.items():
            bg_mask[seg==id] = 1
        pts = utils.depth2xyzmap(depth, self.camera._k)
        pts_no_bg = pts[bg_mask==0].reshape(-1,3) # (N,3)
        if pts_no_bg.shape[0] > self.observation_space.shape[1]:
            num_pts = self.observation_space.shape[1]
            downsample_indxs = utils.farthest_point_sample(pts=pts_no_bg, npoint=num_pts)
            obs = pts_no_bg[downsample_indxs]
        else:
            obs = np.zeros((self.observation_space.shape[1], 3))
        obs = obs.T
        
        return obs


    def _clean_objects(self):
        
        for ob_id in self.obj_ids:
            self._remove_object(obj_id=ob_id)


    def _param_to_grasp(self, grasp_param):
        
        action = grasp_param
        anchor = action[0:3]
        anchor_off = action[3:6] * 0.01 # cm --> m
        axis_y = action[6:9] 
        theta = action[-1] # approach angle
        
        norm_y = np.linalg.norm(axis_y)
        axis_y = axis_y / norm_y if norm_y > 0 else [0, 1, 0]
        axis_x = [axis_y[1], -axis_y[0], 0]
        norm_x = np.linalg.norm(axis_x)
        axis_x = axis_x / norm_x if norm_x > 0 else [1, 0, 0]
        axis_z = np.cross(axis_x, axis_y)
        norm_z = np.linalg.norm(axis_z)
        axis_z = axis_z / norm_z if norm_z > 0 else [0, 0, 1]

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, 0, sin_t],
                    [0    , 1,     0],
                    [-sin_t, 0, cos_t]])

        grasp_in_camera = np.eye(4)
        grasp_in_camera[:3,:3] = np.stack((axis_x, axis_y, axis_z)) @ R
        grasp_in_camera[:3,3] = anchor + anchor_off
        return grasp_in_camera


    def _is_grasp_success(self):

        is_grasped = False
        ob_init_pose = self.pb_utils.get_ob_pose_in_world(self.obj_ids[0])
        for _ in range(50):
            self.pb_utils.add_gravity_to_ob(self.obj_ids[0], gravity=-10)
            self._step_simulation()
        ob_final_pose = self.pb_utils.get_ob_pose_in_world(self.obj_ids[0])
        if np.linalg.norm(ob_init_pose[:3,3]-ob_final_pose[:3,3])<=0.02:
            is_grasped = True

        return is_grasped
        # lower, upper = self.pb_utils.get_joint_limits(self.gripper.id, self.gripper.finger_ids[0])
        # finger_joint_pos = self.pb_utils.get_joint_positions(self.gripper.id, self.gripper.finger_ids)
        # finger_joint_pos = np.array(finger_joint_pos)
        # self.sim.setGravity(0, 0, 0)
        # if np.all(np.abs(finger_joint_pos - upper) < 0.001):
        #     return False
        # return True
    

    def _getCameraImage(self, width, height, viewMatrix, projectionMatrix, shadow, lightDirection):
        
        width, height, rgb, depth, seg = self.sim.getCameraImage(width=width, height=height, viewMatrix=viewMatrix, 
                                                        projectionMatrix=projectionMatrix, shadow=shadow, 
                                                        lightDirection=lightDirection)
        return width, height, rgb, depth, seg


    #---------------------------------------------------------------------------
    # Helper Functions
    #---------------------------------------------------------------------------
    def _load_mesh(self, obj_file, ob_in_world, mass, scale=np.ones(3),has_collision=True,useFixedBase=False,concave=False,collision_margin=0.0001):

        if obj_file in self.mesh_to_urdf:
            urdf_dir = self.mesh_to_urdf[obj_file]
        else:
            urdf_dir = f'/tmp/{os.path.basename(obj_file)}_{uuid4()}.urdf'
            utils.create_urdf_from_mesh(obj_file,out_dir=urdf_dir,mass=mass,has_collision=has_collision,concave=concave,scale=scale)
            self.mesh_to_urdf[obj_file] = urdf_dir
        
        q_wxyz = tf.quaternion_from_matrix(ob_in_world)
        q_xyzw = [q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]
        ob_id = self.sim.loadURDF(urdf_dir, basePosition=ob_in_world[:3,3], baseOrientation=q_xyzw, useFixedBase=useFixedBase, 
                                flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES|p.URDF_USE_MATERIAL_COLORS_FROM_MTL|p.URDF_MAINTAIN_LINK_ORDER)
        self.pb_utils.set_body_pose_in_world(ob_id, ob_in_world)
        self.sim.changeDynamics(ob_id, -1, collisionMargin=collision_margin)
        return ob_id
    

    #---------------------------------------------------------------------------
    # Debug Function
    #---------------------------------------------------------------------------
    def log(self, log_dir='logs'):
        
        pcd = utils.toOpen3dCloud(self.last_obs)
        obs_path = f'{log_dir}/obs_{self._num_step}.ply'
        act_path = obs_path.replace('obs', 'act')
        o3d.io.write_point_cloud(obs_path, pcd)
        self.gripper.export(grasp_in_camera=self.grasp_matrix_in_camera, out_dir=act_path)

   
    def _draw_frame(self, frame_in_world, frame_name, line_width=3, replace_item_ids=None):

        axis_scale = 0.1
        frame_origin= frame_in_world[0:3,3]
        frame_direction = frame_in_world[0:3,0:3]

        kwargs = dict()
        kwargs['lineWidth'] = line_width

        kwargs['lineColorRGB'] = [1, 0, 0]
        if replace_item_ids is not None:
            kwargs['replaceItemUniqueId'] = replace_item_ids[0]
        axis_x_id = self.sim.addUserDebugLine(frame_origin, frame_origin + axis_scale * frame_direction[0:3, 0], **kwargs)

        kwargs['lineColorRGB'] = [0, 1, 0]
        if replace_item_ids is not None:
            kwargs['replaceItemUniqueId'] = replace_item_ids[1]
        axis_y_id = self.sim.addUserDebugLine(frame_origin, frame_origin + axis_scale * frame_direction[0:3, 1], **kwargs)

        kwargs['lineColorRGB'] = [0, 0, 1]
        if replace_item_ids is not None:
            kwargs['replaceItemUniqueId'] = replace_item_ids[2]
        axis_z_id = self.sim.addUserDebugLine(frame_origin, frame_origin + axis_scale * frame_direction[0:3, 2], **kwargs)

        kwargs.clear()
        if replace_item_ids is not None:
            kwargs['replaceItemUniqueId'] = replace_item_ids[3]
        frame_name_id = self.sim.addUserDebugText(frame_name, frame_origin, **kwargs)

        return [axis_x_id, axis_y_id, axis_z_id, frame_name_id]


    def _add_debug_frame(self, frame_name, pose_in_world):

        if self.debug_frame.get(frame_name) is None:
            data = dict()
            replace_item_ids = self._draw_frame(frame_in_world=pose_in_world, frame_name=frame_name)
            data['item_ids'] = replace_item_ids
            data['pose_in_world'] = pose_in_world
            self.debug_frame[frame_name] = data


    def _update_debug_frame(self):
    
        for name, frame_data in self.debug_frame.items():
            self._draw_frame(frame_in_world=frame_data['pose_in_world'], frame_name=name, replace_item_ids=frame_data['item_ids'])


    def _draw_body_pose(self, body_id, link_id=-1, line_width=3):

        if link_id != -1:
            self.sim.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0.1, 0, 0], lineColorRGB=[1, 0, 0], 
                                    lineWidth=line_width, parentObjectUniqueId=body_id, parentLinkIndex=link_id)
            self.sim.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, 0.1, 0], lineColorRGB=[0, 1, 0], 
                                    lineWidth=line_width, parentObjectUniqueId=body_id, parentLinkIndex=link_id)
            self.sim.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, 0, 0.1], lineColorRGB=[0, 0, 1], 
                                    lineWidth=line_width, parentObjectUniqueId=body_id, parentLinkIndex=link_id)
        else:
            self.sim.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0.1, 0, 0], lineColorRGB=[1, 0, 0], 
                                    lineWidth=line_width, parentObjectUniqueId=body_id)
            self.sim.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, 0.1, 0], lineColorRGB=[0, 1, 0], 
                                    lineWidth=line_width, parentObjectUniqueId=body_id)
            self.sim.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, 0, 0.1], lineColorRGB=[0, 0, 1], 
                                    lineWidth=line_width, parentObjectUniqueId=body_id)


    def high_contrast_body(self, body_id, alpha = 0.5):
        
        body_rgba = self.sim.getVisualShapeData(body_id)[0][6]
        for link_index in range(-1, self.sim.getNumJoints(body_id)):
            p.changeVisualShape(body_id, link_index, rgbaColor=[body_rgba[0], body_rgba[1], body_rgba[2], alpha])
    

    