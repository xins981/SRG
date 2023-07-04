import os, time
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bullet_client
import gymnasium as gym
from gymnasium import spaces
from robot import UR5Robotiq85, Gripper
from camera import Camera
from transformations import *
import utils
from utils import Pybullet_Utils
from uuid import uuid4
import open3d as o3d



SIMULATION_STEP_DELAY = 1 / 240.

class Environment(gym.Env):

    def __init__(self, vis=True):

        action_down = np.array([-np.inf, -np.inf, -np.inf, -5, -5, -5, -1, -1, -1, -np.pi/2])
        action_up = np.array([np.inf, np.inf, np.inf, 5, 5, 5, 1, 1, 1, np.pi/2])
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5000,3), dtype=np.float32)
        self.action_space = spaces.Box(action_down, action_up, shape=(10,), dtype=np.float32)

        self.workspace_limits = np.asarray([[0.276, 0.724], [-0.224, 0.224], [-0.0001, 0.4]])

        # Connect to simulator
        self._vis = vis
        conn_mode = p.GUI if vis else p.DIRECT
        self.sim = bullet_client.BulletClient(connection_mode=conn_mode)
        self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.sim.setGravity(0, 0, -10)
        self.bullet_utils = Pybullet_Utils(sim=self.sim)
        self.mesh_to_urdf = {}
        
        
        self.env_body_ids = []
        _plane_id = self.sim.loadURDF("plane.urdf")
        self.env_body_ids.append(_plane_id)


        # self.robot = UR5Robotiq85(sim=self.sim)
        # self.robot.load()
        # self.robot.step_simulation = self.step_simulation

        # load gripper
        self.gripper = Gripper(sim=self.sim, bullet_utils=self.bullet_utils)
        self.gripper.step_simulation = self._step_simulation
        self.env_body_ids.append(self.gripper.id)
        
        # Setup virtual camera in simulation
        K = np.array([2257.7500557850776, 0,                  1032,
                        0,                  2257.4882391629421, 772,
                        0,                  0,                  1]).reshape(3,3)
        H = 1544   # image height
        W = 2064   # image width
        self.camera = Camera(K, H, W)
        self.camera.id = self._load_mesh(f'resources/camera/kinectsensor.obj', 
                                         ob_in_world=self.camera.pose_in_world, mass=0, 
                                         has_collision=False)
        self.camera.getCameraImage = self.getCameraImage
        self.env_body_ids.append(self.camera.id)
        
        # add object
        self.obj_dir = 'resources/objects/blocks'
        self.num_obj = 1
        self.mesh_list = os.listdir(self.obj_dir)
        self.obj_mesh_ind = np.random.randint(0, 8, self.num_obj)
        self.obj_ids = []
        add_success = self._add_objects()
        while add_success == False:
            add_success = self._add_objects()


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
        self._add_objects()
        self._num_step = 0

        observation = self._get_obs()
        self.last_obs = observation
        
        return observation, {}


    def step(self, action_param):
        self._num_step += 1
        self.grasp_in_camera_T, self.grasp_in_camera_quatern  = self._param_to_grasp(grasp_param=action_param)
        _grasp_in_world = self.camera.pose_in_world @ self.grasp_in_camera_T
        gripper_in_world = _grasp_in_world @ np.linalg.inv(self.gripper.grasp_in_gripper)
        
        self.gripper.open()
        self.bullet_utils.set_body_pose_in_world(self.gripper.id, gripper_in_world)
        self.gripper.close()
        
        terminated = self._is_grasp_success()
        observation = self._get_obs()
        self.last_obs = observation
        info = {}
        reward = int(terminated)

        return observation, reward, terminated, False, info


    def close(self):
        
        self.sim.disconnect()


    #---------------------------------------------------------------------------
    # Environment Control Functions
    #---------------------------------------------------------------------------
    def _step_simulation(self):
       
        self.sim.stepSimulation()
        if self._vis:
            time.sleep(SIMULATION_STEP_DELAY)


    def _check_out_workspace(self, obj_id):
        
        ob_in_world = self.bullet_utils.get_ob_pose_in_world(obj_id) 
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
                last_poses[body_id] = self.bullet_utils.get_ob_pose_in_world(body_id) 
                accum_motions[body_id] = 0

            stabled = True
            for _ in range(50):
                p.stepSimulation()
                for body_id in self.obj_ids:
                    cur_pose = self.bullet_utils.get_ob_pose_in_world(body_id)
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

        for object_idx in self.obj_mesh_ind:
            curr_mesh_file = f'resources/objects/blocks/{object_idx}.obj'
            
            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
            R = utils.random_rotation_matrix(np.random.rand(3))
            object_in_world = np.eye(4)
            object_in_world[:3,3] = [drop_x, drop_y, 0.15]
            object_in_world[:3,:3] = R[:3,:3]
            obj_id = self._load_mesh(obj_file=curr_mesh_file, mass=0.1, ob_in_world=object_in_world)
            self.obj_ids.append(obj_id)
            self._simulation_until_stable()
        
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
        for id in self.env_body_ids:
            bg_mask[seg==id] = 1
        pts = utils.depth2xyzmap(depth, self.camera._K)
        pts_no_bg = pts[bg_mask==0].reshape(-1,3)
        downsample_indxs = utils.farthest_point_sample(pts_no_bg, self.observation_space.shape[0])
        return pts_no_bg[downsample_indxs]


    def _clean_objects(self):
        
        for ob_id in self.obj_ids:
            self._remove_object(obj_id=ob_id)


    def _param_to_grasp(self, grasp_param):
        
        action = grasp_param.copy() # ensure action don't change
        center = action[0:3]
        cen_off = action[3:6] # cm
        axis_y = action[6:9] 
        theta = action[-1] # angle
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        R = [[cos_t, 0, sin_t],
            [0    , 1,     0],
            [sin_t, 0, cos_t]]

        norm_y = np.linalg.norm(axis_y)
        axis_y = axis_y / norm_y if norm_y > 0 else [0, 1, 0]

        axis_x = [axis_y[1], -axis_y[0], 0]
        norm_x = np.linalg.norm(axis_x)
        axis_x = axis_x / norm_x if norm_x > 0 else [1, 0, 0]

        axis_z = np.cross(axis_x, axis_y)
        norm_z = np.linalg.norm(axis_z)
        axis_z = axis_z / norm_z if norm_z > 0 else [0, 0, 1]

        matrix = np.stack((axis_x, axis_y, axis_z))
        matrix = np.matmul(matrix, R)

        approach = matrix[:, 0]
        norm_appr = np.linalg.norm(approach)
        approach = approach / norm_appr if norm_appr > 0 else [1, 0, 0]

        bin_normal = np.cross(approach, axis_y)
        norm_bin_normal = np.linalg.norm(bin_normal)
        bin_normal = bin_normal / norm_bin_normal if norm_bin_normal > 0 else [0, 0, 1]
        matrix = np.stack((approach,axis_y,bin_normal)).T
        pose = np.eye(4)
        pose[:3,:3] = matrix
        q_wxyz = utils.quaternion_from_matrix(pose)
        q_xyzw = [q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]
        trans = center + (cen_off * 0.01)
        pose[:3,3] = trans

        return pose, (trans, q_xyzw)


    def _is_grasp_success(self):
        
        lower, upper = self.bullet_utils.get_joint_limits(self.gripper.id, self.gripper.finger_ids[0])
        finger_joint_pos = self.bullet_utils.get_joint_positions(self.gripper.id, self.gripper.finger_ids)
        finger_joint_pos = np.array(finger_joint_pos)
        if np.all(np.abs(finger_joint_pos - upper) < 0.001):
            return False
        return True
    

    def getCameraImage(self, width, height, viewMatrix, projectionMatrix, shadow, lightDirection):
        
        w, h, rgb, depth, seg = self.sim.getCameraImage(width=width, height=height, viewMatrix=viewMatrix, 
                                                        projectionMatrix=projectionMatrix, shadow=shadow, 
                                                        lightDirection=lightDirection)
        return w, h, rgb, depth, seg


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
        
        q_wxyz = utils.quaternion_from_matrix(ob_in_world)
        q_xyzw = [q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]
        ob_id = self.sim.loadURDF(urdf_dir, basePosition=ob_in_world[:3,3], baseOrientation=q_xyzw, useFixedBase=useFixedBase, 
                                flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES|p.URDF_USE_MATERIAL_COLORS_FROM_MTL|p.URDF_MAINTAIN_LINK_ORDER)
        self.bullet_utils.set_body_pose_in_world(ob_id, ob_in_world)
        self.sim.changeDynamics(ob_id, -1, collisionMargin=collision_margin)
        return ob_id
    

    def log(self, log_dir='logs'):
        
        pcd = utils.toOpen3dCloud(self.last_obs)
        obs_path = f'{log_dir}/obs_{self._num_step}.ply'
        act_path = obs_path.replace('obs', 'act')
        o3d.io.write_point_cloud(obs_path, pcd)
        self.gripper.export(grasp_in_camera=self.grasp_in_camera_T, out_dir=act_path)
    

    