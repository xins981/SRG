import os, time
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bullet_client
import gymnasium as gym
from gymnasium import spaces
# from robot import UR5Robotiq85, Gripper
from camera import Camera
import transformations as tf
# import utils
# from utils import Pybullet_Utils
from uuid import uuid4
import open3d as o3d

from pybullet_tools.utils import *

from pybullet_tools.panda_primitives import *

from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver


HOME_JOINT_VALUES = [0.00, 0.074, 0.00, -1.113, 0.00, 1.510, 0.671, 0.04, 0.04]


class Environment(gym.Env):

    def __init__(self, reward_scale, vis=False):

        self.reward_scale = reward_scale

        self.workspace = np.asarray([[0.304, 0.752], 
                                     [-0.02, 0.428], 
                                     [0, 0.4]])
        approach_radian = np.pi / 2
        action_min = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -approach_radian])
        action_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, approach_radian])
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(25600, 3), dtype=np.float32)
        self.action_space = spaces.Box(action_min, action_max, shape=(7, ), dtype=np.float32)

        connect(use_gui=vis)
        draw_global_system()
        set_camera_pose(camera_point=[1, -1, 1])
        add_data_path()
        with LockRenderer():
            with HideOutput(True):
                plane = load_pybullet("models/plane.urdf", fixed_base=True)
                floor = load_pybullet('models/short_floor.urdf', fixed_base=True)
                set_point(floor, [0.5, 0.5, 0.01/2])
                draw_pose(get_pose(floor), length=0.5)
                tray = load_pybullet("models/traybox.urdf", fixed_base=True)
                set_point(tray, [0.5, -0.5, 0.02/2])
                draw_pose(get_pose(tray), length=0.5)
                self.robot = load_pybullet(FRANKA_URDF, fixed_base=True)
                set_point(self.robot, [0, 0, 0.01])
                set_configuration(self.robot, HOME_JOINT_VALUES)
                assign_link_colors(self.robot, max_colors=3, s=0.5, v=1.)
                self.fixed = [plane, floor, tray]
                set_camera_pose([0.5, 0.8, 0.7], target_point=[0.5, 0.5, 0.01])
                # self.camera_pose = get_camera_pose()
                # camera = load_pybullet("models/kinect/kinect.urdf", fixed_base=True)
                # set_pose(camera, get_camera_pose())
                # draw_pose(get_pose(camera), length=0.5)
                
        
        # dump_body(self.robot)
        
        # joint0_id = p.addUserDebugParameter("joint0", -2.897, 2.897, 0)
        # joint1_id = p.addUserDebugParameter("joint1", -1.763, 1.763, 0)
        # joint2_id = p.addUserDebugParameter("joint2", -2.897, 2.897, 0)
        # joint3_id = p.addUserDebugParameter("joint3", -3.072, -0.070, 0)
        # joint4_id = p.addUserDebugParameter("joint4", -2.897, 2.897, 0)
        # joint5_id = p.addUserDebugParameter("joint5", -0.018, 3.752, 0)
        # joint6_id = p.addUserDebugParameter("joint6", -2.897, 2.897, 0)
        # joint9_id = p.addUserDebugParameter("joint9", 0, 0.04, 0)
        # joint10_id = p.addUserDebugParameter("joint10", 0, 0.04, 0)

        # while True:
        #     joint0 = p.readUserDebugParameter(joint0_id)
        #     joint1 = p.readUserDebugParameter(joint1_id)
        #     joint2 = p.readUserDebugParameter(joint2_id)
        #     joint3 = p.readUserDebugParameter(joint3_id)
        #     joint4 = p.readUserDebugParameter(joint4_id)
        #     joint5 = p.readUserDebugParameter(joint5_id)
        #     joint6 = p.readUserDebugParameter(joint6_id)
        #     joint9 = p.readUserDebugParameter(joint9_id)
        #     joint10 = p.readUserDebugParameter(joint10_id)

        #     set_configuration(self.robot, [joint0, joint1, joint2, joint3, joint4, joint5, joint6, joint9, joint10])
        get_image(view_matrix=get_camera_pose(), vertical_fov=30, segment=True)
                        
        self.ik_info = PANDA_INFO
        self.tool_link = link_from_name(self.robot, 'panda_hand')
        self.ik_joints = get_ik_joints(self.robot, self.ik_info, self.tool_link)
        self.moveable_joints = get_movable_joints(self.robot)
        
        self.finger_joints = joints_from_names(self.robot, ["panda_finger_joint1", "panda_finger_joint2"])
        self.ee_close_values = get_min_limits(self.robot, self.finger_joints)
        self.ee_open_values = get_max_limits(self.robot, self.finger_joints)
        
        self.gripper_from_approach = Pose(point=[0,0,-0.05])
        self.gripper_from_grasp = Pose(point=[0,0,0.1])
        
        # draw_pose(Pose(), parent=self.robot, parent_link=11)

        # draw_aabb(get_subtree_aabb(plane))
        # draw_aabb(get_subtree_aabb(floor))
        # draw_aabb(get_subtree_aabb(tray))
        # draw_aabb(get_subtree_aabb(self.robot))

        saved_world = WorldSaver()
        quat = get_link_pose(self.robot, 11)[1]
        # world_from_grasp_bin_overhead = ([0.6, -0.2, 0.4], quat)
        world_from_grasp_bin_overhead = ([0.5, -0.5, 0.4], quat)
        wrold_from_gripper_bin_overhead = multiply(world_from_grasp_bin_overhead, invert(self.gripper_from_grasp))
        draw_pose(wrold_from_gripper_bin_overhead, width=2)
        conf_robot_bin_overhead = get_ik_conf(self.robot, self.ik_info, self.ik_joints, self.tool_link, 
                                            wrold_from_gripper_bin_overhead, obstacles=self.fixed)
        if conf_robot_bin_overhead is None:
            print("calcu ik gripper in bin overhead failed")
        path_bin_overhead = plan_direct_joint_motion(self.robot, self.ik_joints, conf_robot_bin_overhead, obstacles=self.fixed)
        if path_bin_overhead is None:
            print("to bin overhead plan failed")
        self.command_to_bin_overhead = Command([BodyPath(self.robot, path_bin_overhead, joints=self.ik_joints)])
        saved_world.restore()
        self.command_to_bin_overhead.refine(num_steps=10).execute(time_step=0.005)


        # bin_from_gripper = np.eye(4)
        # bin_from_gripper[3,3] = 0.2
        # bin_from_gripper = world_from_bin @ bin_from_gripper
        # conf_bin_overhead = get_ik_conf(self.robot, self.ik_info, self.ik_joints, self.tool_link, 
        #                                 bin_from_gripper, obstacles=self.fixed+[self.robot])
        # saved_world = WorldSaver()
        # path_init_to_bin = plan_direct_joint_motion(self.robot, self.ik_joints, conf_bin_overhead, obstacles=self.fixed+[self.robot])
        # saved_world.restore()
        # self.command_init_to_bin = Command([BodyPath(self.robot, path_init_to_bin, joints=self.ik_joints)])

        # self.sim = bullet_client.BulletClient(connection_mode=(p.GUI if self.vis else p.DIRECT))
        # self.sim.setGravity(0, 0, -10)
        # self.pb_utils = Pybullet_Utils(simulator=self.sim)
        # self.mesh_to_urdf = {}

        # self.env_body = dict()
        # self.env_body['work_floor'] = self.sim.loadURDF('resources/objects/work_floor.urdf', basePosition=[0.528, 0.204, 0], useMaximalCoordinates=True)

        # Setup camera in simulation
        


        # self.camera = Camera(self.sim)
        # self.env_body['camera'] = self._load_mesh(f'resources/camera/kinectsensor.obj', ob_in_world=self.camera.view_matrix, 
        #                                          mass=0, has_collision=False)
        # self.world_from_camera = self.camera.view_matrix
        
        # add object
        self.obj_dir = 'resources/objects/blocks'
        self.num_obj = 5
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
        
        self._num_step = 0
        self.gripper.reset()
        
        self._clean_objects()
        drop_x = (self.workspace[0][1] - self.workspace[0][0] - 0.2) * np.random.random_sample() + self.workspace[0][0] + 0.1
        drop_y = (self.workspace[1][1] - self.workspace[1][0] - 0.2) * np.random.random_sample() + self.workspace[1][0] + 0.1
        self.object_in_world = np.eye(4)
        self.object_in_world[:3,3] = [drop_x, drop_y, 0.20]
        add_success = self._add_objects()
        while add_success == False:
            add_success = self._add_objects()

        observation = self._get_observation()
        
        return observation, {}


    def step(self, action_params):

        camera_from_grasp = utils.to_grasp(action_params)
        world_from_grasp = self.world_from_camera @ camera_from_grasp
        world_from_gripper = world_from_grasp @ self.grasp_from_gripper

        world_from_approach = world_from_gripper @ self.gripper_from_approach
        
        set_configuration(self.robot, HOME_JOINT_VALUES)
        self.open_ee()
        
        saved_world = WorldSaver()
        conf_init = get_joint_positions(self.robot, self.ik_joints)
        
        conf_approach = get_ik_conf(self.robot, self.ik_info, self.ik_joints, self.tool_link, world_from_approach, obstacles=self.fixed+[self.robot])
        conf_grasp = get_ik_conf(self.robot, self.ik_info, self.ik_joints, self.tool_link, world_from_gripper, obstacles=self.fixed+[self.robot])
        if conf_approach is None or conf_grasp is None:
            print("ik failed")

        set_joint_positions(self.robot, self.ik_joints, conf_approach)
        path_approach_to_grasp = plan_direct_joint_motion(self.robot, self.ik_joints, conf_grasp, obstacles=self.fixed+[self.robot])
        if path_approach_to_grasp is None:
            print("to grasp pose plan failed")

        set_joint_positions(self.robot, self.ik_joints, conf_init)
        path_init_to_approach = plan_joint_motion(self.robot, self.ik_joints, conf_approach, obstacles=(self.fixed+self.obj_ids), self_collisions=True)
        if path_init_to_approach is None:
            print("to approach pose plan failed")

        commands_pre = Command([BodyPath(self.robot, path_init_to_approach, joints=self.ik_joints),
                                BodyPath(self.robot, path_approach_to_grasp, joints=self.ik_joints)])
        
        commands_post = Command([BodyPath(self.robot, path_approach_to_grasp[::-1], joints=self.ik_joints),
                                BodyPath(self.robot, path_init_to_approach[::-1], joints=self.ik_joints)])
        saved_world.restore()
        
        commands_pre.refine(num_steps=10).execute(time_step=0.005)
        self.close_ee()
        commands_post.refine(num_steps=10).execute(time_step=0.005)
        
        grasp_success = self._is_grasp_success()
        if grasp_success == True:
            
            self.command_init_to_bin.refine(num_steps=10).execute(time_step=0.005)
            self.open_ee()
        set_configuration(self.robot, HOME_JOINT_VALUES)

        observation = self.get_observation()
        reward = int(grasp_success) * self.reward_scale
        terminated = False
        if not self.has_obj_in_workspace():
            terminated = True
        info = {"is_success": grasp_success}
    
        return observation, reward, terminated, False, info


    def close(self):
        
        disconnect()
    

    #---------------------------------------------------------------------------
    # Environment Control Functions
    #---------------------------------------------------------------------------
    def close_ee(self):

        for _ in joint_controller_hold(self.robot, ["panda_finger_joint1", "panda_finger_joint2"], self.ee_close_values, timeout=(50*self.dt)):
            step_simulation()


    def open_ee(self):
        
        for _ in joint_controller_hold(self.robot, ["panda_finger_joint1", "panda_finger_joint2"], self.ee_open_values, timeout=(50*self.dt)):
            step_simulation()


    def has_obj_in_workspace(self):
        pass

    def _check_out_workspace(self, obj_id):
        
        ob_in_world = self.pb_utils.get_ob_pose_in_world(obj_id) 
        position_x = ob_in_world[0,3]
        position_y = ob_in_world[1,3]
        position_z = ob_in_world[2,3]
        if  position_x > self.workspace[0,1] or position_x < self.workspace[0,0] or\
            position_y > self.workspace[1,1] or position_y > self.workspace[1,1] or\
            position_z > self.workspace[2,1] or position_z > self.workspace[2,1]:
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
        
        # drop_x = (self.workspace[0][1] - self.workspace[0][0] - 0.2) * np.random.random_sample() + self.workspace[0][0] + 0.1
        # drop_y = (self.workspace[1][1] - self.workspace[1][0] - 0.2) * np.random.random_sample() + self.workspace[1][0] + 0.1
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


    def get_observation(self):

        rgb, depth, seg = self.camera.shot()
        pts_scene = utils.depth2xyzmap(depth, self.camera._k)
        bg_mask = depth<0.1
        for name, id in self.env_body.items():
            bg_mask[seg==id] = 1
        pts_obj = pts_scene[bg_mask==False]
        num_sapmple_pts = self.observation_space.shape[0]
        num_obj_pts = len(pts_obj)
        if num_obj_pts == 0:
            observation = np.zeros([num_sapmple_pts, 3])
        elif num_obj_pts < num_sapmple_pts:
            num_padding_pts = num_sapmple_pts - num_obj_pts
            padding_pts = np.repeat([pts_obj[0]], num_padding_pts, axis=0)
            observation = np.concatenate([pts_obj, padding_pts], axis=0)
        else:
            observation = pts_obj[np.random.permutation(num_obj_pts)[:num_sapmple_pts]]
        
        return observation


    def _get_reward(self):
        pass


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
    

    