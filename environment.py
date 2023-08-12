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
import math

from pybullet_tools.utils import *

from pybullet_tools.panda_primitives import *

from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver
from utils import *

HOME_JOINT_VALUES = [0.00, 0.074, 0.00, -1.113, 0.00, 1.510, 0.671, 0.04, 0.04]


class Environment(gym.Env):

    def __init__(self, reward_scale, vis=False):

        self.reward_scale = reward_scale
        approach_radian = np.pi / 2
        action_min = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -approach_radian])
        action_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, approach_radian])
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(25600, 3), dtype=np.float32)
        self.action_space = spaces.Box(action_min, action_max, shape=(7, ), dtype=np.float32)

        method = p.GUI if vis else p.DIRECT
        sim_id = p.connect(method)
        CLIENTS[sim_id] = True if vis else None
        draw_global_system()
        set_camera_pose([1, 1, 1], target_point=[0.5, 0.5, 0.01])
        add_data_path()
        enable_gravity()
        with LockRenderer():
            with HideOutput(True):
                plane = load_pybullet("resources/models/plane.urdf", fixed_base=True)
                self.floor = load_pybullet('resources/models/short_floor.urdf', fixed_base=True)
                set_point(self.floor, [0.5, 0.5, 0.01/2])
                set_color(self.floor, GREY)
                draw_pose(get_pose(self.floor), length=0.5)
                tray = load_pybullet("resources/models/tray/traybox.urdf", fixed_base=True)
                set_point(tray, [0.5, -0.5, 0.02/2])
                draw_pose(get_pose(tray), length=0.5)
                self.robot = load_pybullet(FRANKA_URDF, fixed_base=True)
                set_point(self.robot, [0, 0, 0.01])
                set_configuration(self.robot, HOME_JOINT_VALUES)
                assign_link_colors(self.robot, max_colors=3, s=0.5, v=1.)
                floor_from_camera = Pose(point=[0, 0.75, 1], euler=[-math.radians(145), 0, math.radians(180)])
                world_from_floor = get_pose(self.floor)
                self.world_from_camera = multiply(world_from_floor, floor_from_camera)
                self.camera = Camera(self.world_from_camera)
                self.fixed = [plane, self.floor, tray]
                self.not_in_workspace = [plane, self.robot, tray]
                
        self.workspace = np.asarray([[0.1, 0.9], 
                                     [0.1, 0.9]])
        self.aabb_workspace = aabb_from_extent_center([0.8, 0.8, 0.3], 
                                                      [0.5, 0.5, 0.01+(0.3/2)])
        
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
        # get_image(view_matrix=get_camera_pose(), vertical_fov=30, segment=True)
                        
        self.ik_info = PANDA_INFO
        self.tool_link = link_from_name(self.robot, 'panda_hand')
        self.ik_joints = get_ik_joints(self.robot, self.ik_info, self.tool_link)
        self.moveable_joints = get_movable_joints(self.robot)
        
        self.finger_joints = joints_from_names(self.robot, ["panda_finger_joint1", "panda_finger_joint2"])
        self.ee_close_values = get_min_limits(self.robot, self.finger_joints)
        self.ee_open_values = get_max_limits(self.robot, self.finger_joints)
        
        self.gripper_from_approach = Pose(point=[0,0,-0.03])
        self.grasp_from_gripper = Pose(point=[0,0,-0.1])
        
        # draw_pose(Pose(), parent=self.robot, parent_link=11)

        # draw_aabb(get_subtree_aabb(plane))
        # draw_aabb(get_subtree_aabb(floor))
        # draw_aabb(get_subtree_aabb(tray))
        # draw_aabb(get_subtree_aabb(self.robot))

        # saved_world = WorldSaver()
        # quat = get_link_pose(self.robot, 11)[1]
        # world_from_grasp_bin_overhead = ([0.5, -0.5, 0.4], quat)
        # wrold_from_gripper_bin_overhead = multiply(self.world_from_grasp_bin_overhead, self.grasp_from_gripper)
        # draw_pose(wrold_from_gripper_bin_overhead, width=2)
        # conf_robot_bin_overhead = get_ik_conf(self.robot, self.ik_info, self.ik_joints, self.tool_link, 
        #                                     wrold_from_gripper_bin_overhead, obstacles=self.fixed)
        # if conf_robot_bin_overhead is None:
        #     print("calcu ik gripper in bin overhead failed")
        # path_bin_overhead = plan_direct_joint_motion(self.robot, self.ik_joints, conf_robot_bin_overhead, obstacles=self.fixed)
        # if path_bin_overhead is None:
        #     print("to bin overhead plan failed")
        # self.command_to_bin_overhead = Command([BodyPath(self.robot, path_bin_overhead, joints=self.ik_joints)])
        # saved_world.restore()
        # self.command_to_bin_overhead.refine(num_steps=10).execute(time_step=0.005)
        
        # Define colors for object meshes (Tableau palette)
        self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                                        [89.0, 161.0, 79.0], # green
                                        [156, 117, 95], # brown
                                        [242, 142, 43], # orange
                                        [237.0, 201.0, 72.0], # yellow
                                        [186, 176, 172], # gray
                                        [255.0, 87.0, 89.0], # red
                                        [176, 122, 161], # purple
                                        [118, 183, 178], # cyan
                                        [255, 157, 167]])/255.0 #pink

        # Read files in object mesh directory
        self.mesh_dir = "resources/objects/blocks/obj"
        self.vhacd_dir = "resources/objects/blocks/vhacd"
        self.num_obj = 5
        self.mesh_list = os.listdir(self.mesh_dir)
        self.mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 10, :]
        self.mesh_ids = []
        self.mesh_to_urdf = {}


    #---------------------------------------------------------------------------
    # Standard Gym Functions
    #---------------------------------------------------------------------------
    def seed(self, seed=None):
        
        self._random = np.random.RandomState(seed)
        return seed


    def reset(self, seed = None, options = None):
        
        super().reset(seed=seed, options=options)
        
        set_configuration(self.robot, HOME_JOINT_VALUES)
        self.clean_objects()
        self.add_objects()
        observation = self.get_observation()
        
        return observation, {}


    def step(self, action_params):

        camera_from_grasp = to_grasp(action_params)
        world_from_grasp = multiply(self.world_from_camera, camera_from_grasp)
        world_from_gripper = multiply(world_from_grasp, self.grasp_from_gripper)

        world_from_approach = multiply(world_from_gripper, self.gripper_from_approach)
        
        set_configuration(self.robot, HOME_JOINT_VALUES)
        conf_init = get_joint_positions(self.robot, self.ik_joints)
        saved_world = WorldSaver()
        
        conf_approach = get_ik_conf(self.robot, self.ik_info, self.ik_joints, self.tool_link, world_from_approach, obstacles=self.fixed)
        conf_grasp = get_ik_conf(self.robot, self.ik_info, self.ik_joints, self.tool_link, world_from_gripper, obstacles=self.fixed)
        if conf_approach is None or conf_grasp is None:
            print("ik failed")

        set_joint_positions(self.robot, self.ik_joints, conf_approach)
        path_approach_to_grasp = plan_direct_joint_motion(self.robot, self.ik_joints, conf_grasp, obstacles=self.fixed)
        if path_approach_to_grasp is None:
            print("to grasp pose plan failed")

        set_joint_positions(self.robot, self.ik_joints, conf_init)
        path_init_to_approach = plan_joint_motion(self.robot, self.ik_joints, conf_approach, obstacles=(self.fixed+self.mesh_ids))
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
        
        grasp_success = self.is_grasp_success()
        if grasp_success == True:
            grasped_obj = self.get_grasped_obj()
            # self.command_init_to_bin.refine(num_steps=10).execute(time_step=0.005)
            set_pose(grasped_obj, Pose(point=[0.5, -0.5, 0.4]))
        set_configuration(self.robot, HOME_JOINT_VALUES)

        observation = self.get_observation()
        reward = int(grasp_success) * self.reward_scale
        terminated = False
        if not self.exist_obj_in_workspace():
            terminated = True
        info = {"is_success": grasp_success}
    
        return observation, reward, terminated, False, info


    def close(self):
        
        disconnect()
    

    #---------------------------------------------------------------------------
    # Helper Functions
    #---------------------------------------------------------------------------
    def get_observation(self):

        rgb, depth, seg = self.camera.render()
        pts_scene = depth2xyzmap(depth, self.camera.k)

        bg_mask = depth<0.1
        floor_mask = bg_mask.copy()
        for id in (self.fixed+[self.robot]):
            bg_mask[seg==id] = 1
        floor_mask[seg==self.floor] = 1
        
        pts_objs = pts_scene[bg_mask==False]
        colors_objs = rgb[bg_mask==False]
        pts_floor = pts_scene[floor_mask==True]
        colors_floor = rgb[floor_mask==True]

        o3d.io.write_point_cloud('objs.pcd', toOpen3dCloud(pts_objs, colors_objs))
        o3d.io.write_point_cloud('floor.pcd', toOpen3dCloud(pts_floor, colors_floor))

        num_sapmple_pts = self.observation_space.shape[0]
        num_obj_pts = len(pts_in_workspace)
        if num_obj_pts == 0:
            observation = np.zeros([num_sapmple_pts, 3])
        elif num_obj_pts < num_sapmple_pts:
            num_padding_pts = num_sapmple_pts - num_obj_pts
            padding_pts = np.repeat([pts_in_workspace[0]], num_padding_pts, axis=0)
            observation = np.concatenate([pts_in_workspace, padding_pts], axis=0)
        else:
            observation = pts_in_workspace[np.random.permutation(num_obj_pts)[:num_sapmple_pts]]
        
        return observation
    
    
    def add_objects(self):

        obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
        for object_idx in range(len(obj_mesh_ind)):
            curr_mesh_file = os.path.join(self.mesh_dir, self.mesh_list[obj_mesh_ind[object_idx]])
            curr_vhacd_file = os.path.join(self.vhacd_dir, self.mesh_list[obj_mesh_ind[object_idx]])
            
            drop_x = (self.workspace[0][1] - self.workspace[0][0] - 0.2) * np.random.random_sample() + self.workspace[0][0] + 0.1
            drop_y = (self.workspace[1][1] - self.workspace[1][0] - 0.2) * np.random.random_sample() + self.workspace[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            object_color = [self.mesh_color[object_idx][0], self.mesh_color[object_idx][1], self.mesh_color[object_idx][2], 1]
            obj_id = self.load_mesh(mesh_file=curr_mesh_file, mesh_pose=Pose(object_position, object_orientation), mass=0.1, vhacd_file=curr_vhacd_file)
            set_color(obj_id, object_color)
            p.changeDynamics(obj_id, -1, lateralFriction=0.7, spinningFriction=0.7, collisionMargin=0.0001)
            self.mesh_ids.append(obj_id)
        
        while 1:
            last_pos = {}
            accum_motions = {}
        
            for body_id in self.mesh_ids:
                last_pos[body_id] = np.array(get_point(body_id))
                accum_motions[body_id] = 0

            stabled = True
            for _ in range(50):
                p.stepSimulation()
                for body_id in self.mesh_ids:
                    cur_pos =  np.array(get_point(body_id))
                    motion = np.linalg.norm(cur_pos - last_pos[body_id])
                    accum_motions[body_id] += motion
                    last_pos[body_id] = cur_pos.copy()
                    if accum_motions[body_id]>=0.001:
                        stabled = False
                        break
                if stabled==False:
                    break
        
            if stabled:
                for body_id in self.mesh_ids:
                    p.resetBaseVelocity(body_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0])
                break
    

    def remove_object(self, obj_id):
        
        p.removeBody(obj_id)
        self.mesh_ids.remove(obj_id)


    def clean_objects(self):
        
        for ob_id in self.mesh_ids:
            self.remove_object(obj_id=ob_id)


    def exist_obj_in_workspace(self):
        
        bodies_in_workspace = get_bodies_in_region(self.aabb_workspace)
        bodies_in_workspace = list(set(bodies_in_workspace).difference(set(self.fixed)))
        if len(bodies_in_workspace) > 0:
            return True
        else:
            return False
        

    def get_grasped_obj(self):
        
        for ob_id in self.mesh_ids:
            if body_collision(self.robot, ob_id) == False:
                continue
            return ob_id
    

    def close_ee(self):

        for _ in joint_controller_hold(self.robot, ["panda_finger_joint1", "panda_finger_joint2"], self.ee_close_values, timeout=(50*self.dt)):
            step_simulation()


    def open_ee(self):
        
        for _ in joint_controller_hold(self.robot, ["panda_finger_joint1", "panda_finger_joint2"], self.ee_open_values, timeout=(50*self.dt)):
            step_simulation()


    def is_grasp_success(self):

        finger_joint_pos = np.array(get_joint_positions(self.robot, self.finger_joints))
        if np.all(finger_joint_pos < 0.001):
            return False
        return True
    
   
    def load_mesh(self, mesh_file, mesh_pose, mass, vhacd_file, scale=np.ones(3),has_collision=True,useFixedBase=False,concave=False,collision_margin=0.0001):

        if mesh_file in self.mesh_to_urdf:
            urdf_dir = self.mesh_to_urdf[mesh_file]
        else:
            urdf_dir = f'/tmp/{os.path.basename(mesh_file)}_{uuid4()}.urdf'
            create_urdf_from_mesh(mesh_file, out_dir=urdf_dir, mass=mass, vhacd_dir=vhacd_file, has_collision=has_collision, concave=concave, scale=scale)
            self.mesh_to_urdf[mesh_file] = urdf_dir

        obj_id = p.loadURDF(urdf_dir, useFixedBase=useFixedBase, basePosition=mesh_pose[0], baseOrientation=mesh_pose[1],
                            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES|p.URDF_USE_MATERIAL_COLORS_FROM_MTL|p.URDF_MAINTAIN_LINK_ORDER)
        return obj_id