import os, math, datetime, pkgutil
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
from camera import Camera
from uuid import uuid4
import open3d as o3d
from pybullet_tools.utils import *
from pybullet_tools.panda_primitives import *
from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver
from utils import *
import yaml

egl = pkgutil.get_loader('eglRenderer')

HOME_JOINT_VALUES = [0.00, 0.074, 0.00, -1.113, 0.00, 1.510, 0.671, 0.04, 0.04]
HOME_POSE_GRIPPER = Pose(point=[0, 0, 1], euler=[0, math.radians(180), 0])
CONF_OPEN= [0.04, 0.04]
CONF_CLOSE = [0, 0]


class Environment(gym.Env):

    def __init__(self, reward_scale, vis=False):

        self.reward_scale = reward_scale

        action_min = np.array([-0.05, -0.05, -0.05, -1.0, -1.0, -1.0, 0])
        action_max = np.array([0.05, 0.05, 0.05, 1.0, 1.0, 1.0, np.pi])
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(5000, 3), dtype=np.float32)
        self.action_space = spaces.Box(action_min, action_max, shape=(7,), dtype=np.float32)

        method = p.GUI if vis else p.DIRECT
        sim_id = p.connect(method)
        CLIENTS[sim_id] = True if vis else None
        # draw_global_system(length=0.1)
        # set_camera_pose([1, 1, 1], target_point=[0.5, 0.5, 0.01])
        add_data_path()
        self.plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        enable_gravity()
        with LockRenderer():
            with HideOutput(True):
                plane = load_pybullet('resources/models/plane.urdf', fixed_base=True)
                self.floor = load_pybullet('resources/models/short_floor.urdf', fixed_base=True)
                set_point(self.floor, [0.5, 0.5, 0.01/2])
                set_color(self.floor, GREY)
                # draw_pose(get_pose(self.floor), length=0.5)
                tray = load_pybullet('resources/models/tray/traybox.urdf', fixed_base=True)
                set_point(tray, [0.5, -0.5, 0.02/2])
                # draw_pose(get_pose(tray), length=0.5)
                # self.robot = load_pybullet(f'resources/{FRANKA_URDF}', fixed_base=True)
                # set_point(self.robot, [0, 0, 0.01])
                # set_configuration(self.robot, HOME_JOINT_VALUES)
                # assign_link_colors(self.robot, max_colors=3, s=0.5, v=1.)
                self.gripper = load_pybullet('resources/models/franka_description/robots/hand.urdf', fixed_base=True)
                assign_link_colors(self.gripper, max_colors=3, s=0.5, v=1.)
                set_configuration(self.gripper, CONF_OPEN)
                set_pose(self.gripper, HOME_POSE_GRIPPER)
                draw_pose(unit_pose(), parent=self.gripper, parent_link=link_from_name(self.gripper, 'panda_tcp'), length=0.04, width=3)
                floor_from_camera = Pose(point=[0, 0.75, 1], euler=[-math.radians(145), 0, math.radians(180)])
                world_from_floor = get_pose(self.floor)
                self.world_from_camera = multiply(world_from_floor, floor_from_camera)
                self.camera = Camera(self.world_from_camera)
                self.fixed = [plane, self.floor, tray]
                # draw_pose(self.world_from_camera, length=0.05, width=3)
                
        self.workspace = np.asarray([[0.1, 0.9], 
                                     [0.1, 0.9]])
        self.aabb_workspace = aabb_from_extent_center([0.8, 0.8, 0.3], 
                                                      [0.5, 0.5, 0.01+(0.3/2)])
        # draw_pose(get_pose(self.gripper), length=0.2, width=3)
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

        # set_configuration(self.robot, [joint0, joint1, joint2, joint3, joint4, joint5, joint6, joint9, joint10])
                        
        # self.ik_info = PANDA_INFO
        # self.tool_link = link_from_name(self.robot, 'panda_hand')
        # self.ik_joints = get_ik_joints(self.robot, self.ik_info, self.tool_link)
        # self.moveable_joints = get_movable_joints(self.robot)
        
        # self.finger_joints = joints_from_names(self.robot, ["panda_finger_joint1", "panda_finger_joint2"])
        # self.ee_close_values = get_min_limits(self.robot, self.finger_joints)
        # self.ee_open_values = get_max_limits(self.robot, self.finger_joints)
        
        self.finger_joints = joints_from_names(self.gripper, ["panda_finger_joint1", "panda_finger_joint2"])
        self.finger_links = links_from_names(self.gripper, ['panda_leftfinger', 'panda_rightfinger'])
        # self.gripper_from_approach = Pose(point=[0,0,-0.03])
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
        # self.mesh_dir = "resources/objects/blocks/obj"
        # self.vhacd_dir = "resources/objects/blocks/vhacd"
        mesh_dir = "resources/objects/ycb"
        with open(f'{mesh_dir}/config.yml','r') as ff:
            cfg = yaml.safe_load(ff)
        self.obj_files = []
        for name in cfg['load_obj']:
            self.obj_files.append(f'{mesh_dir}/{name}.obj')
        
        self.mesh_ids = []
        self.mesh_to_urdf = {}

        # self.pcd_dir = f'logs/data/{datetime.datetime.now().strftime("%Y-%m-%d.%H:%M:%S")}'
        # os.makedirs(self.pcd_dir, exist_ok=True)


    #---------------------------------------------------------------------------
    # Standard Gym Functions
    #---------------------------------------------------------------------------
    def seed(self, seed=None):
        
        self._random = np.random.RandomState(seed)
        return seed


    def reset(self, seed = None, options = None):
        
        super().reset(seed=seed, options=options)
        
        # set_configuration(self.robot, HOME_JOINT_VALUES)
        set_pose(self.gripper, HOME_POSE_GRIPPER)
        set_configuration(self.gripper, CONF_OPEN)
        self.clean_objects()
        self.add_objects()
        while self.sim_until_stable() == False:
            self.clean_objects()
            self.add_objects()
        observation = self.get_observation()
        
        return observation, {}


    def step(self, action_params):

        world_from_grasp = to_grasp(action_params)
        world_from_gripper = multiply(world_from_grasp, self.grasp_from_gripper)
        # world_from_grasp = multiply(self.world_from_camera, camera_from_grasp)
        # draw_pose(world_from_grasp, length=0.05, width=3)
        # draw_pose(world_from_gripper, length=0.05, width=3)

        # world_from_approach = multiply(world_from_gripper, self.gripper_from_approach)
        # draw_pose(world_from_approach, length=0.05, width=3)
        
        # set_configuration(self.robot, HOME_JOINT_VALUES)
        
        # conf_init = get_joint_positions(self.robot, self.ik_joints)
        # saved_world = WorldSaver()
        
        # conf_approach = get_ik_conf(self.robot, self.ik_info, self.ik_joints, self.tool_link, world_from_approach, obstacles=self.fixed)
        # if conf_approach != None:
        #     conf_grasp = get_ik_conf(self.robot, self.ik_info, self.ik_joints, self.tool_link, world_from_gripper, obstacles=self.fixed)
        #     if conf_grasp != None:
        #         set_joint_positions(self.robot, self.ik_joints, conf_approach)
        #         path_approach_to_grasp = plan_direct_joint_motion(self.robot, self.ik_joints, conf_grasp, obstacles=self.fixed)
        #         if path_approach_to_grasp != None:
        #             set_joint_positions(self.robot, self.ik_joints, conf_init)
        #             path_init_to_approach = plan_joint_motion(self.robot, self.ik_joints, conf_approach, obstacles=(self.fixed+self.mesh_ids))
        #             if path_init_to_approach != None:
        #                 commands_pre = Command([BodyPath(self.robot, path_init_to_approach, joints=self.ik_joints),
        #                                         BodyPath(self.robot, path_approach_to_grasp, joints=self.ik_joints)])
                        
        #                 commands_post = Command([BodyPath(self.robot, path_approach_to_grasp[::-1], joints=self.ik_joints),
        #                                         BodyPath(self.robot, path_init_to_approach[::-1], joints=self.ik_joints)])
        #                 saved_world.restore()
                        
        #                 commands_pre.refine(num_steps=10).execute(time_step=0.005)
        #                 self.close_ee()
        #                 commands_post.refine(num_steps=10).execute(time_step=0.005)
                        
        #                 grasp_success = self.is_grasp_success()
        #                 if grasp_success == True:
        #                     grasped_obj = self.get_grasped_obj()
        #                     # self.command_init_to_bin.refine(num_steps=10).execute(time_step=0.005)
        #                     set_pose(grasped_obj, Pose(point=[0.5, -0.5, 0.4]))
        #                 set_configuration(self.robot, HOME_JOINT_VALUES)
        #             else:
        #                 print("to approach pose plan failed")
        #                 grasp_success = False
        #         else:
        #             print("to grasp pose plan failed")
        #             grasp_success = False
        #     else:
        #         print("no grasp ik solution")
        #         grasp_success = False
        # else:
        #     print("no approach ik solution")
        #     grasp_success = False

        is_collision = False
        set_pose(self.gripper, world_from_gripper)
        if any(pairwise_collision(self.gripper, b) for b in (self.fixed+self.mesh_ids)):
            grasp_success = False
            is_collision = True
        else:
            self.close_ee()
            grasped_obj = self.get_grasped_obj()
            if grasped_obj != None:
                saved_world = WorldSaver()
                if self.is_grasp_success():
                    world_from_gobj = get_pose(grasped_obj)
                    gripper_from_world = invert(world_from_gripper)
                    gripper_from_gobj = multiply(gripper_from_world, world_from_gobj)
                    set_point(self.gripper, [0.5, 0.5, 0.5])
                    world_from_gripper = get_pose(self.gripper)
                    world_from_gobj = multiply(world_from_gripper, gripper_from_gobj)
                    set_pose(grasped_obj, world_from_gobj)
                    self.sim_until_stable()
                grasp_success = self.is_grasp_success()
                if grasp_success == True:
                    set_point(grasped_obj, [0.5, -0.5, 0.5])
                    self.sim_until_stable()
                else:
                    saved_world.restore()
            else:
                grasp_success = False
            
        set_pose(self.gripper, HOME_POSE_GRIPPER)
        self.open_ee()
        
        observation = self.get_observation()
        reward = int(not grasp_success) * self.reward_scale
        terminated = not self.exist_obj_in_workspace()
        info = {"is_success": grasp_success, 'is_collision': is_collision}
    
        return observation, reward, terminated, False, info


    def close(self):
        
        disconnect()
        p.unloadPlugin(self.plugin)
    

    #---------------------------------------------------------------------------
    # Helper Functions
    #---------------------------------------------------------------------------
    def get_observation(self):
        rgb, depth, seg = self.camera.render()
        pts_scene = depth2xyzmap(depth, self.camera.k)

        bg_mask = depth<0.1
        floor_mask = bg_mask.copy()
        # for id in (self.fixed+[self.robot]):
        for id in (self.fixed+[self.gripper]):
            bg_mask[seg==id] = 1
        floor_mask[seg==self.floor] = 1
        
        camera_from_pts_objs = pts_scene[bg_mask==False]
        camera_from_pts_floor = pts_scene[floor_mask==True]
        world_from_pts_objs = ((tform_from_pose(self.world_from_camera) @ to_homo(camera_from_pts_objs).T).T)[:,:3]
        world_from_pts_floor = ((tform_from_pose(self.world_from_camera) @ to_homo(camera_from_pts_floor).T).T)[:,:3]

        # colors_objs, colors_floor = rgb[bg_mask==False], rgb[floor_mask==True]
        # pcd_objs = toOpen3dCloud(world_from_pts_objs, colors_objs)
        # pcd_floor = toOpen3dCloud(world_from_pts_floor, colors_floor)
        # o3d.io.write_point_cloud(f'{self.pcd_dir}/objs.ply', pcd_objs)
        # o3d.io.write_point_cloud(f'{self.pcd_dir}/floor.ply', pcd_floor)

        num_pts = self.observation_space.shape[0]
        num_obj_pts = int(num_pts * 0.8)
        num_floor_pts = num_pts - num_obj_pts
        if len(world_from_pts_objs) >= num_obj_pts:
            select_obj_index = np.random.choice(len(world_from_pts_objs), num_obj_pts, replace=False)
        elif len(world_from_pts_objs) > 0:
            select_obj_index = np.random.choice(len(world_from_pts_objs), num_obj_pts, replace=True)
        else:
            return world_from_pts_floor[np.random.choice(len(world_from_pts_floor), num_pts, replace=True)]
        
        if len(world_from_pts_floor) >= num_floor_pts:
            select_floor_index = np.random.choice(len(world_from_pts_floor), num_floor_pts, replace=False)
        else:
            select_floor_index = np.random.choice(len(world_from_pts_floor), num_floor_pts, replace=True)

        world_from_pts_objs, world_from_pts_floor = world_from_pts_objs[select_obj_index], world_from_pts_floor[select_floor_index]
        world_from_observation = np.concatenate((world_from_pts_objs, world_from_pts_floor), axis=0)
        
        # colors_objs, colors_floor = colors_objs[select_obj_index], colors_floor[select_obj_index]
        # pcd_objs = toOpen3dCloud(world_from_pts_objs, colors_objs)
        # pcd_floor = toOpen3dCloud(world_from_pts_floor, colors_floor)
        # o3d.io.write_point_cloud(f'{self.pcd_dir}/objs_down.ply', pcd_objs)
        # o3d.io.write_point_cloud(f'{self.pcd_dir}/floor_down.ply', pcd_floor)

        return world_from_observation.astype(np.float32)
    
    
    def add_objects(self):
        for obj_path in self.obj_files:
            vhacd_path = obj_path.replace('.obj', '_vhacd.obj')
            drop_x = (self.workspace[0][1] - self.workspace[0][0] - 0.2) * np.random.random_sample() + self.workspace[0][0] + 0.1
            drop_y = (self.workspace[1][1] - self.workspace[1][0] - 0.2) * np.random.random_sample() + self.workspace[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            obj_id = self.load_mesh(mesh_file=obj_path, mesh_pose=Pose(object_position, object_orientation), 
                                    mass=0.1, vhacd_file=vhacd_path, scale=[1, 1, 1])
            p.changeDynamics(obj_id, -1, lateralFriction=0.7, spinningFriction=0.7, collisionMargin=0.0001)
            self.mesh_ids.append(obj_id)
    

    def sim_until_stable(self):
        
        count = 0
        while count < 2000:
            last_pos = {}
            accum_motions = {}
        
            for body_id in self.mesh_ids:
                last_pos[body_id] = np.array(get_point(body_id))
                accum_motions[body_id] = 0

            stabled = True
            for _ in range(50):
                p.stepSimulation()
                count += 1
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
        return (count < 2000)


    def clean_objects(self):
        
        for ob_id in self.mesh_ids:
            p.removeBody(ob_id)
        self.mesh_ids.clear()


    def exist_obj_in_workspace(self):
        
        bodies_in_workspace = np.array(get_bodies_in_region(self.aabb_workspace))[:, 0]
        bodies_in_workspace = list(set(bodies_in_workspace).difference(set(self.fixed)))
        if len(bodies_in_workspace) > 0:
            return True
        else:
            return False
        

    def get_grasped_obj(self):
        
        for ob_id in self.mesh_ids:
            # if body_collision(self.robot, ob_id) == False:
            if any_link_pair_collision(self.gripper, self.finger_links, ob_id) == True:
                return ob_id


    def close_ee(self):
        
        p.setJointMotorControlArray(self.gripper, jointIndices=self.finger_joints, controlMode=p.POSITION_CONTROL,
                                    targetPositions=CONF_CLOSE, forces=np.ones(2, dtype=float) * 100)
        for _ in range(50):
            p.stepSimulation()
            # time.sleep(0.1)
        
        # for _ in joint_controller_hold(self.robot, [joint_from_name(self.robot, "panda_finger_joint1"), joint_from_name(self.robot, "panda_finger_joint2")], self.ee_close_values, timeout=(50*DEFAULT_TIME_STEP)):
        # for _ in joint_controller_hold(self.gripper, self.finger_joints, CONF_CLOSE, timeout=(50 * DEFAULT_TIME_STEP)):
        #     step_simulation()


    def open_ee(self):

        p.setJointMotorControlArray(self.gripper, jointIndices=self.finger_joints, controlMode=p.POSITION_CONTROL,
                                    targetPositions=CONF_OPEN, forces=np.ones(2, dtype=float) * 100)
        for _ in range(50):
            p.stepSimulation()
        
        # for _ in joint_controller_hold(self.robot, ["panda_finger_joint1", "panda_finger_joint2"], self.ee_open_values, timeout=(50*DEFAULT_TIME_STEP)):
        # for _ in joint_controller_hold(self.gripper, self.finger_joints, CONF_OPEN, timeout=(50 * DEFAULT_TIME_STEP)):
        #     step_simulation()


    def is_grasp_success(self):

        # finger_joint_pos = np.array(get_joint_positions(self.robot, self.finger_joints))
        finger_joint_pos = np.array(get_joint_positions(self.gripper, self.finger_joints))
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