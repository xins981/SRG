import math
from collections import namedtuple
import pybullet as p
import trimesh, copy
from transformations import *
import numpy as np
from autolab_core import RigidTransform
import utils



class RobotBase(object):

    def __init__(self, sim):
        self.sim = sim

    def load(self):
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()
        print(self.joints)

    def step_simulation(self):
        raise RuntimeError('`step_simulation` method of RobotBase Class should be hooked by the environment.')

    def __parse_joint_info__(self):
        numJoints = self.sim.getNumJoints(self.robot_id)
        jointInfo = namedtuple('jointInfo', ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = self.sim.getJointInfo(self.robot_id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(self.robot_id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]

    def __init_robot__(self):
        raise NotImplementedError
    
    def __post_load__(self):
        pass

    def reset(self):
        self.reset_arm()
        self.reset_gripper()

    def reset_arm(self):
        """
        reset to rest poses
        """
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.robot_id, joint_id, rest_pose)

        # Wait for a few steps
        for _ in range(10):
            self.step_simulation()

    def reset_gripper(self):
        self.open_gripper()

    def open_gripper(self):
        self.move_finger(self.gripper_range[1])

    def close_gripper(self):
        self.move_finger(self.gripper_range[0])

    def move_ee(self, action, control_method):
        assert control_method in ('joint', 'end')
        if control_method == 'end':
            x, y, z, roll, pitch, yaw = action
            pos = (x, y, z)
            orn = p.getQuaternionFromEuler((roll, pitch, yaw))
            joint_poses = p.calculateInverseKinematics(self.robot_id, self.eef_id, pos, orn,
                                                       self.arm_lower_limits, self.arm_upper_limits, self.arm_joint_ranges, self.arm_rest_poses,
                                                       maxNumIterations=20)
        elif control_method == 'joint':
            assert len(action) == self.arm_num_dofs
            joint_poses = action
        # arm
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.robot_id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                    force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity)

    def move_finger(self, open_length):
        raise NotImplementedError

    def get_joint_obs(self):
        positions = []
        velocities = []
        for joint_id in self.controllable_joints:
            pos, vel, _, _ = p.getJointState(self.robot_id, joint_id)
            positions.append(pos)
            velocities.append(vel)
        ee_pos = p.getLinkState(self.robot_id, self.eef_id)[0]
        return dict(positions=positions, velocities=velocities, ee_pos=ee_pos)



class UR5Robotiq85(RobotBase):
    
    def __init_robot__(self):
        
        self.robot_id = p.loadURDF('./urdf/ur5_robotiq_85.urdf', useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.end_effecter_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, 
                               -1.3708613585093699, -1.5707970583733368, 0.0009377758247187636]
        self.gripper_width_range = [0, 0.085]
    

    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)


    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.robot_id, self.mimic_parent_id,
                                   self.robot_id, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance


    def move_finger(self, open_length):
        
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(self.robot_id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)


class Gripper:

    def __init__(self, sim, bullet_utils):

        gripper_dir = 'resources/robot/urdf/robotiq_hande'
        
        self._sim = sim
        self._bullet_utils = bullet_utils
        self.rest_pose = np.eye(4)

        finger_mesh1 = trimesh.load(f'{gripper_dir}/finger1.obj')
        finger_mesh2 = copy.deepcopy(finger_mesh1)
        R_z = utils.euler_matrix(0,0,np.pi,axes='sxyz')
        finger_mesh2.apply_transform(R_z)
        self.finger_meshes = [finger_mesh1, finger_mesh2]

        self.id = self._sim.loadURDF(f"{gripper_dir}/gripper.urdf", [0, 0, 0], useFixedBase=True)
        self.finger_ids = np.array([1,2],dtype=int)
        self.gripper_max_force = np.ones(2, dtype=float) * 100
        self.grip_dirs = np.array([[0,1,0],[0,-1,0]])
        self._sim.changeDynamics(self.id, -1, lateralFriction=0.9, spinningFriction=0.9)
        
        T_gripper_grasp = RigidTransform.load(f'{gripper_dir}/T_grasp_gripper.tf')
        if T_gripper_grasp._from_frame=='gripper' and T_gripper_grasp._to_frame=='grasp':
            pass
        elif T_gripper_grasp._from_frame=='grasp' and T_gripper_grasp._to_frame=='gripper':
            T_gripper_grasp = T_gripper_grasp.inverse()
        else:
            raise RuntimeError("gripper_in_grasp from={}, to={}".format(gripper_in_grasp._from_frame, gripper_in_grasp._to_frame))
        gripper_in_grasp = np.eye(4)
        gripper_in_grasp[:3,:3] = T_gripper_grasp.rotation
        gripper_in_grasp[:3,3] = T_gripper_grasp.translation
        self.grasp_in_gripper = np.linalg.inv(gripper_in_grasp)


    def open(self):
        self._move_finger(target_positions=[0, 0], step=50)


    def close(self):
        self._move_finger(target_positions=[1, 1], step=50)


    def _move_finger(self, target_positions, step):
        
        self._sim.setJointMotorControlArray(self.id,jointIndices=self.finger_ids,controlMode=p.POSITION_CONTROL,
                                            targetPositions=target_positions,forces=self.gripper_max_force)
        for _ in range(step):
            self.step_simulation()
    

    def step_simulation(self):
        raise RuntimeError('`step_simulation` method of RobotBase Class should be hooked by the environment.')
    

    def reset(self):
        
        self._bullet_utils.set_body_pose_in_world(self.id, self.rest_pose)
        self.open()


    def export(self, grasp_in_camera, out_dir):
        
        gripper_in_cam = grasp_in_camera @np.linalg.inv(self.grasp_in_gripper)
        gripper_mesh = copy.deepcopy(self.gripper_mesh)
        gripper_mesh.apply_transform(gripper_in_cam)
        gripper_mesh.export(out_dir)

