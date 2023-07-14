import numpy as np
import transformations as tf
import copy
import open3d as o3d
from collections import namedtuple



def create_urdf_from_mesh(mesh_dir,concave=False, out_dir=None, mass=0.1, has_collision=True, scale=np.ones((3))):
    assert '.obj' in mesh_dir, f'mesh_dir={mesh_dir}'

    lateral_friction = 0.8
    spinning_friction = 0.5
    rolling_friction = 0.5

    concave_str = 'no'
    collision_mesh_dir = copy.deepcopy(mesh_dir)
    if concave:
        concave_str = 'yes'
    if mass!=0:
        collision_mesh_dir = mesh_dir.replace('.obj','_vhacd.obj')

    collision_block = ""
    if has_collision:
        collision_block = f"""
            <collision>
                <origin xyz="0 0 0"/>
                <geometry>
                    <mesh filename="{collision_mesh_dir}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
                </geometry>
            </collision>
            """

    link_str = f"""
    <link concave="{concave_str}" name="base_link">
        <inertial>
            <origin xyz="0 0 0" />
            <mass value="{mass}" />
            <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0"/>
            <geometry>
                <mesh filename="{mesh_dir}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
            </geometry>
        </visual>
        {collision_block}
    </link>
    """

    urdf_str = f"""
        <robot name="model.urdf">
            {link_str}
        </robot>
        """

    if out_dir is None:
        out_dir = mesh_dir.replace('.obj','.urdf')
    with open(out_dir,'w') as ff:
        ff.write(urdf_str)

    return out_dir


def depth2xyzmap(depth, K):
    
    invalid_mask = (depth<0.1)
    H, W = depth.shape[:2]
    vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
    zs = depth.reshape(-1)
    xs = (us-K[0,2])*zs/K[0,0]
    ys = (vs-K[1,2])*zs/K[1,1]
    pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
    xyz_map = pts.reshape(H,W,3).astype(np.float32)
    xyz_map[invalid_mask] = 0
    return xyz_map.astype(np.float32)


def farthest_point_sample(pts, npoint):
    
    ret = []
    n = pts.shape[0]
    distance = np.ones(n) * 1e10
    farthest_id = np.random.randint(0, n)
    for _ in range(npoint):
        ret.append(farthest_id)
        distance[farthest_id] = 0
        farthest_xyz = pts[farthest_id]
        dist = np.sum((pts - farthest_xyz) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest_id = np.argmax(distance)
    return ret


def toOpen3dCloud(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
      if colors.max()>1:
        colors = colors/255.0
      cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
      cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def safe_zip(sequence1, sequence2):
    sequence1, sequence2 = list(sequence1), list(sequence2)
    assert len(sequence1) == len(sequence2)
    return list(zip(sequence1, sequence2))


def normalizeRotation(pose):

    new_pose = pose.copy()
    scales = np.linalg.norm(pose[:3,:3],axis=0)
    new_pose[:3,:3] /= scales.reshape(1,3)
    return new_pose



#---------------------------------------------------------------------------
# Pybullet Functions
#---------------------------------------------------------------------------

class Pybullet_Utils:

    INF = np.inf
    PI = np.pi
    EPSILON = 1e-6
    Interval = namedtuple('Interval', ['lower', 'upper']) # AABB
    JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                        'qIndex', 'uIndex', 'flags', 'jointDamping', 
                                        'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                        'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                        'parentFramePos', 'parentFrameOrn', 'parentIndex'])
    JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity', 'jointReactionForces', 'appliedJointMotorTorque'])
    UNIT_LIMITS = Interval(0., 1.)
    CIRCULAR_LIMITS = Interval(-PI, PI)
    UNBOUNDED_LIMITS = Interval(-INF, INF)

    def __init__(self, simulator):
        self.sim = simulator


    def set_body_pose_in_world(self, body_id, ob_in_world):
        
        trans = ob_in_world[:3,3]
        q_wxyz = tf.quaternion_from_matrix(ob_in_world)
        q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
        self.sim.resetBasePositionAndOrientation(body_id, trans, q_xyzw)


    def get_ob_pose_in_world(self, body_id):
        
        trans,q_xyzw = self.sim.getBasePositionAndOrientation(body_id)
        ob_in_world = np.eye(4)
        ob_in_world[:3,3] = trans
        q_wxyz = [q_xyzw[-1],q_xyzw[0],q_xyzw[1],q_xyzw[2]]
        R = tf.quaternion_matrix(q_wxyz)[:3,:3]
        ob_in_world[:3,:3] = R
        return ob_in_world


    def set_joint_positions(self, body, joints, values):
        for joint, value in safe_zip(joints, values):
            self.set_joint_position(body, joint, value)


    def set_joint_position(self, body, joint, value):
        self.sim.resetJointState(body, joint, targetValue=value, targetVelocity=0)


    def get_joint_info(self, body, joint):
        return self.JointInfo(*self.sim.getJointInfo(body, joint))


    def is_circular(self, body, joint):
        joint_info = self.get_joint_info(body, joint)
        if joint_info.jointType == self.sim.JOINT_FIXED:
            return False
        return joint_info.jointUpperLimit < joint_info.jointLowerLimit


    def get_joint_limits(self, body, joint):
        if self.is_circular(body, joint):
            return self.CIRCULAR_LIMITS
        joint_info = self.get_joint_info(body, joint)
        return joint_info.jointLowerLimit, joint_info.jointUpperLimit


    def get_joint_state(self, body, joint):
        return self.JointState(*self.sim.getJointState(body, joint))


    def get_joint_position(self, body, joint):
        return self.get_joint_state(body, joint).jointPosition


    def get_joint_positions(self, body, joints):
        return tuple(self.get_joint_position(body, joint) for joint in joints)
    

    def get_link_name(self, base, link):
    
        if link == -1:
            link_name = self.sim.getBodyInfo(base)[0]
        else:
            link_name = self.sim.getJointInfo(base, link)[12]
        return link_name


    def get_link_pose_in_world(self, base, link):
        
        if link == -1:
            world_inertial_pose = self.sim.getBasePositionAndOrientation(base)
            dynamics_info = self.sim.getDynamicsInfo(base, link)
            local_inertial_pose = (dynamics_info[3], dynamics_info[4])

            local_inertial_pose_inv = self.sim.invertTransform(local_inertial_pose[0], local_inertial_pose[1])
            pos_orn = self.sim.multiplyTransforms(world_inertial_pose[0],
                                                world_inertial_pose[1],
                                                local_inertial_pose_inv[0],
                                                local_inertial_pose_inv[1])
        else:
            state = self.sim.getLinkState(base, link)
            pos_orn = (state[4], state[5])
        return pos_orn


    def add_gravity_to_ob(self, body_id, link_id=-1, gravity=-10):
        ob_mass = self.sim.getDynamicsInfo(body_id,link_id)[0]
        self.sim.applyExternalForce(body_id,link_id,forceObj=[0,0,gravity*ob_mass],posObj=[0,0,0],flags=self.sim.LINK_FRAME)