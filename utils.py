import numpy as np
from transformations import *
import copy, trimesh
from uuid import uuid4
import open3d as o3d
from collections import namedtuple
import math



# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0
# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]
# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


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
	H,W = depth.shape[:2]
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


def safe_zip(sequence1, sequence2): # TODO: *args
    sequence1, sequence2 = list(sequence1), list(sequence2)
    assert len(sequence1) == len(sequence2)
    return list(zip(sequence1, sequence2))


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True
    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)


def random_quaternion(rand=None):
    """Return uniform random unit quaternion.
    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.
    >>> q = random_quaternion()
    >>> numpy.allclose(1.0, vector_norm(q))
    True
    >>> q = random_quaternion(numpy.random.random(3))
    >>> q.shape
    (4,)
    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array((np.sin(t1)*r1,
                        np.cos(t1)*r1,
                        np.sin(t2)*r2,
                        np.cos(t2)*r2), dtype=np.float64)


def random_rotation_matrix(rand=None):
    """Return uniform random rotation matrix.
    rnd: array like
        Three independent random variables that are uniformly distributed
        between 0 and 1 for each returned quaternion.
    >>> R = random_rotation_matrix()
    >>> numpy.allclose(numpy.dot(R.T, R), numpy.identity(4))
    True
    """
    return quaternion_matrix(random_quaternion(rand))


def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True
    """
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M


#---------------------------------------------------------------------------
# Pybullet Functions
#---------------------------------------------------------------------------
INF = np.inf
PI = np.pi
EPSILON = 1e-6
Interval = namedtuple('Interval', ['lower', 'upper']) # AABB
JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                            'qIndex', 'uIndex', 'flags', 'jointDamping', 
                                            'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                            'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                            'parentFramePos', 'parentFrameOrn', 'parentIndex'])
JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                       'jointReactionForces', 'appliedJointMotorTorque'])
UNIT_LIMITS = Interval(0., 1.)
CIRCULAR_LIMITS = Interval(-PI, PI)
UNBOUNDED_LIMITS = Interval(-INF, INF)



class Pybullet_Utils:

    def __init__(self, sim):
        self._sim = sim


    def set_body_pose_in_world(self, body_id, ob_in_world):
        
        trans = ob_in_world[:3,3]
        q_wxyz = quaternion_from_matrix(ob_in_world)
        q_xyzw = [q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]
        self._sim.resetBasePositionAndOrientation(body_id, trans, q_xyzw)


    def get_ob_pose_in_world(self, body_id):
        trans,q_xyzw = self._sim.getBasePositionAndOrientation(body_id)
        ob_in_world = np.eye(4)
        ob_in_world[:3,3] = trans
        q_wxyz = [q_xyzw[-1],q_xyzw[0],q_xyzw[1],q_xyzw[2]]
        R = quaternion_matrix(q_wxyz)[:3,:3]
        ob_in_world[:3,:3] = R
        return ob_in_world


    def set_joint_positions(self, body, joints, values):
        for joint, value in safe_zip(joints, values):
            self.set_joint_position(body, joint, value)


    def set_joint_position(self, body, joint, value):
        # TODO: remove targetVelocity=0
        self._sim.resetJointState(body, joint, targetValue=value, targetVelocity=0, physicsClientId=CLIENT)


    def get_joint_info(self, body, joint):
        return JointInfo(*self._sim.getJointInfo(body, joint))


    def is_circular(self, body, joint):
        joint_info = self.get_joint_info(body, joint)
        if joint_info.jointType == self._sim.JOINT_FIXED:
            return False
        return joint_info.jointUpperLimit < joint_info.jointLowerLimit


    def get_joint_limits(self, body, joint):
        # TODO: make a version for several joints?
        if self.is_circular(body, joint):
            # TODO: return UNBOUNDED_LIMITS
            return CIRCULAR_LIMITS
        joint_info = self.get_joint_info(body, joint)
        return joint_info.jointLowerLimit, joint_info.jointUpperLimit


    def get_joint_state(self, body, joint):
        return JointState(*self._sim.getJointState(body, joint))


    def get_joint_position(self, body, joint):
        return self.get_joint_state(body, joint).jointPosition


    def get_joint_positions(self, body, joints):
        return tuple(self.get_joint_position(body, joint) for joint in joints)


