import numpy as np
import transformations as tf
import copy, os
import open3d as o3d
from collections import namedtuple
import torch
from matplotlib.colors import LinearSegmentedColormap
from pybullet_tools.utils import pose_from_tform


def to_grasp(param):

    p = param[0:3]
    axis_y = param[3:6]
    theta = param[6] # approach angle

    axis_z = [axis_y[1], -axis_y[0], 0]
    axis_x = np.cross(axis_y, axis_z)
    # axis_z = [0, axis_y[1], -axis_y[2]]
    # axis_x = np.cross(axis_y, axis_z)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    perturbation = np.array([[cos_t, 0, sin_t],
                            [0    , 1,     0],
                            [-sin_t, 0, cos_t]])
    # perturbation = np.eye(3)
    r = np.stack((axis_x, axis_y, axis_z)).T
    r_iden = normalize_rotation(r)
    
    T = np.eye(4)
    T[:3,:3] = r_iden @ perturbation
    T[:3,3] = p
    

    return pose_from_tform(T)


def normalize_rotation(pose):
  
    new_pose = pose.copy()
    scales = np.linalg.norm(pose[:3,:3], axis=0)
    scales += 1e-6
    new_pose[:3,:3] /= scales.reshape(1,3)
    return new_pose


def batch_param_to_grasp(grasp_param, device=None):

    if len(grasp_param.shape) < 3: # (B, 10)

        grasp_poses = []
        for param in grasp_param:
            action = param2action(param)
            grasp_poses.append(action)
    else: # (B, N, 10)

        grasp_poses = []
        for batch in grasp_param:
            batch_poses = [] # (N, 7)
            for param in batch:
                action = param2action(param)
                batch_poses.append(action)
            grasp_poses.append(batch_poses)

    grasp_poses = np.array(grasp_poses, dtype=np.float32)
    if device != None:
        grasp_poses = torch.from_numpy(grasp_poses).to(device)
    
    return grasp_poses


def create_urdf_from_mesh(mesh_dir, concave=False, out_dir=None, mass=0.1, vhacd_dir=None, has_collision=True, scale=np.ones((3))):
    
    assert '.obj' in mesh_dir, f'mesh_dir={mesh_dir}'

    lateral_friction = 0.8
    spinning_friction = 0.5
    rolling_friction = 0.5

    concave_str = 'no'
    if concave:
        concave_str = 'yes'

    collision_mesh_dir = mesh_dir
    if mass!=0:
        collision_mesh_dir = vhacd_dir

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


def normalizeRotation(pose):

    new_pose = pose.copy()
    scales = np.linalg.norm(pose[:3,:3],axis=0)
    new_pose[:3,:3] /= scales.reshape(1,3)
    return new_pose


def save_pcd(pts, data_dir, rollout):

    data_dir = f'{data_dir}/pcd'
    for i in range(pts.shape[0]):
        pcd_dir = f'{data_dir}/{i:02d}'
        if not os.path.exists(pcd_dir):
            os.makedirs(pcd_dir)
        elif i == 0:
            break

    for i, batch in enumerate(pts):
        cloud = toOpen3dCloud(batch)
        file_name = os.path.join(f'{data_dir}/{i:02d}', f'{rollout:05d}.pts.pcd')
        o3d.io.write_point_cloud(file_name, cloud)


def save_q_map(pts, q_value, data_dir, rollout):
    
    data_dir = f'{data_dir}/q_map'
    for i in range(pts.shape[0]):
        q_map_dir = f'{data_dir}/{i:02d}'
        if not os.path.exists(q_map_dir):
            os.makedirs(q_map_dir)
        elif i == 0:
            break

    colors = [(0, 0, 1), (1, 0, 0)]  # Blue to red
    n_bins = 100  # Number of bins to represent the gradient
    cmap_name = 'energy_gradient'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    for i, batch in enumerate(pts):
        min_q = np.min(q_value[i,:])
        max_q = np.max(q_value[i,:])
        scaled_q = (q_value[i,:] - min_q) / (max_q - min_q)
        
        color_q = cm(scaled_q)[:,:3] # (N, 3)

        cloud = toOpen3dCloud(points=batch, colors=color_q)
        file_name = os.path.join(f'{data_dir}/{i:02d}', f'{rollout:05d}.q_vlaue.pcd')
        o3d.io.write_point_cloud(file_name, cloud)


def to_homo(pts):
    
    assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
    homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
    return homo
