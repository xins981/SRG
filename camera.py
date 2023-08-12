import numpy as np
import os,sys,yaml,copy,pickle
import pybullet as p
from pybullet_tools.utils import *

cv_from_gl = np.eye(4)
cv_from_gl[1,1] = -1
cv_from_gl[2,2] = -1

# H = 1544   # image height
# W = 2064   # image width
H = 480   # image height
W = 640   # image width
# K = np.array([2257.7500557850776, 0,                  int(W/2),
#               0,                  2257.4882391629421, int(H/2),
#               0,                  0,                  1]).reshape(3,3)
K = np.array([700.0775366775434, 0,                  int(W/2),
              0,                  701.8098152838163, int(H/2),
              0,                  0,                  1]).reshape(3,3)


class Camera:
  def __init__(self, world_from_camera):
    x0 = 0
    y0 = 0
    self.zfar = 3
    self.znear = 0.1
    self.k = K
    
    self.world_from_camera = tform_from_pose(world_from_camera) 

    self.projectionMatrix = np.array([[2*K[0,0]/W, -2*K[0,1]/W, (W - 2*K[0,2] + 2*x0)/W,            0],
                                    [0,     2*K[1,1]/H,  (-H + 2*K[1,2] + 2*y0)/H,                       0],
                                    [0,        0,             (-self.zfar - self.znear)/(self.zfar - self.znear),  -2*self.zfar*self.znear/(self.zfar - self.znear)],
                                    [0,        0,             -1,                                      0]])
    camera_from_world = np.linalg.inv(self.world_from_camera)
    self.cam_gl_from_world = cv_from_gl @ camera_from_world
    self.id = load_pybullet("resources/models/kinect/kinect.urdf", fixed_base=True)
    set_pose(self.id, multiply(world_from_camera, Pose(euler=[0, -math.radians(90), math.radians(90)])))


  def render(self):

    # set_body_pose_in_world(self.cam_id,cam_in_world)

    _,_,rgb,depth,seg = p.getCameraImage(W, H,
                                        viewMatrix=self.cam_gl_from_world.T.reshape(-1),
                                        projectionMatrix=self.projectionMatrix.T.reshape(-1))
    depth = self.zfar * self.znear / (self.zfar - (self.zfar - self.znear) * depth)
    depth[seg<0] = 0
    rgb = rgb[...,:3]
    return rgb,depth,seg