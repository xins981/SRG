import numpy as np
from transformations import *
import pybullet as p



class Camera:
  
    def __init__(self, K, H, W):

        self._H = H
        self._W = W
        self._K = K
        x0 = 0
        y0 = 0
        self.zfar = 100
        self.znear = 0.1
        self.projectionMatrix = \
        np.array([[2*K[0,0]/self._W, -2*K[0,1]/self._W, (self._W - 2*K[0,2] + 2*x0)/self._W,                        0],
                    [          0,     2*K[1,1]/self._H,  (-self._H + 2*K[1,2] + 2*y0)/self._H,                       0],
                    [          0,        0,             (-self.zfar - self.znear)/(self.zfar - self.znear),  -2*self.zfar*self.znear/(self.zfar - self.znear)],
                    [          0,        0,                             -1,                                      0]]).reshape(4,4)
        self.pose_in_world = np.array([-0.0841524825,   0.992909968, -0.0839533508,   0.607104242,
                                        0.981524885,  0.0680680275,  -0.178817391,    0.30635184,
                                        -0.171835035, -0.0974502265,   -0.98029393,   0.705115497,
                                            0,             0,             0,             1]).reshape(4,4)
        self.id = None


    def shot(self):

        gl_in_cv = np.eye(4)
        gl_in_cv[1,1] = -1
        gl_in_cv[2,2] = -1
        world_in_cam = np.linalg.inv(self.pose_in_world)
        world_in_cam_gl = gl_in_cv @ world_in_cam
        _,_,rgb,depth,seg = self.getCameraImage(self._W, self._H, viewMatrix=world_in_cam_gl.T.reshape(-1),
                                                projectionMatrix=self.projectionMatrix.T.reshape(-1),
                                                shadow=1, lightDirection=[1, 1, 1])
        depth = self.zfar * self.znear / (self.zfar - (self.zfar - self.znear) * depth)
        depth[seg<0] = 0
        rgb = rgb[...,:3]
        return rgb,depth,seg
    

    # def getCameraImage(self, width, height, viewMatrix, projectionMatrix, shadow, lightDirection):
    def getCameraImage(self):
        raise RuntimeError('`getCameraImage` method of Camera Class should be hooked by the environment.')