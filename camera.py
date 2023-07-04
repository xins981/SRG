import numpy as np
from transformations import *
import pybullet as p



class Camera:
  
    def __init__(self, K, H, W):

        self._height = H
        self._width = W
        self._k = K
        x0 = 0
        y0 = 0
        self.zfar = 100
        self.znear = 0.1
        self._projection_matrix = \
        np.array([[2*self._k[0,0]/self._width, -2*self._k[0,1]/self._width, (self._width - 2*self._k[0,2] + 2*x0)/self._width,                        0],
                    [          0,     2*self._k[1,1]/self._height,  (-self._height + 2*self._k[1,2] + 2*y0)/self._height,                       0],
                    [          0,        0,             (-self.zfar - self.znear)/(self.zfar - self.znear),  -2*self.zfar*self.znear/(self.zfar - self.znear)],
                    [          0,        0,                             -1,                                      0]]).reshape(4,4)
        self.view_matrix = np.array([-0.0841524825,   0.992909968, -0.0839533508,   0.607104242,
                                        0.981524885,  0.0680680275,  -0.178817391,    0.30635184,
                                        -0.171835035, -0.0974502265,   -0.98029393,   0.705115497,
                                            0,             0,             0,             1]).reshape(4,4)
        self.id = None


    def shot(self):

        gl_in_cv = np.eye(4)
        gl_in_cv[1,1] = -1
        gl_in_cv[2,2] = -1
        world_in_cam = np.linalg.inv(self.view_matrix)
        world_in_cam_gl = gl_in_cv @ world_in_cam
        _, _, rgb, depth, seg = self.getCameraImage(width=self._width, 
                                                    height=self._height, 
                                                    viewMatrix=world_in_cam_gl.T.reshape(-1),
                                                    projectionMatrix=self._projection_matrix.T.reshape(-1),
                                                    shadow=1, 
                                                    lightDirection=[1, 1, 1])
        depth = self.zfar * self.znear / (self.zfar - (self.zfar - self.znear) * depth)
        depth[seg<0] = 0
        rgb = rgb[...,:3]
        return rgb, depth, seg


    def getCameraImage(self):
        raise RuntimeError('`getCameraImage` method of Camera Class should be hooked by the environment.')