import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data

cfg = {
    "dist": 1.3,
    "yaw": 180,
    # "pitch": -41,
    "pitch": 0,
    "roll": 0,
    "look": [
        -0.35,
        -0.58,
        -0.88
    ],
    "fov": 60.0
}

# Draw axes in pybullet for debugging
# https://github.com/caelan/pybullet-planning/blob/master/pybullet_tools/utils.py
from collections import namedtuple
RGBA = namedtuple('RGBA', ['red', 'green', 'blue', 'alpha'])
BLACK = RGBA(0, 0, 0, 1)
CLIENT = 0
NULL_ID = -1
BASE_LINK = -1

def add_line(start, end, color=BLACK, width=1, lifetime=0, parent=NULL_ID, parent_link=BASE_LINK):
    assert (len(start) == 3) and (len(end) == 3)
    return p.addUserDebugLine(start, end, lineColorRGB=color[:3], lineWidth=width,
                            lifeTime=lifetime, parentObjectUniqueId=parent, parentLinkIndex=parent_link,
                            physicsClientId=CLIENT)

def draw_pose(T, length=0.1, d=3, **kwargs):
    origin_world = T @ np.array((0., 0., 0., 1.))
    for k in range(d):
        axis = np.array((0., 0., 0., 1.))
        axis[k] = 1*length
        axis_world = T @ axis
        origin_pt = origin_world[:3] 
        axis_pt = axis_world[:3] 
        color = np.zeros(3)
        color[k] = 1
        add_line(origin_pt, axis_pt, color=color, **kwargs)

# contact-graspnet depth to pc projection using the intrinsic matrix
def depth2pc(depth, K, rgb=None):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    mask = np.where(depth > 0)
    x,y = mask[1], mask[0]
    
    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]
        
    pc = np.vstack((world_x, world_y, world_z)).T
    return (pc, rgb)

# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/deep_mimic/mocap/transformation.py
def unit_vector(data, axis=None, out=None):
  """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
  """
  if out is None:
    data = np.array(data, dtype=np.float64, copy=True)
    if data.ndim == 1:
      data /= math.sqrt(np.dot(data, data))
      return data
  else:
    if out is not data:
      out[:] = np.array(data, copy=False)
    data = out
  length = np.atleast_1d(np.sum(data * data, axis))
  np.sqrt(length, length)
  if axis is not None:
    length = np.expand_dims(length, axis)
  data /= length
  if out is None:
    return data

def rotation_matrix(angle, direction, point=None):
  """Return matrix to rotate about axis defined by point and direction.
  """
  sina = math.sin(angle)
  cosa = math.cos(angle)
  direction = unit_vector(direction[:3])
  # rotation matrix around unit vector
  R = np.diag([cosa, cosa, cosa])
  R += np.outer(direction, direction) * (1.0 - cosa)
  direction *= sina
  R += np.array([[0.0, -direction[2], direction[1]], [direction[2], 0.0, -direction[0]],
                    [-direction[1], direction[0], 0.0]])
  M = np.identity(4)
  M[:3, :3] = R
  if point is not None:
    # rotation not around origin
    point = np.array(point[:3], dtype=np.float64, copy=False)
    M[:3, 3] = point - np.dot(R, point)
  return M

class Env:
    def __init__(self, cfg):
        self.cfg = cfg
        self._timeStep = 1.0 / 1000.0
        self._window_width = 640
        self._window_height = 480

    def reset(self):
        # lookdir = np.array([0, -1, 0]).reshape(3, 1)
        lookdir = np.array([0, 0, -1]).reshape(3, 1)
        updir = np.array([0, 1, 0]).reshape(3, 1)
        # pos = np.array([-.35, .88, -.72]).reshape(3, 1)
        pos = np.array([0.1, 0.3, 0.0]).reshape(3, 1)
        self._view_matrix = p.computeViewMatrix(pos, pos+lookdir, updir)

        self._aspect = float(self._window_width) / self._window_height

        self.near = 0.5
        self.far = 6

        focal_length = 450
        fovh = (np.arctan((self._window_height /
                           2) / focal_length) * 2 / np.pi) * 180

        self._proj_matrix = p.computeProjectionMatrixFOV(
             fovh, self._aspect, self.near, self.far
        )

        # https://www.edmundoptics.com/knowledge-center/application-notes/imaging/understanding-focal-length-and-field-of-view/
        self._intr_matrix = np.eye(3)
        self._intr_matrix[0, 2] = self._window_width / 2
        self._intr_matrix[1, 2] = self._window_height / 2
        # self._intr_matrix[0, 0] = self._window_height / (2 * np.tan(np.deg2rad(fovh) / 2)) * self._aspect
        self._intr_matrix[0, 0] = focal_length
        # self._intr_matrix[1, 1] = self._window_height / (2 * np.tan(np.deg2rad(fovh) / 2))
        self._intr_matrix[1, 1] = focal_length

        # Set table and plane
        p.resetSimulation()
        p.setTimeStep(self._timeStep)

        # Intialize robot and objects
        p.stepSimulation()

        table_file =   "data/objects/table/models"
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.table_id = p.loadURDF(
            "table/table.urdf",
            # os.path.join(table_file, 'model_normalized.urdf'),
            0.1,
            0.3,
            -2,
            # 0.0,
            # -0.5,
            # -1.32,
            0.707,
            0.0,
            0.0,
            0.707,
        )

    def get_obs(self):
        _, _, rgba, zbuffer, mask = p.getCameraImage(
            width=self._window_width,
            height=self._window_height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
        )

        depth_zo = 2. * zbuffer - 1.
        depth = (self.far + self.near - depth_zo * (self.far - self.near))
        depth = (2. * self.near * self.far) / depth
        rgb = rgba[..., :3]

        depth[mask != 0] = 0
        pc, _ = depth2pc(depth, self._intr_matrix, rgb)

        return rgb, depth, pc

    def draw_pc(self, pc):
        T_world2camgl = np.linalg.inv(np.asarray(self._view_matrix).reshape((4, 4), order='F'))
        # draw_pose(T_world2camgl)
        T_camgl2cam = rotation_matrix(angle=np.pi, direction=[1, 0, 0])
        T_world2cam = T_world2camgl @ T_camgl2cam
        draw_pose(T_world2cam)

        idxs = np.random.choice(len(pc), size=1000)
        xyz_cam = pc[idxs]

        xyzh_cam = np.hstack([xyz_cam, np.ones((len(xyz_cam), 1))])
        for xyzh in xyzh_cam[:30]:
            xyz_world = (T_world2cam @ xyzh)[:3]
            T_pt = np.eye(4)
            T_pt[:3, 3] = xyz_world
            draw_pose(T_pt)
        
        # Standardize cameras and cfg
        import IPython; IPython.embed()

if __name__ == '__main__':
    cid = p.connect(p.GUI)
    # p.resetDebugVisualizerCamera(cfg['dist'], cfg['yaw'], cfg['pitch'], cfg['look'])

    env = Env(cfg)
    env.reset()
    rgb, depth, pc = env.get_obs()
    env.draw_pc(pc)

    p.disconnect()

        # TODO visualize T_pt vs. point cloud points
        # fig = plt.figure(figsize=(4,4))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter([0], [0], [0], s=100, c='green')
        # ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=_)

        # ax.axes.set_xlim3d(left=-1.5, right=1.5) 
        # ax.axes.set_ylim3d(bottom=-1.5, top=1.5) 
        # ax.axes.set_zlim3d(bottom=-1.5, top=1.5) 
        # plt.show()
        # fig = plt.figure(figsize=(4,4))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter([0], [0], [0], s=100, c='green')
        # ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=_ / 255.)

        # ax.axes.set_xlim3d(left=-1.5, right=1.5) 
        # ax.axes.set_ylim3d(bottom=-1.5, top=1.5) 
        # ax.axes.set_zlim3d(bottom=-1.5, top=1.5) 
        # plt.show()
