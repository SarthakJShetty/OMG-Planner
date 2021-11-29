
import pybullet as p
# import pytransform3d.rotations as pr
from collections import namedtuple
RGBA = namedtuple('RGBA', ['red', 'green', 'blue', 'alpha'])
BLACK = RGBA(0, 0, 0, 1)
CLIENT = 0
NULL_ID = -1
BASE_LINK = -1

# https://github.com/caelan/pybullet-planning/blob/master/pybullet_tools/utils.py
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
 
# def depth2pc(depth, K, rgb=None):
#     """
#     Convert depth and intrinsics to point cloud and optionally point cloud color
#     :param depth: hxw depth map in m
#     :param K: 3x3 Camera Matrix with intrinsics
#     :returns: (Nx3 point cloud, point cloud color)
#     """

#     mask = np.where(depth > 0)
#     x,y = mask[1], mask[0]
    
#     normalized_x = (x.astype(np.float32) - K[0,2])
#     normalized_y = (y.astype(np.float32) - K[1,2])

#     world_x = normalized_x * depth[y, x] / K[0,0]
#     world_y = normalized_y * depth[y, x] / K[1,1]
#     world_z = depth[y, x]

#     if rgb is not None:
#         rgb = rgb[y,x,:]
        
#     pc = np.vstack((world_x, world_y, world_z)).T
#     return (pc, rgb)
