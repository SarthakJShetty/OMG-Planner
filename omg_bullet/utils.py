import os
import numpy as np
import pybullet as p
import pytransform3d.rotations as pr
from collections import namedtuple
from manifold_grasping.utils import load_mesh
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

def draw_pose(T, length=0.1, d=3, alt_color=False, **kwargs):
    origin_world = T @ np.array((0., 0., 0., 1.))
    dbg_ids = []
    for k in range(d):
        axis = np.array((0., 0., 0., 1.))
        axis[k] = 1*length
        axis_world = T @ axis
        origin_pt = origin_world[:3] 
        axis_pt = axis_world[:3] 
        color = np.zeros(3)
        if alt_color:
            color[k] = 1
            color[(k+1)%(d)] = 1
        else:
            color[k] = 1
        dbg_id = add_line(origin_pt, axis_pt, color=color, **kwargs)
        dbg_ids.append(dbg_id)
    return dbg_ids

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
        
    pc = np.vstack((world_x, world_y, world_z, np.ones((len(world_x))))).T
    return (pc, rgb)

def get_world2cam_transform(env):
    T_world2camgl = np.linalg.inv(np.asarray(env._view_matrix).reshape((4, 4), order='F'))
    T_camgl2cam = np.zeros((4, 4))
    T_camgl2cam[:3, :3] = pr.matrix_from_axis_angle([1, 0, 0, np.pi])
    T_camgl2cam[3, 3] = 1
    T_world2cam = T_world2camgl @ T_camgl2cam
    return T_world2cam

def get_world2bot_transform():
    pos, orn = p.getBasePositionAndOrientation(0)  # panda UID is 0
    mat = np.asarray(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    T_world2bot = np.eye(4)
    T_world2bot[:3, :3] = mat
    T_world2bot[:3, 3] = pos
    return T_world2bot

def bullet_execute_plan(env, plan, write_video, video_writer):
    print('executing...')
    for k in range(plan.shape[0]):
        obs, rew, done, _ = env.step(plan[k].tolist())
        if write_video:
            video_writer.write(obs['rgb'][0][:, :, [2, 1, 0]].astype(np.uint8))
            # video_writer.write(obs['rgb'][:, :, [2, 1, 0]].astype(np.uint8))
    (rew, ret_obs) = env.retract(record=True)
    if write_video: 
        for robs in ret_obs:
            video_writer.write(robs['rgb'][0][:, :, [2, 1, 0]].astype(np.uint8))
            video_writer.write(robs['rgb'][0][:, :, [2, 1, 0]].astype(np.uint8)) # to get enough frames to save
    return rew

def objinfo_from_obj(grasp_h5s, mesh_root, objname):
    """Used in get_bullet_labels.py"""
    # Load object urdf and grasps
    objhash = os.listdir(f'{mesh_root}/meshes/{objname}')[0].replace('.obj', '') # [HASH]
    grasp_prefix = f'{objname}_{objhash}' # Book_[HASH]
    for grasp_h5 in grasp_h5s: # Get file in grasps/ corresponding to this book, hash, and scale
        if grasp_prefix in grasp_h5:
            graspfile = grasp_h5
            scale = graspfile.split('_')[-1].replace('.h5', '')
            break

    obj_mesh, T_ctr2obj = load_mesh(f'{mesh_root}/grasps/{graspfile}', scale=scale, mesh_root_dir=mesh_root, load_processed=True)

    # Load env
    objinfo = {
        'name': f'{grasp_prefix}_{scale}',
        'urdf_dir': f'{mesh_root}/meshes_bullet/{grasp_prefix}_{scale}/model_normalized.urdf',
        'scale': float(scale),
        'T_ctr2obj': T_ctr2obj
    }
    return graspfile, obj_mesh, objinfo
