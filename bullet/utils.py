import os
import numpy as np
import pybullet as p
import pytransform3d.rotations as pr
from collections import namedtuple
from manifold_grasping.utils import load_mesh
from pathlib import Path
import pytransform3d.transformations as pt
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
        add_line(origin_pt, axis_pt, color=color, **kwargs)
 
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

    obj_mesh, T_ctr2obj = load_mesh(f'{mesh_root}/grasps/{graspfile}', scale=scale, mesh_root_dir=mesh_root, load_for_bullet=True)

    # Load env
    objinfo = {
        'name': f'{grasp_prefix}_{scale}',
        'urdf_dir': f'{mesh_root}/meshes_bullet/{grasp_prefix}_{scale}/model_normalized.urdf',
        'scale': float(scale),
        'T_ctr2obj': T_ctr2obj
    }
    return graspfile, obj_mesh, objinfo

def get_object_info(env, objname, mesh_root):
    """Used in multiple_views_acronym_bullet.py"""
    # objhash = os.listdir(f'{mesh_root}/meshes/{objname}')[0].replace('.obj', '')  # [HASH]

    # grasp_h5s = os.listdir(f'{mesh_root}/grasps')
    # for grasp_h5 in grasp_h5s:
        # grasp_id = grasp_h5.replace('.h5', '')
        # scale = float(grasp_id.split('_')[-1])
    grasp_h5 = Path(mesh_root) / 'grasps' / f'{objname}.h5'
    scale = objname.split('_')[-1]
    obj_mesh, T_ctr2obj = load_mesh(str(grasp_h5), scale=scale, mesh_root_dir=mesh_root, load_for_bullet=True)
    objinfo = {
        'name': objname,
        'urdf_dir': f'{mesh_root}/meshes_bullet/{objname}/model_normalized.urdf',
        'scale': float(scale),
        'T_ctr2obj': T_ctr2obj
    }
    return objinfo

def get_random_transform(pos):
    """Currently this is not a random transform"""
    T_rand = np.eye(4)
    T_rand[:3, 3] = pos
    # T_rand[:3, 3] = [0.5, 0.0, 0.5]
    # T_rand[:3, 3] = [0.5, 0.0, 0.1]

    # if cfg.vary_obj_pose:
        # TODO cache and load random scenes
        # raise NotImplementedError

    return T_rand


def place_object(env, target_pos, random=False, gravity=False):
    # place single object
    T_w2b = get_world2bot_transform()

    T_rand = get_random_transform(target_pos)

    # Apply object to centroid transform
    T_ctr2obj = env.objinfos[0]['T_ctr2obj']

    T_w2o = T_w2b @ T_rand @ T_ctr2obj
    # draw_pose(T_w2b @ T_rand)
    # print(T_w2o)
    pq_w2o = pt.pq_from_transform(T_w2o)  # wxyz

    p.resetBasePositionAndOrientation(
        env._objectUids[0],
        pq_w2o[:3],
        pr.quaternion_xyzw_from_wxyz(pq_w2o[3:])
    )
    p.resetBaseVelocity(env._objectUids[0], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    if gravity:
        for i in range(10000):
            p.stepSimulation()
    
    return T_w2o

# def get_object_info(env, objname, dset_root):
#     # Load object urdf and grasps
#     objhash = os.listdir(f'{dset_root}/meshes/{objname}')[0].replace('.obj', '')  # [HASH]
#     grasp_h5s = os.listdir(f'{dset_root}/grasps')
#     grasp_prefix = f'{objname}_{objhash}'  # for example: Book_[HASH]
#     for grasp_h5 in grasp_h5s:  # Get file in grasps/ corresponding to this object, hash, and scale
#         if grasp_prefix in grasp_h5:
#             graspfile = grasp_h5
#             scale = graspfile.split('_')[-1].replace('.h5', '')
#             break

#     obj_mesh, T_ctr2obj = load_mesh(f'{dset_root}/grasps/{graspfile}', scale=scale, mesh_root_dir=dset_root, load_for_bullet=True)

#     # Load env
#     objinfo = {
#         'name': f'{grasp_prefix}_{scale}',
#         'urdf_dir': f'{dset_root}/meshes_bullet/{grasp_prefix}_{scale}/model_normalized.urdf',
#         'scale': float(scale),
#         'T_ctr2obj': T_ctr2obj
#     }
#     return objinfo
