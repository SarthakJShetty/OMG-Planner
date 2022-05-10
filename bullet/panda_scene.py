import os
import argparse
import pybullet as p
import numpy as np
from omg.config import cfg
from omg.core import PlanningScene
from omg.util import tf_quat

from .panda_env import PandaEnv

from acronym_tools import create_gripper_marker
from manifold_grasping.utils import load_mesh
from .utils import get_world2bot_transform, get_random_transform, draw_pose

import pytransform3d.rotations as pr
import pytransform3d.transformations as pt


def init_cfg(args):
    """
    Modify cfg based on command line arguments.
    """
    cfg.vis = False  # set vis to False because panda_env render is True
    cfg.window_width = 200
    cfg.window_height = 200
    cfg.scene_file = ''

    cfg.use_goal_grad = True
    cfg.use_standoff = False
    cfg.grasp_schedule_boost = 1.0
    cfg.cost_schedule_boost = 1.0

    # pq input -> pq output
    cfg.method = ''
    cfg.use_standoff = False
    cfg.fixed_endpoint = False
    if cfg.use_goal_grad:
        cfg.goal_set_proj = False
        cfg.use_min_goal_cost_traj = False
    cfg.use_ik = False
    cfg.grasp_prediction_weights = '/data/manifolds/fb_runs/multirun/pq-pq_bookonly/2022-05-07_223943/lossl1_lr0.0001/default_default/13_13/checkpoints/last.ckpt'

    cfg.ol_alg = None
    # cfg.root_dir = '/data/manifolds/acronym_mini_bookonly/meshes_bullet'
    # debug method using fixed grasp
    # cfg.ol_alg = 'Baseline'
    # cfg.goal_idx = -1

    cfg.get_global_param()


def reset_env_with_object(env, objname, grasp_root, mesh_root):
    # Load object urdf and grasps
    objhash = os.listdir(f'{mesh_root}/meshes/{objname}')[0].replace('.obj', '')  # [HASH]
    grasp_h5s = os.listdir(grasp_root)
    grasp_prefix = f'{objname}_{objhash}'  # for example: Book_[HASH]
    for grasp_h5 in grasp_h5s:  # Get file in grasps/ corresponding to this object, hash, and scale
        if grasp_prefix in grasp_h5:
            graspfile = grasp_h5
            break

    obj_mesh, T_ctr2obj, obj_scale = load_mesh(f'{grasp_root}/{graspfile}', mesh_root_dir=mesh_root, load_for_bullet=True)

    # Load env
    objinfo = {
        'name': grasp_prefix,
        'urdf_dir': f'{mesh_root}/meshes_bullet/{grasp_prefix}/model_normalized.urdf',
        'scale': obj_scale,
        'T_ctr2obj': T_ctr2obj
    }
    env.reset(no_table=True, objinfo=objinfo)
    return objinfo


def randomly_place_object(env):
    # place single object
    T_w2b = get_world2bot_transform(env)
 
    # TODO make actually random
    T_rand = get_random_transform(env)
 
    T_w2o = T_w2b @ T_rand
    draw_pose(T_w2o)
    pq_w2o = pt.pq_from_transform(T_w2o)  # wxyz

    p.resetBasePositionAndOrientation(
        env._objectUids[0],
        pq_w2o[:3],
        pr.quaternion_xyzw_from_wxyz(pq_w2o[3:])
    )
    p.resetBaseVelocity(env._objectUids[0], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mr", "--mesh_root", help="mesh root", type=str, default="/data/manifolds/acronym_mini_bookonly")
    parser.add_argument("-gr", "--grasp_root", help="grasp root", type=str, default="/data/manifolds/acronym_mini_bookonly/grasps") 
    args = parser.parse_args()

    init_cfg(args)

    scene = PlanningScene(cfg)

    # Init environment
    env = PandaEnv(renders=True, gravity=False)

    # Iterate over objects in folder, place randomly in scene.
    for objname in os.listdir(f'{args.mesh_root}/meshes'):
        env.reset(no_table=True)
        objinfo = reset_env_with_object(env, objname, args.grasp_root, args.mesh_root)
        randomly_place_object(env)  # Note: currently not random

        # Scene has separate Env class which is used for planning
        # Add object to planning scene env
        trans, orn = p.getBasePositionAndOrientation(env._objectUids[0])  # x y z w
        obj_prefix = '/data/manifolds/acronym_mini_bookonly/meshes_bullet'
        scene.env.add_object(objinfo['name'], trans, tf_quat(orn), obj_prefix=obj_prefix, abs_path=True)
        scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1, 0, 0, 0]))
        scene.env.combine_sdfs()
        cfg.disable_collision_set = [objinfo['name']]

        # Set grasp selection method for planner
        scene.env.set_target(objinfo['name'])
        scene.reset(lazy=True)


        import IPython; IPython.embed()
        break

    

