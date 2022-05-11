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
from .utils import get_world2bot_transform, get_random_transform, draw_pose, bullet_execute_plan

import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from pathlib import Path
from datetime import datetime
import csv
import cv2


def init_cfg(args):
    """
    Modify cfg based on command line arguments.
    """
    cfg.vis = False  # set vis to False because panda_env render is True
    cfg.window_width = 200
    cfg.window_height = 200

    cfg.method = args.method
    if 'GF' in cfg.method:
        # pq input -> pq output
        cfg.use_goal_grad = True
        cfg.fixed_endpoint = False
        cfg.goal_set_proj = False
        cfg.use_min_goal_cost_traj = False
        cfg.learnedgrasp_weights = '/data/manifolds/fb_runs/multirun/pq-pq_bookonly/2022-05-07_223943/lossl1_lr0.0001/default_default/13_13/checkpoints/last.ckpt'
        cfg.ol_alg = None
        cfg.smoothness_base_weight = 0.01  # 0.1 weight for smoothness cost in total cost
        cfg.base_obstacle_weight = 0.1  # 1.0 weight for obstacle cost in total cost
        cfg.base_grasp_weight = 10.0  # weight for grasp cost in total cost
        cfg.cost_schedule_boost = 1.0  # cost schedule boost for smoothness cost weight
        cfg.base_step_size = 0.5  # initial step size in gradient descent
        cfg.optim_steps = 500 # optimization steps for each planner call
    elif 'OMG' in cfg.method:       # OMG with apples to apples parameters
        cfg.goal_set_proj = True
        cfg.fixed_endpoint = False
        cfg.ol_alg = 'MD'
        cfg.smoothness_base_weight = 0.1  # 0.1 weight for smoothness cost in total cost
        cfg.base_obstacle_weight = 1.0  # 1.0 weight for obstacle cost in total cost
        cfg.cost_schedule_boost = 1.02  # cost schedule boost for smoothness cost weight
        cfg.base_step_size = 0.1  # initial step size in gradient descent
        cfg.optim_steps = 50  # optimization steps for each planner call
        if 'comp' in cfg.method:        # apples to apples parameters
            cfg.use_standoff = False
        elif 'orig' in cfg.method:      # original parameters
            cfg.use_standoff = True
    else:
        raise NotImplementedError
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
            scale = graspfile.split('_')[-1].replace('.h5', '')
            break

    obj_mesh, T_ctr2obj = load_mesh(f'{grasp_root}/{graspfile}', scale=scale, mesh_root_dir=mesh_root, load_for_bullet=True)

    # Load env
    objinfo = {
        'name': f'{grasp_prefix}_{scale}',
        'urdf_dir': f'{mesh_root}/meshes_bullet/{grasp_prefix}_{scale}/model_normalized.urdf',
        'scale': float(scale),
        'T_ctr2obj': T_ctr2obj
    }
    env.reset(no_table=True, objinfo=objinfo)
    return objinfo


def randomly_place_object(env):
    # place single object
    T_w2b = get_world2bot_transform()

    # TODO make actually random
    T_rand = get_random_transform()

    # Apply object to centroid transform
    T_ctr2obj = env.objinfos[0]['T_ctr2obj']

    T_w2o = T_w2b @ T_rand @ T_ctr2obj
    # draw_pose(T_w2b @ T_rand)
    pq_w2o = pt.pq_from_transform(T_w2o)  # wxyz

    p.resetBasePositionAndOrientation(
        env._objectUids[0],
        pq_w2o[:3],
        pr.quaternion_xyzw_from_wxyz(pq_w2o[3:])
    )
    p.resetBaseVelocity(env._objectUids[0], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def init_metrics_entry(object_name, scene_idx):
    return {
        'object_name': object_name,
        'scene_idx': scene_idx,
        'execution': None,
        'planning': None,
        'smoothness': None,
        'collision': None,
        'time': None
    }


def init_video_writer(path, scene_idx):
    return cv2.VideoWriter(
        f"{path}/scene_{scene_idx}.avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (640, 480),
    )


def init_dirs(out_dir, method):
    save_path = Path(f'{out_dir}/{method}_{datetime.now().strftime("%y-%m-%d-%H-%M-%S")}')
    save_path.mkdir(parents=True)
    (save_path / 'info').mkdir()
    (save_path / 'videos').mkdir()
    return str(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="which method to use", required=True, choices=['compOMG_known', 'origOMG_known', 'GF_learned'])
    parser.add_argument("-mr", "--mesh_root", help="mesh root", type=str, default="/data/manifolds/acronym_mini_bookonly")
    parser.add_argument("-gr", "--grasp_root", help="grasp root", type=str, default="/data/manifolds/acronym_mini_bookonly/grasps")
    parser.add_argument("-o", "--out_dir", help="Directory to save experiment to", type=str, default="/data/manifolds/pybullet_eval")
    parser.add_argument("--write_video", help="write video", action="store_true")
    args = parser.parse_args()
    init_cfg(args)

    # Init environment
    env = PandaEnv(renders=True, gravity=False)

    # Init save dir and csv
    save_path = init_dirs(args.out_dir, args.method)
    with open(f'{save_path}/metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['object_name', 'scene_idx', 'execution', 'planning', 'smoothness', 'collision', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Iterate over objects in folder
    scene = PlanningScene(cfg)
    for objname in os.listdir(f'{args.mesh_root}/meshes'):
        for scene_idx in range(1):
            metrics = init_metrics_entry(objname, scene_idx)
            video_writer = init_video_writer(f"{save_path}/videos", scene_idx) if args.write_video else None

            env.reset(no_table=True)
            objinfo = reset_env_with_object(env, objname, args.grasp_root, args.mesh_root)
            randomly_place_object(env)  # Note: currently not random

            # Scene has separate Env class which is used for planning
            # Add object to planning scene env
            trans_w2o, orn_w2o = p.getBasePositionAndOrientation(env._objectUids[0])  # xyzw

            # change to world to object centroid so planning scene env only sees objects in centroid frame
            T_b2w = np.linalg.inv(get_world2bot_transform())
            T_w2o = np.eye(4)
            T_w2o[:3, :3] = pr.matrix_from_quaternion(tf_quat(orn_w2o))
            T_w2o[:3, 3] = trans_w2o
            T_o2c = np.linalg.inv(objinfo['T_ctr2obj'])
            T_b2c = T_b2w @ T_w2o @ T_o2c
            trans = T_b2c[:3, 3]
            orn = pr.quaternion_from_matrix(T_b2c[:3, :3])  # wxyz
            draw_pose(T_w2o @ T_o2c)

            obj_prefix = '/data/manifolds/acronym_mini_bookonly/meshes_bullet'
            scene.env.add_object(objinfo['name'], trans, orn, obj_prefix=obj_prefix, abs_path=True)
            scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1, 0, 0, 0]))
            scene.env.combine_sdfs()
            if 'origOMG' in cfg.method:  # disable collision for target object
                cfg.disable_collision_set = [objinfo['name']]

            # Set grasp selection method for planner
            scene.env.set_target(objinfo['name'])
            scene.reset(lazy=True)

            info = scene.step()
            plan = scene.planner.history_trajectories[-1]
            grasp_success = bullet_execute_plan(env, plan, args.write_video, video_writer)

            # Save data
            np.savez(f'{save_path}/info/{objname}_{scene_idx}', info=info, plan=plan)
            metrics['execution'] = grasp_success
            metrics['planning'] = info[-1]['execute']
            metrics['smoothness'] = info[-1]['smooth']
            metrics['collision'] = info[-1]['obs']
            metrics['time'] = info[-1]['time']
            with open(f'{save_path}/metrics.csv', 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(metrics)

            # Convert avi to high quality gif 
            if args.write_video and info != []:
                os.system(f'ffmpeg -y -i {save_path}/videos/scene_{scene_idx}.avi -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 {save_path}/scene_{scene_idx}.gif')

        import IPython; IPython.embed()
        break



