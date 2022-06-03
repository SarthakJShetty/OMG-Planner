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
from .utils import get_world2bot_transform, draw_pose, bullet_execute_plan, get_object_info, place_object

import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from pathlib import Path
from datetime import datetime
import csv
import cv2
import subprocess
import yaml

def init_cfg(args):
    """
    Modify cfg based on command line arguments.
    """
    # cfg.render = args.render
    cfg.vis = False  
    cfg.window_width = 200
    cfg.window_height = 200

    cfg.eval_type = args.eval_type
    cfg.vary_obj_pose = False if 'fixedpose' in cfg.eval_type else True
    cfg.gravity = False if 'nograv' in cfg.eval_type else True
    if '1obj_float_fixedpose_nograv':
        cfg.table = False
        cfg.cam_look = [-0.05, -0.5, -0.6852]
        cfg.tgt_pos = [0.5, 0.0, 0.5]
    else:
        raise NotImplementedError
        # [-0.05, -0.5, -1.1]

    cfg.method = args.method
    if 'GF' in cfg.method:
        # pq input -> pq output
        cfg.use_goal_grad = True
        cfg.fixed_endpoint = False
        # cfg.use_min_goal_cost_traj = False
        cfg.ol_alg = None
        
        # The following params are controlled as args so these don't reflect the values used
        cfg.smoothness_base_weight = 0.1  # 0.1 weight for smoothness cost in total cost
        cfg.base_obstacle_weight = 0.1  # 1.0 weight for obstacle cost in total cost
        cfg.base_grasp_weight = 5.0  # weight for grasp cost in total cost
        cfg.cost_schedule_boost = 1.0  # cost schedule boost for smoothness cost weight
        cfg.base_step_size = 0.3  # initial step size in gradient descent
        cfg.optim_steps = 250 # optimization steps for each planner call
        cfg.use_initial_ik = False # Use IK to initialize optimization
        cfg.pre_terminate = True  # terminate early if costs are below thresholds
        if 'single' in cfg.method:
            cfg.single_shape_code = True
        if 'learned' in cfg.method:
            cfg.learnedgrasp_weights = args.ckpt
            cfg.goal_set_proj = False
        if 'known' in cfg.method: # Debug with known grasps
            # cfg.goal_set_proj = True
            cfg.remove_flip_grasp = False
    elif 'OMG' in cfg.method:       # OMG with apples to apples parameters
        cfg.goal_set_proj = True
        cfg.fixed_endpoint = False
        cfg.remove_flip_grasp = False
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
            if 'col' in cfg.method:
                cfg.disable_target_collision = False
            else:
                cfg.disable_target_collision = True
    else:
        raise NotImplementedError
        # cfg.root_dir = '/data/manifolds/acronym_mini_bookonly/meshes_bullet'
        # debug method using fixed grasp
        # cfg.ol_alg = 'Baseline'
        # cfg.goal_idx = -1

    # Command-line overrides
    dict_args = vars(args)
    for key in dict_args.keys():
        if key in cfg.keys() and dict_args[key] is not None:
            cfg[key] = dict_args[key]

    cfg.get_global_param()

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


# def randomly_place_object(cfg, env):
#     # place single object
#     T_w2b = get_world2bot_transform()

#     T_rand = get_random_transform(cfg)

#     # Apply object to centroid transform
#     T_ctr2obj = env.objinfos[0]['T_ctr2obj']

#     T_w2o = T_w2b @ T_rand @ T_ctr2obj
#     # draw_pose(T_w2b @ T_rand)
#     # print(T_w2o)
#     pq_w2o = pt.pq_from_transform(T_w2o)  # wxyz

#     p.resetBasePositionAndOrientation(
#         env._objectUids[0],
#         pq_w2o[:3],
#         pr.quaternion_xyzw_from_wxyz(pq_w2o[3:])
#     )
#     p.resetBaseVelocity(env._objectUids[0], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

#     if cfg.gravity:
#         for i in range(10000):
#             p.stepSimulation()


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


def init_video_writer(path, obj_name, scene_idx):
    return cv2.VideoWriter(
        f"{path}/{objname}_{scene_idx}.avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (640, 480),
    )


def init_dirs(out_dir, cfg, prefix=''):
    save_path = Path(f'{out_dir}/{prefix}{cfg.method}_{datetime.now().strftime("%y-%m-%d-%H-%M-%S")}')
    save_path.mkdir(parents=True)
    (save_path / 'info').mkdir()
    (save_path / 'videos').mkdir()
    (save_path / 'gifs').mkdir()
    with open(f'{save_path}/config.yaml', 'w') as yaml_file:
        yaml.dump(cfg, yaml_file)
    return str(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="which method to use", required=True)
        # , choices=['compOMG_known', 'origOMG_known', 'GF_learned_1hot', 'GF_learned_shape', 'GF_known'])
    parser.add_argument("--eval_type", help="which eval to run", required=True, choices=['1obj_float_fixedpose_nograv'])
    parser.add_argument("--ckpt", help="which weights to use for our method", default='/data/manifolds/fb_runs/multirun/pq-pq_mini/2022-05-10_221352/lossl1_lr0.0001/default_default/1_1/checkpoints/epoch=109-step=37605.ckpt')
    parser.add_argument("--dset_root", help="mesh root", type=str, default="/data/manifolds/acronym_mini")
    parser.add_argument("-o", "--out_dir", help="Directory to save experiment to", type=str, default="/data/manifolds/pybullet_eval/dbg")
    parser.add_argument("--render", dest='render', help="render gui", action="store_true", default=True)
    parser.add_argument("--no-render", dest='render', help="don't render gui", action="store_false")
    parser.add_argument("--write_video", help="write video", action="store_true")
    parser.add_argument("--prefix", help="prefix for variant name", default="")
    parser.add_argument("--pc", help="get point cloud with observation", action="store_true")
    parser.add_argument("--run_scenes", help="Run scenes", action="store_true")

    # cfg-specific command line args
    parser.add_argument("--smoothness_base_weight", type=float)
    parser.add_argument("--base_obstacle_weight", type=float)
    parser.add_argument("--base_grasp_weight", type=float)
    parser.add_argument("--base_step_size", type=float)
    parser.add_argument("--optim_steps", type=int)
    parser.add_argument("--goal_thresh", type=float)
    parser.add_argument("--use_min_cost_traj", type=int)
    parser.add_argument("--use_initial_ik", action='store_true')

    args = parser.parse_args()
    init_cfg(args)

    # Init environment
    env = PandaEnv(renders=args.render, gravity=cfg.gravity, cam_look=cfg.cam_look)

    # Init save dir and csv
    save_path = init_dirs(args.out_dir, cfg, prefix=args.prefix)
    with open(f'{save_path}/metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['object_name', 'scene_idx', 'execution', 'planning', 'smoothness', 'collision', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    # save args
    with open(f'{save_path}/args.yaml', 'w') as f:
        yaml.dump(vars(args), f)

    if args.run_scenes:
        init_joints = []
        with open('./data/init_joints_tspace.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                init_joints.append([float(x) for x in row])
        # init_joints = init_joints[:5]  # debug
    else:
        init_joints = [[0.0, -1.285, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.04, 0.04]]

    # Iterate over objects in folder
    for objname in os.listdir(f'{args.dset_root}/meshes_bullet'):
        scene = PlanningScene(cfg)
        for scene_idx in range(len(init_joints)):
            metrics = init_metrics_entry(objname, scene_idx)
            video_writer = init_video_writer(f"{save_path}/videos", objname, scene_idx) if args.write_video else None

            objinfo = get_object_info(env, objname, args.dset_root)
            env.reset(init_joints=init_joints[scene_idx], no_table=not cfg.table, objinfo=objinfo)
            place_object(env, cfg.tgt_pos, random=False, gravity=cfg.gravity)
            obs = env._get_observation(get_pc=args.pc, single_view=False)

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

            obj_prefix = f'{args.dset_root}/meshes_bullet'
            scene.reset_env(joints=init_joints[scene_idx])
            scene.env.add_object(objinfo['name'], trans, orn, obj_prefix=obj_prefix, abs_path=True)
            scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1, 0, 0, 0]))
            scene.env.combine_sdfs()
            if cfg.disable_target_collision:
                cfg.disable_collision_set = [objinfo['name']]

            # Set grasp selection method for planner
            scene.env.set_target(objinfo['name'])
            scene.reset(lazy=True)

            pc = obs['points'] if args.pc else None
            info = scene.step(pc=pc)
            plan = scene.planner.history_trajectories[-1]
            grasp_success = bullet_execute_plan(env, plan, args.write_video, video_writer)

            # Save data
            metrics['execution'] = grasp_success
            metrics['planning'] = info[-1]['execute']
            metrics['smoothness'] = info[-1]['smooth']
            metrics['collision'] = info[-1]['obs']
            metrics['time'] = info[-1]['time']
            with open(f'{save_path}/metrics.csv', 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(metrics)

            np.savez(f'{save_path}/info/{objname}_{scene_idx}', info=info, trajs=scene.planner.history_trajectories)

            # Convert avi to high quality gif 
            if args.write_video and info != []:
                # os.system(f'ffmpeg -y -i {save_path}/videos/{objname}_{scene_idx}.avi -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 {save_path}/gifs/{objname}_{scene_idx}.gif')
                subprocess.Popen(['ffmpeg', '-y', '-i', f'{save_path}/videos/{objname}_{scene_idx}.avi', '-vf', "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse", '-loop', '0', f'{save_path}/gifs/{objname}_{scene_idx}.gif'])

    # import IPython; IPython.embed()


