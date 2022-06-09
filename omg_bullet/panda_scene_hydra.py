import os
# import argparse
import pybullet as p
import numpy as np
from omg.config import cfg
from omg.core import PlanningScene
from omg.util import tf_quat

from omg_bullet.panda_env import PandaEnv

# from acronym_tools import create_gripper_marker
# from manifold_grasping.utils import load_mesh
from omg_bullet.utils import get_world2bot_transform, draw_pose, bullet_execute_plan, get_object_info, place_object

import pytransform3d.rotations as pr
# import pytransform3d.transformations as pt
from pathlib import Path
# from datetime import datetime
import csv
import cv2
import subprocess
import yaml
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig


def merge_cfgs(hydra_cfg, cfg):
    """Two different configs are being used, the config from omg, and hydra
    Hydra handles command line overrides and also redirects the working directory
    Override cfg with hydra cfgs

    Args:
        hydra_cfg (_type_): _description_
        cfg (_type_): _description_
    """
    for key in hydra_cfg.eval.keys():
        if key in cfg.keys():
            val = hydra_cfg.eval[key]
            cfg[key] = val if type(val) != ListConfig else list(val)
    for key in hydra_cfg.variant.keys():
        if key in cfg.keys():
            val = hydra_cfg.variant[key]
            cfg[key] = val if type(val) != ListConfig else list(val)
    for key in hydra_cfg.keys():
        if key in cfg.keys():
            val = hydra_cfg[key]
            cfg[key] = val if type(val) != ListConfig else list(val)

    cfg.get_global_param()

def init_cfg(args):
    """
    Modify cfg based on command line arguments.
    """
    # cfg.render = args.render
    cfg.vis = False  
    cfg.window_width = 200
    cfg.window_height = 200

    cfg.eval_type = args.eval_type
    # cfg.vary_obj_pose = False if 'fixedpose' in cfg.eval_type else True
    cfg.gravity = False if 'nograv' in cfg.eval_type else True
    if '1obj_float' in cfg.eval_type:
        cfg.table = False
        cfg.cam_look = [-0.05, -0.5, -0.6852]
        cfg.tgt_pos = [0.5, 0.0, 0.5]
    else:
        raise NotImplementedError

    if not args.filter_collisions:
        cfg.filter_collision = False
    else:
        cfg.filter_collision = True

    cfg.method = args.method
    if 'GF' in cfg.method:
        # pq input -> pq output
        cfg.use_goal_grad = True
        cfg.fixed_endpoint = False
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
            cfg.goal_set_proj = False
            cfg.remove_flip_grasp = False
    elif 'OMG' in cfg.method:       # OMG with apples to apples parameters
        cfg.goal_set_proj = True
        cfg.remove_flip_grasp = False
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
            if 'col' in cfg.method:
                cfg.disable_target_collision = False
            else:
                cfg.disable_target_collision = True
    else:
        raise NotImplementedError

    # Command-line overrides
    dict_args = vars(args)
    for key in dict_args.keys():
        if key in cfg.keys() and dict_args[key] is not None:
            cfg[key] = dict_args[key]

    cfg.get_global_param()


def init_video_writer(path, obj_name, scene_idx):
    return cv2.VideoWriter(
        f"{path}/{obj_name}_{scene_idx}.avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (640, 480),
    )


def init_dir(hydra_cfg):
    cwd = Path(os.getcwd())
    (cwd / 'info').mkdir() 
    (cwd / 'videos').mkdir() 
    (cwd / 'gifs').mkdir() 
    with open(cwd / 'hydra_config.yaml', 'w') as yaml_file:
        OmegaConf.save(config=hydra_cfg, f=yaml_file.name)
    with open(cwd / 'config.yaml', 'w') as yaml_file:
        yaml.dump(cfg, yaml_file)
    with open(cwd / 'metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['object_name', 'scene_idx', 'execution', 'planning', 'smoothness', 'collision', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def get_scenes(hydra_cfg):
    scenes = []
    if hydra_cfg.run_scenes:
        if hydra_cfg.eval.obj_csv is not None:
            with open(Path(get_original_cwd()) / ".." / hydra_cfg.eval.obj_csv, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    scene = {
                        'joints': [0.0, -1.285, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
                        'obj_rot': pr.quaternion_wxyz_from_xyzw([float(x) for x in row[3:]])
                    }
                    scenes.append(scene)
        elif hydra_cfg.eval.joints_csv is not None:
            with open(Path(get_original_cwd()) / ".." / hydra_cfg.eval.joints_csv, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    scene = {
                        'joints': [float(x) for x in row],
                        'obj_rot': [0, 0, 0, 1]
                    }
                    scenes.append(scene)
    else:
        scenes.append({
            "joints": [0.0, -1.285, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
            "obj_rot": [0, 0, 0, 1]
        })
    return scenes


def save_metrics(objname, scene_idx, grasp_success, info):
    has_plan = info != []
    metrics = {
        'object_name': objname,
        'scene_idx': scene_idx,
        'execution': grasp_success,
        'planning': info[-1]['execute'] if has_plan else np.nan,
        'smoothness': info[-1]['smooth'] if has_plan else np.nan,
        'collision': info[-1]['obs'] if has_plan else np.nan,
        'time': info[-1]['time'] if has_plan else np.nan,
    }
    cwd = Path(os.getcwd())
    with open(cwd / 'metrics.csv', 'a', newline='') as csvfile:
        fieldnames = ['object_name', 'scene_idx', 'execution', 'planning', 'smoothness', 'collision', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(metrics)


def set_scene_env(scene, uid, objinfo, joints, hydra_cfg):
    # Scene has separate Env class which is used for planning
    # Add object to planning scene env
    trans_w2o, orn_w2o = p.getBasePositionAndOrientation(uid)  # xyzw

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

    obj_prefix = Path(hydra_cfg.data_root) / hydra_cfg.dataset / 'meshes_bullet'
    scene.reset_env(joints=joints)
    scene.env.add_object(objinfo['name'], trans, orn, obj_prefix=obj_prefix, abs_path=True)
    scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1, 0, 0, 0]))
    scene.env.combine_sdfs()
    if cfg.disable_target_collision:
        cfg.disable_collision_set = [objinfo['name']]

    # Set grasp selection method for planner
    scene.env.set_target(objinfo['name'])
    scene.reset(lazy=True)
    

@hydra.main(config_path=str(Path(os.path.dirname(__file__)) / '..' / 'config'), 
            config_name="panda_scene", version_base=None)
def main(hydra_cfg):
    merge_cfgs(hydra_cfg, cfg)
    init_dir(hydra_cfg)

    env = PandaEnv(renders=hydra_cfg.render, gravity=cfg.gravity, cam_look=cfg.cam_look)

    scenes = get_scenes(hydra_cfg)
    for objname in os.listdir(Path(hydra_cfg.data_root) / hydra_cfg.dataset / 'meshes_bullet'):
        scene = PlanningScene(cfg)
        for scene_idx in range(len(scenes)):
            objinfo = get_object_info(env, objname, Path(hydra_cfg.data_root) / hydra_cfg.dataset)
            env.reset(init_joints=scenes[scene_idx]['joints'], no_table=not cfg.table, objinfo=objinfo)
            place_object(env, cfg.tgt_pos, q=scenes[scene_idx]['obj_rot'], random=False, gravity=cfg.gravity)
            obs = env._get_observation(get_pc=cfg.pc, single_view=False)

            set_scene_env(scene, env._objectUids[0], objinfo, scenes[scene_idx]['joints'], hydra_cfg)
            pc = obs['points'] if cfg.pc else None
            info = scene.step(pc=pc)
            plan = scene.planner.history_trajectories[-1]

            video_writer = init_video_writer(Path(os.getcwd()) / 'videos', objname, scene_idx) if hydra_cfg.write_video else None
            grasp_success = bullet_execute_plan(env, plan, hydra_cfg.write_video, video_writer)

            save_metrics(objname, scene_idx, grasp_success, info)
            cwd = Path(os.getcwd())
            np.savez(cwd / 'info' / f'{objname}_{scene_idx}', info=info, trajs=scene.planner.history_trajectories)

            # Convert avi to high quality gif 
            if hydra_cfg.write_video and info != []:
                subprocess.Popen(['ffmpeg', '-y', '-i', cwd / 'videos' / f'{objname}_{scene_idx}.avi', '-vf', "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse", '-loop', '0', cwd / 'gifs' / f'{objname}_{scene_idx}.gif'])

if __name__ == '__main__':
    main()
