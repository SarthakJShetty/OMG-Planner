import os
import random
import pybullet as p
import numpy as np
from omg.config import cfg
from omg.core import PlanningScene

from omg_bullet.envs.acronym_env import PandaAcronymEnv
from omg_bullet.envs.ycb_env import PandaYCBEnv

from omg_bullet.utils import bullet_execute_plan

import torch
import pytransform3d.rotations as pr
# import pytransform3d.transformations as pt
from pathlib import Path
import csv
import cv2
import subprocess
import yaml
import hydra
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
# from moviepy.editor import ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips


def set_seeds():
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)


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
    # (cwd / 'trajs').mkdir() 
    with open(cwd / 'hydra_config.yaml', 'w') as yaml_file:
        OmegaConf.save(config=hydra_cfg, f=yaml_file.name)
    with open(cwd / 'config.yaml', 'w') as yaml_file:
        save_cfg = cfg.copy()
        save_cfg['ROBOT'] = None
        yaml.dump(save_cfg, yaml_file) 
    with open(cwd / 'metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['object_name', 'scene_idx', 'execution', 'planning', 'smoothness', 'collision', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


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


def init_env(hydra_cfg):
    if 'acronym' in cfg.eval_env:
        env = PandaAcronymEnv(renders=hydra_cfg.render, gravity=cfg.gravity, cam_look=cfg.cam_look)
    elif 'ycb' in cfg.eval_env:
        env = PandaYCBEnv(renders=hydra_cfg.render, gravity=cfg.gravity)
    else:
        raise NotImplementedError
    return env


@hydra.main(config_path=str(Path(os.path.dirname(__file__)) / '..' / 'config'), 
            config_name="panda_scene", version_base=None)
def main(hydra_cfg):
    set_seeds()
    merge_cfgs(hydra_cfg, cfg)
    init_dir(hydra_cfg)
    env = init_env(hydra_cfg)
    scenes = env.get_scenes(hydra_cfg)
    for scene in scenes:
        if 'Book' in scene['obj_name']:
            continue
        planning_scene = PlanningScene(cfg)
        obs, objname, scene_name = env.init_scene(scene, planning_scene, hydra_cfg)
        # obs, objname, scene_name = env.init_scene(scene, planning_scene, hydra_cfg, single_view=True)

        if cfg.pc:
            pc_dict = {
                'points_world': obs['points'],
                'points_cam2': obs['points_cam2'],
                # 'points_segments': obs['points_segments'],
                'T_world_cam2': obs['T_world_cam2']
            }
        else:
            pc_dict = {}
        category = objname.split('_')[0] if cfg.per_class_models else 'All'
        info = planning_scene.step(category=category, pc_dict=pc_dict, viz_env=env)
        plan = planning_scene.planner.history_trajectories[-1]

        video_writer = init_video_writer(Path(os.getcwd()) / 'videos', objname, scene_name) if hydra_cfg.write_video else None
        grasp_success = bullet_execute_plan(env, plan, hydra_cfg.write_video, video_writer)

        save_metrics(objname, scene_name, grasp_success, info)
        cwd = Path(os.getcwd())
        np.savez(cwd / 'info' / f'{objname}_{scene_name}', info=info, trajs=planning_scene.planner.history_trajectories)

        # Convert avi to high quality gif 
        if hydra_cfg.write_video and info != []:
            subprocess.Popen(['ffmpeg', '-y', '-i', cwd / 'videos' / f'{objname}_{scene_name}.avi', '-vf', "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse", '-loop', '0', cwd / 'gifs' / f'{objname}_{scene_name}.gif'])

        # if not cfg.render:
        #     cfg.window_height = 200
        #     cfg.window_width = 200
        #     planning_scene.setup_renderer()
        #     # planning_scene.cfg.output_video_name = f'{cwd}/trajs/{objname}_{scene_name}.avi'

        #     import time
        #     fps = 24
        #     duration = 0.2
        #     comps = []
        #     for i, traj in enumerate(planning_scene.planner.history_trajectories):
        #         start = time.time()
        #         traj_im = planning_scene.fast_debug_vis(traj=traj, interact=0, write_video=False,
        #                                         nonstop=False, collision_pt=False, goal_set=False, traj_idx=i)
        #         print(f"fast debug vis: {time.time() - start}")
        #         traj_im = cv2.cvtColor(traj_im, cv2.COLOR_RGB2BGR)
        #         start = time.time()
        #         clip = ImageClip(traj_im).set_duration(duration).set_fps(fps)
        #         # https://github.com/Zulko/moviepy/issues/401
        #         txt_clip = (TextClip(f"iter {i}", fontsize=50, color='black')
        #             .set_position('bottom')
        #             .set_duration(duration))
        #         comp = CompositeVideoClip([clip, txt_clip]).set_fps(fps).set_duration(duration)
        #         print(f"video clip: {time.time() - start}")
        #         comps.append(comp)
        #         # cv2.imwrite(f"{args.output_dir}/{exp_name}/{scene_file}/traj_{i+1}.png", traj_im)
        #     for _ in range(3): # add more frames to the end
        #         txt_clip = (TextClip(f"iter {i}*", fontsize=50, color='black')
        #             .set_position('bottom')
        #             .set_duration(duration))
        #         comp = CompositeVideoClip([clip, txt_clip]).set_fps(fps).set_duration(duration)
        #         comps.append(comp)
        #     result = concatenate_videoclips(comps)
        #     result.write_gif(f"{cwd}/trajs/{objname}_{scene_name}.gif")
        #     # result.write_gif(f"{args.output_dir}/{exp_name}/{scene_file}/traj.gif")
        #     result.close()

    env.disconnect()

if __name__ == '__main__':
    main()
