import matplotlib
matplotlib.use('TkAgg')
import os
import random
import pybullet as p
import numpy as np
from omg.config import cfg
from omg.core import PlanningScene

from omg_bullet.envs.acronym_env import PandaAcronymEnv
from omg_bullet.envs.ycb_env import PandaYCBEnv

from omg_bullet.utils import bullet_execute_plan

from datetime import datetime
import torch
import pytransform3d.rotations as pr
# import pytransform3d.transformations as pt
from pathlib import Path
import csv
import cv2
import subprocess
import time
import yaml
import hydra
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
# from moviepy.editor import ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips

from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
    # eval_objects = [x.split(':')[0] for x in cfg.grasp_weights] for ours

    selected_scenes = [
        'Bottle_244894af3ba967ccd957eaf7f4edb205_0.012953570261294404_3',
        'Bottle_244894af3ba967ccd957eaf7f4edb205_0.012953570261294404_6',
        'Bottle_244894af3ba967ccd957eaf7f4edb205_0.012953570261294404_7',
        'Bottle_244894af3ba967ccd957eaf7f4edb205_0.012953570261294404_28',
        'Bowl_9a52843cc89cd208362be90aaa182ec6_0.0008104428339208306_5',
        'Bowl_9a52843cc89cd208362be90aaa182ec6_0.0008104428339208306_9',
        'Bowl_9a52843cc89cd208362be90aaa182ec6_0.0008104428339208306_14',
        'Bowl_9a52843cc89cd208362be90aaa182ec6_0.0008104428339208306_3',
        'Mug_40f9a6cc6b2c3b3a78060a3a3a55e18f_0.0006670441940038386_2',
        'Mug_40f9a6cc6b2c3b3a78060a3a3a55e18f_0.0006670441940038386_15',
        'Mug_40f9a6cc6b2c3b3a78060a3a3a55e18f_0.0006670441940038386_25',
        'Mug_40f9a6cc6b2c3b3a78060a3a3a55e18f_0.0006670441940038386_1',
    ]

    if hydra_cfg.write_video:
        vid_path = Path(hydra_cfg.path) / f"plan_rerun_{datetime.now().strftime('%m-%d-%y_%H-%M-%S')}"
        os.mkdir(vid_path)
    for scene in scenes:
        # if scene['obj_name'].split('_')[0] not in eval_objects:
            # continue
        if 'Book' in scene['obj_name']:
            continue
        # if 'Bottle' not in scene['obj_name']:
        #     continue
        if f'{scene["obj_name"]}_{scene["idx"]}' not in selected_scenes:
            continue
        planning_scene = PlanningScene(cfg)
        obs, objname, scene_name = env.init_scene(scene, planning_scene, hydra_cfg)

        if cfg.pc:
            pc_dict = {
                'points_world': obs['points'],
                'points_cam2': obs['points_cam2'],
                'T_world_cam2': obs['T_world_cam2']
            }
        else:
            pc_dict = {}
        category = objname.split('_')[0] if cfg.per_class_models else 'All'

        # grab the trajectory from the saved run and execute it
        p.removeAllUserDebugItems()
        npz = np.load(Path(hydra_cfg.path) / 'info' / f'{objname}_{scene["idx"]}.npz', allow_pickle=True)
        # keys in npz: info, trajs

        # Visualize planning over time
        # Finding limits for y-axis     
        def get_ymax(ydata):
            ypbot = np.percentile(ydata, 5)
            yptop = np.percentile(ydata, 95)
            ypad = 0.2*(yptop - ypbot)
            # ymin = ypbot - ypad
            ymax = yptop + ypad
            return ymax

        plt.rcParams["axes.titlesize"] = 'x-large'          # controls default text sizes
        plt.rcParams["xtick.labelsize"] = 'large'          # controls default text sizes
        plt.rcParams["ytick.labelsize"] = 'x-large'          # controls default text sizes

        history_trajs = npz['trajs']
        info = npz['info']
        frames = []
        costs = []
        for i, traj in enumerate(history_trajs):
            env.update_panda_viz(torch.tensor(traj), k=6, skip=5)
            obs = env._get_observation(single_view=0)
            color = (0, 160, 0) if i == 500 else (0, 0, 0)
            frame = cv2.putText(obs['rgb'][0].copy(), f"iter: {i}", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA, False)

            coll = info[i]['weighted_obs']
            smth = info[i]['weighted_smooth']
            grsp = info[i]['weighted_grasp']
            costs.append([coll, smth, grsp])
            costs_np = np.array(costs)
            dpi = 100 
            fig = plt.figure(figsize=(640/dpi, 180/dpi), dpi=dpi)
            canvas = FigureCanvas(fig)

            ax1 = fig.add_subplot(131)
            ax1.plot(range(len(costs)), costs_np[:, 2], color='orange', label='Weighted Grasp')           
            ax1.set_title("Grasp")
            ax1.set_ylim(top=get_ymax(costs_np[:, 2]))
            ax2 = fig.add_subplot(132)
            ax2.plot(range(len(costs)), costs_np[:, 0], color='blue', linestyle='dashed', label='Weighted Obs')           
            ax2.set_ylim(top=max(get_ymax(costs_np[:, 0]), 0.01))
            ax2.set_title("Obstacle")
            ax3 = fig.add_subplot(133)
            ax3.plot(range(len(costs)), costs_np[:, 1], color='purple', linestyle='dotted', label='Weighted Smooth')           
            ax3.set_title('Smoothness')
            ax3.set_ylim(top=get_ymax(costs_np[:, 1]))

            plt.tight_layout()
            canvas.draw() 
            width, height = fig.get_size_inches() * fig.get_dpi()
            plot = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

            nframe = frame.copy()
            nframe[300:, :640] = plot
            frames.append(nframe)
            # sm_plot = cv2.resize(plot, (int(width // 3), int(height // 3)))

        clip = ImageSequenceClip(frames, fps=30)
        clip.write_videofile(str(vid_path / f"{objname}_{scene['idx']}.mp4"))

        # Get final trajectory and execute
        # plan = history_trajs[-1]
        # video_writer = init_video_writer(vid_path, objname, scene_name) if hydra_cfg.write_video else None
        # grasp_success = bullet_execute_plan(env, plan, hydra_cfg.write_video, video_writer)

    env.disconnect()

if __name__ == '__main__':
    main()
