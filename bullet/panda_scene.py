# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import random
import yaml
import os
# from gym import spaces
import time
import sys
import argparse
# from . import _init_paths

from omg.core import *
# from omg.util import *
from omg.config import cfg
import numpy as np
import pybullet_data

from PIL import Image
import glob
# import gym
# from panda_gripper import Panda

# import scipy.io as sio
# import pkgutil
from copy import deepcopy

# For backprojection
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from panda_ycb_env import PandaYCBEnv

from utils import *

from acronym_tools import load_mesh
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

def bullet_execute_plan(env, plan, write_video, video_writer):
    print('executing...')
    for k in range(plan.shape[0]):
        obs, rew, done, _ = env.step(plan[k].tolist())
        if write_video:
            video_writer.write(obs['rgb'][:, :, [2, 1, 0]].astype(np.uint8))
    (rew, ret_obs) = env.retract(record=True)
    if write_video: 
        for robs in ret_obs:
            video_writer.write(robs['rgb'][:, :, [2, 1, 0]].astype(np.uint8))
            video_writer.write(robs['rgb'][:, :, [2, 1, 0]].astype(np.uint8)) # to get enough frames to save
    return rew

class KnownGraspInference:
    def __init__(self, obj2grasp_Ts):
        self.obj2grasp_Ts = obj2grasp_Ts

    def inference(self, x):
        """
        x: xyz, xyzw
        """
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-gi", "--grasp_inference", help="which grasp inference method to use", required=True, choices=['acronym', 'contact_graspnet', 'ours_outputpose', 'ours_knowngrasps'])
    parser.add_argument("-gs", "--grasp_selection", help="Which grasp selection algorithm to use", required=True, choices=['Fixed', 'Proj', 'OMG'])
    parser.add_argument("-o", "--output_dir", help="Output directory", type=str, default="./output_videos") 
    parser.add_argument("--traj_init", help="how to init the trajectory", choices=["fixed", "start"], default="fixed")
    parser.add_argument("-s", "--scenes", help="scene(s) to run. If set to '', loops through all 100 scenes", nargs='*', default=["scene_1"])
    parser.add_argument("-r", "--render", help="render", action="store_true")
    parser.add_argument("-v", "--visualize", help="visualize grasps", action="store_true")
    parser.add_argument("--egl", help="use egl render", action="store_true")
    parser.add_argument("-w", "--write_video", help="write video", action="store_true")
    parser.add_argument("-a", "--acronym_dir", help="acronym dataset directory", type=str, default='')
    parser.add_argument("-d", "--debug_exp", help="Override default experiment name with dbg", action="store_true"),  
    parser.add_argument("--debug_traj", help="Visualize intermediate trajectories", action="store_true")
    parser.add_argument("--init_traj_end_at_start", help="Initialize trajectory so end is at start", action="store_true")
    args = parser.parse_args()

    # Set up bullet env
    mkdir_if_missing(args.output_dir)

    # Set up planning scene
    cfg.traj_init = "grasp"
    cfg.vis = False
    cfg.timesteps = 50
    # cfg.timesteps = 5
    cfg.optim_steps = 50
    cfg.extra_smooth_steps = 0
    cfg.get_global_param(cfg.timesteps)
    cfg.scene_file = ''
    cfg.vis = args.debug_traj
    cfg.window_width = 240
    cfg.window_height = 240
    cfg.goal_set_proj = False if args.init_traj_end_at_start else True
    cfg.use_standoff = False if args.init_traj_end_at_start else True
    scene = PlanningScene(cfg)
    scene.reset()

    # Set up env
    if args.scenes == ['acronym_book']:
        env = PandaYCBEnv(renders=args.render, egl_render=args.egl, gravity=False, root_dir=f'{args.acronym_dir}/meshes_omg')
        grasp_root = f"{args.acronym_dir}/grasps"
        objects = ['Book_5e90bf1bb411069c115aef9ae267d6b7_0.0268818133810836']
        grasp_paths = [] # path to grasp file for a given object
        for fn in os.listdir(grasp_root):
            for objname in objects:
                if objname in fn:
                    grasp_paths.append((fn, objname))
                    objects.remove(objname) # Only load first scale of object for now

        # Load acronym objects
        object_infos = []
        for grasp_path, objname in grasp_paths:
            obj_mesh, obj_scale = load_mesh(f"{grasp_root}/{grasp_path}", mesh_root_dir=args.acronym_dir, ret_scale=True)
            object_infos.append((objname, obj_scale, obj_mesh))
        env.reset(object_infos=object_infos)

        # Move object frame to centroid
        obj_mesh = object_infos[0][2]
        obj2ctr_T = np.eye(4)
        # obj2ctr_T[:3, 3] = -obj_mesh.centroid
        obj2ctr_T[:3, 3] = obj_mesh.centroid
    else:
        env = PandaYCBEnv(renders=args.render, egl_render=args.egl, gravity=True)
        env.reset()

    # set initial trajectory start and end positions to be the same
    if args.grasp_inference == "ours_knowngrasps" or args.init_traj_end_at_start:
        scene.traj.end = scene.traj.start
        scene.traj.interpolate_waypoints()

    # Add objects to scene
    for i, name in enumerate(env.obj_path[:-2]):  # load all objects
        name = name.split("/")[-1]
        trans, orn = env.cache_object_poses[i]
        scene.env.add_object(name, trans, tf_quat(orn), compute_grasp=True)
    scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1, 0, 0, 0]))
    scene.env.add_table(np.array([0.55, 0, -0.17]), np.array([0.707, 0.707, 0.0, 0]))
    scene.env.combine_sdfs()

    # Set up save folders
    if args.debug_exp:
        exp_name = 'dbg'
    else:
        exp_name = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}" + \
                    f"_{args.grasp_inference}" + \
                    f"_{args.grasp_selection}"
    mkdir_if_missing(f'{args.output_dir}/{exp_name}')
    with open('args.yml', 'w') as f:
        yaml.dump(args, f)
    # TODO save cfg to exp folder
    
    # Set up grasp inference method
    if args.grasp_inference == 'contact_graspnet':
        from contact_graspnet_infer import *
        grasp_inf_method = ContactGraspNetInference(args.visualize)
    elif args.grasp_inference == 'ours_outputpose':
        from implicit_infer import ImplicitGraspInference
        grasp_inf_method = ImplicitGraspInference()
    elif args.grasp_inference == 'ours_knowngrasps':
        from acronym_tools import load_grasps
        rotgrasp2grasp_T = pt.transform_from(pr.matrix_from_axis_angle([0, 0, 1, -np.pi/2]), [0, 0, 0])
        obj2rotgrasp_Ts, success = load_grasps(f"/data/manifolds/acronym/grasps/Book_5e90bf1bb411069c115aef9ae267d6b7_0.0268818133810836.h5")
        obj2grasp_Ts = obj2rotgrasp_Ts @ rotgrasp2grasp_T
        grasp_inf_method = KnownGraspInference(obj2grasp_Ts)
    else:
        grasp_inf_method = None

    cfg.scene_files = args.scenes
    if cfg.scene_files == []:
        scene_files = ['scene_{}'.format(i) for i in range(100)] # TODO change to listdir
    else:
        scene_files = cfg.scene_files
    cnts, rews = 0, 0
    for scene_file in scene_files:
        print(scene_file)
        mkdir_if_missing(f'{args.output_dir}/{exp_name}/{scene_file}')
        config.cfg.output_video_name = f"{args.output_dir}/{exp_name}/{scene_file}/bullet.avi"
        cfg.scene_file = scene_file
        video_writer = None
        if args.write_video:
            video_writer = cv2.VideoWriter(
                config.cfg.output_video_name,
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (640, 480),
            )
        full_name = os.path.join('data/scenes', scene_file + ".mat")
        env.cache_reset(scene_file=full_name)
        obj_names, obj_poses = env.get_env_info()
        object_lists = [name.split("/")[-1].strip() for name in obj_names]
        object_poses = [pack_pose(pose) for pose in obj_poses]

        if args.scenes == ['acronym_book']:
            # Debug: Manually set book object
            # TODO create scene file
            # TODO make minimum reproducible
            # pos = (0.07345162518699465, -0.4098033797439253, -1.1014019481737773)
            pos = (0.07345162518699465, -0.4098033797439253, -1.10)
            p.resetBasePositionAndOrientation(
                env._objectUids[env.target_idx],
                pos, 
                [0, 0, 0, 1] 
            )
            p.resetBaseVelocity(
                env._objectUids[env.target_idx], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
            )
            
            bot_pos, bot_orn = p.getBasePositionAndOrientation(env._panda.pandaUid)
            mat = np.asarray(p.getMatrixFromQuaternion(bot_orn)).reshape(3, 3)
            T_world2bot = np.eye(4)
            T_world2bot[:3, :3] = mat
            T_world2bot[:3, 3] = bot_pos
            for i, name in enumerate(object_lists):
                if 'Book' in name: 
                    book_worldpose = list(pos) + [1, 0, 0, 0]
                    object_poses[i] = pack_pose(np.linalg.inv(T_world2bot) @ unpack_pose(book_worldpose))
                    break

        # Update planning scene
        exists_ids, placed_poses = [], []
        for i, name in enumerate(object_lists[:-2]):   
            scene.env.update_pose(name, object_poses[i])
            if args.scenes == ['acronym_book']:
                obj_idx = env.obj_path[:-2].index(name)
            else:
                obj_idx = env.obj_path[:-2].index("data/objects/" + name)
            exists_ids.append(obj_idx)
            trans, orn = env.cache_object_poses[obj_idx]
            placed_poses.append(np.hstack([trans, ros_quat(orn)]))

        cfg.disable_collision_set = [
            name.split("/")[-2]
            for obj_idx, name in enumerate(env.obj_path[:-2])
            if obj_idx not in exists_ids
        ]

        obs = env._get_observation(get_pc=True if args.init_traj_end_at_start else False) # TODO change flag to whether pc obs is used
        if args.grasp_inference == 'acronym':
            grasps, grasp_scores = None, None
            inference_duration = 0
        elif args.grasp_inference == 'ours_knowngrasps':
            grasps, grasp_scores = None, None
            inference_duration = 0 # TODO
        elif args.grasp_inference == 'ours_outputpose':
            grasps, grasp_scores = None, None
            inference_duration = 0 # TODO
        elif args.grasp_inference == 'contact_graspnet':
            start_time = time.time()
            idx = env._objectUids[env.target_idx]
            segmask = deepcopy(obs['mask'])
            segmask[segmask != idx] = 0
            segmask[segmask == idx] = 1
            x = {'rgb': obs['rgb'], 'depth': obs['depth'], 'K': env._intr_matrix, 'seg': segmask}
            Ts_cam2grasp, grasp_scores, contact_pts = grasp_inf_method.inference(x)

            # Get transform from world to camera
            T_world2cam = get_world2cam_transform(env)
            # draw_pose(T_world2cam)

            Ts_world2grasp = T_world2cam @ Ts_cam2grasp
            T_world2bot = get_world2bot_transform(env)
            grasps = np.linalg.inv(T_world2bot) @ Ts_world2grasp
            inference_duration = time.time() - start_time
            print(f"inf duration: {inference_duration}")

        # Set grasp selection method for planner
        if args.grasp_selection == 'Fixed':
            scene.planner.cfg.ol_alg = 'Baseline'
            scene.planner.cfg.goal_idx = -1
        elif args.grasp_selection == 'Proj':
            scene.planner.cfg.ol_alg = 'Proj'
        elif args.grasp_selection == 'OMG':
            scene.planner.cfg.ol_alg = 'MD'
        scene.env.set_target(env.obj_path[env.target_idx].split("/")[-1])
        scene.reset(lazy=True, grasps=grasps, grasp_scores=grasp_scores, implicit_model=grasp_inf_method, init_traj_end_at_start=args.init_traj_end_at_start)
        
        if args.init_traj_end_at_start and args.scenes == ["acronym_book"]:
            pc_cam = obs['points']
            T_world2cam = get_world2cam_transform(env)
            T_cam2obj = np.linalg.inv(T_world2cam) @ unpack_pose(book_worldpose)
            pc_obj = (T_cam2obj @ pc_cam.T).T 
            T_world2obj = T_world2cam @ T_cam2obj
        else:
            pc_cam = obs['points']
            T_world2cam = get_world2cam_transform(env)
            T_world2bot = get_world2bot_transform(env)
            T_cam2obj = np.linalg.inv(T_world2cam) @ T_world2bot @ unpack_pose(object_poses[env.target_idx])
            pc_obj = (T_cam2obj @ pc_cam.T).T
            T_bot2obj = np.linalg.inv(T_world2bot) @ T_world2cam @ T_cam2obj
            # draw_pose(T_bot2obj)

        info = scene.step(pc=pc_obj, T_bot2obj=T_bot2obj)
        plan = scene.planner.history_trajectories[-1]

        # Visualize intermediate trajectories
        if args.debug_traj:
            from moviepy.editor import ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips
            fps = 24
            duration = 0.2
            comps = []
            for i, traj in enumerate(scene.planner.history_trajectories):
                start = time.time()
                traj_im = scene.fast_debug_vis(traj=traj, interact=0, write_video=False,
                                               nonstop=False, collision_pt=False, goal_set=False, traj_idx=i)
                print(f"fast debug vis: {time.time() - start}")
                traj_im = cv2.cvtColor(traj_im, cv2.COLOR_RGB2BGR)
                start = time.time()
                clip = ImageClip(traj_im).set_duration(duration).set_fps(fps)
                txt_clip = (TextClip(f"iter {i}", fontsize=50, color='black')
                    .set_position('bottom')
                    .set_duration(duration))
                comp = CompositeVideoClip([clip, txt_clip]).set_fps(fps).set_duration(duration)
                print(f"video clip: {time.time() - start}")
                comps.append(comp)
                cv2.imwrite(f"{args.output_dir}/{exp_name}/{scene_file}/traj_{i+1}.png", traj_im)
            for _ in range(3): # add more frames to the end
                txt_clip = (TextClip(f"iter {i}*", fontsize=50, color='black')
                    .set_position('bottom')
                    .set_duration(duration))
                comp = CompositeVideoClip([clip, txt_clip]).set_fps(fps).set_duration(duration)
                comps.append(comp)
            result = concatenate_videoclips(comps)
            result.write_gif(f"{args.output_dir}/{exp_name}/{scene_file}/traj.gif")
            result.close()

        if info != []:
            rew = bullet_execute_plan(env, plan, args.write_video, video_writer)
        else:
            rew = 0

        for i, name in enumerate(object_lists[:-2]):  # reset planner
            scene.env.update_pose(name, placed_poses[i])
        cnts += 1
        rews += rew
        print('rewards: {} counts: {}'.format(rews, cnts))

        # Save data
        data = [rew, info, plan, inference_duration]
        np.save(f'{args.output_dir}/{exp_name}/{scene_file}/data.npy', data)

        # Convert avi to high quality gif 
        if args.write_video and info != []:
            os.system(f'ffmpeg -y -i {args.output_dir}/{exp_name}/{scene_file}/bullet.avi -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 {args.output_dir}/{exp_name}/{scene_file}/scene.gif')
    env.disconnect()

            # start_time = time.time()

            # pos, orn = p.getBasePositionAndOrientation(env._objectUids[env.target_idx])
            # world2obj_T = np.eye(4)
            # world2obj_T[:3, 3] = pos
            # world2obj_T[:3, :3] = np.asarray(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            # world2ctr_T = world2obj_T @ obj2ctr_T
            # # draw_pose(world2ctr_T)

            # # Get end effector pose frame
            # pos, orn = p.getLinkState(env._panda.pandaUid, env._panda.pandaEndEffectorIndex)[:2]
            # world2ee_T = pt.transform_from_pq(np.concatenate([pos, pr.quaternion_wxyz_from_xyzw(orn)]))
            # # draw_pose(world2ee_T)

            # # Get end effector in centroid frame, do inference
            # ctr2ee_T = np.linalg.inv(world2ctr_T) @ world2ee_T
            # ee_pos = ctr2ee_T[:3, 3]
            # ee_orn = pr.quaternion_from_matrix(ctr2ee_T[:3, :3]) # wxyz
            # ee_pose = np.concatenate([ee_pos, pr.quaternion_xyzw_from_wxyz(ee_orn)])
            # ee_pose = torch.tensor(ee_pose, dtype=torch.float32, device='cuda').unsqueeze(0)

            # fwd_time = time.time()
            # out_pose = grasp_inf_method.inference(ee_pose)
            # fwd_duration = time.time() - fwd_time
            # print(f"fwd pass: {fwd_duration}")
            # out_pose = out_pose.cpu().numpy()

            # # Convert output into world frame
            # out_pos = out_pose[:3]
            # out_orn = out_pose[3:] # xyzw
            # ctr2out_T = pt.transform_from_pq(np.concatenate([out_pos, pr.quaternion_wxyz_from_xyzw(out_orn)]))
            # world2out_T = world2ctr_T @ ctr2out_T
            # # draw_pose(world2out_T)

            # # TODO consolidate
            # pos, orn = p.getBasePositionAndOrientation(env._panda.pandaUid)
            # mat = np.asarray(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            # T_world2bot = np.eye(4)
            # T_world2bot[:3, :3] = mat
            # T_world2bot[:3, 3] = pos
            # grasps = np.linalg.inv(T_world2bot) @ world2out_T[np.newaxis, :]
            # grasp_scores = np.array([1.0])

            # # TODO integrate into planner. 

            # inference_duration = time.time() - start_time
            # print(f"inf pass: {inference_duration}")