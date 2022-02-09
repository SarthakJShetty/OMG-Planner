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

import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from panda_ycb_env import PandaYCBEnv

from utils import *

from acronym_tools import load_mesh
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from moviepy.editor import ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips
# import multiprocessing as m

import trimesh
from acronym_tools import *

from post_process.trimesh_viz import pkls_to_gif

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

def init_cfg(args):
    """
    Modify cfg based on command line arguments.
    """
    if args.render and args.save_all_trajectories:
        print("Error: render and save all trajectories flags both on, save trajectories will fail")
    if args.render and cfg.vis:
        print("Warning: setting vis to False because of render")
        cfg.vis = False

    cfg.method = args.method
    cfg.window_width = 200
    cfg.window_height = 200
    cfg.exp_dir = args.exp_dir
    mkdir_if_missing(cfg.exp_dir)
    if args.scene_files != []:
        cfg.scene_file = args.scene_files[0]
    cfg.use_goal_grad = args.use_goal_grad
    cfg.acronym_dir = args.acronym_dir
    cfg.use_standoff = args.use_standoff

    if 'knowngrasps' in args.method: 
        if 'Fixed' in args.method:
            cfg.goal_set_proj = False
            cfg.ol_alg = 'Baseline'
            cfg.goal_idx = -1
        elif 'Proj' in args.method:
            cfg.ol_alg = 'Proj'
        elif 'OMG' in args.method:
            cfg.ol_alg = 'MD'
    elif 'implicitgrasps' in args.method:
        cfg.use_standoff = False
        cfg.ol_alg = None
        if 'outputposeIK' in args.method:
            cfg.use_ik = True
            cfg.grasp_prediction_weights = '/checkpoint/thomasweng/grasp_manifolds/runs/outputs/nopc_book_logmap_10kfree/2021-11-23_232102_dsetacronym_Book_train0.9_val0.1_free10.0k_sym_distlogmap/default_default/0_0/checkpoints/last.ckpt'
        elif 'outputposegrad' in args.method:
            cfg.use_ik = False
            cfg.grasp_prediction_weights = '/checkpoint/thomasweng/grasp_manifolds/runs/outputs/nopc_book_logmap_10kfree/2021-11-23_232102_dsetacronym_Book_train0.9_val0.1_free10.0k_sym_distlogmap/default_default/0_0/checkpoints/last.ckpt'
        else:
            raise NotImplementedError

        if 'fixed' in args.method: # debug method using fixed grasp
            cfg.ol_alg = 'Baseline'
            cfg.goal_idx = -1

    cfg.get_global_param()

def init_dirs():
    """
    Create output directory and save cfg in directory
    """
    cfg.exp_name = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}" + \
                f"_{cfg.method}"
    mkdir_if_missing(f'{cfg.exp_dir}/{cfg.exp_name}')
    with open(f'{cfg.exp_dir}/{cfg.exp_name}/args.yml', 'w') as f:
        yaml.dump(cfg, f)

def init_video_writer():
    return cv2.VideoWriter(
        f"{cfg.exp_dir}/{cfg.exp_name}/{cfg.scene_file}/bullet.avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (640, 480),
    )

def save_all_trajectories(scene):
    """
    Save history of trajectories over the course of the optimization loop.
    """
    fps = 24
    duration = 0.2
    comps = []

    # comps = [None for i in range(len(scene.planner.history_trajectories))]
    # with m.Pool(4) as pool:
    #     margs = [(scene.fast_debug_vis, traj, i, comps, exp_name, duration, fps) 
    #              for i, traj in enumerate(scene.planner.history_trajectories)]
    #     pool.map(get_clip, margs)

    if len(scene.planner.history_trajectories) < 10:
        trajs = scene.planner.history_trajectories
        ids = range(len(trajs))
    else:
        skip = 1
        trajs = scene.planner.history_trajectories[::skip]
        ids = range(0, len(trajs), skip)

    if not os.path.exists(f"{cfg.exp_dir}/{cfg.exp_name}/{cfg.scene_file}/traj_pngs"):
        os.mkdir(f"{cfg.exp_dir}/{cfg.exp_name}/{cfg.scene_file}/traj_pngs")

    # trajs = scene.planner.history_trajectories if len(scene.planner.history_trajectories) < 10 else scene.planner.history_trajectories[::skip]
    for i, traj in zip(ids, trajs):
        print(i)
        traj_im = scene.fast_debug_vis(traj=traj, interact=0, write_video=False,
                                        nonstop=False, collision_pt=False, goal_set=False, traj_idx=i)
        clip = ImageClip(traj_im).set_duration(duration).set_fps(fps)
        # TextClip not working outside of vscode debugger for some reason
        txt_clip = (TextClip(f"iter {i}", fontsize=20, color='black')
            .set_position('bottom')
            .set_duration(duration))
        # txt_clip = (TextClip(f"iter", fontsize=20, color='black').set_position('bottom').set_duration(duration))
        comp = CompositeVideoClip([clip, txt_clip]).set_fps(fps).set_duration(duration)
        # comp = CompositeVideoClip([clip]).set_fps(fps).set_duration(duration)
        comps.append(comp)
        cv2.imwrite(f"{cfg.exp_dir}/{cfg.exp_name}/{cfg.scene_file}/traj_pngs/traj_{i+1}.png", cv2.cvtColor(traj_im, cv2.COLOR_RGB2BGR))

    for _ in range(3): # add more frames to the end
        # txt_clip = (TextClip(f"iter {i}*", fontsize=50, color='black')
        #     .set_position('bottom')
        #     .set_duration(duration))
        # comp = CompositeVideoClip([clip, txt_clip]).set_fps(fps).set_duration(duration)
        comp = CompositeVideoClip([clip]).set_fps(fps).set_duration(duration)
        comps.append(comp)
    result = concatenate_videoclips(comps)
    result.write_gif(f"{cfg.exp_dir}/{cfg.exp_name}/{cfg.scene_file}/traj.gif")
    result.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="which method to use", required=True), 
        # choices=['knowngrasps_Fixed', 'knowngrasps_Proj', 'knowngrasps_OMG', 'implicitgrasps_outputposegrad', 'implicitgrasps_outputposeIK'])
    parser.add_argument("--exp_dir", help="Output directory", type=str, default="./output_videos") 
    parser.add_argument("--scene_files", help="Which scene to run", nargs="+")
    parser.add_argument("--experiment", help="run all scenes", action="store_true")
    parser.add_argument("--render", help="render", action="store_true")
    # parser.add_argument("--vis", help="visualize usig YCBRenderer, not compatible with render flag", action="store_true")
    # parser.add_argument("--egl", help="use egl render", action="store_true")
    parser.add_argument("--acronym_dir", help="acronym dataset directory", type=str, default='')
    parser.add_argument("--use_standoff", help="use standoff", action="store_true")
    parser.add_argument("--write_video", help="write video", action="store_true")
    parser.add_argument("--save_all_trajectories", help="Save intermediate trajectories", action="store_true")
    parser.add_argument("--use_goal_grad", help="Use goal gradient with implicit method", action="store_true")
    args = parser.parse_args()

    init_cfg(args)
    init_dirs()

    scene = PlanningScene(cfg)

    if args.experiment:   
        scene_files = ['scene_{}'.format(i) for i in range(100)]
    else:
        scene_files = args.scene_files

    # Set up env
    # root_dir = f'{args.acronym_dir}/meshes_omg' if cfg.scene_file == 'acronym_book' else None
    gravity = 'acronym_book' not in scene_files[0]
    # env = PandaYCBEnv(renders=args.render, egl_render=True, root_dir=root_dir, gravity=gravity)
    env = PandaYCBEnv(renders=args.render, gravity=gravity)
    env.reset() 

    # Add objects to scene
    # Scene has separate Env class which is used for planning
    for i, name in enumerate(env.obj_path[:-2]):  # load all objects
        name = name.split("/")[-1]
        trans, orn = env.cache_object_poses[i]
        scene.env.add_object(name, trans, tf_quat(orn), compute_grasp=True)
    scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1, 0, 0, 0]))
    scene.env.add_table(np.array([0.55, 0, -0.17]), np.array([0.707, 0.707, 0.0, 0]))
    scene.env.combine_sdfs()

    cnts, rews = 0, 0
    for scene_idx, scene_file in enumerate(scene_files):
        print(scene_file)
        cfg.scene_file = scene_file
        mkdir_if_missing(f'{cfg.exp_dir}/{cfg.exp_name}/{cfg.scene_file}')
        video_writer = init_video_writer() if args.write_video else None

        ext = ".mat" if 'acronym_book' not in scene_file else ".npy"
        full_name = os.path.join('data/scenes', scene_file + ext)
        env.cache_reset(scene_file=full_name)
        obj_names, obj_poses = env.get_env_info()
        object_lists = [name.split("/")[-1].strip() for name in obj_names]
        object_poses = [pack_pose(pose) for pose in obj_poses]

        # Update planning scene
        exists_ids, placed_poses = [], []
        for i, name in enumerate(object_lists[:-2]):   
            scene.env.update_pose(name, object_poses[i])
            path = "data/objects/" + name
            obj_idx = env.obj_path[:-2].index(path) 
            exists_ids.append(obj_idx)
            # trans, orn = env.cache_object_poses[obj_idx]
            trans, orn = p.getBasePositionAndOrientation(env._objectUids[env.target_idx]) # x y z w
            # placed_poses.append(np.hstack([trans, ros_quat(orn)]))
            placed_poses.append(np.hstack([trans, orn])) # TODO what about not acronym book?

        cfg.disable_collision_set = [
            name.split("/")[-2]
            for obj_idx, name in enumerate(env.obj_path[:-2])
            if obj_idx not in exists_ids
        ]

        if 'acronym_book' in scene_file:
            cfg.disable_collision_set += ['Book_5e90bf1bb411069c115aef9ae267d6b7']

        # obs = env._get_observation(get_pc=True if cfg.method == 'implicitgrasps' else False)

        # Set grasp selection method for planner
        scene.env.set_target(env.obj_path[env.target_idx].split("/")[-1])
        scene.reset(lazy=True)
        
        # acronym book pose
        draw_pose(get_world2bot_transform(env) @ unpack_pose(object_poses[0]))
        draw_pose(get_world2bot_transform(env) @ unpack_pose(object_poses[0]) @ cfg.T_obj2ctr)

        info = scene.step()
        plan = scene.planner.history_trajectories[-1]

        # grasp pose
        end_joints = wrap_value(plan[-1]) # rad2deg
        end_poses = cfg.ROBOT.forward_kinematics_parallel(
            joint_values=end_joints[np.newaxis, :], base_link=cfg.base_link)[0]
        T_bot2ee = end_poses[-3]
        draw_pose(get_world2bot_transform(env) @ T_bot2ee)

        # Visualize transforms
        if False:
            for i in range(len(info)):
                T_bot2ee, T_bot2objfrm, T_objfrm2obj, T_obj2goal, T_obj2ee = info[i]["transforms"]

                grasp_root = f"{cfg.acronym_dir}/grasps"
                obj_name = 'Book_5e90bf1bb411069c115aef9ae267d6b7_0.0268818133810836'
                obj_mesh = load_mesh(f"{grasp_root}/{obj_name}.h5", mesh_root_dir=cfg.acronym_dir)

                T_obj2ctr = np.eye(4)
                T_obj2ctr[:3, 3] = obj_mesh.centroid
                obj_mesh = obj_mesh.apply_transform(T_obj2ctr)

                ee_pose = [create_gripper_marker(color=[0, 0, 255]).apply_transform(T_obj2ee)]
                goal_pose = [create_gripper_marker(color=[255, 0, 0]).apply_transform(T_obj2goal)]

                # # create visual markers for grasps
                # successful_grasps = [
                #     create_gripper_marker(color=[0, 255, 0]).apply_transform(t)
                #     for t in T[np.random.choice(np.where(success == 1)[0], np.min([np.sum(success), 100]))]
                # ]
                # failed_grasps = [
                #     create_gripper_marker(color=[255, 0, 0]).apply_transform(t)
                #     for t in T[np.random.choice(np.where(success == 0)[0], np.min([np.sum(success == 0), 100]))]
                # ]

                trimesh.Scene([obj_mesh] + ee_pose + goal_pose).show()
                # TODO if this doesn't work in render make a separate script...


                # self.info[-1]["transforms"] = [T_bot2ee, T_bot2objfrm, T_objfrm2obj, T_obj2goal, T_obj2ee]

        if args.save_all_trajectories:
            save_all_trajectories(scene)

        rew = bullet_execute_plan(env, plan, args.write_video, video_writer) if info != [] else 0 

        for i, name in enumerate(object_lists[:-2]):  # reset planner
            scene.env.update_pose(name, placed_poses[i])
        cnts += 1
        rews += rew
        print('rewards: {} counts: {}'.format(rews, cnts))

        # Save data
        # data = [rew, info, plan, inference_duration]
        data = [rew, info, plan]
        np.save(f'{cfg.exp_dir}/{cfg.exp_name}/{cfg.scene_file}/data.npy', data)

        # Convert avi to high quality gif 
        if args.write_video and info != []:
            os.system(f'ffmpeg -y -i {cfg.exp_dir}/{cfg.exp_name}/{cfg.scene_file}/bullet.avi -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 {cfg.exp_dir}/{cfg.exp_name}/{cfg.scene_file}/scene.gif')

        # Create gifs of predictions
        if os.path.exists(f'{cfg.exp_dir}/{cfg.exp_name}/{cfg.scene_file}/pred_pkls'):
            # if using conda requires deps to be installed on base env
            os.system(f'python3 ./bullet/post_process/trimesh_viz.py --dir {cfg.exp_dir}/{cfg.exp_name}/{cfg.scene_file}')

    env.disconnect()

# # def get_clip(scene, traj, i, comps, exp_name, duration, fps):
# def get_clip(fast_debug_vis, traj, i, comps, exp_name, duration, fps):
#     print(i)
#     # traj_im = scene.fast_debug_vis(traj=traj, interact=0, write_video=False,
#     traj_im = fast_debug_vis(traj=traj, interact=0, write_video=False,
#                                     nonstop=False, collision_pt=False, goal_set=False, traj_idx=i)
#     traj_im = cv2.cvtColor(traj_im, cv2.COLOR_RGB2BGR)
#     clip = ImageClip(traj_im).set_duration(duration).set_fps(fps)
#     # TextClip not working for some reason
#     # txt_clip = (TextClip(f"iter {i}", fontsize=50, color='black')
#         # .set_position('bottom')
#         # .set_duration(duration))
#     # comp = CompositeVideoClip([clip, txt_clip]).set_fps(fps).set_duration(duration)
#     comp = CompositeVideoClip([clip]).set_fps(fps).set_duration(duration)
#     comps[i] = comp
#     cv2.imwrite(f"{cfg.exp_dir}/{exp_name}/{cfg.scene_file}/traj_{i+1}.png", traj_im)

    # for scene_idx, scene_file in enumerate(scene_files):
    #     pkls_to_gif(f'{cfg.exp_dir}/{cfg.exp_name}/{scene_file}')


        # if args.init_traj_end_at_start and args.scenes == ["acronym_book"]:
        #     pc_cam = obs['points']
        #     T_world2cam = get_world2cam_transform(env)
        #     T_cam2obj = np.linalg.inv(T_world2cam) @ unpack_pose(book_worldpose)
        #     pc_obj = (T_cam2obj @ pc_cam.T).T 
        #     T_world2obj = T_world2cam @ T_cam2obj
        # else:

        # if args.init_traj_end_at_start:
        #     pc_cam = obs['points']
        #     T_world2cam = get_world2cam_transform(env)
        #     T_world2bot = get_world2bot_transform(env)
        #     T_cam2obj = np.linalg.inv(T_world2cam) @ T_world2bot @ unpack_pose(object_poses[env.target_idx])
        #     pc_obj = (T_cam2obj @ pc_cam.T).T
        #     T_bot2obj = np.linalg.inv(T_world2bot) @ T_world2cam @ T_cam2obj
            
        #     # Save for testing shape embeddings 
        #     # print(f"saving scene {scene_idx}")
        #     # np.save(f"scene{scene_idx}_targetpc.npy", {
        #     #     "pc": pc_cam,
        #     #     "obj_path": env.obj_path[env.target_idx],
        #     #     "T_cam2obj": T_cam2obj,
        #     #     "T_world2cam": T_world2cam,
        #     #     "T_world2bot": T_world2bot,
        #     #     "T_bot2obj": T_bot2obj
        #     # })
        #     # x, y, z = list(np.asarray(env._view_matrix).reshape(4, 4).T[:3, 3])
        #     # sceneimg_save_path = f'dbg_data/scene_imgs/cam_x{x:.2f}_y{y:.2f}_z{z:.2f}'
        #     # os.makedirs(sceneimg_save_path, exist_ok=True)
        #     # np.save(f'{sceneimg_save_path}/{scene_idx}.npy', {
        #     #     'rgb': obs['rgb'], 
        #     #     'mask': obs['mask'], 
        #     #     'view_matrix': env._view_matrix, 
        #     #     'proj_matrix': env._proj_matrix,
        #     #     'target_idx': env._objectUids[env.target_idx],
        #     #     'obj_path': env.obj_path[env.target_idx]
        #     # })
        #     # continue
        #     # draw_pose(T_bot2obj)
        # else:
        #     pc_obj = None
        #     T_bot2obj = None



        # if args.grasp_inference == 'acronym':
        #     if args.scenes == ['acronym_book']:
        #         # grasps = load_grasps
        #         from acronym_tools import load_grasps
        #         rotgrasp2grasp_T = pt.transform_from(pr.matrix_from_axis_angle([0, 0, 1, -np.pi/2]), [0, 0, 0])
        #         obj2rotgrasp_Ts, success = load_grasps(f"/data/manifolds/acronym/grasps/Book_5e90bf1bb411069c115aef9ae267d6b7_0.0268818133810836.h5")
        #         obj2grasp_Ts = (obj2rotgrasp_Ts @ rotgrasp2grasp_T)[success == 1]
        #         np.random.shuffle(obj2grasp_Ts)

        #         T_world2cam = get_world2cam_transform(env)
        #         T_world2bot = get_world2bot_transform(env)
        #         T_cam2obj = np.linalg.inv(T_world2cam) @ T_world2bot @ unpack_pose(object_poses[env.target_idx])
        #         T_bot2obj = np.linalg.inv(T_world2bot) @ T_world2cam @ T_cam2obj

        #         grasps = T_bot2obj @ obj2grasp_Ts #[:10] # limit number
        #         # for grasp in grasps: 
        #         #     draw_pose(T_world2bot @ grasp)
        #         grasp_scores = np.ones(len(grasps))
        #     else:
        #         grasps, grasp_scores = None, None
        #     inference_duration = 0
        # # elif args.grasp_inference == 'ours_knowngrasps':
        # #     grasps, grasp_scores = None, None
        # #     inference_duration = 0 # TODO
        # elif args.grasp_inference == 'ours_outputpose':
        #     grasps, grasp_scores = None, None
        #     inference_duration = 0 # TODO
        # elif args.grasp_inference == 'contact_graspnet':
        #     start_time = time.time()
        #     idx = env._objectUids[env.target_idx]
        #     segmask = deepcopy(obs['mask'])
        #     segmask[segmask != idx] = 0
        #     segmask[segmask == idx] = 1
        #     x = {'rgb': obs['rgb'], 'depth': obs['depth'], 'K': env._intr_matrix, 'seg': segmask}
        #     Ts_cam2grasp, grasp_scores, contact_pts = grasp_inf_method.inference(x)

        #     # Get transform from world to camera
        #     T_world2cam = get_world2cam_transform(env)
        #     # draw_pose(T_world2cam)

        #     Ts_world2grasp = T_world2cam @ Ts_cam2grasp
        #     T_world2bot = get_world2bot_transform(env)
        #     grasps = np.linalg.inv(T_world2bot) @ Ts_world2grasp
        #     inference_duration = time.time() - start_time
        #     print(f"inf duration: {inference_duration}")





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