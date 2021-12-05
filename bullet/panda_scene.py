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
from omg.util import *
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

# For 6-DOF graspnet
# import torch
# import grasp_estimator
# from utils import utils as gutils
# from utils import visualization_utils
# import mayavi.mlab as mlab
# mlab.options.offscreen = True

from contact_graspnet_infer import *
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

import torch
import pytorch_lightning
from manifold_grasping.networks import Decoder

class ImplicitGraspInference:
    def __init__(self):
        # TODO config output pose
        ckpt = torch.load('/checkpoint/thomasweng/grasp_manifolds/runs/outputs/nopc_book_logmap_10kfree/2021-11-23_232102_dsetacronym_Book_train0.9_val0.1_free10.0k_sym_distlogmap/default_default/0_0/checkpoints/last.ckpt')
        self.model = Decoder(**ckpt['hyper_parameters'])
        self.model.cuda()
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

    def inference(self, x):
        """
        x: xyz, xyzw
        """
        with torch.no_grad():
            outpose = self.model(x)
        # outpose = outpose.squeeze(0).cpu().numpy()
        outpose = outpose.squeeze(0)
        return outpose


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-gi", "--grasp_inference", help="which grasp inference method to use", required=True, choices=['acronym', 'contact_graspnet', 'ours_outputpose'])
    parser.add_argument("-gs", "--grasp_selection", help="Which grasp selection algorithm to use", required=True, choices=['Fixed', 'Proj', 'OMG'])
    parser.add_argument("-o", "--output_dir", help="Output directory", type=str, default="./output_videos") 
    parser.add_argument("-d", "--debug_exp", help="Override default experiment name with dbg", action="store_true"),  
    parser.add_argument("-s", "--scenes", help="scene(s) to run. If set to '', loops through all 100 scenes", nargs='*', default=["scene_1"])
    parser.add_argument("-r", "--render", help="render", action="store_true")
    parser.add_argument("-v", "--visualize", help="visualize grasps", action="store_true")
    parser.add_argument("--egl", help="use egl render", action="store_true")
    parser.add_argument("-w", "--write_video", help="write video", action="store_true")
    parser.add_argument("--debug_traj", help="Visualize intermediate trajectories", action="store_true")
    args = parser.parse_args()

    # Set up bullet env
    mkdir_if_missing(args.output_dir)
    env = PandaYCBEnv(renders=args.render, egl_render=args.egl, gravity=False)

    if args.scenes == ['acronym_book']:
        grasp_root = "/data/manifolds/acronym/grasps"
        # objects = ['Book_5e90bf1bb411069c115aef9ae267d6b7']
        objects = ['Book_5e90bf1bb411069c115aef9ae267d6b7_0.0268818133810836']
        grasp_paths = [] # path to grasp file for a given object
        for fn in os.listdir(grasp_root):
            for objname in objects:
                if objname in fn:
                    grasp_paths.append((fn, objname))
                    objects.remove(objname) # Only load first scale of object for now

        # Load acronym objects
        object_infos = []
        mesh_root = "/data/manifolds/acronym"
        for grasp_path, objname in grasp_paths:
            obj_mesh, obj_scale = load_mesh(f"{grasp_root}/{grasp_path}", mesh_root_dir=mesh_root, ret_scale=True)
            object_infos.append((objname, obj_scale, obj_mesh))
        env.reset(object_infos=object_infos)

        # Move object frame to centroid
        obj_mesh = object_infos[0][2]
        obj2ctr_T = np.eye(4)
        # obj2ctr_T[:3, 3] = -obj_mesh.centroid
        obj2ctr_T[:3, 3] = obj_mesh.centroid
    else:
        env.reset()

    # Set up planning scene
    cfg.traj_init = "grasp"
    cfg.vis = False
    cfg.timesteps = 50
    cfg.get_global_param(cfg.timesteps)
    cfg.scene_file = ''
    scene = PlanningScene(cfg)
    for i, name in enumerate(env.obj_path[:-2]):  # load all objects
        name = name.split("/")[-1]
        trans, orn = env.cache_object_poses[i]
        scene.env.add_object(name, trans, tf_quat(orn), compute_grasp=True)
    scene.reset()

    # Set up scene
    scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1, 0, 0, 0]))
    scene.env.add_table(np.array([0.55, 0, -0.17]), np.array([0.707, 0.707, 0.0, 0]))
    scene.env.combine_sdfs()

    cfg.scene_files = args.scenes
    if cfg.scene_files == []:
        scene_files = ['scene_{}'.format(i) for i in range(100)] # TODO change to listdir
    else:
        scene_files = cfg.scene_files

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
    
    # if args.debug_traj and not args.use_graspnet: # does not work with use_graspnet due to EGL render issues with mayavi downstream
    # TODO check if this works
    # if args.debug_traj: # does not work with use_graspnet due to EGL render issues with mayavi downstream
        # scene.setup_renderer()
    #     init_traj = scene.planner.traj.data
    #     init_traj_im = scene.fast_debug_vis(traj=init_traj, interact=0, write_video=False,
    #                                 nonstop=False, collision_pt=False, goal_set=False, traj_idx=0)
    #     init_traj_im = cv2.cvtColor(init_traj_im, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(f"{args.output_dir}/{exp_name}/{scene_file}/traj_0.png", init_traj_im)

    # Set up grasp inference method
    if args.grasp_inference == 'contact_graspnet':
        grasp_inf_method = ContactGraspNetInference(args.visualize)
    if args.grasp_inference == 'ours_outputpose':
        grasp_inf_method = ImplicitGraspInference()

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
        obs = env.cache_reset(scene_file=full_name)
        obj_names, obj_poses = env.get_env_info()
        object_lists = [name.split("/")[-1].strip() for name in obj_names]
        object_poses = [pack_pose(pose) for pose in obj_poses]

        if args.scenes == ['acronym_book']:
            # Hotfix: Manually set book object
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
            # for _ in range(1000):
                # p.stepSimulation()

        # Update planning scene
        exists_ids, placed_poses = [], []
        for i, name in enumerate(object_lists[:-2]):   
            scene.env.update_pose(name, object_poses[i])
            obj_idx = env.obj_path[:-2].index("data/objects/" + name)
            exists_ids.append(obj_idx)
            trans, orn = env.cache_object_poses[obj_idx]
            placed_poses.append(np.hstack([trans, ros_quat(orn)]))

        cfg.disable_collision_set = [
            name.split("/")[-2]
            for obj_idx, name in enumerate(env.obj_path[:-2])
            if obj_idx not in exists_ids
        ]

        if args.grasp_inference == 'acronym':
            grasps, grasp_scores = None, None
            inference_duration = 0
        elif args.grasp_inference == 'ours_outputpose':
            start_time = time.time()

            pos, orn = p.getBasePositionAndOrientation(env._objectUids[env.target_idx])
            world2obj_T = np.eye(4)
            world2obj_T[:3, 3] = pos
            world2obj_T[:3, :3] = np.asarray(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            world2ctr_T = world2obj_T @ obj2ctr_T
            # draw_pose(world2ctr_T)

            # Get end effector pose frame
            pos, orn = p.getLinkState(env._panda.pandaUid, env._panda.pandaEndEffectorIndex)[:2]
            world2ee_T = pt.transform_from_pq(np.concatenate([pos, pr.quaternion_wxyz_from_xyzw(orn)]))
            # draw_pose(world2ee_T)

            # Get end effector in centroid frame, do inference
            ctr2ee_T = np.linalg.inv(world2ctr_T) @ world2ee_T
            ee_pos = ctr2ee_T[:3, 3]
            ee_orn = pr.quaternion_from_matrix(ctr2ee_T[:3, :3]) # wxyz
            ee_pose = np.concatenate([ee_pos, pr.quaternion_xyzw_from_wxyz(ee_orn)])
            # ee_pose = torch.tensor(ee_pose, dtype=torch.float32).unsqueeze(0)
            ee_pose = torch.tensor(ee_pose, dtype=torch.float32, device='cuda').unsqueeze(0)

            fwd_time = time.time()
            out_pose = grasp_inf_method.inference(ee_pose)
            fwd_duration = time.time() - fwd_time
            print(f"fwd pass: {fwd_duration}")
            out_pose = out_pose.cpu().numpy()

            # Convert output into world frame
            out_pos = out_pose[:3]
            out_orn = out_pose[3:] # xyzw
            ctr2out_T = pt.transform_from_pq(np.concatenate([out_pos, pr.quaternion_wxyz_from_xyzw(out_orn)]))
            world2out_T = world2ctr_T @ ctr2out_T
            # draw_pose(world2out_T)

            # TODO consolidate
            pos, orn = p.getBasePositionAndOrientation(env._panda.pandaUid)
            mat = np.asarray(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            T_world2bot = np.eye(4)
            T_world2bot[:3, :3] = mat
            T_world2bot[:3, 3] = pos
            grasps = np.linalg.inv(T_world2bot) @ world2out_T[np.newaxis, :]
            grasp_scores = np.array([1.0])

            # TODO integrate into planner. 

            inference_duration = time.time() - start_time
            print(f"inf pass: {inference_duration}")
        elif args.grasp_inference == 'contact_graspnet':
            start_time = time.time()
            idx = env._objectUids[env.target_idx]
            segmask = deepcopy(obs['mask'])
            segmask[segmask != idx] = 0
            segmask[segmask == idx] = 1
            x = {'rgb': obs['rgb'], 'depth': obs['depth'], 'K': env._intr_matrix, 'seg': segmask}
            Ts_cam2grasp, grasp_scores, contact_pts = grasp_inf_method.inference(x)

            # Get transform from world to camera
            T_world2camgl = np.linalg.inv(np.asarray(env._view_matrix).reshape((4, 4), order='F'))
            T_camgl2cam = np.zeros((4, 4))
            T_camgl2cam[:3, :3] = pr.matrix_from_axis_angle([1, 0, 0, np.pi])
            T_camgl2cam[3, 3] = 1
            T_world2cam = T_world2camgl @ T_camgl2cam
            # draw_pose(T_world2cam)

            Ts_world2grasp = T_world2cam @ Ts_cam2grasp
            pos, orn = p.getBasePositionAndOrientation(env._panda.pandaUid)
            mat = np.asarray(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            T_world2bot = np.eye(4)
            T_world2bot[:3, :3] = mat
            T_world2bot[:3, 3] = pos
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
        scene.reset(lazy=True, grasps=grasps, grasp_scores=grasp_scores)

        info = scene.step()
        plan = scene.planner.history_trajectories[-1]

        # # Visualize intermediate trajectories
        # if args.debug_traj:
        #     # TODO make gif
        # #     # if args.use_graspnet:
        #     # scene.setup_renderer()
        #     for i, traj in enumerate(scene.planner.history_trajectories):
        #         traj_im = scene.fast_debug_vis(traj=traj, interact=0, write_video=False,
        #                                        nonstop=False, collision_pt=False, goal_set=True, traj_idx=i)
        #         traj_im = cv2.cvtColor(traj_im, cv2.COLOR_RGB2BGR)
        #         cv2.imwrite(f"{args.output_dir}/{exp_name}/{scene_file}/traj_{i+1}.png", traj_im)

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

        # for grasp in grasps[:30]:
        #     draw_pose(grasp)

        # pts = (T_world2cam @ np.hstack([contact_pts, np.ones((len(contact_pts), 1))]).T).T
        # for pt in pts[:30]:
        #     T = np.eye(4)
        #     T[:, 3] = pt
        #     draw_pose(T)

        # pc, rgb = depth2pc(obs['depth'].squeeze() * (obs['mask'] == 0).squeeze(), env._intr_matrix, obs['rgb']/255.0)
        # idxs = np.random.choice(len(pc), size=1000)
        # xyz_cam = pc[idxs]

        # xyzh_cam = np.hstack([xyz_cam, np.ones((len(xyz_cam), 1))])
        # for xyzh in xyzh_cam[:300]:
        #     xyz_world = (T_world2cam @ xyzh)[:3]
        #     T_pt = np.eye(4)
        #     T_pt[:3, 3] = xyz_world
        #     draw_pose(T_pt) 

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # pc, rgb = depth2pc(obs['depth'].squeeze(), env._intr_matrix, obs['rgb']/255.0)
        # # obs = env._get_observation(pc=True)
        # idxs = np.random.choice(len(pc), size=10000)
        # xyz_cam = pc[idxs]
        # rgb = rgb[idxs]
        # xyzh_cam = np.hstack([xyz_cam, np.ones((len(xyz_cam), 1))])
        # ptsh_cam = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
        # # xyz_world = (np.linalg.inv(T_world2cam) @ xyzh_cam.T).T[:, :3]
        # xyz_world = (T_world2cam @ xyzh_cam.T).T[:, :3]
        # # pts_world = (np.linalg.inv(T_world2cam) @ ptsh_cam.T).T[:, :3]
        # pts_world = (T_world2cam @ ptsh_cam.T).T[:, :3]
        # xyz = xyz_world
        # pts = pts_world

        # fig = plt.figure(figsize=(4,4))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter([0], [0], [0], s=100, c='green')
        # ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb)
        # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=10, c='red')
        # # ax.scatter(Ts_cam2grasp[:, 0, 3], Ts_cam2grasp[:, 1, 3], Ts_cam2grasp[:, 2, 3], c='red')

        # ax.axes.set_xlim3d(left=-1.5, right=1.5) 
        # ax.axes.set_ylim3d(bottom=-1.5, top=1.5) 
        # ax.axes.set_zlim3d(bottom=-1.5, top=1.5) 
        # plt.show()

        # # Save for contact-graspnet
        # np.save('/home/exx/projects/manifolds/contact_graspnet/test_data/pybullet.npy', 
        #     {'rgb': cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR), 'depth': obs['depth'], 'K': env._intr_matrix, 'seg': obs['mask']}, 
        #     allow_pickle=True)

        # if args.use_graspnet:
        #     # TODO update with new obs model
        #     # obs['points']
        #     object_pc, all_pc = env.get_pc()

        #     # Predict grasp using 6-DOF GraspNet
        #     gpath = os.path.dirname(grasp_estimator.__file__)
        #     grasp_sampler_args = gutils.read_checkpoint_args(
        #         os.path.join(gpath, args.grasp_sampler_folder))
        #     grasp_sampler_args['checkpoints_dir'] = os.path.join(gpath, grasp_sampler_args['checkpoints_dir'])
        #     grasp_sampler_args.is_train = False
        #     grasp_evaluator_args = gutils.read_checkpoint_args(
        #         os.path.join(gpath, args.grasp_evaluator_folder))
        #     grasp_evaluator_args['checkpoints_dir'] = os.path.join(gpath, grasp_evaluator_args['checkpoints_dir'])
        #     grasp_evaluator_args.continue_train = True # was in demo file, not sure 
        #     estimator = grasp_estimator.GraspEstimator(grasp_sampler_args,
        #                                                 grasp_evaluator_args, args)

        #     pc = np.asarray(object_pc.points)
        #     start_time = time.time()
        #     generated_grasps, generated_scores = estimator.generate_and_refine_grasps(pc)
        #     graspinf_duration = time.time() - start_time
        #     print("Grasp Inference time: {:.3f}".format(graspinf_duration))

        #     generated_grasps = np.array(generated_grasps)
        #     generated_scores = np.array(generated_scores)

        #     # Set configs according to args
        #     if args.grasp_selection == 'Fixed':
        #         scene.planner.cfg.ol_alg = 'Baseline'
        #         scene.planner.cfg.goal_idx = -1
        #     elif args.grasp_selection == 'Proj':
        #         scene.planner.cfg.ol_alg = 'Proj'
        #     elif args.grasp_selection == 'OMG':
        #         scene.planner.cfg.ol_alg = 'MD'

        #     scene.env.set_target(env.obj_path[env.target_idx].split("/")[-1])
        #     scene.reset(lazy=True, grasps=generated_grasps, grasp_scores=generated_scores)
            
        #     dbg = np.load("{args.output_dir}/dbg.npy", encoding='latin1', allow_pickle=True)
        #     grasp_start, grasp_end, goal_idx, goal_set, goal_quality, grasp_ees = dbg # in joints, not EE pose
        #     offset_pose = np.array(rotZ(-np.pi / 2))  # unrotate gripper for visualization (was rotated in Planner class)
        #     goal_ees_T = [np.matmul(unpack_pose(g), offset_pose) for g in grasp_ees]
        #     # goal_ees_T = [unpack_pose(g) for g in grasp_ees]

        #     # Visualize
        #     # visualization_utils.draw_scene(
        #     #     np.asarray(all_pc.points),
        #     #     pc_color=(np.asarray(all_pc.colors) * 255).astype(int),
        #     #     grasps=goal_ees_T,
        #     #     grasp_scores=goal_quality,
        #     # )

        #     visualization_utils.draw_scene(
        #         np.asarray(object_pc.points),
        #         pc_color=(np.asarray(object_pc.colors) * 255).astype(int),
        #         grasps=[goal_ees_T[goal_idx]],
        #         grasp_scores=[goal_quality[goal_idx]],
        #     )
        #     mlab.savefig(f"{args.output_dir}/{exp_name}/{scene_file}/grasp.png")
        #     # mlab.clf()
        #     mlab.close()
        #     # import IPython; IPython.embed() # Ctrl-D for interactive visualization 
        # else:


    # def make_parser(parser):
    # # parser = argparse.ArgumentParser(
    #     # description='6-DoF GraspNet Demo',
    #     # formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--grasp_sampler_folder',
    #                     type=str,
    #                     default='checkpoints/gan_pretrained/')
    # parser.add_argument('--grasp_evaluator_folder',
    #                     type=str,
    #                     default='checkpoints/evaluator_pretrained/')
    # parser.add_argument('--refinement_method',
    #                     choices={"gradient", "sampling"},
    #                     default='sampling')
    # parser.add_argument('--refine_steps', type=int, default=25)

    # parser.add_argument('--npy_folder', type=str, default='demo/data/')
    # parser.add_argument(
    #     '--threshold',
    #     type=float,
    #     default=0.8,
    #     help=
    #     "When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed"
    # )
    # parser.add_argument(
    #     '--choose_fn',
    #     choices={
    #         "all", "better_than_threshold", "better_than_threshold_in_sequence"
    #     },
    #     default='better_than_threshold',
    #     help=
    #     "If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps"
    # )

    # parser.add_argument('--target_pc_size', type=int, default=1024)
    # parser.add_argument('--num_grasp_samples', type=int, default=200)
    # parser.add_argument(
    #     '--generate_dense_grasps',
    #     action='store_true',
    #     help=
    #     "If enabled, it will create a [num_grasp_samples x num_grasp_samples] dense grid of latent space values and generate grasps from these."
    # )

    # parser.add_argument(
    #     '--batch_size',
    #     type=int,
    #     default=30,
    #     help=
    #     "Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory."
    # )
    # parser.add_argument('--train_data', action='store_true')
    # # opts, _ = parser.parse_known_args()
    # # if opts.train_data:
    # #     parser.add_argument('--dataset_root_folder',
    # #                         required=True,
    # #                         type=str,
    # #                         help='path to root directory of the dataset.')
    # return parser