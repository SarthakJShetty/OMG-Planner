# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import random
import os
from gym import spaces
import time
import sys
import argparse
from . import _init_paths
# from _init_paths import *

from omg.core import *
from omg.util import *
from omg.config import cfg
import pybullet as p
import numpy as np
import pybullet_data

from PIL import Image
import glob
import gym
import IPython
from panda_gripper import Panda

from transforms3d import quaternions
import scipy.io as sio
import pkgutil

import cv2
import matplotlib.pyplot as plt

from collections import namedtuple
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

from acronym_tools import load_grasps, load_mesh, create_gripper_marker

import csv
from copy import deepcopy
import shutil
import h5py

import trimesh

sys.path.append(os.path.dirname(__file__))
from panda_ycb_env import PandaYCBEnv
from utils import *
from panda_scene import bullet_execute_plan

def linear_shake(env, timeSteps, delta_z, record=False):
    pos, orn = p.getLinkState(
        env._panda.pandaUid, env._panda.pandaEndEffectorIndex
    )[:2]

    observations = []
    for t in range(timeSteps):
        jointPoses = env._panda.solveInverseKinematics((pos[0], pos[1], pos[2] + t*delta_z), [1, 0, 0, 0])
        jointPoses[-2:] = [0.0, 0.0]
        env._panda.setTargetPositions(jointPoses)
        p.stepSimulation()
        if env._renders:
            time.sleep(env._timeStep)
        if record and t % 100 == 0:
            observation = env._get_observation()
            observations.append(observation)
    return observations

def rot_shake(env, timeSteps, delta_a, record=False):
    jointPoses = env._panda.getJointStates()[0]

    observations = []
    for t in range(timeSteps):
        jointPoses[-2:] = [0.0, 0.0]
        jointPoses[-4] += delta_a / timeSteps
        env._panda.setTargetPositions(jointPoses)
        p.stepSimulation()
        if env._renders:
            time.sleep(env._timeStep)
        if record and t % 100 == 0:
            observation = env._get_observation()
            observations.append(observation)
    return observations

def load_eval_grasps(filename):
    """Load transformations and qualities of grasps from a JSON file from the dataset.
    Grasps come from predictions of our grasp network. 

    Args:
        filename (str): HDF5 or JSON file name.

    Returns:
        np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
        np.ndarray: List of binary values indicating grasp success in simulation.
    """
    data = h5py.File(filename, "r")
    pos = np.asarray(data['pred_pos'])
    neg = np.asarray(data['pred_neg'])
    free = np.asarray(data['pred_free'])
    T = np.concatenate([pos, neg, free], axis=0)
    success = np.concatenate([
        [1 for i in range(pos.shape[0])],
        [0 for i in range(neg.shape[0])],
        [0 for i in range(free.shape[0])]], axis=0)
    query_type = ["pos" for i in range(pos.shape[0])] + \
        ["neg" for i in range(neg.shape[0])] + \
        ["free" for i in range(free.shape[0])]
    return T, success, query_type

def collect_grasps(margs): 
    args, grasp_path, object_infos = margs
    # gripper faces down
    init_joints = [-0.331397102886776,
                -0.4084196878153061,
                0.27958348738965677,
                -2.100158152658163,
                0.10297468164759681,
                1.7272912586478788,
                0.6990357829811057,
                0.0399,
                0.0399,]

    # setup bullet env
    env = PandaYCBEnv(renders=args.vis, egl_render=args.egl, gravity=False, root_dir=f"{args.mesh_root}/meshes_omg")
    env.reset(init_joints=init_joints, object_infos=object_infos, no_table=True, reset_cache=True)
    initpos, initorn = p.getLinkState(env._panda.pandaUid, env._panda.pandaEndEffectorIndex)[:2]
    initpos = list(initpos)

    # load grasps
    if args.grasp_eval == '': # load acronym grasps
        obj2rotgrasp_Ts, successes = load_grasps(f"{args.grasp_root}/{grasp_path}")
    else: # load predicted grasps
        obj2rotgrasp_Ts, successes, query_types = load_eval_grasps(f"{args.grasp_eval}")

    with h5py.File(f"{args.grasp_root}/{grasp_path}", "r+") as data:
        bullet_group_name = 'grasps/qualities/bullet'
        dset = data[f"{bullet_group_name}/object_in_gripper"]
        collected = data[f"{bullet_group_name}/collected"]

        # bullet_successes = []
        for idx, (obj2rotgrasp_T, success) in enumerate(zip(obj2rotgrasp_Ts, successes)):
            if collected[idx] and not args.overwrite:
                continue

            pq = pt.pq_from_transform(obj2rotgrasp_T)
            print(f"{idx} success label: {success}\tpose: {pq}")

            env._panda.reset(init_joints)
            # env.reset_objects()
            
            # Get end effector pose frame
            pos, orn = p.getLinkState(env._panda.pandaUid, env._panda.pandaEndEffectorIndex)[:2]
            world2ee_T = pt.transform_from_pq(np.concatenate([pos, pr.quaternion_wxyz_from_xyzw(orn)]))
            # draw_pose(world2ee_T)

            # Correct for panda gripper rotation about z axis in bullet
            obj2grasp_T = obj2rotgrasp_T @ rotgrasp2grasp_T
            grasp2obj_T = np.linalg.inv(obj2grasp_T)

            # Move object frame to centroid
            obj_mesh = object_infos[0][2]
            obj2ctr_T = np.eye(4)
            if args.grasp_eval != '':
                obj2ctr_T[:3, 3] = -obj_mesh.centroid


            world2ctr_T = world2ee_T @ grasp2obj_T @ obj2ctr_T
            # draw_pose(world2ctr_T)

            # Place single object
            pq = pt.pq_from_transform(world2ctr_T)
            p.resetBasePositionAndOrientation(
                env._objectUids[0],
                pq[:3],
                pr.quaternion_xyzw_from_wxyz(pq[3:])
            )  # xyzw
            p.resetBaseVelocity(env._objectUids[0], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

            # Close gripper on object
            joints = env._panda.getJointStates()[0]
            joints[-2:] = [0.0, 0.0]
            obs, rew, done, _ = env.step(joints)
            # obs, rew, done, _ = env.step(joints) # run twice
            if rew == 0:
                err_info = 'grasp'
            else:
                err_info = ''

            observations = [obs]

            # Linear shaking
            if err_info == '':
                obs = linear_shake(env, 1000, -0.15/1000., record=args.write_video)
                observations += obs
                obs = linear_shake(env, 1000, 0.15/1000., record=args.write_video)
                observations += obs
                rew = env._reward()
                if rew == 0:
                    err_info = 'linear'

            # Angular shaking
            if err_info == '':
                obs = rot_shake(env, 1000, (np.pi/32.)*15., record=args.write_video)
                observations += obs
                obs = rot_shake(env, 1000, -(np.pi/32.)*15., record=args.write_video)
                observations += obs
                rew = env._reward()
                if rew == 0:
                    err_info = 'shake'

            # if args.write_video and rew == 0: 
            #     video_writer = cv2.VideoWriter(
            #         f'{args.out_dir}/output_videos/{grasp_path}_{idx}.avi',
            #         cv2.VideoWriter_fourcc(*"MJPG"),
            #         1.0, # set low framerate so that videos with low frames will actually save
            #         (640, 480),
            #     )
            #     for obs in observations:
            #         video_writer.write(obs[0][:, :, [2, 1, 0]].astype(np.uint8))
            #         video_writer_all.write(obs[0][:, :, [2, 1, 0]].astype(np.uint8))
                    
            #     # Convert to high quality gif
            #     # os.system(f'ffmpeg -y -i output_videos/{exp_name}/{scene_file}/bullet.avi -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output_videos/{exp_name}/{scene_file}/scene.gif')

            # log pybullet success for this object and pose
            print(f"\tsuccess actual: {rew}")

            dset[idx] = rew
            collected[idx] = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", help="renders", action="store_true")
    parser.add_argument("-w", "--write_video", help="write video", action="store_true")
    parser.add_argument("--egl", help="use egl render", action="store_true")
    parser.add_argument("-mr", "--mesh_root", help="mesh root", type=str, default="/data/manifolds/acronym")
    parser.add_argument("-gr", "--grasp_root", help="grasp root", type=str, default="/data/manifolds/acronym/grasps") 
    parser.add_argument("-o", "--objects", help="objects to load", nargs='*', default=[]) 

    parser.add_argument("-ge", "--grasp_eval", help="grasps to evaluate", type=str, default="")

    parser.add_argument("-od", "--out_dir", help="output directory", type=str, default="/data/manifolds/acronym/bullet_grasps")
    
    parser.add_argument("-ow", "--overwrite", help="overwrite csv", action="store_true")
    parser.add_argument("-dbg", "--debug", help="debug plot", action="store_true")
    parser.add_argument("--workers", help="Pool workers", type=int, default=1)
    parser.add_argument("--start_idx", help="What object by list index to start with", type=int, default=0)

    args = parser.parse_args()

    # Check if csv exists in out dir
    mkdir_if_missing(args.out_dir)

    # # gripper faces down
    # init_joints = [-0.331397102886776,
    #             -0.4084196878153061,
    #             0.27958348738965677,
    #             -2.100158152658163,
    #             0.10297468164759681,
    #             1.7272912586478788,
    #             0.6990357829811057,
    #             0.0399,
    #             0.0399,]

    # # setup bullet env
    # env = PandaYCBEnv(renders=args.vis, egl_render=args.egl, gravity=False, root_dir=f"{args.mesh_root}/meshes_omg")

    # Load acronym object and grasps
    objects = deepcopy(args.objects)
    
    grasp_paths = [] # path to grasp file for a given object
    for fn in os.listdir(args.grasp_root):
        if objects == []: # save all objects
            grasp_paths.append((fn, fn.replace('.h5', '')))
        else:
            for objname in objects:
                if objname in fn:
                    grasp_paths.append((fn, objname))
                    objects.remove(objname) # Only load first scale of object for now

    # Correct for panda gripper rotation about z axis in bullet
    rotgrasp2grasp_T = pt.transform_from(pr.matrix_from_axis_angle([0, 0, 1, -np.pi/2]), [0, 0, 0])

    grasp_and_object_list = []
    for i, (grasp_path, objname) in enumerate(grasp_paths):
        print(i, objname)
        if args.start_idx > i:
            print(f"{i} < {args.start_idx}, skipping")
            continue
        obj_mesh, obj_scale = load_mesh(f"{args.grasp_root}/{grasp_path}", mesh_root_dir=args.mesh_root, ret_scale=True, load_for_bullet=True)
        if obj_mesh == None:
            print(f"{objname} failed to load properly (no urdf), skipping")
            del obj_mesh
            continue
        grasp_and_object = (grasp_path, [[objname, obj_scale, obj_mesh]])

        # load grasps
        if args.grasp_eval == '': # load acronym grasps
            obj2rotgrasp_Ts, successes = load_grasps(f"{args.grasp_root}/{grasp_path}")
        else: # load predicted grasps
            obj2rotgrasp_Ts, successes, query_types = load_eval_grasps(f"{args.grasp_eval}")

        with h5py.File(f"{args.grasp_root}/{grasp_path}", "r+") as data:
            bullet_group_name = 'grasps/qualities/bullet'
            if args.overwrite and bullet_group_name in data:
                del data[f"{bullet_group_name}"]

            if bullet_group_name not in data:
                bullet_group = data.create_group(bullet_group_name)
                # assert len(bullet_successes) == len(data['grasps/qualities/flex/object_in_gripper'])
                dset = bullet_group.create_dataset(f"object_in_gripper", (len(data['grasps/qualities/flex/object_in_gripper']),), dtype='i8')
                # dset.attrs["done"] = False
                collected = bullet_group.create_dataset(f"collected", (len(data['grasps/qualities/flex/object_in_gripper']),), dtype='bool')
            else:
                dset = data[f"{bullet_group_name}/object_in_gripper"]
                if f"{bullet_group_name}/collected" not in data:
                    bullet_group = data[bullet_group_name]
                    collected = bullet_group.create_dataset(f"collected", (len(data['grasps/qualities/flex/object_in_gripper']),), dtype='bool')
                else:
                    collected = data[f"{bullet_group_name}/collected"]

            if np.asarray(collected).sum() == len(collected):
                print(f"All {grasp_path} grasps collected, skipping")
                continue

        grasp_and_object_list.append(grasp_and_object)

        if args.debug: # Debugging plot
            # get transformations and quality of all simulated grasps
            T, success = load_grasps(f"{args.grasp_root}/{grasp_path}")

            # create visual markers for grasps
            successful_grasps = [
                create_gripper_marker(color=[0, 255, 0]).apply_transform(t)
                for t in T[np.random.choice(np.where(success == 1)[0], np.min([np.sum(success), 100]))]
            ]
            failed_grasps = [
                create_gripper_marker(color=[255, 0, 0]).apply_transform(t)
                for t in T[np.random.choice(np.where(success == 0)[0], np.min([np.sum(success == 0), 100]))]
            ]

            trimesh.Scene([obj_mesh] + successful_grasps + failed_grasps).show()

    # import IPython; IPython.embed()

    import multiprocessing as m
    with m.Pool(args.workers) as pool:
        margs = [(args, grasp_path, object_infos) for grasp_path, object_infos in grasp_and_object_list]
        pool.map(collect_grasps, margs)


    # for grasp_path, object_infos in grasp_and_object_list:
    #     # update env
    #     env.reset(init_joints=init_joints, object_infos=object_infos, no_table=True, reset_cache=True)
    #     initpos, initorn = p.getLinkState(env._panda.pandaUid, env._panda.pandaEndEffectorIndex)[:2]
    #     initpos = list(initpos)

    #     if args.write_video:
    #         video_writer_all = cv2.VideoWriter(
    #             f'{args.out_dir}/output_videos/{grasp_path}_all.avi',
    #             cv2.VideoWriter_fourcc(*"MJPG"),
    #             10.0,
    #             (640, 480),
    #         )

    #     with h5py.File(f"{args.grasp_root}/{grasp_path}", "r+") as data:
    #         bullet_group_name = 'grasps/qualities/bullet'
    #         dset = data[f"{bullet_group_name}/object_in_gripper"]
    #         collected = data[f"{bullet_group_name}/collected"]

    #         # bullet_successes = []
    #         for idx, (obj2rotgrasp_T, success) in enumerate(zip(obj2rotgrasp_Ts, successes)):
    #             if collected[idx] and not args.overwrite:
    #                 continue

    #             pq = pt.pq_from_transform(obj2rotgrasp_T)
    #             print(f"{idx} success label: {success}\tpose: {pq}")

    #             env._panda.reset(init_joints)
    #             # env.reset_objects()
                
    #             # Get end effector pose frame
    #             pos, orn = p.getLinkState(env._panda.pandaUid, env._panda.pandaEndEffectorIndex)[:2]
    #             world2ee_T = pt.transform_from_pq(np.concatenate([pos, pr.quaternion_wxyz_from_xyzw(orn)]))
    #             # draw_pose(world2ee_T)

    #             # Correct for panda gripper rotation about z axis in bullet
    #             obj2grasp_T = obj2rotgrasp_T @ rotgrasp2grasp_T
    #             grasp2obj_T = np.linalg.inv(obj2grasp_T)

    #             # Move object frame to centroid
    #             obj_mesh = object_infos[0][2]
    #             obj2ctr_T = np.eye(4)
    #             if args.grasp_eval != '':
    #                 obj2ctr_T[:3, 3] = -obj_mesh.centroid

    #             # if args.debug: # Debugging plot
    #                 # pos_grasp = [create_gripper_marker(color=[0, 255, 0]).apply_transform(obj2rotgrasp_T)]
    #                 # obj_mesh = obj_mesh.apply_transform(obj2ctr_T)
    #                 # trimesh.Scene([obj_mesh] + pos_grasp).show()

    #             world2ctr_T = world2ee_T @ grasp2obj_T @ obj2ctr_T
    #             # draw_pose(world2ctr_T)

    #             # Place single object
    #             pq = pt.pq_from_transform(world2ctr_T)
    #             p.resetBasePositionAndOrientation(
    #                 env._objectUids[0],
    #                 pq[:3],
    #                 pr.quaternion_xyzw_from_wxyz(pq[3:])
    #             )  # xyzw
    #             p.resetBaseVelocity(env._objectUids[0], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    #             # Close gripper on object
    #             joints = env._panda.getJointStates()[0]
    #             joints[-2:] = [0.0, 0.0]
    #             obs, rew, done, _ = env.step(joints)
    #             # obs, rew, done, _ = env.step(joints) # run twice
    #             if rew == 0:
    #                 err_info = 'grasp'
    #             else:
    #                 err_info = ''

    #             observations = [obs]

    #             # Linear shaking
    #             if err_info == '':
    #                 obs = linear_shake(env, 1000, -0.15/1000., record=args.write_video)
    #                 observations += obs
    #                 obs = linear_shake(env, 1000, 0.15/1000., record=args.write_video)
    #                 observations += obs
    #                 rew = env._reward()
    #                 if rew == 0:
    #                     err_info = 'linear'

    #             # Angular shaking
    #             if err_info == '':
    #                 obs = rot_shake(env, 1000, (np.pi/32.)*15., record=args.write_video)
    #                 observations += obs
    #                 obs = rot_shake(env, 1000, -(np.pi/32.)*15., record=args.write_video)
    #                 observations += obs
    #                 rew = env._reward()
    #                 if rew == 0:
    #                     err_info = 'shake'

    #             if args.write_video and rew == 0: 
    #                 video_writer = cv2.VideoWriter(
    #                     f'{args.out_dir}/output_videos/{grasp_path}_{idx}.avi',
    #                     cv2.VideoWriter_fourcc(*"MJPG"),
    #                     1.0, # set low framerate so that videos with low frames will actually save
    #                     (640, 480),
    #                 )
    #                 for obs in observations:
    #                     video_writer.write(obs[0][:, :, [2, 1, 0]].astype(np.uint8))
    #                     video_writer_all.write(obs[0][:, :, [2, 1, 0]].astype(np.uint8))
                        
    #                 # Convert to high quality gif
    #                 # os.system(f'ffmpeg -y -i output_videos/{exp_name}/{scene_file}/bullet.avi -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output_videos/{exp_name}/{scene_file}/scene.gif')

    #             # log pybullet success for this object and pose
    #             print(f"\tsuccess actual: {rew}")
    #             # bullet_successes.append(rew)

    #             # if args.grasp_eval != '':
    #                 # query_type = query_types[idx]
    #             # else:
    #                 # query_type = 'pos' if success else 'neg'
    #             # with open(csvfile, 'a') as f:
    #                 # writer = csv.writer(f, delimiter=',')
    #                 # writer.writerow([grasp_path, idx, success, rew, err_info, pq, query_type])

    #             dset[idx] = rew
    #             collected[idx] = True
