# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import random
import os
from gym import spaces
import time
import sys
import argparse
# from . import _init_paths
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
# from panda_gripper import Panda

from transforms3d import quaternions
import scipy.io as sio
import pkgutil

# For backprojection
import cv2
import matplotlib.pyplot as plt
# import glm
# import open3d as o3d

from collections import namedtuple
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

from acronym_tools import load_grasps, create_gripper_marker
from manifold_grasping.utils import load_mesh

import csv
from copy import deepcopy
import shutil
import h5py

import trimesh

sys.path.append(os.path.dirname(__file__))
from .panda_env import PandaEnv
from .utils import *

def linear_shake(env, timeSteps, delta_z, record=False, second_shake=False):
    pos, orn = p.getLinkState(
        env._panda.pandaUid, env._panda.pandaEndEffectorIndex
    )[:2]

    observations = []
    for t in range(timeSteps):
        jointPoses = env._panda.solveInverseKinematics((pos[0], pos[1], pos[2] + t*delta_z), [1, 0, 0, 0])
        jointPoses[-2:] = [0.0, 0.0]
        env._panda.setTargetPositions(jointPoses)
        p.stepSimulation()
        if second_shake:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", help="render", action="store_true")
    parser.add_argument("--egl", help="use egl render", action="store_true")
    parser.add_argument("--mesh_root", help="mesh root", type=str, required=True, default="/data/manifolds/acronym")
    # parser.add_argument("-o", "--objects", help="objects to load", nargs='*', default=[])

    # parser.add_argument("-ge", "--grasp_eval", help="grasps to evaluate", type=str, default="")

    parser.add_argument("--overwrite", help="overwrite", action="store_true")
    parser.add_argument("--lin", help="How far to linear shake in a second", default=0.15, type=float)
    parser.add_argument("--rot", help="How far to angular shake in a second, in degrees", default=180, type=float)
    # parser.add_argument("-dbg", "--debug", help="debug plot", action="store_true")
    # parser.add_argument("--start_idx", help="What object by list index to start with", type=int, default=0)

    args = parser.parse_args()

    # Check if csv exists in out dir
    # mkdir_if_missing(args.out_dir)

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
    env = PandaEnv(renders=args.render, egl_render=args.egl, gravity=False, root_dir=f"{args.mesh_root}/meshes_bullet")

    # acronym/
    #   meshes/
    #     Book/
    #       [HASH].obj
    #   meshes_bullet/
    #     Book_[HASH]/
    #       model_normalized.urdf

    grasp_h5s = os.listdir(f'{args.mesh_root}/grasps')
    for objname in os.listdir(f'{args.mesh_root}/meshes'):
        # TODO update with code from get_bullet_labels_mp
        graspfile, obj_mesh, objinfo = objinfo_from_obj(grasp_h5s, mesh_root=args.mesh_root, objname=objname)
        env.reset(init_joints=init_joints, no_table=True, objinfo=objinfo)
        rotgrasp2grasp_T = pt.transform_from(pr.matrix_from_axis_angle([0, 0, 1, -np.pi/2]), [0, 0, 0]) # correct wrist rotation

        obj2rotgrasp_Ts, successes = load_grasps(f"{args.mesh_root}/grasps/{graspfile}")

        with h5py.File(f"{args.mesh_root}/grasps/{graspfile}", "r+") as data:
            # Set up bullet dataset
            bullet_group_name = 'grasps/qualities/bullet'
            if args.overwrite and bullet_group_name in data:
                del data[f"{bullet_group_name}"]

            if bullet_group_name not in data:
                bullet_group = data.create_group(bullet_group_name)
                dset = bullet_group.create_dataset(f"object_in_gripper", (len(data['grasps/qualities/flex/object_in_gripper']),), dtype='i8')
                collected = bullet_group.create_dataset(f"collected", (len(data['grasps/qualities/flex/object_in_gripper']),), dtype='bool')
            else:
                dset = data[f"{bullet_group_name}/object_in_gripper"]
                if f"{bullet_group_name}/collected" not in data:
                    bullet_group = data[bullet_group_name]
                    collected = bullet_group.create_dataset(f"collected", (len(data['grasps/qualities/flex/object_in_gripper']),), dtype='bool')
                else:
                    collected = data[f"{bullet_group_name}/collected"]

            # if np.asarray(collected).sum() == len(collected):
            #     print(f"All {grasp_path} grasps collected, skipping")
            #     continue

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

                # Correct for panda gripper rotation about z axis in bullet
                obj2grasp_T = obj2rotgrasp_T @ rotgrasp2grasp_T
                grasp2obj_T = np.linalg.inv(obj2grasp_T)

                # Move object frame to centroid
                # obj_mesh = object_infos[0][2]
                obj2ctr_T = np.eye(4)
                # if args.grasp_eval != '':
                obj2ctr_T[:3, 3] = -obj_mesh.centroid

                # if args.debug: # Debugging plot
                    # pos_grasp = [create_gripper_marker(color=[0, 255, 0]).apply_transform(obj2rotgrasp_T)]
                    # obj_mesh = obj_mesh.apply_transform(obj2ctr_T)
                    # trimesh.Scene([obj_mesh] + pos_grasp).show()

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
                if rew == 0:
                    err_info = 'grasp'
                else:
                    err_info = ''

                observations = [obs]

                # Linear shaking
                if err_info == '':
                    dur = 150
                    obs = linear_shake(env, dur, -args.lin/dur)
                    observations += obs
                    obs = linear_shake(env, dur, args.lin/dur)
                    observations += obs
                    rew = env._reward()
                    if rew == 0:
                        err_info = 'linear'

                # Angular shaking
                if err_info == '':
                    dur = 150
                    obs = rot_shake(env, dur, np.deg2rad(args.rot))
                    observations += obs
                    obs = rot_shake(env, dur, -np.deg2rad(args.rot))
                    observations += obs
                    rew = env._reward()
                    if rew == 0:
                        err_info = 'shake'

                if err_info == '':
                    dur = 150
                    obs = linear_shake(env, dur, -args.lin/dur, second_shake=True)
                    observations += obs
                    obs = linear_shake(env, dur, args.lin/dur, second_shake=True)
                    observations += obs
                    rew = env._reward()
                    if rew == 0:
                        err_info = 'linear2'

                # log pybullet success for this object and pose
                print(f"\tsuccess actual: {rew}")
                dset[idx] = rew
                collected[idx] = True
