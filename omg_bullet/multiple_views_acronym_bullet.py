#!/usr/bin/env python3
"""
The MIT License (MIT)

Copyright (c) 2020 NVIDIA Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import os.path as osp
import sys
import json
import trimesh
import pybullet as p
from omg_bullet.envs.acronym_env import PandaAcronymEnv
# import pyrender
import argparse
import numpy as np
import matplotlib.pyplot as plt

from acronym_tools import load_mesh, create_gripper_marker

from mpl_toolkits import mplot3d
from copy import deepcopy

import h5py

import torch
from datetime import datetime

import pytorch3d
import pytorch3d.ops

# from omg_bullet.utils import place_object


def main(args):
    np.random.seed(0)

    # load object meshes and generate a random scene
    obj_mesh_paths = []
    for obj_name in os.listdir(f"{args.mesh_root}/grasps"):
        obj_mesh_paths.append(f"{args.mesh_root}/grasps/{obj_name}")

    suffix = '_rotate' if args.rotate else ''
    save_dir = f"{args.mesh_root}/shape-dataset_{datetime.now().strftime('%m-%d-%y_%H-%M-%S')}_{args.n_samples / 1000}k{suffix}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cam_look = [-0.05, -0.5, -0.6852]
    env = PandaAcronymEnv(renders=args.visualize, gravity=False, cam_look=cam_look)

    with h5py.File(f'{save_dir}/dataset.hdf5', 'a') as hf:
        
        for mesh_idx, mesh_path in enumerate(obj_mesh_paths):
            # if mesh_idx != 3:
            #     continue
            print(mesh_idx)
            obj_name = mesh_path.split('/')[-1].replace('.h5', '')
            if obj_name in hf:
                if len(hf[obj_name]) == args.n_samples:
                    print(f"skipping {obj_name}")
                    continue
                else:
                    grp = hf[obj_name]
            else:
                grp = hf.create_group(obj_name)
            
            mesh = load_mesh(mesh_path, mesh_root_dir=args.mesh_root)

            T_ctr2obj = np.eye(4)
            T_ctr2obj[:3, 3] = -mesh.centroid
            mesh = mesh.apply_transform(T_ctr2obj)

            dots = np.einsum('ij,ij->i',
                             mesh.center_mass - mesh.convex_hull.triangles_center,
                             mesh.convex_hull.face_normals)
            if not np.all(dots < 0):
                print("Warning: center of mass not inside convex hull, skipping")
                continue

            i = len(grp)
            objinfo = env.get_object_info(obj_name, args.mesh_root)
            while i < args.n_samples:
                print(f"{i}th sample of {obj_name}")

                env.reset(no_table=True, objinfos=[objinfo])
                T_world_obj = env.place_object(env._objectUids[0], target_pos=[0.5, 0.0, 0.5], random=args.rotate, gravity=False)
                # T_world_obj = env.place_object(env._objectUids[0], target_pos=[0.0, 0.0, 0.0], random=args.rotate, gravity=False)
                obs = env._get_observation(get_pc=True, single_view=False)
                pc_world = obs['points']

                # Sample the data randomly and with farthest point sampling
                # pc_world_rnd = regularize_pc_point_count(pc_world, npoints=1500, use_farthest_point=False)  # 3.69s
                pc_world_t = torch.tensor(pc_world, dtype=torch.float32, device='cuda')
                if len(pc_world_t) == 0:
                    continue
                pc_world_fps_t, pc_world_fps_idxs = pytorch3d.ops.sample_farthest_points(pc_world_t.unsqueeze(0), K=1500, random_start_point=True)  # 0.12s
                pc_world_fps_t = pc_world_fps_t.squeeze()
                pc_world_fps_idxs = pc_world_fps_idxs.squeeze()

                # Save data
                subgrp = grp.create_group(f'{i}')
                subgrp.create_dataset('pc_world', data=pc_world_t.cpu(), shape=pc_world_t.shape, compression='gzip', chunks=True)
                subgrp.create_dataset('pc_world_fps_t', data=pc_world_fps_t.cpu(), shape=pc_world_fps_t.shape, compression='gzip', chunks=True)
                subgrp.create_dataset('pc_world_fps_idxs', data=pc_world_fps_idxs.cpu(), shape=pc_world_fps_idxs.shape, compression='gzip', chunks=True)
                # subgrp.create_dataset('pc_world_rnd', data=pc_world_rnd, shape=pc_world_rnd.shape, compression='gzip', chunks=True)
                subgrp.create_dataset('T_world_obj', data=T_world_obj, shape=T_world_obj.shape, compression='gzip', chunks=True)
                # subgrp.create_dataset('T_world_wsm', data=T_world_wsm, shape=T_world_wsm.shape, compression='gzip', chunks=True)
                # subgrp.create_dataset('pc_ctr', data=pc_ctr, shape=pc_ctr.shape, compression='gzip', chunks=True)

                if args.visualize:    # debug visualization
                    # visualize object in world frame
                    pcd_world = trimesh.points.PointCloud(pc_world[:, :3], np.array([0, 255, 0, 255]))
                    # pcd_obj = trimesh.points.PointCloud(pc_obj[:, :3], np.array([0, 0, 255, 255]))
                    trimesh.Scene([pcd_world]).show()

                    # transform to mesh frame
                    pc_obj = (np.linalg.inv(T_world_obj) @ pc_world.T).T
                    pcd_obj = trimesh.points.PointCloud(pc_obj[:, :3], np.array([255, 0, 0, 255]))
                    trimesh.Scene([pcd_obj, pcd_world]).show()

                i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render observations of a randomly generated scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mesh_root", default=".", help="Directory used for loading meshes."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show the scene and camera pose from which observations are rendered.",
    )
    parser.add_argument(
        "--fps",
        action="store_true",
        help="Use furthest-point sampling.",
    )
    parser.add_argument(
        "--n_samples", default=1, type=int,
        help="Number of samples per object"
    )
    parser.add_argument(
        "--target_pc_size", default=1500,
        help="Size of downsampled point cloud"
    )
    parser.add_argument(
        "--rotate", 
        action="store_true",
        help="Randomly rotate object before capturing point cloud"
    )
    args = parser.parse_args()
    main(args)
