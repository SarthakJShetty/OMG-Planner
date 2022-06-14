# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import sys
import os, math, sys
from os.path import *
import numpy as np
import numpy.random as npr

import os
import multiprocessing
import subprocess

try:
    from . import gen_xyz
    from . import gen_sdf
    from . import convert_sdf
    from . import gen_convex_shape
    from . import blender_process
except:
    pass
import platform
import shutil
import trimesh
from pathlib import Path

PYTHON2 = True
if platform.python_version().startswith("3"):
    input_func = input
else:
    input_func = raw_input


def clean_file(dir_list):
    for dir in dir_list:
        if os.path.isdir(dir):
            for file in os.listdir(dir):
                if (
                    file.endswith("model_normalized.obj")
                    or file.endswith("textured_simple.obj")
                    or file.endswith("png")
                    or file.endswith("npy")
                ):
                    continue

                file = os.path.join(dir, file)
                os.system("rm {}".format(file))
        else:
            print("not a dir", dir)


def rename_file(dir_list, save_dir):
    for dir in dir_list:
        if os.path.isdir(dir):
            for file in os.listdir(dir):
                if file.endswith("textured_simple.obj"):
                    file = os.path.join(dir, file)
                    new_file = file.replace("textured_simple", "model_normalized")
                    os.system("cp {} {}".format(file, new_file))
        else:
            print("not a dir", dir)


def cp_urdf(dir_list):
    for dir in dir_list:
        if os.path.isdir(dir):
            for file in os.listdir(dir):
                if "model_normalized" in file:
                    file = os.path.join(dir, "model_normalized.urdf")
                    os.system("cp data/objects/model_normalized.urdf {}".format(file))
        else:
            print("not a dir", dir)


def cp_convex_urdf(dir_list):
    for dir in dir_list:
        if os.path.isdir(dir):
            for file in os.listdir(dir):
                if "model_normalized" in file:
                    file = os.path.join(dir, "model_normalized.urdf")
                    os.system("cp data/objects/model_normalized.urdf {}".format(file))
        else:
            print("not a dir", dir)


import argparse

parser = argparse.ArgumentParser(
    "See README for used 3rd party packages to generate new mesh (importantly"
    "grasp is required to be used as target "
)
parser.add_argument("--mesh_root", help="root directory for meshes", default="/data/manifolds/acronym/meshes/Book")
parser.add_argument("-o", "--overwrite", help="overwrite save root", action="store_true")
parser.add_argument(
    "-f",
    "--file",
    help="filename, needs to be in data/objects and contain model_normalized.obj",
    type=str,
    default="",
)
parser.add_argument("-a", "--all", help="generate all", action="store_true")
parser.add_argument("-c", "--clean", help="clean all", action="store_true")
parser.add_argument("--xyz", help="generate extent and point file", action="store_true")
parser.add_argument("--sdf", help="generate sdf", action="store_true")
parser.add_argument(
    "--convex", help="convexify objects for bullet", action="store_true"
)
parser.add_argument(
    "--blender", help="use blender to fix topology for broken mesh", action="store_true"
)
parser.add_argument("--urdf", help="copy uniform urdf for bullet", action="store_true")
parser.add_argument(
    "--rename",
    help="rename to model_normalized file (shared name structure)",
    action="store_true",
)

args = parser.parse_args()

save_root = f'{args.mesh_root}/../meshes_bullet'
if not os.path.exists(save_root):
    os.mkdir(save_root)

# Iterate over all meshes in acronym/meshes and process them into /outdir/objname_meshid/
for obj_name in os.listdir(args.mesh_root):
    for mesh_file in os.listdir(f"{args.mesh_root}/{obj_name}"):
        mesh_id = mesh_file.replace('.obj', '')

        for grasp_file in os.listdir(f'{args.mesh_root}/../grasps/'):
            if mesh_id in grasp_file:
                scale = grasp_file.replace('.h5', '').split('_')[-1]
                # mesh_path = f"{args.save_root}/{obj_name}_{mesh_id}"
                mesh_path = f"{save_root}/{obj_name}_{mesh_id}_{scale}"

                if os.path.exists(mesh_path):
                    if not args.overwrite and len(os.listdir(mesh_path)) == 9:
                        print(f"skipping {mesh_path}")
                        continue
                    print(f"overwriting {mesh_path}")
                    shutil.rmtree(mesh_path)
                os.mkdir(mesh_path)

                os.system(f"cp {args.mesh_root}/{obj_name}/{mesh_file} {mesh_path}/model_normalized_unscaled.obj")

                # Acronym objects need to be scaled
                mesh = trimesh.load(f'{mesh_path}/model_normalized_unscaled.obj')
                scale = float(Path(mesh_path).parts[-1].split('_')[-1])
                mesh = mesh.apply_scale(scale)
                mesh.export(f'{mesh_path}/model_normalized.obj')

                gen_xyz.generate_extents_points(random_paths=[mesh_path])

                ####### The object SDF is required for CHOMP Scene
                gen_sdf.gen_sdf(random_paths=[mesh_path])
                convert_sdf.convert_sdf([mesh_path])

                ####### These two are mainly for rendering and simulation, needs update urdf if used in bullet
                ####### This can be used for meshes with broken topology and add textures uvs
                blender_process.process_obj(mesh_path)

                ####### The convex shape can be used for bullet.
                try:
                    gen_convex_shape.convexify_model_subprocess([mesh_path])
                except Exception as e:
                    print(e)
                    print("=================> need vhacd from bullet, see README")
                
                cp_urdf([mesh_path])
