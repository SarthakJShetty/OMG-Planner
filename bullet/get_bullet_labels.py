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

# For backprojection
import cv2
import matplotlib.pyplot as plt
# import glm
import open3d as o3d

# For 6-DOF graspnet
# import torch
# import grasp_estimator
# from utils import utils as gutils
# from utils import visualization_utils
# import mayavi.mlab as mlab
# mlab.options.offscreen = True

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", help="renders", action="store_true")
    parser.add_argument("-w", "--write_video", help="write video", action="store_true")
    parser.add_argument("--egl", help="use egl render", action="store_true")
    parser.add_argument("-mr", "--mesh_root", help="mesh root", type=str, default="/data/manifolds/acronym")
    parser.add_argument("-gr", "--grasp_root", help="grasp root", type=str, default="/data/manifolds/acronym/grasps") 
    parser.add_argument("-o", "--objects", help="objects to load", nargs='+', default=[]) 

    parser.add_argument("-ge", "--grasp_eval", help="grasps to evaluate", type=str, default="")

    parser.add_argument("-od", "--out_dir", help="output directory", type=str, default="/data/manifolds/acronym/bullet_grasps")
    
    parser.add_argument("-ow", "--overwrite", help="overwrite csv", action="store_true")

    args = parser.parse_args()

    # Check if csv exists in out dir
    mkdir_if_missing(args.out_dir)
    csvfile = f'{args.out_dir}/grasps.csv'
    if os.path.exists(csvfile):
        if args.overwrite:
            print("WARNING: overwriting bullet_grasps")
            os.remove(csvfile)
            shutil.rmtree(f'{args.out_dir}/output_videos')
            mkdir_if_missing(f'{args.out_dir}/output_videos')
            with open(csvfile, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['grasp_path', 'grasp_i', 'success_label', 'pybullet_success', 'err_info', 'pq', 'query_type'])
        else:
            print("ERROR: grasps.csv exists. Use overwrite flag to overwrite.")
            sys.exit(0)
    else:
        mkdir_if_missing(f'{args.out_dir}/output_videos')
        with open(csvfile, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['grasp_path', 'grasp_i', 'success_label', 'pybullet_success', 'err_info', 'pq', 'query_type'])

    # gripper faces down
    init_joints = [-0.331397102886776,
                -0.4084196878153061,
                0.27958348738965677,
                -2.100158152658163,
                0.10297468164759681,
                1.7272912586478788,
                0.6990357829811057,
                # 0.0, # already inserted by reset 
                0.0399,
                0.0399,]

    # setup bullet env
    env = PandaYCBEnv(renders=args.vis, egl_render=args.egl, gravity=False)
    # p.setRealTimeSimulation(0)

    # Load acronym object and grasps
    objects = deepcopy(args.objects)
    
    grasp_paths = [] # path to grasp file for a given object
    for fn in os.listdir(args.grasp_root):
        if objects == []: # save all objects
            grasp_paths.append((fn, None))
        else:
            for objname in objects:
                if objname in fn:
                    grasp_paths.append((fn, objname))
                    objects.remove(objname) # Only load first scale of object for now

    # Load objects
    object_infos = []
    for grasp_path, objname in grasp_paths:
        obj_mesh, obj_scale = load_mesh(f"{args.grasp_root}/{grasp_path}", mesh_root_dir=args.mesh_root, ret_scale=True)
        object_infos.append((objname, obj_scale, obj_mesh))
    
    env.reset(init_joints=init_joints, object_infos=object_infos)
    initpos, initorn = p.getLinkState(env._panda.pandaUid, env._panda.pandaEndEffectorIndex)[:2]
    initpos = list(initpos)

    # Adjust init joints
    # IPython.embed()
    # pos, orn = p.getLinkState(
    #     env._panda.pandaUid, env._panda.pandaEndEffectorIndex
    # )[:2]
    # pos = list(pos)
    # pos[0] += 0.2
    # ik = env._panda.solveInverseKinematics(pos, orn)
    # env._panda.setTargetPositions(ik)
    # for i in range(200):
    #     p.stepSimulation()
    # out = env._panda.getJointStates()[0]
    # print(out)

    # Correct for panda gripper rotation about z axis in bullet
    rotgrasp2grasp_T = pt.transform_from(pr.matrix_from_axis_angle([0, 0, 1, -np.pi/2]), [0, 0, 0])

    for i, (grasp_path, _) in enumerate(grasp_paths):
        if args.write_video:
            video_writer_all = cv2.VideoWriter(
                f'{args.out_dir}/output_videos/{grasp_path}_all.avi',
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (640, 480),
            )

        # load grasps
        if args.grasp_eval == '': # load acronym grasps
            obj2rotgrasp_Ts, successes = load_grasps(f"{args.grasp_root}/{grasp_path}")
        else: # load predicted grasps
            obj2rotgrasp_Ts, successes, query_types = load_eval_grasps(f"{args.grasp_eval}")
        
        data = h5py.File(f"{args.grasp_root}/{grasp_path}", "r+")
        bullet_group_name = 'grasps/qualities/bullet'
        if bullet_group_name not in data:
            bullet_group = data.create_group(bullet_group_name)
            # assert len(bullet_successes) == len(data['grasps/qualities/flex/object_in_gripper'])
            dset = bullet_group.create_dataset(f"object_in_gripper", (len(data['grasps/qualities/flex/object_in_gripper']),), dtype='i8')
        else:
            dset = data[f"{bullet_group_name}/object_in_gripper"]

        # bullet_successes = []
        for idx, (obj2rotgrasp_T, success) in enumerate(zip(obj2rotgrasp_Ts, successes)):
            pq = pt.pq_from_transform(obj2rotgrasp_T)
            print(f"{idx} success label: {success}\tpose: {pq}")

            env._panda.reset(init_joints)
            env.reset_objects()
            
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

            if False: # Debugging plot
                pos_grasp = [create_gripper_marker(color=[0, 255, 0]).apply_transform(obj2rotgrasp_T)]
                obj_mesh = obj_mesh.apply_transform(obj2ctr_T)
                trimesh.Scene([obj_mesh] + pos_grasp).show()

            world2ctr_T = world2ee_T @ grasp2obj_T @ obj2ctr_T
            # draw_pose(world2ctr_T)

            # Place single object
            pq = pt.pq_from_transform(world2ctr_T)
            p.resetBasePositionAndOrientation(
                env._objectUids[i],
                pq[:3],
                pr.quaternion_xyzw_from_wxyz(pq[3:])
            )  # xyzw
            p.resetBaseVelocity(env._objectUids[i], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

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

            if args.write_video and rew == 0: 
                video_writer = cv2.VideoWriter(
                    f'{args.out_dir}/output_videos/{grasp_path}_{idx}.avi',
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    1.0, # set low framerate so that videos with low frames will actually save
                    (640, 480),
                )
                for obs in observations:
                    video_writer.write(obs[0][:, :, [2, 1, 0]].astype(np.uint8))
                    video_writer_all.write(obs[0][:, :, [2, 1, 0]].astype(np.uint8))
                    
                # Convert to high quality gif
                # os.system(f'ffmpeg -y -i output_videos/{exp_name}/{scene_file}/bullet.avi -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output_videos/{exp_name}/{scene_file}/scene.gif')

            # log pybullet success for this object and pose
            print(f"\tsuccess actual: {rew}")
            # bullet_successes.append(rew)
            if args.grasp_eval != '':
                query_type = query_types[idx]
            else:
                query_type = 'pos' if success else 'neg'
            with open(csvfile, 'a') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([grasp_path, idx, success, rew, err_info, pq, query_type])

            dset[idx] = rew
            # import IPython; IPython.embed()


# RGBA = namedtuple('RGBA', ['red', 'green', 'blue', 'alpha'])
# BLACK = RGBA(0, 0, 0, 1)
# CLIENT = 0
# NULL_ID = -1
# BASE_LINK = -1

# # https://github.com/caelan/pybullet-planning/blob/master/pybullet_tools/utils.py
# def add_line(start, end, color=BLACK, width=1, lifetime=0, parent=NULL_ID, parent_link=BASE_LINK):
#     assert (len(start) == 3) and (len(end) == 3)
#     return p.addUserDebugLine(start, end, lineColorRGB=color[:3], lineWidth=width,
#                             lifeTime=lifetime, parentObjectUniqueId=parent, parentLinkIndex=parent_link,
#                             physicsClientId=CLIENT)

# def draw_pose(T, length=0.1, d=3, **kwargs):
#     origin_world = T @ np.array((0., 0., 0., 1.))
#     for k in range(d):
#         axis = np.array((0., 0., 0., 1.))
#         axis[k] = 1*length
#         axis_world = T @ axis
#         origin_pt = origin_world[:3] 
#         axis_pt = axis_world[:3] 
#         color = np.zeros(3)
#         color[k] = 1
#         add_line(origin_pt, axis_pt, color=color, **kwargs)

# class PandaYCBEnv:
#     """Class for panda environment with ycb objects.
#     adapted from kukadiverse env in pybullet
#     """

#     def __init__(
#         self,
#         urdfRoot=pybullet_data.getDataPath(),
#         actionRepeat=130,
#         isEnableSelfCollision=True,
#         renders=False,
#         isDiscrete=False,
#         maxSteps=800,
#         dtheta=0.1,
#         blockRandom=0.5,
#         target_obj=[1, 2, 3, 4, 10, 11],  # [1,2,4,3,10,11],
#         all_objs=[0, 1, 2, 3, 4, 8, 10, 11],
#         cameraRandom=0,
#         width=640,
#         height=480,
#         numObjects=8,
#         safeDistance=0.13,
#         random_urdf=False,
#         egl_render=False,
#         gui_debug=True,
#         cache_objects=False,
#         isTest=False,
#         gravity=True
#     ):
#         """Initializes the pandaYCBObjectEnv.

#         Args:
#             urdfRoot: The diretory from which to load environment URDF's.
#             actionRepeat: The number of simulation steps to apply for each action.
#             isEnableSelfCollision: If true, enable self-collision.
#             renders: If true, render the bullet GUI.
#             isDiscrete: If true, the action space is discrete. If False, the
#                 action space is continuous.
#             maxSteps: The maximum number of actions per episode.
#             blockRandom: A float between 0 and 1 indicated block randomness. 0 is
#                 deterministic.
#             cameraRandom: A float between 0 and 1 indicating camera placement
#                 randomness. 0 is deterministic.
#             width: The observation image width.
#             height: The observation image height.
#             numObjects: The number of objects in the bin.
#             isTest: If true, use the test set of objects. If false, use the train
#                 set of objects.
#         """

#         self._timeStep = 1.0 / 1000.0
#         self._urdfRoot = urdfRoot
#         self._observation = []
#         self._renders = renders
#         self._maxSteps = maxSteps
#         self._actionRepeat = actionRepeat
#         self._env_step = 0
#         self._gravity = gravity

#         self._cam_dist = 1.3
#         # self._cam_dist = 2
#         self._cam_yaw = 180
#         self._cam_pitch = -41
#         self._safeDistance = safeDistance
#         self._root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
#         self._p = p
#         self._target_objs = target_obj
#         self._all_objs = all_objs
#         self._window_width = width
#         self._window_height = height
#         self._blockRandom = blockRandom
#         self._cameraRandom = cameraRandom
#         self._numObjects = numObjects
#         self._shift = [0.5, 0.5, 0.5]  # to work without axis in DIRECT mode
#         # self._shift = [0., 0., 0.]  # to work without axis in DIRECT mode
#         self._egl_render = egl_render

#         self._cache_objects = cache_objects
#         self._object_cached = False
#         self._gui_debug = gui_debug
#         self.target_idx = 0
#         self.connect()

#     def connect(self):
#         if self._renders:
#             self.cid = p.connect(p.SHARED_MEMORY)
#             if self.cid < 0:
#                 self.cid = p.connect(p.GUI)
#                 # p.resetDebugVisualizerCamera(self._cam_dist, self.cam_yaw, self.cam_pitch, [-0.35, -0.58, -0.88])
#                 p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [-0.0, -0.58, -0.88])
#             if not self._gui_debug:
#                 p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

#         else:
#             self.cid = p.connect(p.DIRECT)

#         egl = pkgutil.get_loader("eglRenderer")
#         if self._egl_render and egl:
#             p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

#         self.connected = True

#     def disconnect(self):
#         p.disconnect()
#         self.connected = False

#     def cache_objects(self, object_infos=[]):
#         """
#         Load all YCB objects and set up (only work for single apperance)
#         """
#         obj_path = self._root_dir + "/data/objects/"
#         if object_infos == []:
#             objects = sorted([m for m in os.listdir(obj_path) if m.startswith("0")])
#             paths = ["data/objects/" + objects[i] for i in self._all_objs]
#             scales = [1 for i in range(len(paths))]
#         else:
#             objects = []
#             paths = []
#             scales = []
#             fnames = os.listdir(obj_path)
#             for name, scale, _ in object_infos:
#                 obj = next(filter(lambda x: x.startswith(name), fnames), None)
#                 if obj != None:
#                     objects.append(obj)
#                     paths.append('data/objects/' + obj)
#                     scales.append(scale)
#             # objects = sorted[m for m in os.listdir(obj_path) if m.startswith("Book")] # hardcode to load book object
#         # ycb_path = ['data/objects/' + ycb_objects[i] for i in [0]]
#         # scale = 0.0268818134 # from load_mesh in acronym

#         pose = np.zeros([len(paths), 3])
#         pose[:, 0] = -2.0 - np.linspace(0, 4, len(paths))  # place in the back
#         pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
#         paths = [p_.strip() for p_ in paths]
#         objectUids = []
#         self.obj_path = paths  
#         self.cache_object_poses = []
#         self.cache_object_extents = []

#         for i, name in enumerate(paths):
#             trans = pose[i] + np.array(pos)  # fixed position
#             self.cache_object_poses.append((trans.copy(), np.array(orn).copy()))
#             uid = self._add_mesh(
#                 os.path.join(self._root_dir, name, "model_normalized.urdf"), trans, orn, scale=scales[i]
#             )  # xyzw
#             objectUids.append(uid)
#             self.cache_object_extents.append(
#                 np.loadtxt(
#                     os.path.join(self._root_dir, name, "model_normalized.extent.txt")
#                 )
#             )
#             p.setCollisionFilterPair(
#                 uid, self.plane_id, -1, -1, 0
#             )  # unnecessary simulation effort
#             p.changeDynamics(
#                 uid,
#                 -1,
#                 restitution=0.1,
#                 mass=0.5,
#                 spinningFriction=0,
#                 rollingFriction=0,
#                 lateralFriction=0.9,
#             )

#         self._object_cached = True
#         # In this code we don't use cached_objects for the location in a scene,
#         # We use it as the reset position. So any objects added are always True cache
#         # self.cached_objects = [False] * len(self.obj_path)
#         self.cached_objects = [True] * len(self.obj_path)

#         return objectUids

#     def reset(self, init_joints=None, scene_file=None, object_infos=[]):
#         """Environment reset"""

#         # Set the camera settings.
#         look = [
#             -0.35,
#             -0.58,
#             -0.88,
#         ]   
#         distance = self._cam_dist   
#         pitch = self._cam_pitch
#         yaw = self._cam_yaw  
#         roll = 0
#         fov = 60.0 + self._cameraRandom * np.random.uniform(-2, 2)
#         self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
#             look, distance, yaw, pitch, roll, 2
#         )

#         aspect = float(self._window_width) / self._window_height

#         self.near = 0.5
#         self.far = 6
#         self._proj_matrix = p.computeProjectionMatrixFOV(
#             fov, aspect, self.near, self.far
#         )

#         # Set table and plane
#         p.resetSimulation()
#         p.setTimeStep(self._timeStep)

#         # Intialize robot and objects
#         if self._gravity:
#             p.setGravity(0, 0, -9.81)
#         p.stepSimulation()
#         if init_joints is None:
#             self._panda = Panda(stepsize=self._timeStep, base_shift=self._shift)
#         else:
#             self._panda = Panda(
#                 stepsize=self._timeStep, init_joints=init_joints, base_shift=self._shift
#             )
#             for _ in range(1000):
#                 p.stepSimulation()

#         plane_file =  "data/objects/floor" 
#         # table_file =   "data/objects/table/models"
#         self.plane_id = p.loadURDF(
#             os.path.join(plane_file, 'model_normalized.urdf'), 
#             [0 - self._shift[0], 0 - self._shift[1], -0.82 - self._shift[2]]
#         )
#         # self.table_id = p.loadURDF(
#         #     os.path.join(table_file, 'model_normalized.urdf'),
#         #     0.5 - self._shift[0],
#         #     0.0 - self._shift[1],
#         #     -0.82 - self._shift[2],
#         #     0.707,
#         #     0.0,
#         #     0.0,
#         #     0.707,
#         # )

#         if not self._object_cached:
#             self._objectUids = self.cache_objects(object_infos=object_infos)

#         # self.obj_path += [plane_file, table_file]
#         self.obj_path += [plane_file]
#         # self._objectUids += [self.plane_id, self.table_id]
#         self._objectUids += [self.plane_id]
        
#         self._env_step = 0
#         return self._get_observation()

#     def reset_objects(self):
#         for idx, obj in enumerate(self._objectUids): 
#             if idx >= len(self.cached_objects): continue
#             if self.cached_objects[idx]:
#                 p.resetBasePositionAndOrientation(
#                     obj,
#                     self.cache_object_poses[idx][0],
#                     self.cache_object_poses[idx][1],
#                 )
#             # self.cached_objects[idx] = False

#     def cache_reset(self, init_joints=None, scene_file=None):
#         self._panda.reset(init_joints)
#         self.reset_objects()

#         if scene_file is None or not os.path.exists(scene_file):
#             self._randomly_place_objects(self._get_random_object(self._numObjects))
#         else:
#             self.place_objects_from_scene(scene_file)
#         self._env_step = 0
#         self.obj_names, self.obj_poses = self.get_env_info()
#         return self._get_observation()

#     def place_objects_from_scene(self, scene_file):
#         """place objects with pose based on the scene file"""
#         scene = sio.loadmat(scene_file)
#         poses = scene["pose"]
#         path = scene["path"]

#         pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
#         objectUids = []
#         objects_paths = [
#             _p.strip() for _p in path if "table" not in _p and "floor" not in _p
#         ]

#         for i, name in enumerate(objects_paths):
#             obj_idx = self.obj_path.index(name)
#             pose = poses[i]
#             trans = pose[:3, 3] + np.array(pos)  # fixed position
#             orn = ros_quat(mat2quat(pose[:3, :3]))
#             p.resetBasePositionAndOrientation(self._objectUids[obj_idx], trans, orn)
#             self.cached_objects[obj_idx] = True
#         if 'target_name' in scene: 
#             target_idx = [idx for idx, name in enumerate(objects_paths) if 
#                                    str(scene['target_name'][0]) in str(name)][0]
#         else:
#             target_idx = 0
#         self.target_idx = self.obj_path.index(objects_paths[target_idx])
#         if "states" in scene:
#             init_joints = scene["states"][0]
#             self._panda.reset(init_joints)
        
#         for _ in range(2000):
#             p.stepSimulation()
#         return objectUids

#     def _check_safe_distance(self, xy, pos, obj_radius, radius):
#         dist = np.linalg.norm(xy - pos, axis=-1)
#         safe_distance = obj_radius + radius -0.02 # avoid being too conservative
#         return not np.any(dist < safe_distance)

#     def _randomly_place_objects(self, urdfList, scale=1, poses=None):
#         """
#         Randomize positions of each object urdf.
#         """

#         xpos = 0.6 + 0.2 * (self._blockRandom * random.random() - 0.5) - self._shift[0]
#         ypos = 0.5 * self._blockRandom * (random.random() - 0.5) - self._shift[0]
#         orn = p.getQuaternionFromEuler([0, 0, 0])  #
#         p.resetBasePositionAndOrientation(
#             self._objectUids[self.target_idx],
#             [xpos, ypos, -0.44 - self._shift[2]],
#             [orn[0], orn[1], orn[2], orn[3]],
#         )
#         p.resetBaseVelocity(
#             self._objectUids[self.target_idx], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
#         )
#         self.cached_objects[self.obj_path.index(urdfList[0])] = True

#         for _ in range(3000):
#             p.stepSimulation()
#         pos, _ = p.getLinkState(
#             self._panda.pandaUid, self._panda.pandaEndEffectorIndex
#         )[:2]
#         pos = np.array([[pos[0], pos[1]], [xpos, ypos]])
#         k = 0
#         max_cnt = 50
#         obj_radius = [
#             0.05,
#             np.max(self.cache_object_extents[self.obj_path.index(urdfList[0])]) / 2,
#         ]
#         assigned_orn = [orn]
#         placed_indexes = [self.target_idx]

#         for i, name in enumerate(urdfList[1:]):
#             obj_idx = self.obj_path.index(name)
#             radius = np.max(self.cache_object_extents[obj_idx]) / 2

#             cnt = 0
#             if self.cached_objects[obj_idx]:
#                 continue

#             while cnt < max_cnt:
#                 cnt += 1
#                 xpos_ = xpos - self._blockRandom * 1.0 * random.random()
#                 ypos_ = ypos - self._blockRandom * 3 * (random.random() - 0.5)  # 0.5
#                 xy = np.array([[xpos_, ypos_]])
#                 if (
#                     self._check_safe_distance(xy, pos, obj_radius, radius)
#                     and (
#                         xpos_ > 0.35 - self._shift[0] and xpos_ < 0.65 - self._shift[0]
#                     )
#                     and (
#                         ypos_ < 0.20 - self._shift[1] and ypos_ > -0.20 - self._shift[1]
#                     )
#                 ):  # 0.15
#                     break
#             if cnt == max_cnt:
#                 continue  # check 1

#             xpos_ = xpos_ + 0.05  # closer and closer to the target
#             angle = np.random.uniform(-np.pi, np.pi)
#             orn = p.getQuaternionFromEuler([0, 0, angle])
#             p.resetBasePositionAndOrientation(
#                 self._objectUids[obj_idx],
#                 [xpos, ypos_, -0.44 - self._shift[2]],
#                 [orn[0], orn[1], orn[2], orn[3]],
#             )  # xyzw

#             p.resetBaseVelocity(
#                 self._objectUids[obj_idx], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
#             )
#             for _ in range(3000):
#                 p.stepSimulation()

#             _, new_orn = p.getBasePositionAndOrientation(self._objectUids[obj_idx])
#             ang = (
#                 np.arccos(2 * np.power(np.dot(tf_quat(orn), tf_quat(new_orn)), 2) - 1)
#                 * 180.0
#                 / np.pi
#             )
#             if ang > 20:
#                 p.resetBasePositionAndOrientation(
#                     self._objectUids[obj_idx],
#                     [xpos, ypos, -10.0 - self._shift[2]],
#                     [orn[0], orn[1], orn[2], orn[3]],
#                 )
#                 continue  # check 2
#             self.cached_objects[obj_idx] = True
#             obj_radius.append(radius)
#             pos = np.concatenate([pos, xy], axis=0)
#             xpos = xpos_
#             placed_indexes.append(obj_idx)
#             assigned_orn.append(orn)

#         for _ in range(10000):
#             p.stepSimulation()

#         return True

#     def _get_random_object(self, num_objects, ycb=True):
#         """
#         Randomly choose an object urdf from the selected objects
#         """

#         self.target_idx = self._all_objs.index(
#             self._target_objs[np.random.randint(0, len(self._target_objs))]
#         )  #
#         obstacle = np.random.choice(
#             range(len(self._all_objs)), self._numObjects - 1
#         ).tolist()
#         selected_objects = [self.target_idx] + obstacle
#         selected_objects_filenames = [
#             self.obj_path[selected_object] for selected_object in selected_objects
#         ]
#         return selected_objects_filenames

#     def retract(self, record=False):
#         """Retract step."""
#         cur_joint = np.array(self._panda.getJointStates()[0])
#         cur_joint[-2:] = 0

#         self.step(cur_joint.tolist())  # grasp
#         pos, orn = p.getLinkState(
#             self._panda.pandaUid, self._panda.pandaEndEffectorIndex
#         )[:2]
#         observations = []
#         for i in range(10):
#             pos = (pos[0], pos[1], pos[2] + 0.03)
#             jointPoses = np.array(
#                 p.calculateInverseKinematics(
#                     self._panda.pandaUid, self._panda.pandaEndEffectorIndex, pos
#                 )
#             )
#             jointPoses[-2:] = 0.0

#             self.step(jointPoses.tolist())
#             observation = self._get_observation()
#             if record:
#                 observations.append(observation)

#         return (self._reward(), observations) if record else self._reward()

#     def step(self, action, obs=True):
#         """Environment step."""
#         self._env_step += 1
#         self._panda.setTargetPositions(action)
#         for _ in range(self._actionRepeat):
#             p.stepSimulation()
#             if self._renders:
#                 time.sleep(self._timeStep)
#         if not obs:
#             observation = None
#         else:
#             observation = self._get_observation()
#         done = self._termination()
#         reward = self._reward()

#         return observation, reward, done, None

#     def _get_observation(self):
#         _, _, rgba, depth, mask = p.getCameraImage(
#             width=self._window_width,
#             height=self._window_height,
#             viewMatrix=self._view_matrix,
#             projectionMatrix=self._proj_matrix,
#             physicsClientId=self.cid,
#         )

#         depth = self.far * self.near / (self.far - (self.far - self.near) * depth)
#         joint_pos, joint_vel = self._panda.getJointStates()
#         obs = np.concatenate(
#             [rgba[..., :3], depth[..., None], mask[..., None]], axis=-1
#         )
#         return (obs, joint_pos)

#     def get_pc(self):
#         target_id = env._objectUids[env.target_idx]
#         _, _, rgba, depth, mask = p.getCameraImage(
#             width=self._window_width,
#             height=self._window_height,
#             viewMatrix=self._view_matrix,
#             projectionMatrix=self._proj_matrix,
#             physicsClientId=self.cid,
#         )
#         rgba = rgba / 255.0

#         imgH, imgW = depth.shape
#         projGLM = np.asarray(self._proj_matrix).reshape([4, 4], order='F')
#         view = np.asarray(self._view_matrix).reshape([4, 4], order='F')

#         pos, orn = p.getBasePositionAndOrientation(env._panda.pandaUid)

#         all_pc = []
#         stepX = 1
#         stepY = 1
#         for h in range(0, imgH, stepY):
#             for w in range(0, imgW, stepX):
#                 win = glm.vec3(w, imgH - h, depth[h][w])
#                 position = glm.unProject(win, glm.mat4(view), glm.mat4(projGLM), glm.vec4(0, 0, imgW, imgH))
#                 all_pc.append([position[0], position[1], position[2], rgba[h, w, 0], rgba[h, w, 1], rgba[h, w, 2]])

#         all_pc = np.array(all_pc)
#         all_pc[:, :3] -= pos
        
#         all_pcd = o3d.geometry.PointCloud()
#         all_pcd.points = o3d.utility.Vector3dVector(all_pc[:, :3])
#         all_pcd.colors = o3d.utility.Vector3dVector(all_pc[:, 3:])

#         pc = []
#         obj_idxs = np.where(mask == target_id)
#         for i in range(len(obj_idxs[0])):
#             h = obj_idxs[0][i]
#             w = obj_idxs[1][i]
#             win = glm.vec3(w, imgH - h, depth[h][w])
#             position = glm.unProject(win, glm.mat4(view), glm.mat4(projGLM), glm.vec4(0, 0, imgW, imgH))
#             pc.append([position[0], position[1], position[2], rgba[h, w, 0], rgba[h, w, 1], rgba[h, w, 2]])
#         pc = np.array(pc)
#         pc[:, :3] -= pos
        
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
#         pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:])
#         # print("Visualizing point cloud...")
#         # o3d.visualization.draw_geometries([pcd])
#         # downpcd = pcd.voxel_down_sample(voxel_size=0.05)
#         # o3d.visualization.draw_geometries([downpcd])
#         # print("Point cloud visualized.")
#         return pcd, all_pcd

#     def _get_target_obj_pose(self):
#         return p.getBasePositionAndOrientation(self._objectUids[self.target_idx])[0]

#     def _reward(self):
#         """Calculates the reward for the episode.

#         The reward is 1 if one of the objects is above height .2 at the end of the
#         episode.
#         """
#         reward = 0
#         hand_pos, _ = p.getLinkState(
#             self._panda.pandaUid, self._panda.pandaEndEffectorIndex
#         )[:2]
#         pos, _ = p.getBasePositionAndOrientation(
#             self._objectUids[self.target_idx]
#         )  # target object
#         grip_pos = self._panda.getJointStates()[0][-2:]
#         if (
#             # np.linalg.norm(np.subtract(pos, hand_pos)) < 0.2
#             min(grip_pos) > 0.001
#             # and pos[2] > -0.35 - self._shift[2]
#         ):
#             reward = 1
#         return reward

#     def _termination(self):
#         """Terminates the episode if we have tried to grasp or if we are above
#         maxSteps steps.
#         """
#         return self._env_step >= self._maxSteps

#     def _add_mesh(self, obj_file, trans, quat, scale=1):
#         try:
#             bid = p.loadURDF(obj_file, trans, quat, globalScaling=scale)
#             return bid
#         except:
#             print("load {} failed".format(obj_file))

#     def get_env_info(self ):
#         pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
#         base_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
#         poses = []
#         obj_dir = []

#         for idx, uid in enumerate(self._objectUids):
#             if   idx >= len(self.cached_objects)  or self.cached_objects[idx]:
#                 pos, orn = p.getBasePositionAndOrientation(uid)  # center offset of base
#                 obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
#                 poses.append(inv_relative_pose(obj_pose, base_pose))
#                 obj_dir.append(self.obj_path[idx])  # .encode("utf-8")

#         return obj_dir, poses

# def bullet_execute_plan(env, plan, write_video, video_writer):
#     print('executing...')
#     for k in range(plan.shape[0]):
#         obs, rew, done, _ = env.step(plan[k].tolist())
#         if write_video:
#             video_writer.write(obs[0][:, :, [2, 1, 0]].astype(np.uint8))
#     (rew, ret_obs) = env.retract(record=True)
#     if write_video: 
#         for robs in ret_obs:
#             video_writer.write(robs[0][:, :, [2, 1, 0]].astype(np.uint8))
#             video_writer.write(robs[0][:, :, [2, 1, 0]].astype(np.uint8)) # to get enough frames to save
#     return rew

    # for i in range(10000):
        # p.stepSimulation()
        # time.sleep(1./240.)

    # pos, orn = p.getLinkState(
    #     env._panda.pandaUid, env._panda.pandaEndEffectorIndex
    # )[:2]
    # pos = list(pos)
    # pos[2] += 0.4
    # ik = env._panda.solveInverseKinematics(pos, orn)
    # # env._panda.setTargetPositions(ik)
    # out = env.step(ik)
    # print(out)


    # pregrasp2grasp_T = np.eye(4)
    # pregrasp2grasp_T[2, 3] = 0.1
    # world2obj_T = world2ee_T @ pregrasp2grasp_T @ grasp2obj_T
    # draw_pose(world2ee_T @ pregrasp2grasp_T)

    # world2obj_T = pt.transform_from_pq(np.concatenate([pos, pr.quaternion_wxyz_from_xyzw(orn)]))
    # T_world_obj = np.eye(4)
    # T_world_obj[:3, 3] = -obj_mesh.centroid
    # obj_mesh = obj_mesh.apply_transform(T_world_obj)

    


    # # setup planner
    # cfg.traj_init = "grasp"
    # cfg.scene_file = args.file
 
    # cfg.vis = False
    # cfg.timesteps = 50  
    # cfg.get_global_param(cfg.timesteps)
    # scene = PlanningScene(cfg)
    # for i, name in enumerate(env.obj_path[:-2]):  # load all objects
    #     name = name.split("/")[-1]
    #     trans, orn = env.cache_object_poses[i]
    #     scene.env.add_object(name, trans, tf_quat(orn), compute_grasp=True)

    # scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1, 0, 0, 0]))
    # scene.env.add_table(np.array([0.55, 0, -0.17]), np.array([0.707, 0.707, 0.0, 0]))
    # scene.env.combine_sdfs()
    # # if args.experiment:   
    #     # scene_files = ['scene_{}'.format(i) for i in range(100)]
    #     # exp_name = f"{time.strftime('%Y-%m-%d/%H-%M-%S', time.localtime())}" + \
    #             #    f"_{'infgrasps' if args.use_graspnet else 'knowngrasps'}" + \
    #             #    f"_{args.grasp_selection}"
    #     # mkdir_if_missing(f'output_videos/{exp_name}')
    # # else:
    # scene_files = [scene_file]
    # exp_name = "dbg"
    
    # # if args.debug_traj and not args.use_graspnet: # does not work with use_graspnet due to EGL render issues with mayavi downstream
    # #     scene.setup_renderer()
    # #     init_traj = scene.planner.traj.data
    # #     init_traj_im = scene.fast_debug_vis(traj=init_traj, interact=0, write_video=False,
    # #                                 nonstop=False, collision_pt=False, goal_set=False, traj_idx=0)
    # #     init_traj_im = cv2.cvtColor(init_traj_im, cv2.COLOR_RGB2BGR)
    # #     cv2.imwrite(f"output_videos/{exp_name}/{scene_file}/traj_0.png", init_traj_im)

    # cnts, rews = 0, 0
    # for scene_file in scene_files:
    #     mkdir_if_missing(f'output_videos/{exp_name}/{scene_file}')
    #     config.cfg.output_video_name = f"output_videos/{exp_name}/{scene_file}/bullet.avi"
    #     cfg.scene_file = scene_file
    #     video_writer = None
    #     if args.write_video:
    #         video_writer = cv2.VideoWriter(
    #             config.cfg.output_video_name,
    #             cv2.VideoWriter_fourcc(*"MJPG"),
    #             10.0,
    #             (640, 480),
    #         )
    #     full_name = os.path.join('data/scenes', scene_file + ".mat")
    #     env.cache_reset(scene_file=full_name)
    #     obj_names, obj_poses = env.get_env_info()
    #     object_lists = [name.split("/")[-1].strip() for name in obj_names]
    #     object_poses = [pack_pose(pose) for pose in obj_poses]

    #     exists_ids, placed_poses = [], []
    #     for i, name in enumerate(object_lists[:-2]):  # update planning scene
    #         # if i == 0:
    #             # print(name)
    #         object_poses[i][-2] += 2.0
    #         object_poses[i][-1] += 2.0
    #         scene.env.update_pose(name, object_poses[i])
    #         obj_idx = env.obj_path[:-2].index("data/objects/" + name)
    #         exists_ids.append(obj_idx)
    #         trans, orn = env.cache_object_poses[obj_idx]
    #         placed_poses.append(np.hstack([trans, ros_quat(orn)]))

    #     cfg.disable_collision_set = [
    #         name.split("/")[-2]
    #         for obj_idx, name in enumerate(env.obj_path[:-2])
    #         if obj_idx not in exists_ids
    #     ]

    #     import IPython; IPython.embed()
    #     # import sys
    #     # sys.exit(1)

    #     # Get point clouds
    #     if args.use_graspnet:
    #         object_pc, all_pc = env.get_pc()

    #         # Predict grasp using 6-DOF GraspNet
    #         gpath = os.path.dirname(grasp_estimator.__file__)
    #         grasp_sampler_args = gutils.read_checkpoint_args(
    #             os.path.join(gpath, args.grasp_sampler_folder))
    #         grasp_sampler_args['checkpoints_dir'] = os.path.join(gpath, grasp_sampler_args['checkpoints_dir'])
    #         grasp_sampler_args.is_train = False
    #         grasp_evaluator_args = gutils.read_checkpoint_args(
    #             os.path.join(gpath, args.grasp_evaluator_folder))
    #         grasp_evaluator_args['checkpoints_dir'] = os.path.join(gpath, grasp_evaluator_args['checkpoints_dir'])
    #         grasp_evaluator_args.continue_train = True # was in demo file, not sure 
    #         estimator = grasp_estimator.GraspEstimator(grasp_sampler_args,
    #                                                     grasp_evaluator_args, args)

    #         pc = np.asarray(object_pc.points)
    #         start_time = time.time()
    #         generated_grasps, generated_scores = estimator.generate_and_refine_grasps(pc)
    #         graspinf_duration = time.time() - start_time
    #         print("Grasp Inference time: {:.3f}".format(graspinf_duration))

    #         generated_grasps = np.array(generated_grasps)
    #         generated_scores = np.array(generated_scores)

    #         # Set configs according to args
    #         if args.grasp_selection == 'Fixed':
    #             scene.planner.cfg.ol_alg = 'Baseline'
    #             scene.planner.cfg.goal_idx = -1
    #         elif args.grasp_selection == 'Proj':
    #             scene.planner.cfg.ol_alg = 'Proj'
    #         elif args.grasp_selection == 'OMG':
    #             scene.planner.cfg.ol_alg = 'MD'

    #         scene.env.set_target(env.obj_path[env.target_idx].split("/")[-1])
    #         scene.reset(lazy=True, grasps=generated_grasps, grasp_scores=generated_scores)
            
    #         dbg = np.load("output_videos/dbg.npy", encoding='latin1', allow_pickle=True)
    #         grasp_start, grasp_end, goal_idx, goal_set, goal_quality, grasp_ees = dbg # in joints, not EE pose
    #         offset_pose = np.array(rotZ(-np.pi / 2))  # unrotate gripper for visualization (was rotated in Planner class)
    #         goal_ees_T = [np.matmul(unpack_pose(g), offset_pose) for g in grasp_ees]
    #         # goal_ees_T = [unpack_pose(g) for g in grasp_ees]

    #         # Visualize
    #         # visualization_utils.draw_scene(
    #         #     np.asarray(all_pc.points),
    #         #     pc_color=(np.asarray(all_pc.colors) * 255).astype(int),
    #         #     grasps=goal_ees_T,
    #         #     grasp_scores=goal_quality,
    #         # )

    #         visualization_utils.draw_scene(
    #             np.asarray(object_pc.points),
    #             pc_color=(np.asarray(object_pc.colors) * 255).astype(int),
    #             grasps=[goal_ees_T[goal_idx]],
    #             grasp_scores=[goal_quality[goal_idx]],
    #         )
    #         mlab.savefig(f"output_videos/{exp_name}/{scene_file}/grasp.png")
    #         # mlab.clf()
    #         mlab.close()
    #         # import IPython; IPython.embed() # Ctrl-D for interactive visualization 
    #     else:
    #         # Set configs according to args
    #         if args.grasp_selection == 'Fixed':
    #             scene.planner.cfg.ol_alg = 'Baseline'
    #             scene.planner.cfg.goal_idx = -1
    #         elif args.grasp_selection == 'Proj':
    #             scene.planner.cfg.ol_alg = 'Proj'
    #         elif args.grasp_selection == 'OMG':
    #             scene.planner.cfg.ol_alg = 'MD'

    #         scene.env.set_target(env.obj_path[env.target_idx].split("/")[-1])
    #         scene.reset(lazy=True)

    #     info = scene.step()
    #     plan = scene.planner.history_trajectories[-1]

    #     # Visualize intermediate trajectories
    #     if args.debug_traj:
    #         if args.use_graspnet:
    #             scene.setup_renderer()
    #         for i, traj in enumerate(scene.planner.history_trajectories):
    #             traj_im = scene.fast_debug_vis(traj=traj, interact=0, write_video=False,
    #                                            nonstop=False, collision_pt=False, goal_set=True, traj_idx=i)
    #             traj_im = cv2.cvtColor(traj_im, cv2.COLOR_RGB2BGR)
    #             cv2.imwrite(f"output_videos/{exp_name}/{scene_file}/traj_{i+1}.png", traj_im)

    #     rew = bullet_execute_plan(env, plan, args.write_video, video_writer)
    #     for i, name in enumerate(object_lists[:-2]):  # reset planner
    #         scene.env.update_pose(name, placed_poses[i])
    #     cnts += 1
    #     rews += rew
    #     print('rewards: {} counts: {}'.format(rews, cnts))

    #     # Save data
    #     if args.use_graspnet and 'time' in info[-1].keys():
    #         info[-1]['time'] += graspinf_duration
    #     np.save(f'output_videos/{exp_name}/{scene_file}/data.npy', [rew, info, plan])

    #     # Convert avi to high quality gif 
    #     os.system(f'ffmpeg -y -i output_videos/{exp_name}/{scene_file}/bullet.avi -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output_videos/{exp_name}/{scene_file}/scene.gif')
    # env.disconnect()


# def make_parser(parser):
#     # parser = argparse.ArgumentParser(
#         # description='6-DoF GraspNet Demo',
#         # formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--grasp_sampler_folder',
#                         type=str,
#                         default='checkpoints/gan_pretrained/')
#     parser.add_argument('--grasp_evaluator_folder',
#                         type=str,
#                         default='checkpoints/evaluator_pretrained/')
#     parser.add_argument('--refinement_method',
#                         choices={"gradient", "sampling"},
#                         default='sampling')
#     parser.add_argument('--refine_steps', type=int, default=25)

#     parser.add_argument('--npy_folder', type=str, default='demo/data/')
#     parser.add_argument(
#         '--threshold',
#         type=float,
#         default=0.8,
#         help=
#         "When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed"
#     )
#     parser.add_argument(
#         '--choose_fn',
#         choices={
#             "all", "better_than_threshold", "better_than_threshold_in_sequence"
#         },
#         default='better_than_threshold',
#         help=
#         "If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps"
#     )

#     parser.add_argument('--target_pc_size', type=int, default=1024)
#     parser.add_argument('--num_grasp_samples', type=int, default=200)
#     parser.add_argument(
#         '--generate_dense_grasps',
#         action='store_true',
#         help=
#         "If enabled, it will create a [num_grasp_samples x num_grasp_samples] dense grid of latent space values and generate grasps from these."
#     )

#     parser.add_argument(
#         '--batch_size',
#         type=int,
#         default=30,
#         help=
#         "Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory."
#     )
#     parser.add_argument('--train_data', action='store_true')
#     # opts, _ = parser.parse_known_args()
#     # if opts.train_data:
#     #     parser.add_argument('--dataset_root_folder',
#     #                         required=True,
#     #                         type=str,
#     #                         help='path to root directory of the dataset.')
#     return parser