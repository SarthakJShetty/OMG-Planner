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
import glm
import open3d as o3d

# For 6-DOF graspnet
import torch
import grasp_estimator
from utils import utils as gutils
from utils import visualization_utils

class PandaYCBEnv:
    """Class for panda environment with ycb objects.
    adapted from kukadiverse env in pybullet
    """

    def __init__(
        self,
        urdfRoot=pybullet_data.getDataPath(),
        actionRepeat=130,
        isEnableSelfCollision=True,
        renders=False,
        isDiscrete=False,
        maxSteps=800,
        dtheta=0.1,
        blockRandom=0.5,
        target_obj=[1, 2, 3, 4, 10, 11],  # [1,2,4,3,10,11],
        all_objs=[0, 1, 2, 3, 4, 8, 10, 11],
        cameraRandom=0,
        width=640,
        height=480,
        numObjects=8,
        safeDistance=0.13,
        random_urdf=False,
        egl_render=False,
        gui_debug=True,
        cache_objects=False,
        isTest=False,
    ):
        """Initializes the pandaYCBObjectEnv.

        Args:
            urdfRoot: The diretory from which to load environment URDF's.
            actionRepeat: The number of simulation steps to apply for each action.
            isEnableSelfCollision: If true, enable self-collision.
            renders: If true, render the bullet GUI.
            isDiscrete: If true, the action space is discrete. If False, the
                action space is continuous.
            maxSteps: The maximum number of actions per episode.
            blockRandom: A float between 0 and 1 indicated block randomness. 0 is
                deterministic.
            cameraRandom: A float between 0 and 1 indicating camera placement
                randomness. 0 is deterministic.
            width: The observation image width.
            height: The observation image height.
            numObjects: The number of objects in the bin.
            isTest: If true, use the test set of objects. If false, use the train
                set of objects.
        """

        self._timeStep = 1.0 / 1000.0
        self._urdfRoot = urdfRoot
        self._observation = []
        self._renders = renders
        self._maxSteps = maxSteps
        self._actionRepeat = actionRepeat
        self._env_step = 0

        self._cam_dist = 1.3
        # self._cam_dist = 2
        self._cam_yaw = 180
        self._cam_pitch = -41
        self._safeDistance = safeDistance
        self._root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self._p = p
        self._target_objs = target_obj
        self._all_objs = all_objs
        self._window_width = width
        self._window_height = height
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
        self._numObjects = numObjects
        self._shift = [0.5, 0.5, 0.5]  # to work without axis in DIRECT mode
        # self._shift = [0., 0., 0.]  # to work without axis in DIRECT mode
        self._egl_render = egl_render

        self._cache_objects = cache_objects
        self._object_cached = False
        self._gui_debug = gui_debug
        self.target_idx = 0
        self.connect()

    def connect(self):
        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if self.cid < 0:
                self.cid = p.connect(p.GUI)
                # p.resetDebugVisualizerCamera(self._cam_dist, self.cam_yaw, self.cam_pitch, [-0.35, -0.58, -0.88])
                p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [-0.0, -0.58, -0.88])
            if not self._gui_debug:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        else:
            self.cid = p.connect(p.DIRECT)

        egl = pkgutil.get_loader("eglRenderer")
        if self._egl_render and egl:
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        self.connected = True

    def disconnect(self):
        p.disconnect()
        self.connected = False

    def cache_objects(self):
        """
        Load all YCB objects and set up (only work for single apperance)
        """
        obj_path = self._root_dir + "/data/objects/"
        ycb_objects = sorted([m for m in os.listdir(obj_path) if m.startswith("0")])
        ycb_path = ["data/objects/" + ycb_objects[i] for i in self._all_objs]

        pose = np.zeros([len(ycb_path), 3])
        pose[:, 0] = -2.0 - np.linspace(0, 4, len(ycb_path))  # place in the back
        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        objects_paths = [p_.strip() for p_ in ycb_path]
        objectUids = []
        self.obj_path = objects_paths  
        self.cache_object_poses = []
        self.cache_object_extents = []

        for i, name in enumerate(objects_paths):
            trans = pose[i] + np.array(pos)  # fixed position
            self.cache_object_poses.append((trans.copy(), np.array(orn).copy()))
            uid = self._add_mesh(
                os.path.join(self._root_dir, name, "model_normalized.urdf"), trans, orn
            )  # xyzw
            objectUids.append(uid)
            self.cache_object_extents.append(
                np.loadtxt(
                    os.path.join(self._root_dir, name, "model_normalized.extent.txt")
                )
            )
            p.setCollisionFilterPair(
                uid, self.plane_id, -1, -1, 0
            )  # unnecessary simulation effort
            p.changeDynamics(
                uid,
                -1,
                restitution=0.1,
                mass=0.5,
                spinningFriction=0,
                rollingFriction=0,
                lateralFriction=0.9,
            )

        self._object_cached = True
        self.cached_objects = [False] * len(self.obj_path)
        return objectUids

    def reset(self, init_joints=None, scene_file=None):
        """Environment reset"""

        # Set the camera settings.
        look = [
            -0.35,
            -0.58,
            -0.88,
        ]   
        distance = self._cam_dist   
        pitch = self._cam_pitch
        yaw = self._cam_yaw  
        roll = 0
        fov = 60.0 + self._cameraRandom * np.random.uniform(-2, 2)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            look, distance, yaw, pitch, roll, 2
        )

        aspect = float(self._window_width) / self._window_height

        self.near = 0.5
        self.far = 6
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov, aspect, self.near, self.far
        )

        # Set table and plane
        p.resetSimulation()
        p.setTimeStep(self._timeStep)

        # Intialize robot and objects
        p.setGravity(0, 0, -9.81)
        p.stepSimulation()
        if init_joints is None:
            self._panda = Panda(stepsize=self._timeStep, base_shift=self._shift)

        else:
            for _ in range(1000):
                p.stepSimulation()
            self._panda = Panda(
                stepsize=self._timeStep, init_joints=init_joints, base_shift=self._shift
            )

        plane_file =  "data/objects/floor" 
        table_file =   "data/objects/table/models"
        self.plane_id = p.loadURDF(
            os.path.join(plane_file, 'model_normalized.urdf'), 
            [0 - self._shift[0], 0 - self._shift[1], -0.82 - self._shift[2]]
        )
        self.table_id = p.loadURDF(
            os.path.join(table_file, 'model_normalized.urdf'),
            0.5 - self._shift[0],
            0.0 - self._shift[1],
            -0.82 - self._shift[2],
            0.707,
            0.0,
            0.0,
            0.707,
        )

        if not self._object_cached:
            self._objectUids = self.cache_objects()

        self.obj_path += [plane_file, table_file]

        self._objectUids += [self.plane_id, self.table_id]
        
        self._env_step = 0
        return self._get_observation()

    def reset_objects(self):
        for idx, obj in enumerate(self._objectUids): 
            if idx >= len(self.cached_objects): continue
            if self.cached_objects[idx]:
                p.resetBasePositionAndOrientation(
                    obj,
                    self.cache_object_poses[idx][0],
                    self.cache_object_poses[idx][1],
                )
            self.cached_objects[idx] = False

    def cache_reset(self, init_joints=None, scene_file=None):
        self._panda.reset(init_joints)
        self.reset_objects()

        if scene_file is None or not os.path.exists(scene_file):
            self._randomly_place_objects(self._get_random_object(self._numObjects))
        else:
            self.place_objects_from_scene(scene_file)
        self._env_step = 0
        self.obj_names, self.obj_poses = self.get_env_info()
        return self._get_observation()

    def place_objects_from_scene(self, scene_file):
        """place objects with pose based on the scene file"""
        scene = sio.loadmat(scene_file)
        poses = scene["pose"]
        path = scene["path"]

        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        objectUids = []
        objects_paths = [
            _p.strip() for _p in path if "table" not in _p and "floor" not in _p
        ]

        for i, name in enumerate(objects_paths):
            obj_idx = self.obj_path.index(name)
            pose = poses[i]
            trans = pose[:3, 3] + np.array(pos)  # fixed position
            orn = ros_quat(mat2quat(pose[:3, :3]))
            p.resetBasePositionAndOrientation(self._objectUids[obj_idx], trans, orn)
            self.cached_objects[obj_idx] = True
        if 'target_name' in scene: 
            target_idx = [idx for idx, name in enumerate(objects_paths) if 
                                   str(scene['target_name'][0]) in str(name)][0]
        else:
            target_idx = 0 
        self.target_idx = self.obj_path.index(objects_paths[target_idx])
        if "states" in scene:
            init_joints = scene["states"][0]
            self._panda.reset(init_joints)
        
        for _ in range(2000):
            p.stepSimulation()
        return objectUids

    def _check_safe_distance(self, xy, pos, obj_radius, radius):
        dist = np.linalg.norm(xy - pos, axis=-1)
        safe_distance = obj_radius + radius -0.02 # avoid being too conservative
        return not np.any(dist < safe_distance)

    def _randomly_place_objects(self, urdfList, scale=1, poses=None):
        """
        Randomize positions of each object urdf.
        """

        xpos = 0.6 + 0.2 * (self._blockRandom * random.random() - 0.5) - self._shift[0]
        ypos = 0.5 * self._blockRandom * (random.random() - 0.5) - self._shift[0]
        orn = p.getQuaternionFromEuler([0, 0, 0])  #
        p.resetBasePositionAndOrientation(
            self._objectUids[self.target_idx],
            [xpos, ypos, -0.44 - self._shift[2]],
            [orn[0], orn[1], orn[2], orn[3]],
        )
        p.resetBaseVelocity(
            self._objectUids[self.target_idx], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        )
        self.cached_objects[self.obj_path.index(urdfList[0])] = True

        for _ in range(3000):
            p.stepSimulation()
        pos, _ = p.getLinkState(
            self._panda.pandaUid, self._panda.pandaEndEffectorIndex
        )[:2]
        pos = np.array([[pos[0], pos[1]], [xpos, ypos]])
        k = 0
        max_cnt = 50
        obj_radius = [
            0.05,
            np.max(self.cache_object_extents[self.obj_path.index(urdfList[0])]) / 2,
        ]
        assigned_orn = [orn]
        placed_indexes = [self.target_idx]

        for i, name in enumerate(urdfList[1:]):
            obj_idx = self.obj_path.index(name)
            radius = np.max(self.cache_object_extents[obj_idx]) / 2

            cnt = 0
            if self.cached_objects[obj_idx]:
                continue

            while cnt < max_cnt:
                cnt += 1
                xpos_ = xpos - self._blockRandom * 1.0 * random.random()
                ypos_ = ypos - self._blockRandom * 3 * (random.random() - 0.5)  # 0.5
                xy = np.array([[xpos_, ypos_]])
                if (
                    self._check_safe_distance(xy, pos, obj_radius, radius)
                    and (
                        xpos_ > 0.35 - self._shift[0] and xpos_ < 0.65 - self._shift[0]
                    )
                    and (
                        ypos_ < 0.20 - self._shift[1] and ypos_ > -0.20 - self._shift[1]
                    )
                ):  # 0.15
                    break
            if cnt == max_cnt:
                continue  # check 1

            xpos_ = xpos_ + 0.05  # closer and closer to the target
            angle = np.random.uniform(-np.pi, np.pi)
            orn = p.getQuaternionFromEuler([0, 0, angle])
            p.resetBasePositionAndOrientation(
                self._objectUids[obj_idx],
                [xpos, ypos_, -0.44 - self._shift[2]],
                [orn[0], orn[1], orn[2], orn[3]],
            )  # xyzw

            p.resetBaseVelocity(
                self._objectUids[obj_idx], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
            )
            for _ in range(3000):
                p.stepSimulation()

            _, new_orn = p.getBasePositionAndOrientation(self._objectUids[obj_idx])
            ang = (
                np.arccos(2 * np.power(np.dot(tf_quat(orn), tf_quat(new_orn)), 2) - 1)
                * 180.0
                / np.pi
            )
            if ang > 20:
                p.resetBasePositionAndOrientation(
                    self._objectUids[obj_idx],
                    [xpos, ypos, -10.0 - self._shift[2]],
                    [orn[0], orn[1], orn[2], orn[3]],
                )
                continue  # check 2
            self.cached_objects[obj_idx] = True
            obj_radius.append(radius)
            pos = np.concatenate([pos, xy], axis=0)
            xpos = xpos_
            placed_indexes.append(obj_idx)
            assigned_orn.append(orn)

        for _ in range(10000):
            p.stepSimulation()

        return True

    def _get_random_object(self, num_objects, ycb=True):
        """
        Randomly choose an object urdf from the selected objects
        """

        self.target_idx = self._all_objs.index(
            self._target_objs[np.random.randint(0, len(self._target_objs))]
        )  #
        obstacle = np.random.choice(
            range(len(self._all_objs)), self._numObjects - 1
        ).tolist()
        selected_objects = [self.target_idx] + obstacle
        selected_objects_filenames = [
            self.obj_path[selected_object] for selected_object in selected_objects
        ]
        return selected_objects_filenames

    def retract(self, record=False):
        """Retract step."""
        cur_joint = np.array(self._panda.getJointStates()[0])
        cur_joint[-2:] = 0

        self.step(cur_joint.tolist())  # grasp
        pos, orn = p.getLinkState(
            self._panda.pandaUid, self._panda.pandaEndEffectorIndex
        )[:2]
        observations = []
        for i in range(10):
            pos = (pos[0], pos[1], pos[2] + 0.03)
            jointPoses = np.array(
                p.calculateInverseKinematics(
                    self._panda.pandaUid, self._panda.pandaEndEffectorIndex, pos
                )
            )
            jointPoses[-2:] = 0.0

            self.step(jointPoses.tolist())
            observation = self._get_observation()
            if record:
                observations.append(observation)

        return (self._reward(), observations) if record else self._reward()

    def step(self, action, obs=True):
        """Environment step."""
        self._env_step += 1
        self._panda.setTargetPositions(action)
        for _ in range(self._actionRepeat):
            p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
        if not obs:
            observation = None
        else:
            observation = self._get_observation()
        done = self._termination()
        reward = self._reward()

        return observation, reward, done, None

    def _get_observation(self):
        _, _, rgba, depth, mask = p.getCameraImage(
            width=self._window_width,
            height=self._window_height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            physicsClientId=self.cid,
        )

        depth = self.far * self.near / (self.far - (self.far - self.near) * depth)
        joint_pos, joint_vel = self._panda.getJointStates()
        obs = np.concatenate(
            [rgba[..., :3], depth[..., None], mask[..., None]], axis=-1
        )
        return (obs, joint_pos)

    def get_pc(self, id=5):
        _, _, rgba, depth, mask = p.getCameraImage(
                    width=self._window_width,
                    height=self._window_height,
                    viewMatrix=self._view_matrix,
                    projectionMatrix=self._proj_matrix,
                    physicsClientId=self.cid,
                )
        rgba = rgba / 255.0

        imgH, imgW = depth.shape
        projGLM = np.asarray(self._proj_matrix).reshape([4, 4], order='F')
        view = np.asarray(self._view_matrix).reshape([4, 4], order='F')

        pos, orn = p.getBasePositionAndOrientation(env._panda.pandaUid)
        # base_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]

        all_pc = []
        stepX = 1
        stepY = 1
        for h in range(0, imgH, stepY):
            for w in range(0, imgW, stepX):
                win = glm.vec3(w, imgH - h, depth[h][w])
                position = glm.unProject(win, glm.mat4(view), glm.mat4(projGLM), glm.vec4(0, 0, imgW, imgH))
                all_pc.append([position[0], position[1], position[2], rgba[h, w, 0], rgba[h, w, 1], rgba[h, w, 2]])

        all_pc = np.array(all_pc)
        all_pc[:, :3] -= pos
        
        all_pcd = o3d.geometry.PointCloud()
        all_pcd.points = o3d.utility.Vector3dVector(all_pc[:, :3])
        all_pcd.colors = o3d.utility.Vector3dVector(all_pc[:, 3:])

        pc = []
        obj_idxs = np.where(mask == id)
        for i in range(len(obj_idxs[0])):
            h = obj_idxs[0][i]
            w = obj_idxs[1][i]
            win = glm.vec3(w, imgH - h, depth[h][w])
            position = glm.unProject(win, glm.mat4(view), glm.mat4(projGLM), glm.vec4(0, 0, imgW, imgH))
            pc.append([position[0], position[1], position[2], rgba[h, w, 0], rgba[h, w, 1], rgba[h, w, 2]])
        pc = np.array(pc)
        pc[:, :3] -= pos
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:])
        # print("Visualizing point cloud...")
        # o3d.visualization.draw_geometries([pcd])
        # downpcd = pcd.voxel_down_sample(voxel_size=0.05)
        # o3d.visualization.draw_geometries([downpcd])
        # print("Point cloud visualized.")
        return pcd, all_pcd

    def _get_target_obj_pose(self):
        return p.getBasePositionAndOrientation(self._objectUids[self.target_idx])[0]

    def _reward(self):
        """Calculates the reward for the episode.

        The reward is 1 if one of the objects is above height .2 at the end of the
        episode.
        """
        reward = 0
        hand_pos, _ = p.getLinkState(
            self._panda.pandaUid, self._panda.pandaEndEffectorIndex
        )[:2]
        pos, _ = p.getBasePositionAndOrientation(
            self._objectUids[self.target_idx]
        )  # target object
        if (
            np.linalg.norm(np.subtract(pos, hand_pos)) < 0.2
            and pos[2] > -0.35 - self._shift[2]
        ):
            reward = 1
        return reward

    def _termination(self):
        """Terminates the episode if we have tried to grasp or if we are above
        maxSteps steps.
        """
        return self._env_step >= self._maxSteps

    def _add_mesh(self, obj_file, trans, quat, scale=1):
        try:
            bid = p.loadURDF(obj_file, trans, quat, globalScaling=scale)
            return bid
        except:
            print("load {} failed".format(obj_file))

    def get_env_info(self ):
        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        base_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        poses = []
        obj_dir = []

        for idx, uid in enumerate(self._objectUids):
            if   idx >= len(self.cached_objects)  or self.cached_objects[idx]:
                pos, orn = p.getBasePositionAndOrientation(uid)  # center offset of base
                obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
                poses.append(inv_relative_pose(obj_pose, base_pose))
                obj_dir.append(self.obj_path[idx])  # .encode("utf-8")

        return obj_dir, poses

def bullet_execute_plan(env, plan, write_video, video_writer):
    print('executing...')
    for k in range(plan.shape[0]):
        obs, rew, done, _ = env.step(plan[k].tolist())
        if write_video:
            video_writer.write(obs[0][:, :, [2, 1, 0]].astype(np.uint8))
    (rew, ret_obs) = env.retract(record=True)
    if write_video: 
        for robs in ret_obs:
            video_writer.write(robs[0][:, :, [2, 1, 0]].astype(np.uint8))
            video_writer.write(robs[0][:, :, [2, 1, 0]].astype(np.uint8)) # to get enough frames to save
    return rew

def make_parser(parser):
    # parser = argparse.ArgumentParser(
        # description='6-DoF GraspNet Demo',
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--grasp_sampler_folder',
                        type=str,
                        default='checkpoints/gan_pretrained/')
    parser.add_argument('--grasp_evaluator_folder',
                        type=str,
                        default='checkpoints/evaluator_pretrained/')
    parser.add_argument('--refinement_method',
                        choices={"gradient", "sampling"},
                        default='sampling')
    parser.add_argument('--refine_steps', type=int, default=25)

    parser.add_argument('--npy_folder', type=str, default='demo/data/')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help=
        "When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed"
    )
    parser.add_argument(
        '--choose_fn',
        choices={
            "all", "better_than_threshold", "better_than_threshold_in_sequence"
        },
        default='better_than_threshold',
        help=
        "If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps"
    )

    parser.add_argument('--target_pc_size', type=int, default=1024)
    parser.add_argument('--num_grasp_samples', type=int, default=200)
    parser.add_argument(
        '--generate_dense_grasps',
        action='store_true',
        help=
        "If enabled, it will create a [num_grasp_samples x num_grasp_samples] dense grid of latent space values and generate grasps from these."
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=30,
        help=
        "Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory."
    )
    parser.add_argument('--train_data', action='store_true')
    # opts, _ = parser.parse_known_args()
    # if opts.train_data:
    #     parser.add_argument('--dataset_root_folder',
    #                         required=True,
    #                         type=str,
    #                         help='path to root directory of the dataset.')
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="filename", type=str, default="scene_1")
    parser.add_argument(
        "-d", "--dir", help="filename", type=str, default="data/scenes/"
    ) 
    parser.add_argument("-v", "--vis", help="renders", action="store_true")
    parser.add_argument("-w", "--write_video", help="write video", action="store_true")
    parser.add_argument("-exp", "--experiment", help="loop through the 100 scenes", action="store_true")
    parser.add_argument("--egl", help="use egl render", action="store_true")

    # GraspNet args
    parser = make_parser(parser)

    args = parser.parse_args()

    # setup bullet env
    mkdir_if_missing('output_videos')
    env = PandaYCBEnv(renders=args.vis, egl_render=args.egl)
    scene_file = args.file
    env.reset()

    # setup planner
    cfg.traj_init = "grasp"
    cfg.scene_file = args.file
 
    cfg.vis = False
    cfg.timesteps = 50  
    cfg.get_global_param(cfg.timesteps)
    scene = PlanningScene(cfg)

    for i, name in enumerate(env.obj_path[:-2]):  # load all objects
        name = name.split("/")[-1]
        trans, orn = env.cache_object_poses[i]
        scene.env.add_object(name, trans, tf_quat(orn), compute_grasp=True)

    scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1, 0, 0, 0]))
    scene.env.add_table(np.array([0.55, 0, -0.17]), np.array([0.707, 0.707, 0.0, 0]))
    scene.env.combine_sdfs()
    if args.experiment:   
        scene_files = ['scene_{}'.format(i) for i in range(100)]
    else:
        scene_files = [scene_file]

    cnts, rews = 0, 0
    for scene_file in scene_files:
        config.cfg.output_video_name = "output_videos/bullet_" + scene_file + ".avi"
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

        exists_ids, placed_poses = [], []
        for i, name in enumerate(object_lists[:-2]):  # update planning scene
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
        # scene.env.set_target(env.obj_path[env.target_idx].split("/")[-1])
        # scene.reset(lazy=True)

        # Get point clouds
        obj_id = 5 # 7
        object_pc, all_pc = env.get_pc(id=obj_id)

        # Predict grasp using 6-DOF GraspNet
        gpath = os.path.dirname(grasp_estimator.__file__)
        grasp_sampler_args = gutils.read_checkpoint_args(
            os.path.join(gpath, args.grasp_sampler_folder))
        grasp_sampler_args['checkpoints_dir'] = os.path.join(gpath, grasp_sampler_args['checkpoints_dir'])
        grasp_sampler_args.is_train = False
        grasp_evaluator_args = gutils.read_checkpoint_args(
            os.path.join(gpath, args.grasp_evaluator_folder))
        grasp_evaluator_args['checkpoints_dir'] = os.path.join(gpath, grasp_evaluator_args['checkpoints_dir'])
        grasp_evaluator_args.continue_train = True # was in demo file, not sure 
        estimator = grasp_estimator.GraspEstimator(grasp_sampler_args,
                                                    grasp_evaluator_args, args)

        pc = np.asarray(object_pc.points)
        generated_grasps, generated_scores = estimator.generate_and_refine_grasps(
            pc)

        # Choose grasp
        select_criteria = 'distance' # 'distance' to EE vs. prediction 'score'
        if select_criteria == 'distance':
            # Get each grasp as a control point
            control_points = gutils.transform_control_points(
                torch.Tensor(generated_grasps), 
                len(generated_grasps), mode='rt')
            
            # Get current EE position as a control point
            pos, orn = p.getLinkState(
                env._panda.pandaUid, env._panda.pandaEndEffectorIndex
            )[:2]
            T_ee = quat2rotmat(orn)
            T_ee[:3, 3] = pos
            ee_control_points = gutils.transform_control_points(
                torch.Tensor([T_ee]),
                1, mode='rt')

            # Get grasp with lowest L1 control point error
            error = control_points - ee_control_points # (N, 6, 4)
            error = torch.sum(torch.abs(error), -1)  # (N, 6)
            error = torch.mean(error, dim=-1) # (N)
            g_id = np.argmin(error)
            grasp = generated_grasps[g_id]

        elif select_criteria =='score':
            g_id = np.argmax(generated_scores)
            grasp = generated_grasps[g_id]

        generated_grasps = np.array(generated_grasps)
        generated_scores = np.array(generated_scores)

        target_id = 2 # 4
        scene.env.set_target(env.obj_path[target_id].split("/")[-1])

        scene.planner.cfg.ol_alg = 'Proj'
        # scene.planner.cfg.ol_alg = 'Baseline'

        # size = 30
        # idxs = np.random.choice(range(len(generated_grasps)), size=size)
        # sampled_grasps = generated_grasps[idxs]
        # sampled_scores = generated_scores[idxs]

        scene.reset(lazy=True, grasps=generated_grasps, grasp_scores=generated_scores)
        # scene.reset(lazy=True, grasps=np.array([grasp]))

        dbg = np.load("output_videos/dbg.npy", encoding='latin1', allow_pickle=True)
        grasp_start, grasp_end, goal_idx, goal_set, goal_quality, grasp_ees = dbg # in joints, not EE pose
        goal_ees_T = [unpack_pose(g) for g in grasp_ees]
        # goal_end_T = scene.planner.env.robot.robot_kinematics.forward_kinematics_parallel(wrap_value(grasp_end)[None, :], base_link=cfg.base_link)[0, -1, :, :]
        # goal_start_T = scene.planner.env.robot.robot_kinematics.forward_kinematics_parallel(wrap_value(grasp_start)[None, :], base_link=cfg.base_link)[0, -1, :, :]

        # Visualize
        # visualization_utils.draw_scene(
        #     np.asarray(all_pc.points),
        #     pc_color=(np.asarray(all_pc.colors) * 255).astype(int),
        #     # grasps=[grasp, generated_grasps[0]],
        #     # grasps=sampled_grasps,
        #     grasps=goal_ees_T,
        #     # grasp_scores=[generated_scores[g_id], generated_scores[0]],
        #     grasp_scores=goal_quality,
        # )

        visualization_utils.draw_scene(
            np.asarray(all_pc.points),
            pc_color=(np.asarray(all_pc.colors) * 255).astype(int),
            # grasps=[grasp, generated_grasps[0]],
            # grasps=sampled_grasps,
            grasps=[goal_ees_T[goal_idx]],
            # grasps=[goal_end_T, goal_start_T],
            # grasp_scores=[generated_scores[g_id], generated_scores[0]],
            grasp_scores=[goal_quality[goal_idx]],
        )
        # import IPython; IPython.embed() # Ctrl-D for interactive visualization 

        info = scene.step()
        plan = scene.planner.history_trajectories[-1]

        rew = bullet_execute_plan(env, plan, args.write_video, video_writer)
        for i, name in enumerate(object_lists[:-2]):  # reset planner
            scene.env.update_pose(name, placed_poses[i])
        cnts += 1
        rews += rew
        print('rewards: {} counts: {}'.format(rews, cnts))

        import IPython; IPython.embed()
    env.disconnect()

        # from mpl_toolkits import mplot3d
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.scatter(generated_grasps[:, 0, 3], generated_grasps[:, 1, 3], generated_grasps[:, 2, 3])
