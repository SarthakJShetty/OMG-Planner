import random
import os
# from gym import spaces
import time
import sys
import argparse
# from . import _init_paths

from omg.core import *
from omg.util import *
from omg.config import cfg
import pybullet as p
import numpy as np
import pybullet_data

# from PIL import Image
# import glob
from omg_bullet.panda_gripper import Panda

# import scipy.io as sio
import pkgutil
from omg_bullet.utils import depth2pc, draw_pose, get_world2bot_transform
from copy import deepcopy
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

import trimesh
from acronym_tools import create_gripper_marker
from pathlib import Path
import hydra
from hydra.utils import get_original_cwd
import csv
from manifold_grasping.utils import load_mesh


def get_random_transform(pos, q=None, random=False):
    T_rand = np.eye(4)
    T_rand[:3, 3] = pos

    if q is not None:
        T_rand[:3, :3] = pr.matrix_from_quaternion(q)
    elif random: 
        q = pr.random_quaternion()
        T_rand[:3, :3] = pr.matrix_from_quaternion(q)

    return T_rand

class PandaEnv:
    """Class for panda environment.
    adapted from kukadiverse env in pybullet
    """

    def __init__(
        self,
        renders=False,
        gui_debug=True,
        egl_render=False,
    ):
        self._renders = renders
        self._gui_debug = gui_debug
        self._egl_render = egl_render
        self.connect()

    def connect(self):
        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if self.cid < 0:
                self.cid = p.connect(p.GUI)
                p.resetDebugVisualizerCamera(1.3, 180, 0, [-0.35, -0.58, -0.88])
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
        # wait in case gripper closes
        for i in range(10):
            self.step(jointPoses.tolist())
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

    def _reward(self):
        """Calculates the reward for the episode.

        The reward is 1 if the gripper has significant width at the end of the episode.
        """
        reward = 0
        grip_pos = self._panda.getJointStates()[0][-2:]
        if (
            min(grip_pos) > 0.001
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

    def _get_target_obj_pose(self):
        return p.getBasePositionAndOrientation(self._objectUids[self.target_idx])[0]


class PandaAcronymEnv(PandaEnv):
    """Class for panda environment.
    adapted from kukadiverse env in pybullet
    """

    def __init__(
        self,
        urdfRoot=pybullet_data.getDataPath(),
        actionRepeat=130,
        isEnableSelfCollision=True,
        isDiscrete=False,
        maxSteps=800,
        dtheta=0.1,
        blockRandom=0.5,
        cameraRandom=0,
        width=640,
        height=480,
        numObjects=8,
        safeDistance=0.13,
        random_urdf=False,
        gravity=True,
        root_dir=None,
        cam_look=[-0.05, -0.5, -1.1],
        *args,
        **kwargs
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
        """
        super(PandaAcronymEnv, self).__init__(*args, **kwargs)
        self._timeStep = 1.0 / 1000.0
        self._urdfRoot = urdfRoot
        self._observation = []
        self._maxSteps = maxSteps
        self._actionRepeat = actionRepeat
        self._env_step = 0
        self._gravity = gravity

        # observation cameras
        self._cam_look = cam_look
        self._cams = [
            {
                'look': cam_look,
                'dist': 0.9,
                'yaw': 45,
                'pitch': -34,
                'roll': 0
            },
            {
                'look': cam_look,
                'dist': 0.9,
                'yaw': 225,
                'pitch': -34,
                'roll': 0
            },
            {
                'look': cam_look,
                'dist': 0.9,
                'yaw': 135,
                'pitch': -34,
                'roll': 0
            },
            {
                'look': cam_look,
                'dist': 0.9,
                'yaw': 325,
                'pitch': -34,
                'roll': 0
            },
        ]

        self._safeDistance = safeDistance
        self._root_dir = root_dir if root_dir is not None else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        self._window_width = width
        self._window_height = height
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
        self._numObjects = numObjects
        self._shift = [0.5, 0.5, 0.5]  # to work without axis in DIRECT mode

        self.target_idx = 0

    def load_object(self, objinfo=None):
        if objinfo is None:
            return []

        uid = self._add_mesh(
            # objinfo['urdf_dir'], [0, 0, 0], [0, 0, 0, 1], scale=objinfo['scale']
            objinfo['urdf_dir'], [0, 0, 0], [0, 0, 0, 1], scale=1 # object is pre-scaled in .obj
        )  # xyzw
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
        return [uid]

    def reset_perception(self):
        self._fov = 60.0 + self._cameraRandom * np.random.uniform(-2, 2)

        # Draw target axes
        T = np.eye(4)
        T[:3, 3] = self._cam_look
        draw_pose(T)

        self._view_matrices = []
        self._extrinsics = []
        for cam in self._cams:
            # http://www.songho.ca/opengl/gl_transform.html#modelview
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cam['look'], cam['dist'], cam['yaw'], cam['pitch'], cam['roll'], 2
            )
            self._view_matrices.append(view_matrix)
            view_matrix = np.reshape(np.array(view_matrix), (4, 4)).T
            extrinsic_matrix = np.linalg.inv(view_matrix)
            draw_pose(extrinsic_matrix)
            self._extrinsics.append(extrinsic_matrix)

        self._aspect = float(self._window_width) / self._window_height

        self.near = 0.5
        self.far = 6

        focal_length = 450
        fovh = (np.arctan((self._window_height /
                           2) / focal_length) * 2 / np.pi) * 180

        self._proj_matrix = p.computeProjectionMatrixFOV(
            fovh, self._aspect, self.near, self.far
        )

        # https://www.edmundoptics.com/knowledge-center/application-notes/imaging/understanding-focal-length-and-field-of-view/
        self._intr_matrix = np.eye(3)
        self._intr_matrix[0, 2] = self._window_width / 2
        self._intr_matrix[1, 2] = self._window_height / 2
        self._intr_matrix[0, 0] = focal_length
        self._intr_matrix[1, 1] = focal_length

    def reset(self, init_joints=None, no_table=False, objinfos=[]):
        """Environment reset"""
        self.reset_perception()

        # Set table and plane
        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        if self._gravity:
            p.setGravity(0, 0, -9.81)
        p.stepSimulation()

        # Intialize robot
        if init_joints is None:
            self._panda = Panda(stepsize=self._timeStep, base_shift=self._shift)
        else:
            self._panda = Panda(
                stepsize=self._timeStep, init_joints=init_joints, base_shift=self._shift
            )
        self._panda_vizs = []

        # Initialize objects
        plane_file = "data/objects/floor"
        table_file = "data/objects/table/models"
        self.plane_id = p.loadURDF(
            str(Path(self._root_dir) / plane_file / 'model_normalized.urdf'),
            [0 - self._shift[0], 0 - self._shift[1], -0.82 - self._shift[2]]
        )
        table_z = -5 if no_table else -0.82 - self._shift[2]
        self.table_id = p.loadURDF(
            str(Path(self._root_dir) / table_file / 'model_normalized.urdf'),
            0.5 - self._shift[0],
            0.0 - self._shift[1],
            table_z,
            0.707,
            0.0,
            0.0,
            0.707,
        )

        # TODO turn off for mustard bottle
        self.objinfos = objinfos
        self._objectUids = []
        for objinfo in objinfos:
            self._objectUids += self.load_object(objinfo=objinfo)
        self._objectUids += [self.plane_id, self.table_id]

        self._env_step = 0
        return self._get_observation()

    def _get_observation(self, get_pc=False, single_view=True):
        rgbs = []
        depths = []
        masks = []
        pcs = []

        for i, cam in enumerate(self._cams):
            _, _, rgba, zbuffer, mask = p.getCameraImage(
                width=self._window_width,
                height=self._window_height,
                viewMatrix=self._view_matrices[i],
                projectionMatrix=self._proj_matrix,
                physicsClientId=self.cid,
            )

            # The depth provided by getCameraImage() is in normalized device coordinates from 0 to 1.
            # To get the metric depth, scale to [-1, 1] and then apply inverse of projection matrix.
            # https://stackoverflow.com/questions/51315865/glreadpixels-how-to-get-actual-depth-instead-of-normalized-values
            # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/getCameraImageTest.py
            # https://stackoverflow.com/questions/51315865/glreadpixels-how-to-get-actual-depth-instead-of-normalized-values
            depth_zo = 2. * zbuffer - 1.
            depth = (self.far + self.near - depth_zo * (self.far - self.near))
            depth = (2. * self.near * self.far) / depth
            rgb = rgba[..., :3]
            depth_masked = deepcopy(depth)
            depth_masked[~(mask == self._objectUids[self.target_idx])] = 0

            rgbs.append(rgb)
            depths.append(depth)  # width x height
            masks.append(mask)

            if get_pc:
                pc = depth2pc(depth_masked, self._intr_matrix)[0] # N x 7 (XYZ, RGB, Mask ID)
                pcs.append(pc)

            if single_view:
                break

        joint_pos, joint_vel = self._panda.getJointStates()

        # Get point clouds in object frame: centered at mean of point cloud with world axes
        if get_pc:
            T_rotx = np.eye(4)
            T_rotx[:3, :3] = pr.matrix_from_euler_xyz([np.pi, 0, 0])

            T_cams = []
            pcs_world = []
            for i, pc_cam in enumerate(pcs):
                T_world_cam = self._extrinsics[i]
                T_world_cam_rot = T_world_cam @ T_rotx
                draw_pose(T_world_cam_rot)
                T_cams.append(T_world_cam_rot)
                pc_world = (T_world_cam_rot @ pc_cam.T).T
                pcs_world.append(pc_world)
            pc_world = np.concatenate(pcs_world, axis=0)

            if False:  # Debug visualization
                pcd_world = trimesh.points.PointCloud(pc_world[:, :3], colors=np.array([0, 0, 255, 255]))
                # camera poses represented using the gripper marker
                cams = [create_gripper_marker(color=[255, 0, 0]).apply_transform(T_cam) for T_cam in T_cams]
                trimesh.Scene([pcd_world] + cams).show()
        else:
            pc_world = None

        obs = {
            'rgb': rgbs,
            'depth': depths,
            'mask': masks,
            # 'rgb': rgb,
            # 'depth': depth,
            # 'mask': mask,
            'points': pc_world,
            'joint_pos': joint_pos,
            'joint_vel': joint_vel
        }
        return obs

    def get_env_info(self):
        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        base_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        poses = []
        # obj_dir = []

        # Only one object loaded
        uid = self._objectUids[0]
        pos, orn = p.getBasePositionAndOrientation(uid)  # center offset of base
        obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        poses.append(inv_relative_pose(obj_pose, base_pose))

        # return obj_dir, poses
        return [uid], poses

    def update_panda_viz(self, traj, k=1):
        # Visualize last k steps
        traj_len = traj.data.shape[0]
        count = 0
        for tstep in range(traj_len - k, traj_len): 
            if len(self._panda_vizs) < k:
                self._panda_vizs.append(Panda(stepsize=self._timeStep, base_shift=self._shift, viz=True))
            self._panda_vizs[count].reset(traj.data[tstep])
            count += 1
            # self._panda_vizs.append(Panda(stepsize=self._timeStep, base_shift=self._shift, viz=True))

        #     self._panda_vizs[tstep].reset(traj.data[tstep])
        # joints = traj.data[-1]
        # self._panda_viz.reset(joints)
    
    @staticmethod
    def get_scenes(hydra_cfg):
        objnames = os.listdir(Path(hydra_cfg.data_root) / hydra_cfg.dataset / 'meshes_bullet')
        scenes = []
        if hydra_cfg.run_scenes:
            if hydra_cfg.eval.obj_csv is not None:
                with open(Path(get_original_cwd()) / ".." / hydra_cfg.eval.obj_csv, 'r') as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        for objname in objnames:   
                            scene = {
                                'idx': i, 
                                'obj_name': objname,
                                'joints': [0.0, -1.285, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
                                'obj_rot': pr.quaternion_wxyz_from_xyzw([float(x) for x in row[3:]])
                            }
                            scenes.append(scene)
            elif hydra_cfg.eval.joints_csv is not None:
                with open(Path(get_original_cwd()) / ".." / hydra_cfg.eval.joints_csv, 'r') as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        for objname in objnames:
                            scene = {
                                'idx': i, 
                                'obj_name': objname,
                                'joints': [float(x) for x in row],
                                'obj_rot': [0, 0, 0, 1]
                            }
                            scenes.append(scene)
        else:
            for objname in objnames:
                scenes.append({
                    "idx": 0,
                    "obj_name": objname,
                    "joints": [0.0, -1.285, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
                    "obj_rot": [0, 0, 0, 1]
                })
        return scenes
    #     if 'Mug' not in objname: 
    #         continue
    #         if scene_idx != 4:
    #             continue

    
    def init_scene(self, scene, planning_scene, hydra_cfg):
        objinfos = []
        objinfo = self.get_object_info(scene['obj_name'], Path(hydra_cfg.data_root) / hydra_cfg.dataset)
        objinfos.append(objinfo)

        if planning_scene.cfg.float_obstacle:
            objinfo = self.get_object_info('Bottle_244894af3ba967ccd957eaf7f4edb205_0.012953570261294404', 
                Path(hydra_cfg.data_root) / hydra_cfg.dataset)
            objinfos.append(objinfo)
    
        self.reset(init_joints=scene['joints'], no_table=not cfg.table, objinfos=objinfos)
        uids = []
        uid = self._objectUids[0]
        self.place_object(uid, cfg.tgt_pos, q=scene['obj_rot'], random=False, gravity=cfg.gravity)
        uids.append(uid)

        if planning_scene.cfg.float_obstacle:
            obs_pos = deepcopy(cfg.tgt_pos)
            obs_pos[0] -= 0.1
            uid = self._objectUids[1]
            self.place_object(uid, obs_pos, q=[1, 0, 0, 0], random=False, gravity=cfg.gravity)
            uids.append(uid)

        obs = self._get_observation(get_pc=cfg.pc, single_view=False)
        self.set_scene_env(planning_scene, uids, objinfos, scene['joints'], hydra_cfg)
        return obs, scene['obj_name'], scene['idx']

    def get_object_info(self, objname, mesh_root):
        """Used in multiple_views_acronym_bullet.py"""
        grasp_h5 = Path(mesh_root) / 'grasps' / f'{objname}.h5'
        scale = objname.split('_')[-1]
        _, T_ctr2obj = load_mesh(str(grasp_h5), scale=scale, mesh_root_dir=mesh_root, load_for_bullet=True)
        objinfo = {
            'name': objname,
            'urdf_dir': f'{mesh_root}/meshes_bullet/{objname}/model_normalized.urdf',
            'scale': float(scale),
            'T_ctr2obj': T_ctr2obj,
            'T_ctr2obj_com': None
        }
        return objinfo

    def place_object(self, uid, target_pos, q=None, random=False, gravity=False):
        """_summary_

        Args:
            env (_type_): _description_
            target_pos (_type_): _description_
            q (_type_, optional): _description_. Defaults to None. The object rotation as an wxyz quaternion
            random (bool, optional): _description_. Defaults to False. Whether to sample a random rotation
            gravity (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # place single object
        T_w2b = get_world2bot_transform()
    
        T_rand = get_random_transform(target_pos, q=q, random=random)

        # Apply object to centroid transform
        T_ctr2obj = self.objinfos[0]['T_ctr2obj'] # TODO
        # T_ctr2obj = env.objinfos[0]['T_ctr2obj_com']

        T_w2o = T_w2b @ T_rand @ T_ctr2obj
        pq_w2o = pt.pq_from_transform(T_w2o)  # wxyz

        p.resetBasePositionAndOrientation(
            uid,
            pq_w2o[:3],
            pr.quaternion_xyzw_from_wxyz(pq_w2o[3:])
        )
        p.resetBaseVelocity(uid, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        if gravity:
            for i in range(10000):
                p.stepSimulation()
        
        return T_w2o

    def set_scene_env(self, planning_scene, uids, objinfos, joints, hydra_cfg):
        obj_prefix = Path(hydra_cfg.data_root) / hydra_cfg.dataset / 'meshes_bullet'
        planning_scene.reset_env(joints=joints)
        T_b2w = np.linalg.inv(get_world2bot_transform())
        for i, objinfo in enumerate(objinfos):
            # Scene has separate Env class which is used for planning
            # Add object to planning scene env
            trans_w2o, orn_w2o = p.getBasePositionAndOrientation(uids[i])  # xyzw

            # change to world to object centroid so planning scene env only sees objects in centroid frame
            T_w2o = np.eye(4)
            T_w2o[:3, :3] = pr.matrix_from_quaternion(tf_quat(orn_w2o))
            T_w2o[:3, 3] = trans_w2o
            T_o2c = np.linalg.inv(objinfo['T_ctr2obj']) # TODO
            # T_o2c = np.linalg.inv(objinfo['T_ctr2obj_com'])
            T_b2c = T_b2w @ T_w2o @ T_o2c
            trans = T_b2c[:3, 3]
            orn = pr.quaternion_from_matrix(T_b2c[:3, :3])  # wxyz
            draw_pose(T_w2o @ T_o2c)

            planning_scene.env.add_object(objinfo['name'], trans, orn, obj_prefix=obj_prefix, abs_path=True)
            
        planning_scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1, 0, 0, 0]))
        planning_scene.env.combine_sdfs()
        if cfg.disable_target_collision:
            cfg.disable_collision_set = [objinfos[0]['name']]

        # Set grasp selection method for planner
        planning_scene.env.set_target(objinfos[0]['name'])
        planning_scene.reset(lazy=True)
