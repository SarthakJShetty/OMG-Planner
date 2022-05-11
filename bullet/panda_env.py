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

from PIL import Image
import glob
import IPython
from .panda_gripper import Panda

import scipy.io as sio
import pkgutil
from .utils import depth2pc
from copy import deepcopy
import pytransform3d.rotations as pr


class PandaEnv:
    """Class for panda environment.
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
        cameraRandom=0,
        width=640,
        height=480,
        numObjects=8,
        safeDistance=0.13,
        random_urdf=False,
        egl_render=False,
        gui_debug=True,
        gravity=True,
        root_dir=None,
        cam_look=[-0.35, -0.58, -0.88],
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
        self._timeStep = 1.0 / 1000.0
        self._urdfRoot = urdfRoot
        self._observation = []
        self._renders = renders
        self._maxSteps = maxSteps
        self._actionRepeat = actionRepeat
        self._env_step = 0
        self._gravity = gravity

        self._cam_look = cam_look
        self._cam_dist = 0.9 # 1.3
        self._cam_yaw = 180
        self._cam_pitch = -41
        self._safeDistance = safeDistance
        self._root_dir = root_dir if root_dir is not None else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self._p = p
        self._window_width = width
        self._window_height = height
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
        self._numObjects = numObjects
        self._shift = [0.5, 0.5, 0.5]  # to work without axis in DIRECT mode
        self._egl_render = egl_render

        self._gui_debug = gui_debug
        self.target_idx = 0
        self.connect()

    def connect(self):
        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if self.cid < 0:
                self.cid = p.connect(p.GUI)
                p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, 0, [-0.35, -0.58, -0.88])
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

    def load_object(self, objinfo=None):
        if objinfo is None:
            return []

        uid = self._add_mesh(
            objinfo['urdf_dir'], [0, 0, 0], [0, 0, 0, 1], scale=objinfo['scale']
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
        look = self._cam_look
        distance = self._cam_dist
        pitch = self._cam_pitch
        yaw = self._cam_yaw
        roll = 0
        self._fov = 60.0 + self._cameraRandom * np.random.uniform(-2, 2)
        # http://www.songho.ca/opengl/gl_transform.html#modelview
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            look, distance, yaw, pitch, roll, 2
        )

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

    def reset(self, init_joints=None, no_table=False, objinfo=None):
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
            for _ in range(1000):
                p.stepSimulation()
            self._panda = Panda(
                stepsize=self._timeStep, init_joints=init_joints, base_shift=self._shift
            )

        # Initialize objects
        plane_file = "data/objects/floor"
        table_file = "data/objects/table/models"
        self.plane_id = p.loadURDF(
            os.path.join(plane_file, 'model_normalized.urdf'),
            [0 - self._shift[0], 0 - self._shift[1], -0.82 - self._shift[2]]
        )
        table_z = -5 if no_table else -0.82 - self._shift[2]
        self.table_id = p.loadURDF(
            os.path.join(table_file, 'model_normalized.urdf'),
            0.5 - self._shift[0],
            0.0 - self._shift[1],
            table_z,
            0.707,
            0.0,
            0.0,
            0.707,
        )

        self.objinfos = [objinfo]
        self._objectUids = self.load_object(objinfo=objinfo)
        self._objectUids += [self.plane_id, self.table_id]

        self._env_step = 0
        return self._get_observation()

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

    def _get_observation(self, get_pc=False):
        _, _, rgba, zbuffer, mask = p.getCameraImage(
            width=self._window_width,
            height=self._window_height,
            viewMatrix=self._view_matrix,
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

        joint_pos, joint_vel = self._panda.getJointStates()

        depth_masked = deepcopy(depth)
        depth_masked[~(mask == self._objectUids[self.target_idx])] = 0
        pc = depth2pc(depth_masked, self._intr_matrix)[0] if get_pc else None # N x 7 (XYZ, RGB, Mask ID)

        if False and pc is not None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2])
            plt.show()

        obs = {
            'rgb': rgb,
            'depth': depth[..., None],
            'mask': mask[..., None],
            'points': pc,
            'joint_pos': joint_pos,
            'joint_vel': joint_vel
        }
        return obs

    def _get_target_obj_pose(self):
        return p.getBasePositionAndOrientation(self._objectUids[self.target_idx])[0]

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
