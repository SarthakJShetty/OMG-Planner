import random
import os
import time
import sys
import argparse

from omg.core import *
from omg.util import *
from omg.config import cfg
import pybullet as p
import numpy as np
import pybullet_data

# import glob
from omg_bullet.panda_gripper import Panda

import scipy.io as sio
import pkgutil
from omg_bullet.utils import depth2pc
from copy import deepcopy
import pytransform3d.rotations as pr

from omg_bullet.envs.acronym_env import PandaAcronymEnv
from pathlib import Path
from omg_bullet.utils import draw_pose, get_world2bot_transform

class PandaYCBEnv(PandaAcronymEnv):
    """Class for panda environment.
    adapted from kukadiverse env in pybullet
    """
    def __init__(self,
                 target_obj=[1, 2, 3, 4, 10, 11],  # [1,2,4,3,10,11],
                 all_objs=[0, 1, 2, 3, 4, 8, 10, 11],
                 cache_objects=False,
                 *args, **kwargs):
        super(PandaYCBEnv, self).__init__(*args, **kwargs)
        self._target_objs = target_obj
        self._all_objs = all_objs
        self._cache_objects = cache_objects
        self._object_cached = False

    def cache_objects(self):
        """
        Load all YCB objects and set up (only work for single apperance)
        """
        obj_path = self._root_dir
        objects = sorted([m for m in os.listdir(f'{obj_path}/data/objects') if m.startswith("0")])
        paths = ['data/objects/' + objects[i] for i in self._all_objs]
        scales = [1 for i in range(len(paths))]

        pose = np.zeros([len(paths), 3])
        pose[:, 0] = -2.0 - np.linspace(0, 4, len(paths))  # place in the back
        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        paths = [p_.strip() for p_ in paths]
        objectUids = []
        self.obj_path = paths
        self.cache_object_poses = []
        self.cache_object_extents = []

        for i, name in enumerate(paths):
            trans = pose[i] + np.array(pos)  # fixed position
            self.cache_object_poses.append((trans.copy(), np.array(orn).copy()))

            uid = self._add_mesh(
                os.path.join(self._root_dir, name, "model_normalized.urdf"), trans, orn, scale=scales[i]
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
        # In this code we don't use cached_objects for the location in a scene,
        # We use it as the reset position. So any objects added are always True cache
        # self.cached_objects = [False] * len(self.obj_path)
        self.cached_objects = [True] * len(self.obj_path)

        return objectUids

    def reset(self, init_joints=None, scene_file=None, no_table=False, reset_cache=False):
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
            str(Path(self._root_dir) / plane_file / 'model_normalized.urdf'),
            [0 - self._shift[0], 0 - self._shift[1], -0.82 - self._shift[2]]
        )
        table_z = -5 if no_table else -0.82 - self._shift[2]
        self.table_id = p.loadURDF(
            str(Path(self._root_dir)/ table_file / 'model_normalized.urdf'),
            0.5 - self._shift[0],
            0.0 - self._shift[1],
            table_z,
            0.707,
            0.0,
            0.0,
            0.707,
        )

        if not self._object_cached or reset_cache:
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
            # self.cached_objects[idx] = False
            self.cached_objects[idx] = True # consider objects always cached

    def cache_reset(self, init_joints=None, scene_file=None):
        self._panda.reset(init_joints)
        self.reset_objects()

        if scene_file is None or not os.path.exists(scene_file):
            self._randomly_place_objects(self._get_random_object(self._numObjects))
        else:
            self.place_objects_from_scene(scene_file)
        self._env_step = 0
        self.obj_names, self.obj_poses = self.get_env_info()

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

    def get_env_info(self):
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

    def get_scenes(self, hydra_cfg):
        if hydra_cfg.run_scenes:
            scene_files = ['scene_{}'.format(i) for i in range(100)]
        else:
            scene_files = [None]
        return scene_files

    def init_scene(self, scene, planning_scene, hydra_cfg):
        full_name = Path(self._root_dir) / 'data' / 'scenes' / f'{scene}.mat'
        self.reset(reset_cache=True)
        self.cache_reset(scene_file=full_name)
        obj_names, obj_poses = self.get_env_info()
        object_lists = [name.split("/")[-1].strip() for name in obj_names]
        object_poses = [pack_pose(pose) for pose in obj_poses]

        # load objects into planning scene
        for i, name in enumerate(self.obj_path[:-2]):
            name = name.split("/")[-1]
            trans, orn = self.cache_object_poses[i]
            planning_scene.env.add_object(name, trans, tf_quat(orn), compute_grasp=True)

        exists_ids, placed_poses = [], []
        for i, name in enumerate(object_lists[:-2]):  # update planning scene
            planning_scene.env.update_pose(name, object_poses[i])
            obj_idx = self.obj_path[:-2].index("data/objects/" + name)
            exists_ids.append(obj_idx)
            trans, orn = self.cache_object_poses[obj_idx]
            placed_poses.append(np.hstack([trans, ros_quat(orn)]))
        
        cfg.disable_collision_set = [
            name.split("/")[-2]
            for obj_idx, name in enumerate(self.obj_path[:-2])
            if obj_idx not in exists_ids
        ]
        
        planning_scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1, 0, 0, 0]))
        planning_scene.env.add_table(np.array([0.55, 0, -0.17]), np.array([0.707, 0.707, 0.0, 0]))
        planning_scene.env.combine_sdfs()
        planning_scene.env.set_target(self.obj_path[self.target_idx].split("/")[-1])
        
        planning_scene.reset(lazy=True)
        obj_name = str(Path(self.obj_path[self.target_idx]).parts[-1])
        obs = self._get_observation()

        # coords = planning_scene.env.objects[self.target_idx].sdf.visualize()

        # T_w2b = get_world2bot_transform()
        # T_b2o = unpack_pose(planning_scene.env.objects[self.target_idx].pose)
        # draw_pose(T_w2b @ T_b2o)
        
        # for i in range(coords.shape[0]):
        #     coord = coords[i]
        #     coord = np.concatenate([coord, [1]])
        #     T_c = np.eye(4)
        #     T_c[:, 3] = coord
        #     draw_pose(T_w2b @ T_b2o @ T_c)

        return obs, obj_name, scene


if __name__ == '__main__':
    env = PandaYCBEnv(renders=True, gravity=True, root_dir='/data/manifolds/ycb_mini')
    env.reset(reset_cache=True)
    env.cache_reset()

    # Load grasps
    target_name = Path(env.obj_path[env.target_idx]).parts[-1]
    simulator_path = (
        env._root_dir
        + "/grasps/simulated/{}.npy".format(target_name)
    )
    try:
        simulator_grasp = np.load(simulator_path, allow_pickle=True)
        pose_grasp = simulator_grasp.item()["transforms"]
    except:
        simulator_grasp = np.load(
            simulator_path,
            allow_pickle=True,
            fix_imports=True,
            encoding="bytes",
        )
        pose_grasp = simulator_grasp.item()[b"transforms"]

    offset_pose = np.array(rotZ(np.pi / 2)) # rotate about z axis
    pose_grasp = np.matmul(pose_grasp, offset_pose)  # flip x, y
    pose_grasp = ycb_special_case(pose_grasp, target_name)

    # Visualize grasps
    T_w2b = get_world2bot_transform()
    _, poses = env.get_env_info()
    T_b2o = poses[env.target_idx]
    draw_pose(T_w2b @ T_b2o)

    for T_obj2grasp in pose_grasp[:30]:
        draw_pose(T_w2b @ T_b2o @ T_obj2grasp)

    import IPython; IPython.embed()
