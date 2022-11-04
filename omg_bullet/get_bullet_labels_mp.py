# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import argparse
import os
import sys
import time

import h5py
import numpy as np
import pybullet as p
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import trimesh
from omg.core import *
from omg.util import *

sys.path.append(os.path.dirname(__file__))
from acronym_tools import create_gripper_marker, load_grasps, load_mesh

from omg_bullet.envs.acronym_env import PandaAcronymEnv
from utils import *


def linear_shake(env, timeSteps, delta_z, record=False, second_shake=False):
    pos, orn = p.getLinkState(env._panda.pandaUid, env._panda.pandaEndEffectorIndex)[:2]

    observations = []
    for t in range(timeSteps):
        jointPoses = env._panda.solveInverseKinematics(
            (pos[0], pos[1], pos[2] + t * delta_z), [1, 0, 0, 0]
        )
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


def run_grasp(env, mesh, mctr_obj_T, mesh_root, obj, mctr_grasp_T, lin=0.15, rot=180):
    # end_T is in pybullet wrist convention
    # object is loaded in obj frame
    # grasp is centroid frame

    # gripper faces down
    init_joints = [
        -0.331397102886776,
        -0.4084196878153061,
        0.27958348738965677,
        -2.100158152658163,
        0.10297468164759681,
        1.7272912586478788,
        0.6990357829811057,
        0.0399,
        0.0399,
    ]

    objinfo = {
        "name": obj,
        "urdf_dir": f"{mesh_root}/meshes_bullet/{obj}/model_normalized.urdf",
    }

    env.reset(init_joints=init_joints, no_table=True, objinfos=[objinfo])

    obj_grasp_T = np.linalg.inv(mctr_obj_T) @ mctr_grasp_T

    # Get end effector pose frame
    pos, orn = p.getLinkState(env._panda.pandaUid, env._panda.pandaEndEffectorIndex)[:2]
    world2ee_T = pt.transform_from_pq(
        np.concatenate([pos, pr.quaternion_wxyz_from_xyzw(orn)])
    )

    world2ctr_T = world2ee_T @ np.linalg.inv(obj_grasp_T)

    # Place single object
    pq = pt.pq_from_transform(world2ctr_T)  # mesh still object centered
    p.resetBasePositionAndOrientation(
        env._objectUids[0], pq[:3], pr.quaternion_xyzw_from_wxyz(pq[3:])
    )  # xyzw
    p.resetBaseVelocity(env._objectUids[0], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    # Close gripper on object
    joints = env._panda.getJointStates()[0]
    joints[-2:] = [0.0, 0.0]
    obs, rew, done, _ = env.step(joints)
    if rew == 0:
        err_info = "grasp"
    else:
        err_info = ""

    # wait after grasping
    for i in range(3):
        env.step(joints)

    observations = [obs]

    # Linear shaking
    if err_info == "":
        dur = 150
        obs = linear_shake(env, dur, -lin / dur)
        observations += obs
        obs = linear_shake(env, dur, lin / dur)
        observations += obs
        rew = env._reward()
        if rew == 0:
            err_info = "linear"

    # Angular shaking
    if err_info == "":
        dur = 150
        obs = rot_shake(env, dur, np.deg2rad(rot))
        observations += obs
        obs = rot_shake(env, dur, -np.deg2rad(rot))
        observations += obs
        rew = env._reward()
        if rew == 0:
            err_info = "shake"

    # Linear shake again
    if err_info == "":
        dur = 150
        obs = linear_shake(env, dur, -lin / dur, second_shake=True)
        observations += obs
        obs = linear_shake(env, dur, lin / dur, second_shake=True)
        observations += obs
        rew = env._reward()
        if rew == 0:
            err_info = "linear2"

    return rew, err_info


def collect_grasps(margs):
    args, graspfile, objinfo = margs

    # gripper faces down
    init_joints = [
        -0.331397102886776,
        -0.4084196878153061,
        0.27958348738965677,
        -2.100158152658163,
        0.10297468164759681,
        1.7272912586478788,
        0.6990357829811057,
        0.0399,
        0.0399,
    ]

    # setup bullet env
    env = PandaAcronymEnv(renders=args.render, egl_render=args.egl, gravity=False)
    env.reset(init_joints=init_joints, no_table=True, objinfos=[objinfo])

    # load grasps
    obj2rotgrasp_Ts, successes = load_grasps(f"{args.mesh_root}/grasps/{graspfile}")

    with h5py.File(f"{args.mesh_root}/grasps/{graspfile}", "r+") as data:
        bullet_group_name = f"grasps/qualities/bullet"
        dset = data[f"{bullet_group_name}/object_in_gripper"]
        collected = data[f"{bullet_group_name}/collected"]

        for idx, (obj2rotgrasp_T, success) in enumerate(
            zip(obj2rotgrasp_Ts, successes)
        ):
            if collected[idx] and not args.overwrite:
                continue

            pq = pt.pq_from_transform(obj2rotgrasp_T)
            print(f"{idx} success label: {success}\tpose: {pq}")

            env._panda.reset(init_joints)

            # Get end effector pose frame
            pos, orn = p.getLinkState(
                env._panda.pandaUid, env._panda.pandaEndEffectorIndex
            )[:2]
            world2ee_T = pt.transform_from_pq(
                np.concatenate([pos, pr.quaternion_wxyz_from_xyzw(orn)])
            )

            # trimesh and bullet panda gripper convention difference
            # Correct for panda gripper rotation about z axis in bullet
            rotgrasp2grasp_T = pt.transform_from(
                pr.matrix_from_axis_angle([0, 0, 1, -np.pi / 2]), [0, 0, 0]
            )  # correct wrist rotation
            obj2grasp_T = obj2rotgrasp_T @ rotgrasp2grasp_T

            # Move object frame to centroid
            mesh = objinfo["mesh"]
            obj2ctr_T = np.eye(4)
            obj2ctr_T[:3, 3] = -mesh.centroid

            if args.debug:  # Debugging plot
                grasp = [
                    create_gripper_marker(
                        color=[0, 255, 0], tube_radius=0.003
                    ).apply_transform(obj2rotgrasp_T)
                ]
                trimesh.Scene([mesh] + grasp).show()

            world2ctr_T = world2ee_T @ np.linalg.inv(obj2grasp_T)

            # Place single object
            pq = pt.pq_from_transform(world2ctr_T)  # mesh still object centered
            p.resetBasePositionAndOrientation(
                env._objectUids[0], pq[:3], pr.quaternion_xyzw_from_wxyz(pq[3:])
            )  # xyzw
            p.resetBaseVelocity(env._objectUids[0], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

            # Close gripper on object
            joints = env._panda.getJointStates()[0]
            joints[-2:] = [0.0, 0.0]
            obs, rew, done, _ = env.step(joints)
            if rew == 0:
                err_info = "grasp"
            else:
                err_info = ""

            observations = [obs]

            # Linear shaking
            if err_info == "":
                dur = 150
                obs = linear_shake(env, dur, -args.lin / dur)
                observations += obs
                obs = linear_shake(env, dur, args.lin / dur)
                observations += obs
                rew = env._reward()
                if rew == 0:
                    err_info = "linear"

            # Angular shaking
            if err_info == "":
                dur = 150
                obs = rot_shake(env, dur, np.deg2rad(args.rot))
                observations += obs
                obs = rot_shake(env, dur, -np.deg2rad(args.rot))
                observations += obs
                rew = env._reward()
                if rew == 0:
                    err_info = "shake"

            # Linear shake again
            if err_info == "":
                dur = 150
                obs = linear_shake(env, dur, -args.lin / dur, second_shake=True)
                observations += obs
                obs = linear_shake(env, dur, args.lin / dur, second_shake=True)
                observations += obs
                rew = env._reward()
                if rew == 0:
                    err_info = "linear2"

            # log pybullet success for this object and pose
            print(f"\tsuccess actual: {rew}")
            dset[idx] = rew
            collected[idx] = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", help="render", action="store_true")
    parser.add_argument("--egl", help="use egl render", action="store_true")
    parser.add_argument(
        "--mesh_root", help="mesh root", type=str, default="/data/manifolds/acronym"
    )
    parser.add_argument("--overwrite", help="overwrite", action="store_true")
    parser.add_argument("--workers", help="Pool workers", type=int, default=1)
    parser.add_argument(
        "--lin", help="How far to linear shake in a second", default=0.15, type=float
    )
    parser.add_argument(
        "--rot",
        help="How far to angular shake in a second, in degrees",
        default=180,
        type=float,
    )
    parser.add_argument(
        "--debug", help="show debug visualizations", action="store_true"
    )

    args = parser.parse_args()

    # For each grasp file, load mesh and get pybullet labels for all grasps
    grasp_and_object_list = []
    grasp_h5s = os.listdir(f"{args.mesh_root}/grasps")
    for objname in os.listdir(f"{args.mesh_root}/meshes"):
        for grasp_h5 in grasp_h5s:
            grasp_id = grasp_h5.replace(".h5", "")
            scale = float(grasp_id.split("_")[-1])
            try:
                obj_mesh, T_ctr2obj = load_mesh(
                    f"{args.mesh_root}/grasps/{grasp_h5}",
                    mesh_root_dir=args.mesh_root,
                    load_for_bullet=True,
                )
            except Exception as e:
                print(e)
                continue
            objinfo = {
                "name": grasp_h5,
                "urdf_dir": f"{args.mesh_root}/meshes_bullet/{grasp_id}/model_normalized.urdf",
                "scale": scale,
                "T_ctr2obj": T_ctr2obj,
                "mesh": obj_mesh,
            }
            grasp_and_object = (grasp_h5, objinfo)

            with h5py.File(f"{args.mesh_root}/grasps/{grasp_h5}", "r+") as data:
                # Set up bullet dataset
                bullet_group_name = "grasps/qualities/bullet"
                if args.overwrite and bullet_group_name in data:
                    del data[f"{bullet_group_name}"]

                try:
                    data.attrs["lin_shake_duration"]
                except KeyError:
                    data.attrs.create("lin_shake_duration", 150.0 / 1000.0)
                    data.attrs.create("rot_shake_duration", 150.0 / 1000.0)

                if bullet_group_name not in data:
                    bullet_group = data.create_group(bullet_group_name)
                    dset = bullet_group.create_dataset(
                        f"object_in_gripper",
                        (len(data["grasps/qualities/flex/object_in_gripper"]),),
                        dtype="i8",
                    )
                    collected = bullet_group.create_dataset(
                        f"collected",
                        (len(data["grasps/qualities/flex/object_in_gripper"]),),
                        dtype="bool",
                    )
                else:
                    dset = data[f"{bullet_group_name}/object_in_gripper"]
                    if f"{bullet_group_name}/collected" not in data:
                        bullet_group = data[bullet_group_name]
                        collected = bullet_group.create_dataset(
                            f"collected",
                            (len(data["grasps/qualities/flex/object_in_gripper"]),),
                            dtype="bool",
                        )
                    else:
                        collected = data[f"{bullet_group_name}/collected"]

                if np.asarray(collected).sum() == len(collected):
                    print(f"All {grasp_h5} grasps collected, skipping")
                    continue

            grasp_and_object_list.append(grasp_and_object)

    import multiprocessing as m

    with m.Pool(args.workers) as pool:
        margs = [
            (args, grasp_prefix, objinfo)
            for grasp_prefix, objinfo in grasp_and_object_list
        ]
        pool.map(collect_grasps, margs)
