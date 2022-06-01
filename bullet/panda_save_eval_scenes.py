import os
import argparse
import csv
import numpy as np
import pybullet as p
import pytransform3d.rotations as pr
from .panda_env import PandaEnv
from .utils import get_world2bot_transform, draw_pose, get_object_info, place_object
import torch
import theseus as th
from differentiable_robot_model.robot_model import (
    DifferentiableFrankaPanda,
)


def save_joints(joints):
    with open('./data/init_joints.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(joints)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_root", help="mesh root", type=str, default="/data/manifolds/acronym_mini_relabel")
    args = parser.parse_args()

env = PandaEnv(renders=True, gravity=False, cam_look=[-0.05, -0.5, -0.6852])

# Change this to get more complex scenes
# use the first object as a stand-in
objname = os.listdir(f'{args.dset_root}/meshes_bullet')[0] 
objinfo = get_object_info(env, objname, args.dset_root)

urdf_path = DifferentiableFrankaPanda().urdf_path.replace('_no_gripper', '')
robot_model = th.eb.UrdfRobotModel(urdf_path)

count = 0
n_samples = 30
while count < n_samples:
    # Place object in scene
    env.reset(no_table=True, objinfo=objinfo)
    target_pos = [0.5, 0.0, 0.5]
    place_object(env, target_pos, random=False, gravity=False)
    pos, orn = p.getBasePositionAndOrientation(env._objectUids[0])  # xyzw

    # Randomly sample joints for robot
    if True:  # end effector
        from manifold_grasping.generate_grasp_data.collect_dataset import random_transform
        T_world_bot = get_world2bot_transform()

        T_bot_tgt = np.eye(4)
        T_bot_tgt[:3, 3] = target_pos
        draw_pose(T_world_bot @ T_bot_tgt)

        T_tgt_rand = random_transform(max_radius=0.5)
        T_world_rand = T_world_bot @ T_bot_tgt @ T_tgt_rand
        draw_pose(T_world_rand)

        pos = T_world_rand[:3, 3]
        orn_wxyz = pr.quaternion_from_matrix(T_world_rand[:3, :3])  # wxyz
        orn = pr.quaternion_xyzw_from_wxyz(orn_wxyz)

        for i in range(30):  # iterate to a better solution
            joints = env._panda.solveInverseKinematics(pos, orn)
            env._panda.reset(joints=joints)

        pose_ee = robot_model.forward_kinematics(torch.tensor([joints]))['panda_hand']
        T_bot_ee = pose_ee.to_matrix().numpy()
        T_world_ee = T_world_bot @ T_bot_ee
        if not np.allclose(T_world_ee, T_world_rand, atol=1e-05):
            continue

        user_inp = input("save? (y/n): ")
        if user_inp == 'y':
            save_joints(joints)
            count += 1
    else:
        q_min = env._panda.q_min[:7]
        q_max = env._panda.q_max[:7]
        joints = []
        for i in range(len(q_min)):
            q = np.random.uniform(low=q_min[i], high=q_max[i])
            joints.append(q)

        joints += [0.04, 0.04]  # add parallel jaw joints
        env._panda.reset(joints=joints)

        user_inp = input("save? (y/n): ")
        if user_inp == 'y':
            save_joints(joints)
            count += 1
