import os
import argparse
import csv
import numpy as np
import pybullet as p
import pytransform3d.rotations as pr
from .panda_env import PandaEnv
from .utils import get_world2bot_transform, draw_pose, get_object_info, place_object
import torch
from manifold_grasping.generate_grasp_data.collect_dataset import random_transform
import theseus as th
from differentiable_robot_model.robot_model import (
    DifferentiableFrankaPanda,
)


def save_scene(pos_orn):
    pos, orn = pos_orn # xyzw
    with open('./data/scenes.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(pos + orn)
        # writer.writerow(joints)


# def save_joints(joints):
#     with open('./data/init_joints.csv', 'a') as f:
#         writer = csv.writer(f)
#         writer.writerow(joints)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_root", help="mesh root", type=str, default="/data/manifolds/acronym_mini_relabel")
    parser.add_argument("--table", help="toggle table", action='store_true')
    parser.add_argument("--rotate_objects", help="sample object rotations", action='store_true')
    parser.add_argument("--sample_joints", help="sample robot joints", action='store_true')
    parser.add_argument("--samples", help="number of samples", type=int, default=30)
    args = parser.parse_args()

    # If using the table, turn gravity on and set camera parameters
    if args.table:
        gravity = True
        cam_look = [-0.05, -0.5, -1.1]
        target_pos = [0.5, 0.0, 0.1]
    else:
        gravity = False
        cam_look = [-0.05, -0.5, -0.6852]
        target_pos = [0.5, 0.0, 0.5]

    env = PandaEnv(renders=True, gravity=gravity, cam_look=cam_look)

    # Change this to get more complex scenes
    # use the first object as a stand-in
    objname = os.listdir(f'{args.dset_root}/meshes_bullet')[3]
    objinfo = get_object_info(env, objname, args.dset_root)

    urdf_path = DifferentiableFrankaPanda().urdf_path.replace('_no_gripper', '')
    robot_model = th.eb.UrdfRobotModel(urdf_path)

    count = 0
    while count < args.samples:
        # Place object in scene
        env.reset(no_table=not args.table, objinfo=objinfo)
        place_object(env, target_pos, random=args.rotate_objects, gravity=gravity)
        pos, orn = p.getBasePositionAndOrientation(env._objectUids[0])  # xyzw

        # Randomly sample joints for robot
        if args.sample_joints:  # end effector
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
            if not np.allclose(T_world_ee, T_world_rand, atol=1e-04):
                continue

        # user_inp = input("save? (y/n): ")
        # if user_inp == 'y':
        # save_joints(joints)
        # save_obj_pose(pos, orn)
        save_scene((pos, orn))
        count += 1

