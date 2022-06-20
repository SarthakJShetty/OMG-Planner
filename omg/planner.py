# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from .optimizer import Optimizer
from .cost import Cost
from .util import *
from .online_learner import Learner

from . import config
import time
import multiprocessing
from copy import deepcopy

import torch
# from liegroups.torch import SE3
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

from omg_bullet.utils import draw_pose, get_world2bot_transform
import numpy as np
# from .viz_trimesh import visualize_predicted_grasp, trajT_to_grasppredT, grasppredT_to_trajT
# import pytorch3d.transforms as ptf
import pybullet as p
# from manifold_grasping.control_pts import *

from manifold_grasping.utils import load_grasps, load_mesh

import pathlib

import theseus as th
from differentiable_robot_model.robot_model import (
    DifferentiableFrankaPanda,
)


def solve_one_pose_ik(input):
    """
    solve for one ik
    """
    (
        end_pose,
        standoff_grasp,
        one_trial,
        init_seed,
        attached,
        reach_tail_len,
        ik_seed_num,
        use_standoff,
        seeds,
    ) = input

    r = config.cfg.ROBOT
    joint = 0.04
    finger_joint = np.array([joint, joint])
    finger_joints = np.tile(finger_joint, (reach_tail_len, 1))
    reach_goal_set = []
    standoff_goal_set = []
    any_ik = False

    for seed in seeds:
        if use_standoff:
            standoff_pose = pack_pose(standoff_grasp[-1])
            standoff_ik = r.inverse_kinematics(
                standoff_pose[:3], ros_quat(standoff_pose[3:]), seed=seed
            )  #
            standoff_iks = [standoff_ik]  # this one can often be off

            if standoff_ik is not None:
                for k in range(reach_tail_len):
                    standoff_pose = pack_pose(standoff_grasp[k])
                    standoff_ik_k = r.inverse_kinematics(
                        standoff_pose[:3],
                        ros_quat(standoff_pose[3:]),
                        seed=standoff_iks[-1],
                    )  #
                    if standoff_ik_k is not None:
                        standoff_iks.append(np.array(standoff_ik_k))
                    else:
                        break
            standoff_iks = standoff_iks[1:]

            if len(standoff_iks) == reach_tail_len:
                if not attached:
                    standoff_iks = standoff_iks[::-1]
                reach_traj = np.stack(standoff_iks)
                diff = np.linalg.norm(np.diff(reach_traj, axis=0))

                if diff < 2:  # smooth
                    standoff_ = standoff_iks[0] if not attached else standoff_iks[-1]
                    reach_traj = np.concatenate([reach_traj, finger_joints], axis=-1)
                    reach_goal_set.append(reach_traj)
                    standoff_goal_set.append(
                        np.concatenate([standoff_, finger_joint])
                    )  # [-1]
                    any_ik = True

        else:
            goal_ik = r.inverse_kinematics(
                end_pose[:3], ros_quat(end_pose[3:]), seed=seed
            )
            if goal_ik is not None:
                reach_goal_set.append(np.concatenate([goal_ik, finger_joint]))
                standoff_goal_set.append(np.concatenate([goal_ik, finger_joint]))
                any_ik = True
    return reach_goal_set, standoff_goal_set, any_ik


class Planner(object):
    """
    Planner class that plans a grasp trajectory
    Tricks such as standoff pregrasp, flip grasps are for real world experiments.
    """

    def __init__(self, env, traj, lazy=False):
        self.cfg = env.cfg
        self.env = env
        self.traj = traj
        self.cost = Cost(env)
        self.optim = Optimizer(env, self.cost)
        self.lazy = lazy

        # Planning methods
        if 'known' in self.cfg.method:
            start_time_ = time.time()
            self.load_grasp_set(env)
            self.setup_goal_set(env, filter_collision=self.cfg.filter_collision)
            self.grasp_init(env)
            self.setup_time = time.time() - start_time_
            self.learner = Learner(env, self.traj, self.cost)
        elif 'learned' in self.cfg.method:
            from bullet.methods.learnedgrasp import LearnedGrasp
            self.grasp_predictor = LearnedGrasp(ckpt_path=self.cfg.learnedgrasp_weights, single_shape_code=self.cfg.single_shape_code, dset_root=self.cfg.dset_root)
            self.setup_time = 0
        elif 'CG' in self.cfg.method:
            from bullet.methods.contact_graspnet import ContactGraspNetInference
            self.grasp_predictor = ContactGraspNetInference()
            # Run grasp predictor to set up grasp set
            self.setup_time = 0
        else:
            raise NotImplementedError

        self.history_trajectories = []
        self.info = []
        self.ik_cache = []
        self.dbg_ids = []

    def grasp_init(self, env=None):
        """
        Use precomputed grasps to initialize the end point and goal set
        """
        grasp_ees = []
        if len(env.objects) > 0:
            self.traj.goal_set = env.objects[env.target_idx].grasps
            self.traj.goal_potentials = env.objects[env.target_idx].grasp_potentials
            if bool(env.objects[env.target_idx].grasps_scores): # not None or empty
                self.traj.goal_quality = env.objects[env.target_idx].grasps_scores
                grasp_ees = env.objects[env.target_idx].grasp_ees
            if self.cfg.use_standoff:
                if len(env.objects[env.target_idx].reach_grasps) > 0:
                    self.traj.goal_set = env.objects[env.target_idx].reach_grasps[:, -1]

        if len(self.traj.goal_set) > 0:
            proj_dist = np.linalg.norm(
                (self.traj.start - np.array(self.traj.goal_set))
                * self.cfg.link_smooth_weight,
                axis=-1,
            )

            if self.traj.goal_quality is None or self.traj.goal_quality == []: # is None or empty
                self.traj.goal_quality = np.ones(len(self.traj.goal_set))

            if self.cfg.goal_idx >= 0: # manual specify
                self.traj.goal_idx = self.cfg.goal_idx

            elif self.cfg.goal_idx == -1:  # initial
                costs = (
                    self.traj.goal_potentials + self.cfg.dist_eps * proj_dist
                )
                self.traj.goal_idx = np.argmin(costs)

            else:
                self.traj.goal_idx = 0

            if self.cfg.ol_alg == "Proj":  #
                self.traj.goal_idx = np.argmin(proj_dist)

    def flip_grasp(self, old_grasps):
        """
        flip wrist in joint space for augmenting symmetry grasps
        """
        grasps = np.array(old_grasps[:])
        neg_mask, pos_mask = (grasps[..., -3] < 0), (grasps[..., -3] > 0)
        grasps[neg_mask, -3] += np.pi
        grasps[pos_mask, -3] -= np.pi
        limits = (grasps[..., -3] < 2.8973 - self.cfg.soft_joint_limit_padding) * (
            grasps[..., -3] > -2.8973 + self.cfg.soft_joint_limit_padding
        )
        return grasps, limits

    def solve_goal_set_ik(
        self, target_obj, env, pose_grasp, grasp_scores=[], one_trial=False, z_upsample=False, y_upsample=False,
        in_global_coords=False
    ):
        """
        Solve the IKs to the goals
        """

        object_pose = unpack_pose(target_obj.pose)
        start_time = time.time()
        init_seed = self.traj.start[:7]
        reach_tail_len = self.cfg.reach_tail_length
        reach_goal_set = []
        standoff_goal_set = []
        score_set = []
        grasp_set = []
        reach_traj_set = []
        cnt = 0
        anchor_seeds = util_anchor_seeds[: self.cfg.ik_seed_num].copy()

        if one_trial == True:
            seeds = init_seed[None, :]
        else:
            seeds = np.concatenate([init_seed[None, :], anchor_seeds[:, :7]], axis=0)

        """ IK prep """
        if in_global_coords:
            pose_grasp_global = pose_grasp
        else:
            pose_grasp_global = np.matmul(object_pose, pose_grasp)  # gripper -> object

        if z_upsample:
            # Added upright/gravity (support from base for placement) upsampling by object global z rotation
            bin_num = 50
            global_rot_z = np.linspace(-np.pi, np.pi, bin_num)
            global_rot_z = np.stack([rotZ(z_ang) for z_ang in global_rot_z], axis=0)
            translation = object_pose[:3, 3]
            pose_grasp_global[:, :3, 3] = (
                pose_grasp_global[:, :3, 3] - object_pose[:3, 3]
            )  # translate to object origin
            pose_grasp_global = np.matmul(global_rot_z, pose_grasp_global)  # rotate
            pose_grasp_global[:, :3, 3] += translation  # translate back

        if y_upsample:
            # Added upsampling by local y rotation around finger antipodal contact
            bin_num = 10
            global_rot_y = np.linspace(-np.pi / 4, np.pi / 4, bin_num)
            global_rot_y = np.stack([rotY(y_ang) for y_ang in global_rot_y], axis=0)
            finger_translation = pose_grasp_global[:, :3, :3].dot(np.array([0, 0, 0.13])) + pose_grasp_global[:, :3, 3]
            local_rotation = np.matmul(pose_grasp_global[:, :3, :3], global_rot_y[:, None, :3, :3])
            delta_translation  = local_rotation.dot(np.array([0, 0, 0.13]))
            pose_grasp_global = np.tile(pose_grasp_global[:,None], (1, bin_num, 1, 1))
            pose_grasp_global[:,:,:3,3]  = (finger_translation[None] - delta_translation).transpose((1,0,2))
            pose_grasp_global[:,:,:3,:3] = local_rotation.transpose((1,0,2,3))
            pose_grasp_global = pose_grasp_global.reshape(-1, 4, 4)

        # standoff
        pose_standoff = np.tile(np.eye(4), (reach_tail_len, 1, 1, 1))
        if self.cfg.use_standoff:
            pose_standoff[:, 0, 2, 3] = (
                -1
                * self.cfg.standoff_dist
                * np.linspace(0, 1, reach_tail_len, endpoint=False)
            )
        standoff_grasp_global = np.matmul(pose_grasp_global, pose_standoff)
        parallel = self.cfg.ik_parallel
        # parallel = False
        seeds_ = seeds[:]

        if not parallel:
            hand_center = np.empty((0, 3))
            for grasp_idx in range(pose_grasp_global.shape[0]):
                end_pose = pack_pose(pose_grasp_global[grasp_idx])
                if (
                    len(standoff_goal_set) > 0
                    and len(hand_center) > 0
                    and self.cfg.increment_iks
                ):  # augment
                    dists = np.linalg.norm(end_pose[:3] - hand_center, axis=-1)
                    closest_idx, _ = np.argsort(dists)[:5], np.amin(dists)
                    seeds_ = np.concatenate(
                        [
                            seeds,
                            np.array(standoff_goal_set)[closest_idx, :7].reshape(-1, 7),
                        ],
                        axis=0,
                    )

                standoff_pose = standoff_grasp_global[:, grasp_idx]
                reach_goal_set_i, standoff_goal_set_i, any_ik = \
                solve_one_pose_ik(
                    [
                        end_pose,
                        standoff_pose,
                        one_trial,
                        init_seed,
                        target_obj.attached,
                        self.cfg.reach_tail_length,
                        self.cfg.ik_seed_num,
                        self.cfg.use_standoff,
                        seeds_,
                    ]
                )
                reach_goal_set.extend(reach_goal_set_i)
                standoff_goal_set.extend(standoff_goal_set_i)
                if grasp_scores != []:
                    score_set.extend([grasp_scores[grasp_idx] for _ in range(len(standoff_goal_set_i))])
                    grasp_set.extend([pose_grasp_global[grasp_idx] for _ in range(len(standoff_goal_set_i))])

                if not any_ik:
                    cnt += 1
                else:
                    hand_center = np.concatenate(
                        [
                            hand_center,
                            np.tile(end_pose[:3], (len(standoff_goal_set_i), 1)),
                        ],
                        axis=0,
                    )

        else:
            processes = 4 # multiprocessing.cpu_count() // 2
            reach_goal_set = (
                np.zeros([0, self.cfg.reach_tail_length, 9])
                if self.cfg.use_standoff
                else np.zeros([0, 9])
            )
            standoff_goal_set = np.zeros([0, 9])
            grasp_set = np.zeros([0, 7])
            any_ik, cnt = [], 0
            p = multiprocessing.Pool(processes=processes)

            num = pose_grasp_global.shape[0]
            for i in range(0, num, processes):
                param_list = [
                    [
                        pack_pose(pose_grasp_global[idx]),
                        standoff_grasp_global[:, idx],
                        one_trial,
                        init_seed,
                        target_obj.attached,
                        self.cfg.reach_tail_length,
                        self.cfg.ik_seed_num,
                        self.cfg.use_standoff,
                        seeds_,
                    ]
                    for idx in range(i, min(i + processes, num - 1))
                ]

                res = p.map(solve_one_pose_ik, param_list)
                any_ik += [s[2] for s in res]

                if np.sum([s[2] for s in res]) > 0:
                    reach_goal_set = np.concatenate(
                        (
                            reach_goal_set,
                            np.concatenate(
                                [np.array(s[0]) for s in res if len(s[0]) > 0],
                                axis=0,
                            ),
                        ),
                        axis=0,
                    )
                    standoff_goal_set = np.concatenate(
                        (
                            standoff_goal_set,
                            np.concatenate(
                                [s[1] for s in res if len(s[1]) > 0], axis=0
                            ),
                        ),
                        axis=0,
                    )
                    if grasp_scores != []:
                        # grasp score for every new reach goal set result
                        new_score_set = np.concatenate(
                                    [[grasp_scores[i+idx] for _ in range(len(s[1]))]
                                        for idx, s in enumerate(res) if len(s[1]) > 0], axis=0)
                        score_set = np.concatenate(
                            (
                                score_set,
                                new_score_set
                            ),
                            axis=0,
                        )
                        new_grasp_set = np.concatenate(
                                    [[pack_pose(pose_grasp_global[i+idx]) for _ in range(len(s[1]))]
                                        for idx, s in enumerate(res) if len(s[1]) > 0], axis=0)
                        grasp_set = np.concatenate(
                            (
                                grasp_set,
                                new_grasp_set
                            ),
                            axis=0
                        )

                if self.cfg.increment_iks:
                    max_index = np.random.choice(
                        np.arange(len(standoff_goal_set)),
                        min(len(standoff_goal_set), 20),
                    )
                    seeds_ = np.concatenate(
                        (seeds, standoff_goal_set[max_index, :7])
                    )
            p.terminate()
            cnt = np.sum(1 - np.array(any_ik))
        if not self.cfg.silent:
            print(
            "{} IK init time: {:.3f}, failed_ik: {}, goal set num: {}/{}".format(
                target_obj.name,
                time.time() - start_time,
                cnt,
                len(reach_goal_set),
                pose_grasp_global.shape[0],
            )
        )
        return list(reach_goal_set), list(standoff_goal_set), list(score_set), list(grasp_set)

    def load_grasp_set(self, env):
        """
        Example to load precomputed grasps for YCB Objects.
        """
        for i, target_obj in enumerate(env.objects):
            if target_obj.compute_grasp and (i == env.target_idx or not self.lazy):

                if not target_obj.attached:

                    """ simulator generated poses """
                    if len(target_obj.grasps_poses) == 0:
                        """ acronym objects """
                        if 'acronym' in target_obj.mesh_path:
                            mesh_root = pathlib.Path(target_obj.mesh_path).parents[2]
                            grasps_path = str(mesh_root / f"grasps/{target_obj.name}.h5")
                            obj_mesh, T_ctr2obj = load_mesh(grasps_path, mesh_root_dir=mesh_root)
                            Ts_obj2rotgrasp, _, success = load_grasps(grasps_path)
                            Ts_obj2rotgrasp = Ts_obj2rotgrasp[success == 1]
                            pose_grasp = T_ctr2obj @ Ts_obj2rotgrasp

                            if False:  # debug visualization
                                import trimesh
                                from acronym_tools import create_gripper_marker
                                grasps_v = [create_gripper_marker(color=[0, 0, 255]).apply_transform(T) for T in (T_ctr2obj @ Ts_obj2rotgrasp)[:50]]
                                m = obj_mesh.apply_transform(T_ctr2obj)
                                trimesh.Scene([m] + grasps_v).show()
                        else:
                            simulator_path = (
                                self.cfg.robot_model_path
                                + "/../grasps/simulated/{}.npy".format(target_obj.name)
                            )
                            if not os.path.exists(simulator_path):
                                continue
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

                        if False:
                            T = get_world2bot_transform()
                            T_b2o = pt.transform_from_pq(target_obj.pose)
                            T_w2o = T @ T_b2o
                            [draw_pose(T @ T_b2o @ x) for x in pose_grasp[:50]]

                        offset_pose = np.array(rotZ(np.pi / 2)) # rotate about z axis
                        pose_grasp = np.matmul(pose_grasp, offset_pose)  # flip x, y
                        pose_grasp = ycb_special_case(pose_grasp, target_obj.name)
                        target_obj.grasps_poses = pose_grasp
                    else:
                        pose_grasp = target_obj.grasps_poses
                    z_upsample = False

                else:  # placement
                    pose_grasp = np.linalg.inv(unpack_pose(target_obj.rel_hand_pose))[
                        None
                    ]
                    z_upsample = True

                target_obj.reach_grasps, target_obj.grasps, _, _ = self.solve_goal_set_ik(
                    target_obj, env, pose_grasp, z_upsample=z_upsample, y_upsample=self.cfg.y_upsample
                )
                target_obj.grasp_potentials = []

                if (
                    self.cfg.augment_flip_grasp
                    and not target_obj.attached
                    and len(target_obj.reach_grasps) > 0
                ):
                    """ add augmenting symmetry grasps in C space """
                    flip_grasps, flip_mask = self.flip_grasp(target_obj.grasps)
                    flip_reach, flip_reach_mask = self.flip_grasp(
                        target_obj.reach_grasps
                    )
                    mask = flip_mask
                    target_obj.reach_grasps.extend(list(flip_reach[mask]))
                    target_obj.grasps.extend(list(flip_grasps[mask]))
                target_obj.reach_grasps = np.array(target_obj.reach_grasps)
                target_obj.grasps = np.array(target_obj.grasps)

                if (
                    self.cfg.remove_flip_grasp
                    and len(target_obj.reach_grasps) > 0
                    and not target_obj.attached
                ):
                    """ remove grasps in task space that have large rotation change """
                    start_hand_pose = (
                        self.env.robot.robot_kinematics.forward_kinematics_parallel(
                            wrap_value(self.traj.start)[None]
                        )[0][7]
                    )
                    if self.cfg.use_standoff:
                        n = 5
                        interpolated_traj = multi_interpolate_waypoints(
                            self.traj.start,
                            np.array(target_obj.reach_grasps[:, -1]),
                            n,
                            self.traj.dof, # 9,
                            "linear",
                        )
                        target_hand_pose = (
                            self.env.robot.robot_kinematics.forward_kinematics_parallel(
                                wrap_values(interpolated_traj)
                            )[:, 7]
                        )
                        target_hand_pose = target_hand_pose.reshape(-1, n, 4, 4)
                    else:
                        target_hand_pose = (
                            self.env.robot.robot_kinematics.forward_kinematics_parallel(
                                wrap_values(np.array(target_obj.grasps))
                            )[:, 7]
                        )

                    if len(target_hand_pose.shape) == 3:
                        target_hand_pose = target_hand_pose[:,None]

                    # difference angle
                    R_diff = np.matmul(target_hand_pose[..., :3, :3], start_hand_pose[:3,:3].transpose(1,0))
                    angle = np.abs(np.arccos((np.trace(R_diff, axis1=2, axis2=3) - 1 ) /  2))
                    angle = angle * 180 / np.pi
                    rot_masks = angle > self.cfg.target_hand_filter_angle
                    z = target_hand_pose[..., :3, 0] / np.linalg.norm(target_hand_pose[..., :3, 0], axis=-1, keepdims=True)
                    downward_masks = z[:,:,-1] < -0.3
                    masks = (rot_masks + downward_masks).sum(-1) > 0
                    target_obj.reach_grasps = list(target_obj.reach_grasps[~masks])
                    target_obj.grasps = list(target_obj.grasps[~masks])

    def setup_goal_set(self, env, filter_collision=True, filter_diversity=True):
        """
        Remove the goals that are in collision
        """
        """ collision """
        for i, target_obj in enumerate(env.objects):
            goal_set = target_obj.grasps
            reach_goal_set = target_obj.reach_grasps

            if len(goal_set) > 0 and target_obj.compute_grasp:  # goal_set
                potentials, _, vis_points, collide = self.cost.batch_obstacle_cost(
                    goal_set, special_check_id=i, uncheck_finger_collision=-1
                )  # n x (m + 1) x p (x 3)

                threshold = (
                    0.5
                    * (self.cfg.epsilon - self.cfg.ik_clearance) ** 2
                    / self.cfg.epsilon
                )  #
                collide = collide.sum(-1).sum(-1).detach().cpu().numpy()
                potentials = potentials.sum(dim=(-2, -1)).detach().cpu().numpy()
                ik_goal_num = len(goal_set)

                if filter_collision:
                    collision_free = (
                        collide <= self.cfg.allow_collision_point
                    ).nonzero()  # == 0

                    # new_goal_set = []
                    ik_goal_num = len(goal_set)
                    goal_set = [goal_set[idx] for idx in collision_free[0]]
                    reach_goal_set = [reach_goal_set[idx] for idx in collision_free[0]]
                    if target_obj.grasps_scores is not None and target_obj.grasps_scores != []:
                        try:
                            grasp_scores = [target_obj.grasps_scores[idx] for idx in collision_free[0]]
                            grasp_ees = [target_obj.grasp_ees[idx] for idx in collision_free[0]]
                        except Exception as e:
                            import IPython; IPython.embed()
                    potentials = potentials[collision_free[0]]
                    vis_points = vis_points[collision_free[0]]

                """ diversity """
                diverse = False
                sample = False
                num = len(goal_set)
                indexes = range(num)

                if filter_diversity:
                    if num > 0:
                        diverse = True
                        unique_grasps = [goal_set[0]]  # diversity
                        indexes = []

                        for j, joint in enumerate(goal_set):
                            dists = np.linalg.norm(
                                np.array(unique_grasps) - joint, axis=-1
                            )
                            min_dist = np.amin(dists)
                            if min_dist < 0.5:  # 0.01
                                continue
                            unique_grasps.append(joint)
                            indexes.append(j)
                        num = len(indexes)

                    """ sample """
                if num > 0:
                    sample = True
                    sample_goals = np.random.choice(
                        indexes, min(num, self.cfg.goal_set_max_num), replace=False
                    )

                    target_obj.grasps = [goal_set[int(idx)] for idx in sample_goals]
                    target_obj.reach_grasps = [
                        reach_goal_set[int(idx)] for idx in sample_goals
                    ]
                    if target_obj.grasps_scores is not None and target_obj.grasps_scores != []:
                        target_obj.grasps_scores = [grasp_scores[int(idx)] for idx in sample_goals]
                        target_obj.grasp_ees = [grasp_ees[int(idx)] for idx in sample_goals]
                    target_obj.seeds += target_obj.grasps
                    # compute 5 step interpolation for final reach
                    target_obj.reach_grasps = np.array(target_obj.reach_grasps)
                    target_obj.grasp_potentials.append(potentials[sample_goals])
                    target_obj.grasp_vis_points.append(vis_points[sample_goals])
                    if not self.cfg.silent:
                        print(
                        "{} IK FOUND collision-free goal num {}/{}/{}/{}".format(
                            env.objects[i].name,
                            len(target_obj.reach_grasps),
                            len(target_obj.grasps),
                            num,
                            ik_goal_num,
                            )
                        )
                else:
                    print("{} IK FAIL".format(env.objects[i].name))

                if not sample:
                    target_obj.grasps = []
                    target_obj.reach_grasps = []
                    target_obj.grasps_scores = []
                    target_obj.grasp_ees = []
                    target_obj.grasp_potentials = []
                    target_obj.grasp_vis_points = []
            target_obj.compute_grasp = False

            if False:  # visualization
                T_w2b = get_world2bot_transform()
                draw_pose(T_w2b)
                T_b2o = unpack_pose(target_obj.pose)
                draw_pose(T_w2b @ T_b2o)

                for T_obj2grasp in target_obj.grasps_poses[:30]:
                    draw_pose(T_w2b @ T_b2o @ T_obj2grasp)

    def get_T_obj2bot(self):
        """
        Returns: numpy matrix
        """
        # T_bot2objfrm = pt.transform_from_pq(self.env.objects[self.env.target_idx].pose)
        # T_objfrm2obj = self.cfg.T_obj2ctr
        # T_obj2bot = np.linalg.inv(T_bot2objfrm @ T_objfrm2obj)
        T_bot2obj = pt.transform_from_pq(self.env.objects[self.env.target_idx].pose)
        T_obj2bot = np.linalg.inv(T_bot2obj)
        return T_obj2bot

    def get_T_obj2goal(self, fixed_goal=False):
        """
        Get transform from robot frame to desired grasp frame
        Returns: numpy matrix
        """
        if fixed_goal: # use goal from pre-existing grasp set
            goal_joints = wrap_value(self.traj.goal_set[self.traj.goal_idx]) # degrees
            goal_poses = self.cfg.ROBOT.forward_kinematics_parallel(
                joint_values=goal_joints[np.newaxis, :], base_link=self.cfg.base_link)[0]
            T_bot2goal = goal_poses[-3]

            # if True:
            #     import pybullet as p
            #     pos, orn = p.getBasePositionAndOrientation(0)
            #     T_world2bot = np.eye(4)
            #     T_world2bot[:3, :3] = np.asarray(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            #     T_world2bot[:3, 3] = pos
            #     draw_pose(T_world2bot @ T_bot2goal)

            T_obj2bot = self.get_T_obj2bot()
            T_obj2goal = T_obj2bot @ T_bot2goal

        else: # predict grasp
            pass

        return T_obj2goal

    def CHOMP_update(self, traj, pose_goal, robot_model):
        q_curr = torch.tensor(traj.data[-1], device='cpu', dtype=torch.float32).unsqueeze(0)
        q_curr.requires_grad = True

        def fn(q, vis=False):
            pose_ee = robot_model.forward_kinematics(q)['panda_hand'] # SE(3) 3x4
            if vis: # visualize
                T_b2e_np = pose_ee.to_matrix().detach().squeeze().numpy()
                # T_b2g_np = pose_goal.to_matrix().detach().squeeze().numpy()
                draw_pose(self.T_w2b_np @ T_b2e_np) # ee in world frame
                # draw_pose(self.T_w2b_np @ T_b2g_np, alt_color=True) # goal in world frame
            residual = pose_goal.local(pose_ee) # SE(3) 1 x 6
            # residual = pose_ee.local(pose_goal) # SE(3) 1 x 6
            # residual[:, :3] = (1.0 / 0.08) * np.pi
            return residual
        residual = fn(q_curr, vis=True) # 1 x 6
        mse = torch.linalg.norm(residual, ord='fro')
        if mse.item() == 0:
            traj.goal_cost = 0
            traj.goal_grad = np.zeros((1, 7))
        else:
            mse.backward()
            traj.goal_cost = mse.item()
            traj.goal_grad = q_curr.grad.cpu().numpy()[:, :7]

    def plan(self, traj, pc=None, viz_env=None):
        """
        Run chomp optimizer to do trajectory optmization
        """
        self.traj = traj
        self.history_trajectories = [np.copy(traj.data)]
        self.info = []
        self.selected_goals = []
        start_time_ = time.time()
        alg_switch = self.cfg.ol_alg != "Baseline" and 'GF_learned' not in self.cfg.method

        best_traj_idx = -1
        best_traj = None # Save lowest cost trajectory
        best_cost = 1000
        if (not self.cfg.goal_set_proj) or len(self.traj.goal_set) > 0 \
            or 'GF' in self.cfg.method:
            self.T_w2b_np = get_world2bot_transform()

            urdf_path = DifferentiableFrankaPanda().urdf_path.replace('_no_gripper', '')
            robot_model = th.eb.UrdfRobotModel(urdf_path)

            # Get shape code for point cloud
            if pc is not None:
                # mean_pc = np.mean(pc, axis=0)
                # pc_obj = deepcopy(pc)
                # pc_obj[:, :3] -= mean_pc[:3]
                shape_code, mean_pc = self.grasp_predictor.get_shape_code(pc)
                T_w2pc = np.eye(4)
                T_w2pc[:3, 3] = mean_pc[:3]
                T_b2pc = np.linalg.inv(self.T_w2b_np) @ T_w2pc
                draw_pose(T_w2pc)
            
            self.optim.init(self.traj)
            ran_initial_ik = False

            for t in range(self.cfg.optim_steps + self.cfg.extra_smooth_steps):
                print(f"plan step {t}")
                start_time = time.time()

                if (
                    # self.cfg.goal_set_proj and
                    alg_switch and t < self.cfg.optim_steps
                ):
                    self.learner.update_goal()
                    self.selected_goals.append(self.traj.goal_idx)

                if 'GF_learned' in self.cfg.method:
                    # Get input pose to network as position+quaternion in object frame
                    q = torch.tensor(self.traj.data[-1], device='cpu').unsqueeze(0)
                    pose_b2e = robot_model.forward_kinematics(q)['panda_hand'] # SE(3) 3x4
                    T_b2e_np = pose_b2e.to_matrix().detach().squeeze().numpy()

                    # object frame is known in 1hot encoding
                    # object frame is estimated with shape codes

                    T_o2b_np = self.get_T_obj2bot()
                    T_o2e_np = T_o2b_np @ T_b2e_np
                    try:
                        pr.check_matrix(T_o2e_np[:3, :3])
                    except ValueError as e:
                        print(e)
                        print("normalizing matrix")
                        T_o2e_np[:3, :3] = pr.norm_matrix(T_o2e_np[:3, :3])
                    pq_o2e_np = pt.pq_from_transform(T_o2e_np) # xyz wxyz
                        
                    input_pq = torch.tensor(pq_o2e_np, device='cuda', dtype=torch.float32)

                    # Run network
                    shape_info = {}
                    if pc is not None:  # shape code
                        shape_info['shape_code'] = shape_code
                    elif self.cfg.single_shape_code: # single shape code
                        objname = self.env.objects[0].name
                        shape_info['shape_key'] = objname
                    else:  # 1 hot
                        objname = self.env.objects[0].name
                        shape_info['1hot'] = objname
                    output = self.grasp_predictor.forward(input_pq, shape_info)

                    if pc is not None:
                        pq_pc2g = output
                        pq_pc2g_np = pq_pc2g.detach().cpu().numpy()

                        # Get predicted grasp in bot frame
                        T_pc2g_np = np.eye(4)
                        T_pc2g_np[:3, :3] = pr.matrix_from_quaternion(pq_pc2g_np[3:8])
                        T_pc2g_np[:3, 3] = pq_pc2g_np[:3]
                        T_b2g_np = T_b2pc @ T_pc2g_np
                        # draw_pose(self.T_w2b_np @ np.linalg.inv(T_o2b_np) @ T_pc2g_np)
                        pose_b2g = th.SE3(data=torch.tensor(T_b2g_np[:3]).float().unsqueeze(0))
                    else:  # 1 hot
                        pq_o2g = output
                        pq_o2g_np = pq_o2g.detach().cpu().numpy()

                        # Get predicted grasp in bot frame
                        T_o2g_np = np.eye(4)
                        T_o2g_np[:3, :3] = pr.matrix_from_quaternion(pq_o2g_np[3:8])
                        T_o2g_np[:3, 3] = pq_o2g_np[:3]
                        T_b2g_np = np.linalg.inv(T_o2b_np) @ T_o2g_np
                        pose_b2g = th.SE3(data=torch.tensor(T_b2g_np[:3]).float().unsqueeze(0))
                    # Visualization
                    draw_pose(self.T_w2b_np @ T_b2g_np, alt_color=True) # goal in world frame

                    # self.CHOMP_update(self.traj, pose_b2g, robot_model)
                elif 'GF_known' in self.cfg.method:
                    q_goal = torch.tensor(self.traj.goal_set[self.traj.goal_idx], device='cpu', dtype=torch.float32)
                    pose_b2g = robot_model.forward_kinematics(q_goal)['panda_hand']
                    T_b2g_np = pose_b2g.to_matrix().squeeze(0).cpu().numpy()
                    draw_pose(self.T_w2b_np @ T_b2g_np, alt_color=True) # goal in world frame

                # Update trajectory goal cost and gradient with either IK or differentiable robot model
                if 'GF' in self.cfg.method and self.cfg.initial_ik and not ran_initial_ik:
                    pq_b2g = pt.pq_from_transform(T_b2g_np) # wxyz
                    seed = self.traj.start[:7]
                    goal_ik = config.cfg.ROBOT.inverse_kinematics(
                        pq_b2g[:3], ros_quat(pq_b2g[3:]), seed=seed
                    )
                    # if IK fails, just run CHOMP update and try again next iter
                    if goal_ik is None:
                        print(f"Initial IK failed in iter {t}")
                        self.CHOMP_update(self.traj, pose_b2g, robot_model) 
                    else:
                        traj.set_goal_and_interp(goal_ik)
                        traj.goal_cost = 0
                        traj.goal_grad = np.zeros((1, 7))
                        ran_initial_ik = True
                else:
                    self.CHOMP_update(self.traj, pose_b2g, robot_model)

                info_t = self.optim.optimize(self.traj, force_update=True, tstep=t+1)

                if 'GF' in self.cfg.method:
                    info_t['pred_grasp'] = pose_b2g.to_matrix().detach().cpu().numpy()
                self.info.append(info_t)
                self.history_trajectories.append(np.copy(traj.data))
                if self.cfg.use_min_cost_traj:
                    if info_t['cost'] < best_cost:
                        best_cost = info_t['cost']
                        best_traj = np.copy(traj.data)
                        best_traj_idx = t

                if self.cfg.report_time:
                    print("plan optimize:", time.time() - start_time)


                # if viz_env:
                    # viz_env.update_panda_viz(self.traj, k=1)

                if viz_env and info_t['collide'] > 0 and False:
                    while len(self.dbg_ids) > 0:
                        dbg_id = self.dbg_ids.pop(0)
                        p.removeUserDebugItem(dbg_id)
                    # p.removeAllUserDebugItems()

                    fngr_col_idxs = np.where(info_t['collision_pts'][:, :, :, 3] == 255)
                    col_pts = info_t['collision_pts'][fngr_col_idxs][:, :6]
                    for col_pt in col_pts:
                        col_pt_h = np.concatenate((col_pt[:3], [1]))
                        w2col_pt = self.T_w2b_np @ col_pt_h
                        col_pt[3:6] /= 255.0
                        dbg_id = p.addUserDebugLine(
                                w2col_pt[:3], 
                                w2col_pt[:3]+np.array([0.001, 0.001, 0.001]),
                                lineWidth=5.0,
                                lineColorRGB=col_pt[3:6])
                        self.dbg_ids.append(dbg_id)


                # if viz_env and \
                    # (self.info[-1]["terminate"] or t == self.cfg.optim_steps + self.cfg.extra_smooth_steps - 1):
                    # viz_env.update_panda_viz(self.traj)
                    # robot = self.cost.env.robot
                    # robot_pts = robot.collision_points.transpose([0, 2, 1])
                    # (
                    #     robot_poses,
                    #     joint_origins,
                    #     joint_axis,
                    # ) = self.cost.env.robot.robot_kinematics.forward_kinematics_parallel(
                    #     wrap_values(self.traj.data), return_joint_info=True)
                    # ws_positions = self.cost.forward_points(
                    #     robot_poses, robot_pts
                    # )  # p x (m + 1) x n x 3
                    # ws_pos = ws_positions.copy()
                    # ws_pos = ws_pos.transpose([2, 1, 0, 3])
                    # last_col_pts = ws_pos[-1]
                    # fngr_col_pts = last_col_pts[-2:]

                    # p.removeAllUserDebugItems()

                    # for fngr_idx in range(fngr_col_pts.shape[0]):
                    #     for col_pt in fngr_col_pts[fngr_idx]:
                    #         col_pt = np.concatenate([col_pt, [1]])
                    #         # T_col_pt = np.eye(4)
                    #         # T_col_pt[:, 3] = col_pt
                    #         # draw_pose(self.T_w2b_np @ col_pt)
                    #         w2col_pt = self.T_w2b_np @ col_pt
                    #         p.addUserDebugLine(
                    #             w2col_pt[:3], 
                    #             w2col_pt[:3]+np.array([0.001, 0.001, 0.001]),
                    #             lineWidth=2.0,
                    #             lineColorRGB=(1.0, 0, 0))


                if self.info[-1]["terminate"] and t > 0:
                    break

            # compute information for the final
            if not self.info[-1]["terminate"]:
                if self.cfg.use_min_cost_traj:
                    print("Replacing final traj with lowest cost traj")
                    traj.data = best_traj
                    self.info.append(self.optim.optimize(traj, info_only=True))
                    self.history_trajectories.append(best_traj)
                    # with open(f'{self.cfg.exp_dir}/{self.cfg.exp_name}/{self.cfg.scene_file}/{best_traj_idx}.txt', 'w') as f:
                    #     f.write('')
                else:
                    self.info.append(self.optim.optimize(self.traj, info_only=True))
            # else:
                # del self.history_trajectories[-1]

            plan_time = time.time() - start_time_
            res = (
                "SUCCESS BE GENTLE"
                if self.info[-1]["terminate"]
                else "FAIL DONT EXECUTE"
            )
            if not self.cfg.silent:
                print(
                "planning time: {:.3f} PLAN {} Length: {}".format(
                    plan_time, res, len(self.history_trajectories[-1])
                )
            )
            self.info[-1]["time"] = plan_time + self.setup_time

        else:
            if not self.cfg.silent: print("planning not run...")
        return self.info


    # def get_T_bot2ee(self, traj, idx=-1):
    #     """
    #     Returns: numpy matrix
    #     """
    #     angles = traj.data[idx]
    #     end_joints = wrap_value(angles) # rad2deg
    #     end_poses = self.cfg.ROBOT.forward_kinematics_parallel(
    #         joint_values=end_joints[np.newaxis, :], base_link=self.cfg.base_link)[0]
    #     T_bot2ee = end_poses[-3]
    #     return T_bot2ee

    # def pq_from_tau(self, tau):
    #     pq = torch.zeros((7,), device='cuda', dtype=torch.float64)
    #     T = ptf.se3_exp_map(tau.unsqueeze(0), eps=1e-10).squeeze().T
    #     pq[:3] = T[:3, 3]
    #     pq[3:] = ptf.matrix_to_quaternion(T[:3, :3].unsqueeze(0)).squeeze(0) # qw qx qy qz
    #     return pq

    # def pq_from_T(self, T):
    #     pq = torch.zeros((7,), device='cuda', dtype=torch.float64)
    #     p[:3] = T[:3, 3]
    #     pq[3:] = ptf.matrix_to_quaternion(T[:3, :3].unsqueeze(0)).squeeze(0) # qw qx qy qz
    #     return pq

    # def compute_loss(self, tau_b2e, tau_b2g, loss_fn='logmap'):
    #     if loss_fn == 'logmap_split':
    #         # nu := translational component of the exp. coords
    #         # omega := rotational component of the exp. coords
    #         alpha = 0.01
    #         nu_diff = tau_b2e[:3] - tau_b2g[:3]
    #         omega_diff = tau_b2e[3:] - tau_b2g[3:]
    #         loss = 0.5*torch.linalg.norm(nu_diff)**2 + 0.5*torch.linalg.norm(omega_diff)**2
    #     elif loss_fn == 'logmap':
    #         alpha = 0.01
    #         loss = 0.5*torch.linalg.norm(tau_b2e - tau_b2g)**2
    #     elif loss_fn == 'pq':
    #         alpha = 0.05
    #         pq_b2e = self.pq_from_tau(tau_b2e)
    #         pq_b2g = self.pq_from_tau(tau_b2g)
    #         loss = 0.5*torch.linalg.norm(pq_b2e - pq_b2g)**2
    #     elif loss_fn == 'control_points':
    #         alpha = 0.02
    #         T_b2e = ptf.se3_exp_map(tau_b2e.unsqueeze(0), eps=1e-10).squeeze().T
    #         T_b2g = ptf.se3_exp_map(tau_b2g.unsqueeze(0), eps=1e-10).squeeze().T
    #         cp_b2e = transform_control_points(T_b2e.unsqueeze(0).float(), 1, mode='rt', device='cuda', rotate=True)
    #         cp_b2g = transform_control_points(T_b2g.unsqueeze(0).float(), 1, mode='rt', device='cuda', rotate=True)
    #         # loss = control_point_l1_loss(cp_b2e, cp_b2g)
    #         loss = control_point_l2_loss(cp_b2e, cp_b2g)
    #         if True:
    #             T_b2e_np = T_b2e.detach().cpu().numpy()
    #             for cp in cp_b2e[0]:
    #                 T = np.eye(4)
    #                 T[:3, :3] = T_b2e_np[:3, :3]
    #                 T[:, 3] = cp.detach().cpu().numpy()
    #                 draw_pose(self.T_world2bot @ T)
    #     return loss




        # tau_b2e.requires_grad = True

        # # Compute loss
        # loss = self.compute_loss(tau_b2e, tau_b2g, loss_fn=loss_fn)

        # # Backprop the loss and update the query pose
        # loss.backward()
        # tau_b2e_grad = tau_b2e.grad.detach()

        # # Finite difference check
        # if True:
        #     fn = lambda x: (0.5*torch.linalg.norm(x - tau_b2g)**2).unsqueeze(0)
        #     jac = jacobian(f=fn, initial=tau_b2e)

        # # Gradient descent in pose space
        # #   Get step size for pose space gradient
        # if loss_fn == 'logmap_split' or loss_fn == 'logmap':
        #     alpha = 0.05
        # elif loss_fn == 'pq':
        #     alpha = 0.05
        # elif loss_fn == 'control_points':
        #     alpha = 0.02
        # tau_b2e = (tau_b2e - alpha * tau_b2e_grad).detach()
        # if True: # visualize
        #     T_b2e = ptf.se3_exp_map(tau_b2e.unsqueeze(0)).squeeze().T
        #     T_b2e_np = T_b2e.detach().cpu().numpy()
        #     draw_pose(self.T_world2bot @ T_b2e_np, alt_color=True) # ee in world frame
        # return tau_b2e





    # def grad_joints_update(self, tau_b2e, tau_b2g, q_curr, loss_fn='logmap'):
    #     """
    #     update the input transform to move toward goal pose, agnostic to traj opt loop.
    #     """
    #     tau_b2e.requires_grad = True

    #     # Compute loss
    #     loss = self.compute_loss(tau_b2e, tau_b2g, loss_fn=loss_fn)
    #     loss.backward()
    #     tau_b2e_grad = tau_b2e.grad.detach()

    #     # Finite difference check
    #     if True:
    #         def fn(q):
    #             '''return loss in exponential coordinates from joint angles'''
    #             T_b2e_np = self.cfg.ROBOT.forward_kinematics_parallel(
    #                 joint_values=wrap_value(q.unsqueeze(0).cpu().numpy()), base_link=self.cfg.base_link)[0][-3]
    #             T_b2e_fd = torch.tensor(T_b2e_np, device='cuda', dtype=torch.float64)
    #             tau_b2e_fd = ptf.se3_log_map(T_b2e_fd.T.unsqueeze(0), eps=1e-10, cos_bound=1e-10).squeeze(0) # [nu omega]
    #             return (0.5*torch.linalg.norm(tau_b2e_fd - tau_b2g)**2).unsqueeze(0)
    #         jac = jacobian(f=fn, initial=torch.tensor(q_curr[0], device='cuda', dtype=torch.float64))

    #         def fn_e(q):
    #             '''return exponential coordinates from joint angles'''
    #             T_b2e_np = self.cfg.ROBOT.forward_kinematics_parallel(
    #                 joint_values=wrap_value(q.unsqueeze(0).cpu().numpy()), base_link=self.cfg.base_link)[0][-3]
    #             T_b2e_fd = torch.tensor(T_b2e_np, device='cuda', dtype=torch.float64)
    #             tau_b2e_fd = ptf.se3_log_map(T_b2e_fd.T.unsqueeze(0), eps=1e-10, cos_bound=1e-10).squeeze(0) # [nu omega]
    #             return tau_b2e_fd
    #         jac_e = jacobian(f=fn_e, initial=torch.tensor(q_curr[0], device='cuda', dtype=torch.float64))

    #     # Gradient descent in joint space using the manipulator Jacobian
    #     J = self.cfg.ROBOT.jacobian(q_curr.squeeze(0)) # radians
    #     tau_b2e_np = tau_b2e.detach().unsqueeze(1).cpu().numpy() # 6 x 1
    #     tau_b2g_np = tau_b2g.detach().unsqueeze(1).cpu().numpy() # 6 x 1
    #     # tau_b2e_grad = tau_b2e_grad.detach().unsqueeze(1).cpu().numpy() # 6 x 1
    #     T_b2e_np = self.cfg.ROBOT.forward_kinematics_parallel(
    #         joint_values=wrap_value(q_curr), base_link=self.cfg.base_link)[0][-3]

    #     # Geometric jacobian
    #     transforms = self.cfg.ROBOT.forward_kinematics_parallel(
    #         joint_values=wrap_value(q_curr), base_link=self.cfg.base_link)[0]

    #     joints_pos = transforms[1:7 + 1, :3, 3]
    #     ee_pos = transforms[-1, :3, 3]
    #     axes = transforms[1:7 + 1, :3, 2]
    #     joints_pos = transforms[:7, :3, 3]
    #     ee_pos = transforms[3, :3, 3]
    #     axes = transforms[:7, :3, 2]

    #     J = np.r_[np.cross(axes, ee_pos - joints_pos).T, axes.T]

    #     # Use J from manual backprop calculation instead of J inv
    #     # q_b2e_grad = (tau_b2e_np - tau_b2g_np).T @ J # 1 x 7
    #     q_b2e_grad = (tau_b2e_np - tau_b2g_np).T @ jac_e.cpu().numpy() # 1 x 7

    #     q_next = deepcopy(q_curr) # 1 x 7
    #     # q_next = q_curr - 0.1*jac.cpu().numpy()# 1 x 7
    #     q_next = q_curr - 0.1*q_b2e_grad # 1 x 7

    #     T_b2e_np = self.cfg.ROBOT.forward_kinematics_parallel(
    #         joint_values=wrap_value(q_next), base_link=self.cfg.base_link)[0][-3]
    #     T_b2e = torch.tensor(T_b2e_np, device='cuda', dtype=torch.float64)
    #     tau_b2e = ptf.se3_log_map(T_b2e.T.unsqueeze(0), eps=1e-10, cos_bound=1e-10).squeeze(0) # [nu omega]
    #     draw_pose(self.T_world2bot @ T_b2e_np) # ee in world frame

    #     return tau_b2e, q_next






    # def CHOMP_update(self, traj, tau_b2g, loss='pq'):
    #     q_curr = traj.data[-1][np.newaxis, :7]

    #     # Get current end effector pose in exp coordinates
    #     T_b2e_np = self.get_T_bot2ee(traj, idx=-1)
    #     T_b2e = torch.tensor(T_b2e_np, device='cuda', dtype=torch.float64)
    #     tau_b2e = ptf.se3_log_map(T_b2e.T.unsqueeze(0), eps=1e-10, cos_bound=1e-10).squeeze(0)
    #     draw_pose(self.T_world2bot @ T_b2e_np) # ee in world frame

    #     # Finite difference gradient with L2 loss on end effector pose
    #     def fn(q):
    #         '''return loss in exponential coordinates from joint angles'''
    #         T_b2e_np = self.cfg.ROBOT.forward_kinematics_parallel(
    #             joint_values=wrap_value(q.unsqueeze(0).cpu().numpy()), base_link=self.cfg.base_link)[0][-3]
    #         T_b2e_fd = torch.tensor(T_b2e_np, device='cuda', dtype=torch.float64)
    #         tau_b2e_fd = ptf.se3_log_map(T_b2e_fd.T.unsqueeze(0), eps=1e-10, cos_bound=1e-10).squeeze(0) # [nu omega]
    #         pq_b2e_fd = self.pq_from_tau(tau_b2e_fd)
    #         pq_b2g_fd = self.pq_from_tau(tau_b2g)
    #         # return (0.5*torch.linalg.norm(tau_b2e_fd - tau_b2g)**2).unsqueeze(0)
    #         return (0.5*torch.linalg.norm(pq_b2e_fd - pq_b2g_fd)**2).unsqueeze(0)
    #     jac = jacobian(f=fn, initial=torch.tensor(q_curr[0], device='cuda', dtype=torch.float64))

    #     # Compute loss
    #     if loss == 'logmap_split':
    #         # nu := translational component of the exp. coords
    #         # omega := rotational component of the exp. coords
    #         nu_diff = tau_b2e[:3] - tau_b2g[:3]
    #         omega_diff = tau_b2e[3:] - tau_b2g[3:]
    #         loss = torch.linalg.norm(nu_diff) + torch.linalg.norm(omega_diff)
    #     elif loss == 'logmap':
    #         loss = 0.5*torch.linalg.norm(tau_b2e - tau_b2g)**2
    #     elif loss == 'pq':
    #         pq_b2e = self.pq_from_tau(tau_b2e)
    #         pq_b2g = self.pq_from_tau(tau_b2g)
    #         loss = 0.5*torch.linalg.norm(pq_b2e - pq_b2g)**2

    #     traj.goal_cost = loss.item()
    #     traj.goal_grad = jac.cpu().numpy()
    #     print(f"cost: {traj.goal_cost}, grad: {traj.goal_grad}")
