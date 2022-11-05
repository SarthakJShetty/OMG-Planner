# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import multiprocessing
import pathlib
import time

import numpy as np
import pybullet as p
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import theseus as th
import torch
from differentiable_robot_model.robot_model import DifferentiableFrankaPanda
from ngdf.control_pts import *
from ngdf.utils import load_grasps, load_mesh
from omg_bullet.utils import draw_pose, get_world2bot_transform

from . import config
from .cost import Cost
from .online_learner import Learner
from .optimizer import Optimizer
from .util import *


# The returned tensor will have 7 elements, [x, y, z, qw, qx, qy, qz] where
# [x y z] corresponds to the translation and [qw qx qy qz] to the quaternion
# using the [w x y z] convention
def pose_to_pq(pose):
    pq = th.SE3().tensor.new_zeros(1, 7)
    pq[:, :3] = pose[:, :, 3]
    pq[:, 3:] = th.SO3(tensor=pose[:, :, :3]).to_quaternion()
    return pq


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

    def __init__(self, env, traj, lazy=False, hydra_cfg=None):
        self.cfg = env.cfg
        self.env = env
        self.traj = traj
        self.cost = Cost(env)
        self.optim = Optimizer(env, self.cost)
        self.lazy = lazy

        # Planning methods
        if "known" in self.cfg.method:
            start_time_ = time.time()
            self.load_grasp_set(env)
            self.setup_goal_set(env, filter_collision=self.cfg.filter_collision)
            self.grasp_init(env)
            self.setup_time = time.time() - start_time_
            self.learner = Learner(env, self.traj, self.cost)
        elif "NGDF" in self.cfg.method:
            self.use_double = True
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
            from omg_bullet.methods.learnedgrasp import LearnedGrasp

            self.grasp_predictor = LearnedGrasp(
                ckpt_paths=self.cfg.grasp_weights,
                use_double=self.use_double,
                hydra_cfg=hydra_cfg,
            )
            self.setup_time = 0
        elif "CG" in self.cfg.method:
            from omg_bullet.methods.contact_graspnet import ContactGraspNetInference

            self.grasp_predictor = ContactGraspNetInference()
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
            if bool(env.objects[env.target_idx].grasps_scores):  # not None or empty
                self.traj.goal_quality = env.objects[env.target_idx].grasps_scores
                grasp_ees = env.objects[env.target_idx].grasp_ees
            if self.cfg.goal_set_proj and self.cfg.use_standoff:
                if len(env.objects[env.target_idx].reach_grasps) > 0:
                    self.traj.goal_set = env.objects[env.target_idx].reach_grasps[:, -1]

        if len(self.traj.goal_set) > 0:
            proj_dist = np.linalg.norm(
                (self.traj.start - np.array(self.traj.goal_set))
                * self.cfg.link_smooth_weight,
                axis=-1,
            )

            if (
                self.traj.goal_quality is None or self.traj.goal_quality == []
            ):  # is None or empty
                self.traj.goal_quality = np.ones(len(self.traj.goal_set))

            if self.cfg.goal_idx >= 0:  # manual specify
                self.traj.goal_idx = self.cfg.goal_idx

            elif self.cfg.goal_idx == -1:  # initial
                costs = self.traj.goal_potentials + self.cfg.dist_eps * proj_dist
                self.traj.goal_idx = np.argmin(costs)

            elif self.cfg.goal_idx == -4:  # max score
                costs = self.traj.goal_quality
                self.traj.goal_idx = np.argmax(costs)

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
        self,
        target_obj,
        env,
        pose_grasp,
        grasp_scores=[],
        one_trial=False,
        z_upsample=False,
        y_upsample=False,
        in_global_coords=False,
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
            finger_translation = (
                pose_grasp_global[:, :3, :3].dot(np.array([0, 0, 0.13]))
                + pose_grasp_global[:, :3, 3]
            )
            local_rotation = np.matmul(
                pose_grasp_global[:, :3, :3], global_rot_y[:, None, :3, :3]
            )
            delta_translation = local_rotation.dot(np.array([0, 0, 0.13]))
            pose_grasp_global = np.tile(pose_grasp_global[:, None], (1, bin_num, 1, 1))
            pose_grasp_global[:, :, :3, 3] = (
                finger_translation[None] - delta_translation
            ).transpose((1, 0, 2))
            pose_grasp_global[:, :, :3, :3] = local_rotation.transpose((1, 0, 2, 3))
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
                reach_goal_set_i, standoff_goal_set_i, any_ik = solve_one_pose_ik(
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
                    score_set.extend(
                        [
                            grasp_scores[grasp_idx]
                            for _ in range(len(standoff_goal_set_i))
                        ]
                    )
                grasp_set.extend(
                    [
                        pose_grasp_global[grasp_idx]
                        for _ in range(len(standoff_goal_set_i))
                    ]
                )

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
            processes = 4  # multiprocessing.cpu_count() // 2
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
                            [
                                [grasp_scores[i + idx] for _ in range(len(s[1]))]
                                for idx, s in enumerate(res)
                                if len(s[1]) > 0
                            ],
                            axis=0,
                        )
                        score_set = np.concatenate(
                            (score_set, new_score_set),
                            axis=0,
                        )
                    new_grasp_set = np.concatenate(
                        [
                            [
                                pack_pose(pose_grasp_global[i + idx])
                                for _ in range(len(s[1]))
                            ]
                            for idx, s in enumerate(res)
                            if len(s[1]) > 0
                        ],
                        axis=0,
                    )
                    grasp_set = np.concatenate((grasp_set, new_grasp_set), axis=0)

                if self.cfg.increment_iks:
                    max_index = np.random.choice(
                        np.arange(len(standoff_goal_set)),
                        min(len(standoff_goal_set), 20),
                    )
                    seeds_ = np.concatenate((seeds, standoff_goal_set[max_index, :7]))
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
        return (
            list(reach_goal_set),
            list(standoff_goal_set),
            list(score_set),
            list(grasp_set),
        )

    def load_grasp_set_cg(self, env, pred_grasps_cam, scores, T_world_cam2):
        """
        Process predicted contact graspnet grasps
        """
        target_obj = env.objects[env.target_idx]

        T_b2w = np.linalg.inv(get_world2bot_transform())
        pose_grasp = (
            T_b2w @ T_world_cam2 @ pred_grasps_cam
        )  # pose_grasp is in bot frame
        target_obj.grasps_poses = pose_grasp

        # target_obj.pose is in bot frame
        z_upsample = False
        (
            target_obj.reach_grasps,
            target_obj.grasps,
            target_obj.grasp_scores,
            grasp_set,
        ) = self.solve_goal_set_ik(
            target_obj,
            env,
            pose_grasp,
            grasp_scores=scores,
            z_upsample=z_upsample,
            y_upsample=self.cfg.y_upsample,
            in_global_coords=True,
        )
        target_obj.reach_grasps = np.array(target_obj.reach_grasps)
        target_obj.grasps = np.array(target_obj.grasps)
        target_obj.grasp_scores = np.array(target_obj.grasp_scores)

        target_obj.grasp_potentials = [0 for _ in range(len(target_obj.grasps))]
        return grasp_set

    def load_grasp_set(self, env):
        """
        Example to load precomputed grasps for YCB Objects.
        """
        for i, target_obj in enumerate(env.objects):
            if target_obj.compute_grasp and (i == env.target_idx or not self.lazy):

                if not target_obj.attached:

                    """simulator generated poses"""
                    if len(target_obj.grasps_poses) == 0:
                        """acronym objects"""
                        if "acronym" in target_obj.mesh_path:
                            mesh_root = pathlib.Path(target_obj.mesh_path).parents[2]
                            grasps_path = str(
                                mesh_root / f"grasps/{target_obj.name}.h5"
                            )
                            obj_mesh, T_ctr2obj = load_mesh(
                                grasps_path,
                                mesh_root_dir=mesh_root,
                                load_for_bullet=True,
                            )
                            Ts_obj2rotgrasp, _, success = load_grasps(grasps_path)
                            Ts_obj2rotgrasp = Ts_obj2rotgrasp[success == 1]
                            pose_grasp = Ts_obj2rotgrasp  # load grasps in original mesh frame, not mesh centroid frame

                            if False:  # debug visualization
                                import trimesh
                                from acronym_tools import create_gripper_marker

                                # grasps_v = [create_gripper_marker(color=[0, 0, 255]).apply_transform(T) for T in (T_ctr2obj @ Ts_obj2rotgrasp)[:50]]
                                grasps_v = [
                                    create_gripper_marker(
                                        color=[0, 0, 255]
                                    ).apply_transform(T_ctr2obj @ T)
                                    for T in (pose_grasp)[:50]
                                ]
                                # m = obj_mesh.apply_transform(np.linalg.inv(T_ctr2obj))
                                m = obj_mesh
                                trimesh.Scene([m] + grasps_v).show()
                        else:
                            simulator_path = (
                                self.cfg.robot_model_path
                                + "/../grasps/simulated/{}.npy".format(target_obj.name)
                            )
                            if not os.path.exists(simulator_path):
                                continue
                            try:
                                simulator_grasp = np.load(
                                    simulator_path, allow_pickle=True
                                )
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
                            [draw_pose(T @ T_b2o @ x) for x in pose_grasp[:50]]

                        offset_pose = np.array(rotZ(np.pi / 2))  # rotate about z axis
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

                (
                    target_obj.reach_grasps,
                    target_obj.grasps,
                    _,
                    _,
                ) = self.solve_goal_set_ik(
                    target_obj,
                    env,
                    pose_grasp,
                    z_upsample=z_upsample,
                    y_upsample=self.cfg.y_upsample,
                )
                target_obj.grasp_potentials = []

                if (
                    self.cfg.augment_flip_grasp
                    and not target_obj.attached
                    and len(target_obj.reach_grasps) > 0
                ):
                    """add augmenting symmetry grasps in C space"""
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
                    """remove grasps in task space that have large rotation change"""
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
                            self.traj.dof,  # 9,
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
                        target_hand_pose = target_hand_pose[:, None]

                    # difference angle
                    R_diff = np.matmul(
                        target_hand_pose[..., :3, :3],
                        start_hand_pose[:3, :3].transpose(1, 0),
                    )
                    angle = np.abs(
                        np.arccos((np.trace(R_diff, axis1=2, axis2=3) - 1) / 2)
                    )
                    angle = angle * 180 / np.pi
                    rot_masks = angle > self.cfg.target_hand_filter_angle
                    z = target_hand_pose[..., :3, 0] / np.linalg.norm(
                        target_hand_pose[..., :3, 0], axis=-1, keepdims=True
                    )
                    downward_masks = z[:, :, -1] < -0.3
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

                    ik_goal_num = len(goal_set)
                    goal_set = [goal_set[idx] for idx in collision_free[0]]
                    reach_goal_set = [reach_goal_set[idx] for idx in collision_free[0]]
                    if (
                        target_obj.grasps_scores is not None
                        and target_obj.grasps_scores != []
                    ):
                        try:
                            grasp_scores = [
                                target_obj.grasps_scores[idx]
                                for idx in collision_free[0]
                            ]
                            grasp_ees = [
                                target_obj.grasp_ees[idx] for idx in collision_free[0]
                            ]
                        except Exception as e:
                            import IPython

                            IPython.embed()
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
                    if (
                        target_obj.grasps_scores is not None
                        and target_obj.grasps_scores != []
                    ):
                        target_obj.grasps_scores = [
                            grasp_scores[int(idx)] for idx in sample_goals
                        ]
                        target_obj.grasp_ees = [
                            grasp_ees[int(idx)] for idx in sample_goals
                        ]
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

    def CHOMP_update(self, traj, pose_goal, robot_model):
        q_curr = torch.tensor(
            traj.data[-1], device="cpu", dtype=torch.float32
        ).unsqueeze(0)
        q_curr.requires_grad = True

        def fn(q, pose_goal_, vis=False):
            pose_ee = robot_model.forward_kinematics(q)["panda_hand"]  # SE(3) 3x4
            if vis:  # visualize
                T_b2e_np = pose_ee.to_matrix().detach().squeeze().numpy()
                draw_pose(self.T_w2b_np @ T_b2e_np)  # ee in world frame
            if self.cfg.dist_func == "control_points":
                T_ee = pose_ee.to_matrix()
                T_goal = pose_goal_.to_matrix()
                ee_control_pts = transform_control_points(
                    T_ee, batch_size=T_ee.shape[0], mode="rt", device="cpu"
                )  # N x 6 x 4
                goal_control_pts = transform_control_points(
                    T_goal, batch_size=T_goal.shape[0], mode="rt", device="cpu"
                )  # N x 6 x 4
                loss = control_point_l1_loss(
                    ee_control_pts, goal_control_pts, mean_batch=False
                )
            return loss

        loss = fn(q_curr, pose_goal, vis=True)  # 1 x 6
        if loss.item() == 0:
            traj.goal_cost = 0
            traj.goal_grad = np.zeros((1, 7))
        else:
            loss.backward()
            traj.goal_cost = loss.item()
            traj.goal_grad = q_curr.grad.cpu().numpy()[:, :7]

    def plan(self, traj, category="All", pc_dict={}, viz_env=None):
        """
        Run chomp optimizer to do trajectory optmization
        """
        self.traj = traj
        self.history_trajectories = [np.copy(traj.data)]
        self.info = []
        self.selected_goals = []
        start_time_ = time.time()
        alg_switch = self.cfg.ol_alg != "Baseline" and "NGDF" not in self.cfg.method

        best_traj = None  # Save lowest cost trajectory
        best_cost = 1000
        if (
            (not self.cfg.goal_set_proj)
            or len(self.traj.goal_set) > 0
            or ("CG" in self.cfg.method)
            or "NGDF" in self.cfg.method
        ):
            self.T_w2b_np = get_world2bot_transform()

            urdf_path = DifferentiableFrankaPanda().urdf_path.replace("_no_gripper", "")
            robot_model = th.eb.UrdfRobotModel(urdf_path, device="cuda")

            if "NGDF" in self.cfg.method and pc_dict is not {}:
                # Get shape code for point cloud
                shape_code, mean_pc = self.grasp_predictor.get_shape_code(
                    pc_dict["points_world"], category=category
                )
                T_w2pc = np.eye(4)
                T_w2pc[:3, 3] = mean_pc[:3]
                draw_pose(T_w2pc)
                T_b2pc_np = np.linalg.inv(self.T_w2b_np) @ T_w2pc
                T_pc2b_np = np.linalg.inv(T_w2pc) @ self.T_w2b_np

                dtype = torch.float32 if not self.use_double else torch.float64
                device = self.grasp_predictor.device()
                T_pc2b = torch.tensor(T_pc2b_np, dtype=dtype, device=device)
                pose_pc2b = th.SE3(tensor=T_pc2b[:3].unsqueeze(0))

                # Get transform from mesh obj frame (not centroid!) to bot frame
                # T_o2b = torch.tensor(T_o2b_np, dtype=dtype, device=device)
                # pose_o2b = th.SE3(tensor=T_o2b[:3].unsqueeze(0))

                # correct wrist rotation - TODO make sure transform is correct for what we need, acronym is rotgrasp convention
                T_rotgrasp2grasp = pt.transform_from(
                    pr.matrix_from_axis_angle([0, 0, 1, -np.pi / 2]), [0, 0, 0]
                )
                pose_h2t = th.SE3(
                    tensor=torch.tensor(T_rotgrasp2grasp)[:3].unsqueeze(0)
                )
                # T_h2t = wrist_to_tip(dtype=dtype, device=device) # TODO transform if use_tip

                if self.cfg.initial_ik:
                    # set init end pose to current orientation but at offset to point cloud centroid frame
                    pose_b2start = robot_model.forward_kinematics(
                        torch.tensor(self.traj.start)
                    )["panda_hand"]
                    T_b2start = pose_b2start.to_matrix().squeeze().cpu().numpy()
                    T_b2pc_np[:3, :3] = T_b2start[:3, :3]

                    # offset
                    T_offset = np.eye(4)
                    T_offset[2, 3] = -0.28
                    T_b2init = T_b2pc_np @ T_offset

                    pq_b2pc = pt.pq_from_transform(T_b2init)  # wxyz
                    seed = self.traj.start[:7]
                    goal_ik = config.cfg.ROBOT.inverse_kinematics(
                        pq_b2pc[:3], ros_quat(pq_b2pc[3:]), seed=seed
                    )
                    if goal_ik is None:
                        print(f"Warning: initial IK failed")
                    else:
                        traj.set_goal_and_interp(goal_ik)

                if False:  # visualize pc in world frame
                    self.dbg_ids = []
                    while len(self.dbg_ids) > 0:
                        dbg_id = self.dbg_ids.pop(0)
                        p.removeUserDebugItem(dbg_id)

                    pc_idx_sample = np.random.choice(
                        range(len(pc_dict["points_world"])), 100
                    )
                    for pc_idx in pc_idx_sample:
                        w2pc = pc_dict["points_world"][pc_idx]
                        dbg_id = p.addUserDebugLine(
                            w2pc[:3],
                            w2pc[:3] + np.array([0.001, 0.001, 0.001]),
                            lineWidth=5.0,
                            lineColorRGB=(255.0, 0, 0),
                        )
                        self.dbg_ids.append(dbg_id)
            elif "CG" in self.cfg.method and pc_dict is not {}:
                # get grasp set using contact graspnet
                start_time_ = time.time()
                pred_grasps_cam, scores, _ = self.grasp_predictor.inference(
                    pc_dict["points_cam2"]
                )
                # pred_grasps_cam, scores, _ = self.grasp_predictor.inference(pc_dict['points_cam2'], pc_dict['points_segments'])
                T_world_cam2 = pc_dict["T_world_cam2"]

                grasp_set = self.load_grasp_set_cg(
                    self.env, pred_grasps_cam, scores, T_world_cam2
                )
                if len(grasp_set) == 0:
                    self.info.append(self.optim.optimize(self.traj, info_only=True))
                    self.info[-1]["time"] = np.isnan
                    print("planning not run, no grasp set")
                    return self.info

                self.grasp_init(self.env)
                self.learner = Learner(self.env, self.traj, self.cost)
                self.setup_time = time.time() - start_time_

                # Visualize predicted grasps
                if True:
                    if self.dbg_ids is None:
                        self.dbg_ids = []
                    while len(self.dbg_ids) > 0:
                        dbg_id = self.dbg_ids.pop(0)
                        p.removeUserDebugItem(dbg_id)

                    T_w2b = get_world2bot_transform()
                    pred_grasp = unpack_pose(grasp_set[self.traj.goal_idx])
                    dbg_ids = draw_pose(T_w2b @ pred_grasp)
                    # dbg_ids = [draw_pose(T_w2b @ unpack_pose(g)) for g in grasp_set[:50]]
                    self.dbg_ids += dbg_ids

            self.optim.init(self.traj)

            for t in range(self.cfg.optim_steps + self.cfg.extra_smooth_steps):
                sys.stdout.write(f"plan step {t} ")
                start_time = time.time()

                if alg_switch and t < self.cfg.optim_steps:
                    self.learner.update_goal()
                    self.selected_goals.append(self.traj.goal_idx)

                if "NGDF" in self.cfg.method:
                    q = torch.tensor(
                        self.traj.data[-1], dtype=dtype, device=device
                    ).unsqueeze(0)
                    q.requires_grad = True

                    def fn(q):
                        pose_b2h = robot_model.forward_kinematics(q)[
                            "panda_hand"
                        ]  # SE(3) 3x4
                        if self.use_double:
                            pose_b2h = th.SE3(tensor=pose_b2h.tensor.double())
                        # input end effector needs to be in point cloud centroid frame
                        pose_pc2t = pose_pc2b.compose(pose_b2h)
                        x_dict = {
                            "pq": pose_to_pq(pose_pc2t).squeeze(),
                            "shape_code": shape_code,
                            "category": category,
                        }
                        dist = self.grasp_predictor.forward(x_dict)
                        loss = torch.abs(dist.mean(dim=1, keepdim=True))

                        if (
                            False
                        ):  # visualize pc centroid and final gripper pose in world frame
                            if self.dbg_ids is None:
                                self.dbg_ids = []
                            while len(self.dbg_ids) > 0:
                                dbg_id = self.dbg_ids.pop(0)
                                p.removeUserDebugItem(dbg_id)

                            dbg_ids = draw_pose(self.T_w2b_np @ T_b2pc_np)
                            self.dbg_ids += dbg_ids
                            dbg_ids = draw_pose(
                                self.T_w2b_np
                                @ T_b2pc_np
                                @ pose_pc2t.to_matrix().detach().cpu().squeeze().numpy()
                            )
                            self.dbg_ids += dbg_ids

                        return loss

                    loss = fn(q)
                    loss.backward()

                    traj.goal_cost = loss.item()
                    traj.goal_grad = q.grad.float().squeeze()[:7].cpu().numpy()

                info_t = self.optim.optimize(self.traj, force_update=True, tstep=t + 1)

                if self.cfg.report_time:
                    print("plan optimize:", time.time() - start_time)

                # Visualize points in collision with robot
                if viz_env and info_t["collide"] > 0 and False:
                    while len(self.dbg_ids) > 0:
                        dbg_id = self.dbg_ids.pop(0)
                        p.removeUserDebugItem(dbg_id)
                    # p.removeAllUserDebugItems()

                    fngr_col_idxs = np.where(info_t["collision_pts"][:, :, :, 3] == 255)
                    col_pts = info_t["collision_pts"][fngr_col_idxs][:, :6]
                    for col_pt in col_pts:
                        col_pt_h = np.concatenate((col_pt[:3], [1]))
                        w2col_pt = self.T_w2b_np @ col_pt_h
                        col_pt[3:6] /= 255.0
                        dbg_id = p.addUserDebugLine(
                            w2col_pt[:3],
                            w2col_pt[:3] + np.array([0.001, 0.001, 0.001]),
                            lineWidth=5.0,
                            lineColorRGB=col_pt[3:6],
                        )
                        self.dbg_ids.append(dbg_id)

                    # # draw sdf points
                    # coords = self.env.objects[self.env.target_idx].sdf.visualize()
                    # T_b2o = unpack_pose(self.env.objects[self.env.target_idx].pose)
                    # draw_pose(self.T_w2b_np @ T_b2o)

                    # for i in range(coords.shape[0]):
                    #     coord = coords[i]
                    #     coord = np.concatenate([coord, [1]])
                    #     T_c = np.eye(4)
                    #     T_c[:, 3] = coord
                    #     draw_pose(self.T_w2b_np @ T_b2o @ T_c)

                # visualize robot's collision points
                # if viz_env and \
                #     (self.info[-1]["terminate"] or t == self.cfg.optim_steps + self.cfg.extra_smooth_steps - 1):
                if False:
                    viz_env.update_panda_viz(self.traj)
                    robot = self.cost.env.robot
                    robot_pts = robot.collision_points.transpose([0, 2, 1])
                    (
                        robot_poses,
                        joint_origins,
                        joint_axis,
                    ) = self.cost.env.robot.robot_kinematics.forward_kinematics_parallel(
                        wrap_values(self.traj.data), return_joint_info=True
                    )
                    ws_positions = self.cost.forward_points(
                        robot_poses, robot_pts
                    )  # p x (m + 1) x n x 3
                    ws_pos = ws_positions.copy()
                    ws_pos = ws_pos.transpose([2, 1, 0, 3])
                    last_col_pts = ws_pos[-1]
                    fngr_col_pts = last_col_pts[-2:]

                    p.removeAllUserDebugItems()

                    for fngr_idx in range(fngr_col_pts.shape[0]):
                        for col_pt in fngr_col_pts[fngr_idx]:
                            col_pt = np.concatenate([col_pt, [1]])
                            # T_col_pt = np.eye(4)
                            # T_col_pt[:, 3] = col_pt
                            # draw_pose(self.T_w2b_np @ col_pt)
                            w2col_pt = self.T_w2b_np @ col_pt
                            p.addUserDebugLine(
                                w2col_pt[:3],
                                w2col_pt[:3] + np.array([0.001, 0.001, 0.001]),
                                lineWidth=2.0,
                                lineColorRGB=(1.0, 0, 0),
                            )

                if self.info[-1]["terminate"] and t > 0:
                    break

            if viz_env:
                viz_env.remove_panda_viz()

            # compute information for the final
            if not self.info[-1]["terminate"]:
                if self.cfg.use_min_cost_traj:
                    print("Replacing final traj with lowest cost traj")
                    traj.data = best_traj
                    self.info.append(self.optim.optimize(traj, info_only=True))
                    self.history_trajectories.append(best_traj)
                else:
                    self.info.append(self.optim.optimize(self.traj, info_only=True))

            plan_time = time.time() - start_time_
            if not self.cfg.silent:
                print(
                    "planning time: {:.3f} PLAN Length: {}".format(
                        plan_time, len(self.history_trajectories[-1])
                    )
                )
            self.info[-1]["time"] = plan_time + self.setup_time

        else:
            if not self.cfg.silent:
                print("planning not run...")

        return self.info
