# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
from .optimizer import Optimizer
from .cost import Cost
import math
from .util import *
import torch
import time
import IPython
import theseus as th
from differentiable_robot_model.robot_model import (
    DifferentiableFrankaPanda,
)
from ngdf.control_pts import *

np.set_printoptions(4)


def find_zero(f, x0, x1, eps=1e-6, max_iter=100):
    """
    bisect search
    """
    x = (x0 + x1) / 2
    s = (x1 - x0) / 4
    for i in range(max_iter):
        y = f(x)
        if abs(y) < eps:
            return x
        x -= s * np.sign(y)
        s /= 2
    return x


def bp(x, v, delta, w, max_iter=100, err=1e-6):
    """
    Bregman projection onto simplex using the weighted AND shifted entropy
    delta = shift vector
    """

    n = len(x)
    alpha = np.zeros(n)

    for i in range(max_iter):
        z = (alpha - v) / w
        target = 1 + np.sum(delta)
        shiftx = x + delta

        f = lambda L: np.sum(shiftx * np.exp(L / w + z)) - target
        L = find_zero(f, 0, np.max(w + v), err, max_iter)
        y = shiftx * np.exp((L + alpha - v) / w) - delta

        # take a gradient descent step toward the correct alphas
        alpha_prime = np.maximum(0, v - L + w * np.log(delta / shiftx))
        if np.linalg.norm(alpha - alpha_prime, ord=2) < err:
            break
        alpha = alpha_prime

    y = np.maximum(y, 0)
    y = y / np.sum(y)
    return y


class Learner(object):
    """
    An online learner that updates the goal distribution for current trajectory.
    """

    def __init__(self, env, traj, cost):
        self.cfg = env.cfg
        self.env = env
        self.traj = traj
        self.cost = cost
        self.alg_name = self.cfg.ol_alg
        self.N = len(traj.goal_set)
        self.T = self.cfg.optim_steps
        self.Ti = np.zeros(self.N)
        self.Tis = []
        self.weights = np.ones(self.N)
        self.t = 0.0

        self.p = np.ones(self.N) / self.N
        self.sum_costs = np.zeros(self.N)
        self.last_leader = 0
        self.eta = np.sqrt(np.log(self.N + 1) / (self.T))

        self.etas = [self.eta * (2 ** x) for x in [-2, -1, 0, 2, 4]]
        self.delta = np.ones(self.N) / (4 * self.N + 1)
        self.num_experts = len(self.etas)

        self.experts_p = [np.ones(self.N) / self.N for _ in range(len(self.etas))]
        self.experts_costs = np.zeros(self.num_experts)

        # mixture of experts
        self.q = np.ones(self.num_experts) / self.num_experts

        if (
            ((self.cfg.goal_idx == -2 and (self.alg_name == "Baseline" or self.alg_name == "MD" or self.alg_name == 'FTC' or self.alg_name == 'Exp' or self.alg_name == 'FTL')))  # online learning algs
            and (len(self.env.objects) > 0 and len(self.env.objects[self.env.target_idx].reach_grasps) > 0)
        ):
            costs = self.cost_vector()
            sorted_idx = np.argsort(costs)
            self.traj.goal_idx = np.argmin(costs)  
    
        if self.alg_name is not None:
            self.traj.end = self.traj.goal_set[self.traj.goal_idx]  #
            self.traj.interpolate_waypoints()

        if 'GF_known' in self.env.cfg.method:
            urdf_path = DifferentiableFrankaPanda().urdf_path.replace('_no_gripper', '')
            self.robot_model = th.eb.UrdfRobotModel(urdf_path)

    def cost_vector(self):
        """
        objective cost estimate
        """
        clamp = 1  # 1
        start = clamp + int((self.t / self.cfg.optim_steps) * (self.cfg.timesteps)) - 1
        start = min(start, self.cfg.timesteps - clamp)
        if self.cfg.report_time:
            print("online learning, start time step", start)
        traj_start = self.traj.data[start]
        n = self.cfg.timesteps - start
        m = traj_start.shape[0]
        if (
            len(self.env.objects[self.env.target_idx].reach_grasps) == 0
            and self.cfg.traj_init == "grasp"
        ):
            return np.zeros(1)

        goal_set = (
            self.env.objects[self.env.target_idx].reach_grasps[:, -1, :]
            if self.cfg.use_standoff
            else self.traj.goal_set
        )
        s = time.time()

        interpolated_traj = multi_interpolate_waypoints(
            traj_start, goal_set, n, m, "linear"
        )
        if self.cfg.report_time:
            print("goal selection interpolate time:", time.time() - s, n)
        s = time.time()
        collision_potentials = self.cost.batch_obstacle_cost(
            interpolated_traj,
            arc_length=n,
            special_check_id=self.env.target_idx,
            uncheck_finger_collision=0,
            start=traj_start,
            end=goal_set,
        )[
            0
        ]  # n x (m + 1) x p
        if self.cfg.report_time:
            print("goal selection potentials time:", time.time() - s, n)

        collision_potentials = (
            torch.sum(collision_potentials, (-2, -1))
            .reshape([-1, n])
            .sum(-1)
            .detach()
            .cpu()
            .numpy()
        )
        smooth = (
            np.linalg.norm(
                np.diff(traj_start - np.array(self.traj.goal_set), axis=-1), axis=-1
            )
            ** 2
            * 0.3
        )
        potentials = (
            self.cfg.base_obstacle_weight * collision_potentials
            + self.cfg.smoothness_base_weight * self.cfg.dist_eps * smooth
        )
        if self.cfg.normalize_cost:
            potentials = potentials / np.linalg.norm(
                potentials
            )  # normalize cost vector

        return potentials

    def update_goal_dist(self):
        """
        Run online learning algorithm to update goal distribution.
        """
        if self.alg_name == "Proj":
            self.Proj()
        elif self.alg_name is None: # for implicit grasps
            self.Proj()
        else:
            cv = self.cost_vector()
            if self.alg_name == "FTL":
                self.FTL(cv)
            elif self.alg_name == "FTC":
                self.FTC(cv)
            elif self.alg_name == "Exp":
                self.Exp(cv)
            elif self.alg_name == "MD":
                self.MD(cv)

    def FTL(self, cv):
        """
        Follow the leader.
        """
        self.sum_costs = self.sum_costs + cv
        self.last_leader = np.argmin(self.sum_costs)
        self.p = np.zeros(self.N)
        self.p[self.last_leader] = 1

    def FTC(self, cv):
        """
        Follow the cheapest
        """
        self.last_leader = np.argmin(cv)
        self.p = np.zeros(self.N)
        self.p[self.last_leader] = 1

    def Proj(self):
        """
        Goal set projection 
        """
        if 'GF_known' in self.env.cfg.method:
            q_curr = torch.tensor(self.traj.data[-1], device='cpu', dtype=torch.float32).unsqueeze(0)
            pose_ee = self.robot_model.forward_kinematics(q_curr)['panda_hand']

            q_goals = torch.tensor(self.traj.goal_set, device='cpu', dtype=torch.float32)
            pose_goals = self.robot_model.forward_kinematics(q_goals)['panda_hand']
            if self.env.cfg.dist_func == 'control_points':
                T_ee = pose_ee.to_matrix()
                T_goals = pose_goals.to_matrix()
                ee_control_pts = transform_control_points(T_ee, batch_size=T_ee.shape[0], mode='rt', device='cpu') # N x 6 x 4
                goal_control_pts = transform_control_points(T_goals, batch_size=T_goals.shape[0], mode='rt', device='cpu') # N x 6 x 4
                norms = control_point_l1_loss(ee_control_pts, goal_control_pts, mean_batch=False)
            target_idx = torch.argmin(norms)
        else:
            cur_end_point = self.traj.data[-1]
            try:
                diff = cur_end_point - np.array(self.traj.goal_set)
            except Exception as e: # no grasps in the goal set
                print(e)
                import IPython; IPython.embed()
            dists = np.linalg.norm(diff, axis=-1)
            target_idx = np.argmin(dists)

        self.p = np.zeros(self.N)
        self.p[target_idx] = 1

    def Exp(self, cv):
        """
        Exponential Weight
        """
        self.sum_costs = self.sum_costs + cv
        norm_sum_cost = safe_div(self.sum_costs, np.sum(self.sum_costs))
        p_old = self.p
        p_new = np.exp(-self.eta * cv) * p_old
        self.p = p_new * 0.999 + norm_sum_cost * 0.001  # soft reset
        self.p = safe_div(self.p, np.sum(self.p))  # normalize

    def MD(self, cv):
        """
        Mirror Descent over experts
        """
        for i in range(self.num_experts):
            p = bp(self.experts_p[i], self.etas[i] * cv, self.delta, self.weights)
            self.experts_costs[i] = np.dot(cv, p) + np.dot(
                self.weights, np.abs(p - self.experts_p[i])
            )
            self.experts_p[i] = p

            self.q = self.q * np.exp(-1 * self.experts_costs)
            self.q = self.q / np.sum(self.q)

            self.p = sum(self.experts_p[i] * self.q[i] for i in range(self.num_experts))
            self.p = self.p / np.sum(self.p)

    def update_goal(self):
        """
        Take the argmax of the goal distribution
        """
        self.t += 1
        self.update_goal_dist()

        goal_idx_old = self.traj.goal_idx
        self.traj.goal_idx = np.argmax(self.p)
        if 'GF_known' not in self.cfg.method: # Don't update endpoint of trajectory, let gradient do it
            self.traj.end = self.traj.goal_set[self.traj.goal_idx]
        self.Ti[self.traj.goal_idx] += 1
        self.Tis.append(self.Ti)
        return self.traj.goal_idx != goal_idx_old

    def reset(self, traj):
        """
        Reset the online learner
        """
        self.alg_name = self.cfg.ol_alg
        self.N = len(traj.goal_set)
        self.T = self.cfg.optim_steps
        self.weights = np.ones(self.N)
        self.t = 0.0
        self.traj = traj
        self.p = np.ones(self.N) / self.N
        self.sum_costs = np.zeros(self.N)
        self.last_leader = 0
