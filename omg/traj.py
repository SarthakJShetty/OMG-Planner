import numpy as np

from . import config
from .util import *


class Trajectory(object):
    """
    Trajectory class that wraps an object or an obstacle
    """

    def __init__(
        self,
        timesteps=100,
        dof=9,
        start_end_equal=False,
        start=[0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
    ):
        """
        Initialize fixed endpoint trajectory.
        """
        self.timesteps = config.cfg.timesteps
        self.dof = dof
        self.data = np.zeros([self.timesteps, dof])
        self.goal_set = []
        self.goal_quality = []

        self.start = np.array(start)
        if start_end_equal:
            self.end = self.start.copy()
        else:
            self.end = np.array(
                [-0.99, -1.74, -0.61, -3.04, 0.88, 1.21, -1.12, 0.04, 0.04]
            )
        self.interpolate_waypoints(mode=config.cfg.traj_interpolate)

    def update(self, grad):
        """
        Update trajectory based on functional gradient.
        """
        self.data[:, :-2] += grad[:, :-2]
        self.data[:, -2:] = np.minimum(np.maximum(self.data[:, -2:], 0), 0.04)

    def set(self, new_traj):
        """
        Set trajectory by given data.
        """
        self.data = new_traj

    def set_goal_and_interp(self, goal):
        """Set goal of trajectory and interpolate trajectory from start to goal

        Args:
            goal (_type_): goal in joint angles
        """
        if goal.shape[0] == 7:
            goal = np.concatenate([goal, [0.04, 0.04]], axis=0)

        self.end = goal
        self.interpolate_waypoints(mode=config.cfg.traj_interpolate)

    def interpolate_waypoints(self, mode="cubic"):
        """
        Interpolate the waypoints using interpolation.
        """
        timesteps = config.cfg.timesteps
        if config.cfg.dynamic_timestep:
            timesteps = min(
                max(
                    int(np.linalg.norm(self.start - self.end) / config.cfg.traj_delta),
                    config.cfg.traj_min_step,
                ),
                config.cfg.traj_max_step,
            )
            config.cfg.timesteps = timesteps
            self.data = np.zeros([timesteps, self.dof])  # fixed start and end
            config.cfg.get_global_param(timesteps)
            self.timesteps = timesteps
        self.data = interpolate_waypoints(
            np.stack([self.start, self.end]), timesteps, self.start.shape[0], mode=mode
        )
