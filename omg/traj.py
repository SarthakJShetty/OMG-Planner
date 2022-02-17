import numpy as np
from . import config
from .util import *

class Trajectory(object):
    """
    Trajectory class that wraps an object or an obstacle
    """

    def __init__(self, timesteps=100, dof=9, start_end_equal=False):
        """
        Initialize fixed endpoint trajectory.
        """
        self.timesteps = config.cfg.timesteps
        self.dof = dof
        self.data = np.zeros([self.timesteps, dof])  # fixed start and end
        self.goal_set = []
        self.goal_quality = []

        self.start = np.array([0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
        if start_end_equal:
            self.end = self.start.copy()
            # When trajectory end does not match goal
            # TODO check that these are necessary
            self.goal_pose = None
            self.goal_joints = None
            self.goal_cost = None
            self.goal_grad = None
        else:
            self.end = np.array([-0.99, -1.74, -0.61, -3.04, 0.88, 1.21, -1.12, 0.04, 0.04])
        self.interpolate_waypoints(mode=config.cfg.traj_interpolate)

    def update(self, grad):
        """
        Update trajectory based on functional gradient.
        """
        if config.cfg.consider_finger:
            self.data += grad
        else:
            self.data[:, :-2] += grad[:, :-2]
        self.data[:, -2:] = np.minimum(np.maximum(self.data[:, -2:], 0), 0.04)

    def set(self, new_traj):
        """
        Set trajectory by given data.
        """
        self.data = new_traj

    def interpolate_waypoints(self, waypoints=None, mode="cubic"):  # linear
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
