import numpy as np
from .sdf_tools import *
from .util import *
from . import config

class Model(object):
    """
    Model class that wraps an object or an obstacle
    """

    def __init__(
        self, path=None, id=0, target=True, pose=None, compute_grasp=True, name=None
    ):
        path = config.cfg.root_dir + path
        self.mesh_path = path + "model_normalized.obj"
        self.pose_mat = np.eye(4)
        self.pose_mat[:3, 3] = [0.1, 0.04, 0.15]  # 0.15  [0.3, 0.04, 0.55]
        self.pose_mat[:3, :3] = euler2mat(-np.pi / 2, np.pi, np.pi)
        if pose is not None:
            self.pose_mat = pose

        self.pose = pack_pose(self.pose_mat)
        self.type = target
        self.model_name = path.split("/")[-2]
        if name is None:
            self.name = self.model_name
        else:
            self.name = name
        self.id = id
        self.extents = np.loadtxt(path + "model_normalized.extent.txt").astype(
            np.float32
        )
        self.resize = (
            config.cfg.target_size if target else config.cfg.obstacle_size
        )

        self.sdf = SignedDensityField.from_pth(path + "model_normalized_chomp.pth")  #
        self.sdf.resize(config.cfg.target_size)
        self.sdf.data[self.sdf.data < 0] *= config.cfg.penalize_constant
        self.compute_grasp = compute_grasp
        self.grasps = []
        self.reach_grasps = []
        self.grasps_scores = []
        self.seeds = []
        self.grasp_potentials = []
        self.grasp_vis_points = []
        self.attached = False
        self.rel_hand_pose = None
        self.grasps_poses = []
        if self.name.startswith("0"):
            self.points = np.loadtxt(path + "model_normalized.xyz")
            self.points = self.points[random.sample(range(self.points.shape[0]), 500)]

    def world_to_obj(self, points):
        return np.swapaxes(
            (np.matmul(
                    self.pose_mat[:3, :3].T,
                    (np.swapaxes(points, -1, -2) - self.pose_mat[:3, [3]]),
                )), -1, -2, )

    def update_pose(self, pose):
        self.pose = pose
        self.pose_mat = unpack_pose(pose)
