# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytorch3d.ops
import torch
from ngdf.networks import Decoder
from omegaconf import OmegaConf


class NGDFPrediction:
    def __init__(self, ckpt_paths=[], use_double=False, hydra_cfg=None):
        self.use_double = use_double
        self.hydra_cfg = (
            hydra_cfg
            if hydra_cfg is not None
            else OmegaConf.load(
                Path(__file__).parents[4] / "config" / "panda_scene.yaml"
            )
        )
        # data structure of ckpt_paths: list of "category:path" strings
        self.models = {}
        for item in ckpt_paths:
            category, ckpt_path = item.split(":")
            self.models[category] = self.load_model(ckpt_path)

    def init_model(self):
        return Decoder(**self.cfg.net).cuda()

    def load_model(self, ckpt_path):
        # load separate models for each object
        self.ckpt_path = ckpt_path
        cfg_path = Path(self.ckpt_path).parents[3] / ".hydra" / "config.yaml"
        self.cfg = OmegaConf.load(cfg_path)

        # update project_root
        self.cfg.project_root = self.hydra_cfg.project_root
        self.cfg.data_root = self.hydra_cfg.data_root
        self.arch = self.cfg.net.arch

        model = self.init_model()
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"])
        if self.use_double:
            model.double()
        model.eval()
        return model

    def get_input(self, x_dict={}):
        pq = x_dict["pq"]
        latent = x_dict["shape_code"]
        input_x = torch.cat([pq.unsqueeze(0), latent], axis=1)
        return input_x

    def forward(self, x_dict={}):
        input_x = self.get_input(x_dict)
        category = x_dict["category"]
        output = self.models[category](input_x)
        return output

    def get_shape_code(self, pc, category="All"):
        """input is not mean_centered, point cloud is in world frame"""
        # Farthest point sampling
        pc_t = torch.tensor(pc[:, :3]).unsqueeze(0)  # 1 x N x 3
        try:
            pc_fps, pc_fps_idxs = pytorch3d.ops.sample_farthest_points(
                pc_t, K=1500, random_start_point=True
            )
        except Exception as e:
            print(e)
            import IPython

            IPython.embed()

        # mean center the point cloud
        mean_pc = pc_fps.mean(axis=1)
        pc_fps -= mean_pc

        # Run VN-OccNets
        if self.use_double:
            pc_fps = pc_fps.double()
        shape_mi = {"point_cloud": pc_fps.cuda()}
        with torch.no_grad():
            latent = self.models[category].shape_model.extract_latent(shape_mi)
            latent = torch.reshape(latent, (latent.shape[0], -1))
        return latent, mean_pc.squeeze().cpu().numpy()

    def device(self):
        return self.models[
            list(self.models.keys())[0]
        ].device  # all models should be on the same device
