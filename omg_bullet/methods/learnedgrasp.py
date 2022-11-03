import torch
from ngdf.networks import Decoder
# from manifold_grasping.pointset.pointnet2_segmentation import PointNet2_Seg

from pathlib import Path
from omegaconf import OmegaConf
# import h5py

import pytorch3d.ops


class LearnedGrasp:
    def __init__(self, ckpt_paths=[], use_double=False, hydra_cfg=None):
        self.use_double = use_double
        self.hydra_cfg = hydra_cfg if hydra_cfg is not None else OmegaConf.load(Path(__file__).parents[4] / "config" / "panda_scene.yaml")
        # data structure of ckpt_paths: list of "category:path" strings
        self.models = {}
        for item in ckpt_paths:
            category, ckpt_path = item.split(':')
            self.models[category] = self.load_model(ckpt_path)

    def init_model(self):
        return Decoder(**self.cfg.net).cuda()

    def load_model(self, ckpt_path):
        # load separate models for each object
        self.ckpt_path = ckpt_path
        if 'hpc' in self.ckpt_path:
            cfg_path = Path(self.ckpt_path).parent / ".hydra" / "config.yaml"
        else:
            cfg_path = Path(self.ckpt_path).parents[3] / ".hydra" / "config.yaml"
        self.cfg = OmegaConf.load(cfg_path)

        # update project_root
        self.cfg.project_root = self.hydra_cfg.project_root
        self.cfg.data_root = self.hydra_cfg.data_root
        self.arch = self.cfg.net.arch

        model = self.init_model()
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
        if self.use_double: 
            model.double()
        model.eval()
        return model
    
    def get_input(self, x_dict={}):
        pq = x_dict['pq']
        latent = x_dict['shape_code']
        input_x = torch.cat([pq.unsqueeze(0), latent], axis=1)
        return input_x

    def forward(self, x_dict={}):
        input_x = self.get_input(x_dict)
        category = x_dict['category']
        output = self.models[category](input_x)
        # TODO get dist from output
        return output

    def get_shape_code(self, pc, category='All'):
        if self.arch == 'deepsdf':
            """input is not mean_centered, point cloud is in world frame"""
            # Farthest point sampling
            pc_t = torch.tensor(pc[:, :3]).unsqueeze(0)  # 1 x N x 3
            try:
                pc_fps, pc_fps_idxs = pytorch3d.ops.sample_farthest_points(pc_t, K=1500, random_start_point=True)
            except Exception as e:
                print(e)
                import IPython; IPython.embed()

            # mean center the point cloud
            mean_pc = pc_fps.mean(axis=1)
            pc_fps -= mean_pc

            # Run VN-OccNets
            if self.use_double:
                pc_fps = pc_fps.double()
            shape_mi = {'point_cloud': pc_fps.cuda()}
            with torch.no_grad():
                latent = self.models[category].shape_model.extract_latent(shape_mi)
                latent = torch.reshape(latent, (latent.shape[0], -1))
            return latent, mean_pc.squeeze().cpu().numpy()
        elif self.arch == 'pointnet2_seg':
            raise NotImplementedError

    def device(self):
        return self.models[list(self.models.keys())[0]].device # all models should be on the same device



# class LearnedGrasp_PointNet2Seg(LearnedGraspBase):
#     def __init__(self): 
#         super().__init__()

#     def init_model(self):
#         return PointNet2_Seg(**self.cfg.net).cuda()

#     def forward(self, x_dict={}):
#         """
#         pq (torch.Tensor): xyz, wxyz
#         objname for known object model
#         """
#         # TODO
#         raise NotImplementedError
#         # query_pose = x_dict['pq']
#         # pc = 
#         # category = x_dict['category']
#         # input_x = torch.cat([pq.unsqueeze(0), latent], axis=1)
#         # dist = self.models[category](input_x)
#         # return dist


# class LearnedGrasp_DeepSDF(LearnedGraspBase):
#     def __init__(self): 
#         super().__init__()

#     def init_model(self):
#         return Decoder(**self.cfg.net).cuda()

#     def forward(self, x_dict={}):
#         """
#         pq (torch.Tensor): xyz, wxyz
#         objname for known object model
#         """
#         pq = x_dict['pq']
#         latent = x_dict['shape_code']
#         category = x_dict['category']
#         input_x = torch.cat([pq.unsqueeze(0), latent], axis=1)
#         dist = self.models[category](input_x)
#         return dist

#     def get_shape_code(self, pc, category='All'):
#         """input is not mean_centered, point cloud is in world frame"""
#         # Farthest point sampling
#         pc_t = torch.tensor(pc[:, :3]).unsqueeze(0)  # 1 x N x 3
#         try:
#             pc_fps, pc_fps_idxs = pytorch3d.ops.sample_farthest_points(pc_t, K=1500, random_start_point=True)
#         except Exception as e:
#             print(e)
#             import IPython; IPython.embed()

#         # mean center the point cloud
#         mean_pc = pc_fps.mean(axis=1)
#         pc_fps -= mean_pc

#         # Run VN-OccNets
#         if self.use_double:
#             pc_fps = pc_fps.double()
#         shape_mi = {'point_cloud': pc_fps.cuda()}
#         with torch.no_grad():
#             latent = self.models[category].shape_model.extract_latent(shape_mi)
#             latent = torch.reshape(latent, (latent.shape[0], -1))
#         return latent, mean_pc.squeeze().cpu().numpy()




# class LearnedGrasp_Old:
#     def __init__(self, ckpt_path, use_double=False):
#         self.ckpt_path = ckpt_path
#         self.use_double = use_double
#         cfg_path = Path(self.ckpt_path).parents[3] / ".hydra" / "config.yaml"
#         self.cfg = OmegaConf.load(cfg_path)

#         # update project_root
#         hydra_cfg = OmegaConf.load(Path(__file__).parents[4] / "config" / "panda_scene.yaml")
#         self.cfg.project_root = hydra_cfg.project_root
#         self.cfg.data_root = hydra_cfg.data_root

#         self.model = Decoder(**self.cfg.net).cuda()
#         ckpt = torch.load(ckpt_path)
#         self.model.load_state_dict(ckpt['state_dict'])
#         if self.use_double: 
#             self.model.double()
#         self.model.eval()

#     def forward(self, pq, shape_info):
#         """
#         pq (torch.Tensor): xyz, wxyz
#         objname for known object model
#         """
#         latent = shape_info['shape_code']
#         input_x = torch.cat([pq.unsqueeze(0), latent], axis=1)
#         dist = self.model(input_x)
#         return dist

#     def get_shape_code(self, pc):
#         """input is not mean_centered, point cloud is in world frame"""
#         # Farthest point sampling
#         pc_t = torch.tensor(pc[:, :3]).unsqueeze(0)  # 1 x N x 3
#         try:
#             pc_fps, pc_fps_idxs = pytorch3d.ops.sample_farthest_points(pc_t, K=1500, random_start_point=True)
#         except Exception as e:
#             print(e)
#             import IPython; IPython.embed()

#         # mean center the point cloud
#         mean_pc = pc_fps.mean(axis=1)
#         pc_fps -= mean_pc

#         # Run VN-OccNets
#         if self.use_double:
#             pc_fps = pc_fps.double()
#         shape_mi = {'point_cloud': pc_fps.cuda()}
#         with torch.no_grad():
#             latent = self.model.shape_model.extract_latent(shape_mi)
#             latent = torch.reshape(latent, (latent.shape[0], -1))
#         return latent, mean_pc.squeeze().cpu().numpy()
    
#     def device(self):
#         return self.model.device

