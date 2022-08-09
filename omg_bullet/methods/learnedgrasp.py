import torch
from manifold_grasping.networks import Decoder

from pathlib import Path
from omegaconf import OmegaConf
import h5py

import pytorch3d.ops


class LearnedGrasp:
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        cfg_path = Path(self.ckpt_path).parents[3] / ".hydra" / "config.yaml"
        self.cfg = OmegaConf.load(cfg_path)

        # update project_root
        hydra_cfg = OmegaConf.load(Path(__file__).parents[4] / "config" / "panda_scene.yaml")
        self.cfg.project_root = hydra_cfg.project_root
        self.cfg.data_root = hydra_cfg.data_root

        self.model = Decoder(**self.cfg.net).cuda()
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

        # # Load shape dataset
        # if self.single_shape_code:
        #     self.shape_codes = {}
        #     self.dset_shape = h5py.File(f'{self.cfg.shape_data_path}/dataset.hdf5', 'r')
        #     for key in list(self.dset_shape.keys()):
        #         code = self.dset_shape[key]['0']['shape_code']  # 1 x 256 x 3
        #         self.shape_codes[key] = code
        # else:  # Init onehot index
        #     self.onehot = [
        #         'Book_b1611143b4da5c783143cfc9128d0843_0.023835858278933857',
        #         'Bottle_244894af3ba967ccd957eaf7f4edb205_0.012953570261294404',
        #         'Bowl_9a52843cc89cd208362be90aaa182ec6_0.0008104428339208306',
        #         'Mug_40f9a6cc6b2c3b3a78060a3a3a55e18f_0.0006670441940038386']


    def forward(self, pq, shape_info):
        """
        pq (torch.Tensor): xyz, wxyz
        objname for known object model
        """
        # if '1hot' in shape_info:
        #     objname = shape_info['1hot']
        #     latent = torch.tensor([self.onehot.index(objname)], device=pq.device).unsqueeze(0)
        # elif 'shape_key' in shape_info:
        #     objname = shape_info['shape_key']
        #     latent = torch.tensor(self.shape_codes[objname], device=pq.device)
        #     latent = torch.reshape(latent, (1, -1))
        # elif 'shape_code' in shape_info:
        #     latent = shape_info['shape_code']
        latent = shape_info['shape_code']
        # pq.requires_grad = True
        input_x = torch.cat([pq.unsqueeze(0), latent], axis=1)

        dist = self.model(input_x)

        # if self.use_shape_code:
        #     shape_code = torch.tensor(self.shape_codes[objname][0], device=pq.device).flatten()
        #     input_x = torch.cat([pq, shape_code])
        # else:
        #     input_x = torch.cat([pq, torch.tensor([self.onehot.index(objname)], device=pq.device)])

        # with torch.no_grad():
        #     outpose = self.model(input_x.unsqueeze(0))
        dist = dist.squeeze(0)
        return dist

    def get_shape_code(self, pc):
        """input is not mean_centered"""
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
        shape_mi = {'point_cloud': pc_fps.cuda()}
        with torch.no_grad():
            latent = self.model.shape_model.extract_latent(shape_mi)
            latent = torch.reshape(latent, (latent.shape[0], -1))
        return latent, mean_pc.squeeze().cpu().numpy()


# class ImplicitGrasp_NoVision:
#     def __init__(self):
#         # TODO config output pose
#         ckpt = torch.load('/checkpoint/thomasweng/grasp_manifolds/runs/outputs/nopc_book_logmap_10kfree/2021-11-23_232102_dsetacronym_Book_train0.9_val0.1_free10.0k_sym_distlogmap/default_default/0_0/checkpoints/last.ckpt')
#         self.model = Decoder(**ckpt['hyper_parameters'])
#         self.model.cuda()
#         self.model.load_state_dict(ckpt['state_dict'])
#         self.model.eval()

#     def forward(self, x):
#         """
#         x: xyz, xyzw
#         """
#         with torch.no_grad():
#             outpose = self.model(x)
#         # outpose = outpose.squeeze(0).cpu().numpy()
#         outpose = outpose.squeeze(0)
#         return outpose

# class ImplicitGrasp_OutputPose: # Also valid for OutputDist
#     def __init__(self, ckpt_path):
#         # ckpt = torch.load('/checkpoint/thomasweng/grasp_manifolds/runs/outputs/nopc_book_logmap_10kfree/2021-11-23_232102_dsetacronym_Book_train0.9_val0.1_free10.0k_sym_distlogmap/default_default/0_0/checkpoints/last.ckpt')
#         ckpt = torch.load(ckpt_path)
#         self.model = Decoder(**ckpt['hyper_parameters'], intype='pq').double() # TODO remove intype for non pose
#         self.model.cuda()
#         self.model.load_state_dict(ckpt['state_dict'])
#         self.model.eval()

#     def forward(self, x):
#         """
#         x: xyz, xyzw
#         """
#         # with torch.no_grad():
#         outpose = self.model(x)
#         # outpose = outpose.squeeze(0).cpu().numpy()
#         outpose = outpose.squeeze(0)
#         return outpose

# class ImplicitGrasp_OutputDist:
#     def __init__(self, ckpt_path):
#         # ckpt = torch.load('/checkpoint/thomasweng/grasp_manifolds/runs/outputs/nopc_book_logmap_10kfree/2021-11-23_232102_dsetacronym_Book_train0.9_val0.1_free10.0k_sym_distlogmap/default_default/0_0/checkpoints/last.ckpt')
#         ckpt = torch.load(ckpt_path)
#         self.model = Decoder(**ckpt['hyper_parameters'])
#         self.model.cuda()
#         self.model.load_state_dict(ckpt['state_dict'])
#         self.model.eval()

#     def forward(self, x):
#         """
#         x: xyz, xyzw
#         """
#         with torch.no_grad():
#             outpose = self.model(x)
#         # outpose = outpose.squeeze(0).cpu().numpy()
#         outpose = outpose.squeeze(0)
#         return outpose