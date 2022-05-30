import torch
import pytorch_lightning
from manifold_grasping.networks import Decoder

from pathlib import Path
from omegaconf import OmegaConf
import h5py

import pytorch3d.ops


class LearnedGrasp:
    def __init__(self, ckpt_path, single_shape_code=False, dset_root=''):
        self.ckpt_path = ckpt_path
        self.single_shape_code = single_shape_code

        # import IPython; IPython.embed()
        if 'hpc' in self.ckpt_path:
            cfg_path = Path(self.ckpt_path).parent / ".hydra" / "config.yaml"
        else:
            cfg_path = Path(self.ckpt_path).parents[3] / ".hydra" / "config.yaml"
        self.cfg = OmegaConf.load(cfg_path)

        # Cluster vs. local
        if 'project_root' not in self.cfg or 'private' in self.cfg.project_root:
            self.cfg.project_root = '/home/exx/projects/manifolds'
            self.cfg.data_root = '/data/manifolds'

        self.model = Decoder(**self.cfg.net).cuda()
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

        # Load shape dataset
        if self.single_shape_code:
            self.shape_codes = {}
            self.dset_shape = h5py.File(f'{self.cfg.shape_data_path}/dataset.hdf5', 'r')
            for key in list(self.dset_shape.keys()):
                code = self.dset_shape[key]['0']['shape_code']  # 1 x 256 x 3
                self.shape_codes[key] = code

            # pass  # already handled inside self.model, refactor

            # cfg_path = Path(ckpt_path).parents[3] / ".hydra" / "config.yaml"
            # self.cfg = OmegaConf.load(cfg_path)

            # self.shape_codes = {}
            # if self.cfg.net.latent.size > 1:
                # import IPython; IPython.embed()

                                

                # dset_parent = str(Path(dset_root).parent)
                # shape_data_path = self.cfg.shape_data_path.replace(self.cfg.data_root, dset_parent)

                # # For debugging
                # # shape_data_path = '/data/manifolds/acronym_mini_relabel/shape-dataset_05-18-22_00-18-55'
                # shape_data_path = '/data/manifolds/acronym_mini_relabel/shape-dataset_05-15-22_00-40-42'
                
                # self.dset_shape = h5py.File(f'{shape_data_path}/dataset.hdf5', 'r')
                # for key in list(self.dset_shape.keys()):
                #     code = self.dset_shape[key]['0']['shape_code']  # 1 x 256 x 3
                #     self.shape_codes[key] = code
        else:  # Init onehot index
            self.onehot = [
                'Book_b1611143b4da5c783143cfc9128d0843_0.023835858278933857',
                'Bottle_244894af3ba967ccd957eaf7f4edb205_0.012953570261294404',
                'Bowl_9a52843cc89cd208362be90aaa182ec6_0.0008104428339208306',
                'Mug_40f9a6cc6b2c3b3a78060a3a3a55e18f_0.0006670441940038386']


    def forward(self, pq, shape_info):
        """
        pq (torch.Tensor): xyz, wxyz
        objname for known object model
        """
        if '1hot' in shape_info:
            objname = shape_info['1hot']
            latent = torch.tensor([self.onehot.index(objname)], device=pq.device).unsqueeze(0)
        elif 'shape_key' in shape_info:
            objname = shape_info['shape_key']
            latent = torch.tensor(self.shape_codes[objname], device=pq.device)
            latent = torch.reshape(latent, (1, -1))
        elif 'shape_code' in shape_info:
            latent = shape_info['shape_code']
        input_x = torch.cat([pq.unsqueeze(0), latent], axis=1)

        with torch.no_grad():
            outpose = self.model(input_x)

        # if self.use_shape_code:
        #     shape_code = torch.tensor(self.shape_codes[objname][0], device=pq.device).flatten()
        #     input_x = torch.cat([pq, shape_code])
        # else:
        #     input_x = torch.cat([pq, torch.tensor([self.onehot.index(objname)], device=pq.device)])

        # with torch.no_grad():
        #     outpose = self.model(input_x.unsqueeze(0))
        outpose = outpose.squeeze(0)
        return outpose

    def get_shape_code(self, pc):
        # Farthest point sampling
        pc_t = torch.tensor(pc[:, :3]).unsqueeze(0)  # 1 x N x 3
        try:
            pc_fps, pc_fps_idxs = pytorch3d.ops.sample_farthest_points(pc_t, K=1500, random_start_point=True)
        except Exception as e:
            print(e)
            import IPython; IPython.embed()

        # Run VN-OccNets
        shape_mi = {'point_cloud': pc_fps.cuda()}
        with torch.no_grad():
            latent = self.model.shape_model.extract_latent(shape_mi)
            latent = torch.reshape(latent, (latent.shape[0], -1))
        return latent


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