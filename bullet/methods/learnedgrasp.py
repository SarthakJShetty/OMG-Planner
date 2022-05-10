import torch
import pytorch_lightning
from manifold_grasping.networks import Decoder


class LearnedGrasp:
    def __init__(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.model = Decoder(**ckpt['hyper_parameters'])
        self.model.cuda()
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

    def forward(self, x):
        """
        x: xyz, wxyz
        """
        with torch.no_grad():
            outpose = self.model(x)
        outpose = outpose.squeeze(0)
        return outpose


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