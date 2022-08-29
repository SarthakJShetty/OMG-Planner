import os
import argparse
import contact_graspnet
from contact_graspnet import config_utils
from contact_graspnet.inference import inference as cg_inference
from contact_graspnet.inference import init as cg_init

class ContactGraspNetInference:
    def __init__(self, save_results=False, visualize=False):
        self.args = self.get_args()
        self.global_config = config_utils.load_config(self.args.ckpt_dir, batch_size=self.args.forward_passes, arg_configs=self.args.arg_configs)

        # move some of inference to init
        sess, grasp_estimator = cg_init(self.global_config, self.args.ckpt_dir)
        self.sess = sess
        self.grasp_estimator = grasp_estimator

    def inference(self, pc, 
        # save_results=False, visualize=False
        ):
        # pred_grasps_cam, scores, contact_pts = cg_inference(self.sess, self.grasp_estimator, pc, z_range=eval(str(self.args.z_range)),
        #                                                     K=self.args.K, local_regions=self.args.local_regions, filter_grasps=self.args.filter_grasps, segmap_id=self.args.segmap_id, 
        #                                                     forward_passes=self.args.forward_passes, skip_border_objects=self.args.skip_border_objects,
        #                                                     save_results=save_results, visualize=visualize)
        
        print('Generating Grasps...')
        pc_full = pc[:, :3]
        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(self.sess, pc_full, 
                                                                                          local_regions=self.args.local_regions, filter_grasps=self.args.filter_grasps, 
                                                                                          forward_passes=self.args.forward_passes)  
                                                                                        #   pc_segments=pc_segments, 


        return pred_grasps_cam[-1], scores[-1], contact_pts[-1]

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--ckpt_dir', default=f'{os.path.dirname(contact_graspnet.__file__)}/../checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
        parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
        parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
        parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
        parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
        # parser.add_argument('--local_regions', action='store_true', default=True, help='Crop 3D local regions around given segments.')
        parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
        parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
        # parser.add_argument('--filter_grasps', action='store_true', default=True,  help='Filter grasp contacts according to segmap.')
        parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
        parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
        parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
        parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
        args = parser.parse_args([])
        return args