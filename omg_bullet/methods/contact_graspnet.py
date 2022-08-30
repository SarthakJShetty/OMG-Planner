import os
import argparse
import contact_graspnet
from contact_graspnet import config_utils
# from contact_graspnet.inference import inference as cg_inference
# from contact_graspnet.inference import init as cg_init
from keras import backend as K 
import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt


# import multiprocessing

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# def run_inference(args, global_config, pc):
#     # args, global_config, pc = margs

#     sess, grasp_estimator = cg_init(global_config, args.ckpt_dir)

#     print('Generating Grasps...')
#     pc_full = pc[:, :3]
#     pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, 
#                                                                                     local_regions=args.local_regions, filter_grasps=args.filter_grasps, 
#                                                                                     forward_passes=args.forward_passes)  
#                                                                                     #   pc_segments=pc_segments, 

#     # Correct wrist rotation convention discrepancy
#     T_rotgrasp2grasp = pt.transform_from(pr.matrix_from_axis_angle([0, 0, 1, -np.pi/2]), [0, 0, 0])  # correct wrist rotation
#     pred_grasps = pred_grasps_cam[-1]
#     pred_grasps = pred_grasps @ T_rotgrasp2grasp

#     # tf.reset_default_graph()
#     # self.delete()

#     return pred_grasps, scores[-1], contact_pts[-1]


class ContactGraspNetInference:
    def __init__(self, save_results=False, visualize=False):
        self.args = self.get_args()
        self.global_config = config_utils.load_config(self.args.ckpt_dir, batch_size=self.args.forward_passes, arg_configs=self.args.arg_configs)

    # def inference1(self, pc):
    #     # https://github.com/tensorflow/tensorflow/issues/19731
    #     manager = multiprocessing.Manager()
    #     return_dict = manager.dict()
    #     p = multiprocessing.Process(
    #         target=run_inference,
    #         args=(self.args, self.global_config, pc)
    #     )
    #     p.start()
    #     p.join()
    #     print(return_dict.values())

    def inference(self, pc):
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        with tf.Graph().as_default(), tf.Session(config=config) as sess:
            from contact_graspnet.contact_grasp_estimator import GraspEstimator
            grasp_estimator = GraspEstimator(self.global_config)
            grasp_estimator.build_network()

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(save_relative_paths=True)

            # Load weights
            grasp_estimator.load_weights(sess, saver, self.args.ckpt_dir, mode='test')

            print('Generating Grasps...')
            pc_full = pc[:, :3]
            pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, 
                                                                                            local_regions=self.args.local_regions, filter_grasps=self.args.filter_grasps, 
                                                                                            forward_passes=self.args.forward_passes)  
                                                                                            #   pc_segments=pc_segments, 

            # Correct wrist rotation convention discrepancy
            T_rotgrasp2grasp = pt.transform_from(pr.matrix_from_axis_angle([0, 0, 1, -np.pi/2]), [0, 0, 0])  # correct wrist rotation
            pred_grasps = pred_grasps_cam[-1]
            pred_grasps = pred_grasps @ T_rotgrasp2grasp

            return pred_grasps, scores[-1], contact_pts[-1]

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
