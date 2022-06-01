#!/usr/bin/env python3
"""
The MIT License (MIT)

Copyright (c) 2020 NVIDIA Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import os.path as osp
import sys
import json
import trimesh
import pybullet as p
from .panda_env import PandaEnv
# import pyrender
import argparse
import numpy as np
import matplotlib.pyplot as plt

from acronym_tools import load_mesh, create_gripper_marker

from mpl_toolkits import mplot3d
from copy import deepcopy

import h5py
# from utils import regularize_pc_point_count

import torch
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.utils import trimesh_util
from datetime import datetime

# from pyrender_scene import PyrenderScene
# from scene_renderer import SceneRenderer
import pytorch3d
import pytorch3d.ops

from .utils import get_object_info, place_object


# def load_model():
#     model_path = '/home/exx/projects/manifolds/ndf_robot/src/ndf_robot/model_weights/multi_category_weights.pth'
#     model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model


# def visualize_scene(scene, target_obj, trimesh_camera, T_world2cam, T_world2cam2):
#     # show scene, including a marker representing the camera
#     trimesh_scene = scene.colorize({target_obj: [255, 0, 0]}).as_trimesh_scene()
#     trimesh_scene.add_geometry(
#         trimesh.creation.camera_marker(trimesh_camera),
#         node_name="camera1",
#         transform=T_world2cam)
#     trimesh_scene.add_geometry(
#         trimesh.creation.camera_marker(trimesh_camera),
#         node_name="camera2",
#         transform=T_world2cam2)
#     # trimesh_scene.add_geometry(
#     #     trimesh.creation.camera_marker(trimesh_camera),
#     #     node_name="obj",
#     #     transform=T_world2cam @ T_cam2objcom)
#     trimesh_scene.show()


# def visualize_pcs(pc_data, octr_objfrm, T_octr2pctr, pc_sampled):
#     pc = np.concatenate([pc_data['cam1_pc'], pc_data['cam2_pc']], axis=0)[:, :3]
#     pc_mean = np.mean(pc, axis=0)
#     p_objfrm = pc_data['cam1_pc'][:, :3]
#     p2_objfrm = pc_data['cam2_pc'][:, :3]
#     p_objfrm -= pc_mean
#     p2_objfrm -= pc_mean


#     fig = plt.figure(figsize=plt.figaspect(0.5))
#     ax = fig.add_subplot(1, 2, 1, projection='3d')
#     ax.scatter(p_objfrm[::10, 0], p_objfrm[::10, 1], p_objfrm[::10, 2], s=1)
#     ax.scatter(p2_objfrm[::10, 0], p2_objfrm[::10, 1], p2_objfrm[::10, 2], s=1)
#     if octr_objfrm is not None:
#         ax.scatter(octr_objfrm[0], octr_objfrm[1], octr_objfrm[2], s=50, c='green', alpha=0.7, label='obj_ctr')
#         ax.quiver(
#             octr_objfrm[0], octr_objfrm[1], octr_objfrm[2], # <-- starting point of vector
#             T_octr2pctr[0, 3], T_octr2pctr[1, 3], T_octr2pctr[2, 3],
#             color='blue', alpha=.8, lw=3,
#         )
#     lim = 0.5
#     ax.axes.set_xlim3d(left=-lim, right=lim)
#     ax.axes.set_ylim3d(bottom=-lim, top=lim)
#     ax.axes.set_zlim3d(bottom=-lim, top=lim)

#     ax = fig.add_subplot(1, 2, 2, projection='3d')
#     ax.scatter(pc_sampled[:, 0], pc_sampled[:, 1], pc_sampled[:, 2], s=1)
#     plt.show()


# def visualize_occ(pc_sampled, shape_mi, model, thresh=0.1):
#     pcd = trimesh.PointCloud(pc_sampled)
#     bb = pcd.bounding_box
#     eval_pts = bb.sample_volume(10000)
#     shape_mi['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().cuda().detach()
#     out = model(shape_mi)

#     in_inds = torch.where(out['occ'].squeeze() > thresh)[0].cpu().numpy()
#     in_pts = eval_pts[in_inds]
#     # out_inds = torch.where(out['occ'].squeeze() < thresh)[0].cpu().numpy()
#     # out_pts = eval_pts[out_inds]

#     # Center on mean of predicted volume
#     mean = in_pts.mean(axis=0)
#     pc_sampled -= mean
#     in_pts -= mean
#     trimesh_util.trimesh_show([in_pts, pc_sampled])


def main(args):
    np.random.seed(0)

    # load object meshes and generate a random scene
    obj_mesh_paths = []
    for obj_name in os.listdir(f"{args.mesh_root}/grasps"):
        obj_mesh_paths.append(f"{args.mesh_root}/grasps/{obj_name}")
    # table_dims = [2.0, 2.2, 0.6]
    # support_mesh = trimesh.creation.box(table_dims)

    save_dir = f"{args.mesh_root}/shape-dataset_{datetime.now().strftime('%m-%d-%y_%H-%M-%S')}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cam_look = [-0.05, -0.5, -0.6852]
    env = PandaEnv(renders=args.visualize, gravity=False, cam_look=cam_look)

    with h5py.File(f'{save_dir}/dataset.hdf5', 'a') as hf:
        
        for mesh_idx, mesh_path in enumerate(obj_mesh_paths):
            print(mesh_idx)
            obj_name = mesh_path.split('/')[-1].replace('.h5', '')
            if obj_name in hf:
                if len(hf[obj_name]) == args.n_samples:
                    print(f"skipping {obj_name}")
                    continue
                else:
                    grp = hf[obj_name]
            else:
                grp = hf.create_group(obj_name)
            
            # if 'Mug' not in obj_name:
                # continue

            mesh = load_mesh(mesh_path, mesh_root_dir=args.mesh_root)

            T_ctr2obj = np.eye(4)
            T_ctr2obj[:3, 3] = -mesh.centroid
            mesh = mesh.apply_transform(T_ctr2obj)

            dots = np.einsum('ij,ij->i',
                             mesh.center_mass - mesh.convex_hull.triangles_center,
                             mesh.convex_hull.face_normals)
            if not np.all(dots < 0):
                print("Warning: center of mass not inside convex hull, skipping")
                continue

            i = len(grp)
            # warn = 0
            # warn_total = 0
            # warn_obj = 0
            objinfo = get_object_info(env, obj_name, args.mesh_root)
            while i < args.n_samples:
                print(f"{i}th sample of {obj_name}")

                env.reset(no_table=True, objinfo=objinfo)
                T_world_obj = place_object(env, [0.5, 0.0, 0.5], random=False, gravity=False)
                obs = env._get_observation(get_pc=True, single_view=False)
                pc_world = obs['points']

                # try:
                #     scene = PyrenderScene.random_arrangement([mesh], support_mesh)
                #     warn = 0
                # except Exception as e:
                #     print(e)
                #     print("Warning: random arrangement failed")
                #     warn += 1
                #     warn_total += 1
                #     if warn > 5 or warn_total > 10: # 10 consecutive or 100 total failures
                #         warn = 0
                #         break
                #     continue

                # target_obj = "obj0"
                # if "obj0" not in scene._objects.keys():
                #     print("Warning: target object not in scene")
                #     warn_obj += 1
                #     warn_total += 1
                #     if warn_obj > 5 or warn_total > 10:
                #         warn_obj = 0
                #         break
                #     continue
                # else:
                #     warn_obj = 0

                # # choose camera intrinsics and extrinsics
                # # renderer = SceneRenderer(scene)
                # trimesh_camera = renderer.get_trimesh_camera()
                # roll = np.random.uniform(low=3 * np.pi / 8, high=4 * np.pi / 8)
                # yaw = np.random.uniform(low=-np.pi, high=np.pi)
                # rot_mat1 = trimesh.transformations.euler_matrix(roll, 0, yaw)
                # rand_dist = np.random.uniform(low=0.6, high=0.8)
                # camera_pose1 = trimesh_camera.look_at(
                #     points=[scene.get_transform(target_obj, frame="centroid")[:3, 3]],
                #     rotation=rot_mat1,
                #     distance=rand_dist,
                # )
                # rot_mat2 = trimesh.transformations.euler_matrix(roll, 0, yaw + np.pi)
                # camera_pose2 = trimesh_camera.look_at(
                #     points=[scene.get_transform(target_obj, frame="centroid")[:3, 3]],
                #     rotation=rot_mat2,
                #     distance=rand_dist,
                # )
                # rot_mat3 = trimesh.transformations.euler_matrix(roll, 0, yaw + np.pi/2)
                # camera_pose3 = trimesh_camera.look_at(
                #     points=[scene.get_transform(target_obj, frame="centroid")[:3, 3]],
                #     rotation=rot_mat3,
                #     distance=rand_dist,
                # )
                # rot_mat4 = trimesh.transformations.euler_matrix(roll, 0, yaw - np.pi/2)
                # camera_pose4 = trimesh_camera.look_at(
                #     points=[scene.get_transform(target_obj, frame="centroid")[:3, 3]],
                #     rotation=rot_mat4,
                #     distance=rand_dist,
                # )

                # # render observations
                # color1, depth1, _, segmentation1 = renderer.render(
                #     camera_pose=camera_pose1, target_id=target_obj)
                # color2, depth2, _, segmentation2 = renderer.render(
                #     camera_pose=camera_pose2, target_id=target_obj)
                # color3, depth3, _, segmentation3 = renderer.render(
                #     camera_pose=camera_pose3, target_id=target_obj)
                # color4, depth4, _, segmentation4 = renderer.render(
                #     camera_pose=camera_pose4, target_id=target_obj)

                # T_obj_world = scene.get_transform(target_obj, frame="centroid")
                # T_cam1_world = camera_pose1.dot(trimesh.transformations.euler_matrix(np.pi, 0, 0))
                # T_cam2_world = camera_pose2.dot(trimesh.transformations.euler_matrix(np.pi, 0, 0))
                # T_cam3_world = camera_pose3.dot(trimesh.transformations.euler_matrix(np.pi, 0, 0))
                # T_cam4_world = camera_pose4.dot(trimesh.transformations.euler_matrix(np.pi, 0, 0))

                # pc1_cam1 = renderer._to_pointcloud(depth1*segmentation1)          # N x [x y z 1]
                # pc2_cam2 = renderer._to_pointcloud(depth2*segmentation2)
                # pc3_cam3 = renderer._to_pointcloud(depth3*segmentation3)
                # pc4_cam4 = renderer._to_pointcloud(depth4*segmentation4)

                # # Save point cloud in object centroid / center of mass frame.
                # # Also save transformation from this frame to point cloud frame,
                # # where points are in workspace orientation, but with origin at the point cloud mean.
                # T_world_obj = np.linalg.inv(T_obj_world)
                # pc1_world = (T_cam1_world @ pc1_cam1.T).T
                # pc2_world = (T_cam2_world @ pc2_cam2.T).T
                # pc3_world = (T_cam3_world @ pc3_cam3.T).T
                # pc4_world = (T_cam4_world @ pc4_cam4.T).T
                # pc_world = np.concatenate([pc1_world, pc2_world, pc3_world, pc4_world], axis=0)  # N x 4




                # pc_ctr = np.mean(pc_world, axis=0)                              # (4,)

                # pc in object frame
                # pc_obj = (T_world_obj @ pc_world.T).T                           # N x 4

                # pc in workspace_mean frame
                # T_world_wsm = np.eye(4)
                # T_world_wsm[:3, 3] = -pc_ctr[:3]
                # pc_wsm = (T_world_wsm @ pc_world.T).T

                # Sample the data randomly and with farthest point sampling
                # pc_world_rnd = regularize_pc_point_count(pc_world, npoints=1500, use_farthest_point=False)  # 3.69s
                pc_world_t = torch.tensor(pc_world, dtype=torch.float32, device='cuda')
                pc_world_fps_t, pc_world_fps_idxs = pytorch3d.ops.sample_farthest_points(pc_world_t.unsqueeze(0), K=1500, random_start_point=True)  # 0.12s
                pc_world_fps_t = pc_world_fps_t.squeeze()
                pc_world_fps_idxs = pc_world_fps_idxs.squeeze()


                # Save data
                subgrp = grp.create_group(f'{i}')
                subgrp.create_dataset('pc_world', data=pc_world_t.cpu(), shape=pc_world_t.shape, compression='gzip', chunks=True)
                subgrp.create_dataset('pc_world_fps_t', data=pc_world_fps_t.cpu(), shape=pc_world_fps_t.shape, compression='gzip', chunks=True)
                subgrp.create_dataset('pc_world_fps_idxs', data=pc_world_fps_idxs.cpu(), shape=pc_world_fps_idxs.shape, compression='gzip', chunks=True)
                # subgrp.create_dataset('pc_world_rnd', data=pc_world_rnd, shape=pc_world_rnd.shape, compression='gzip', chunks=True)
                subgrp.create_dataset('T_world_obj', data=T_world_obj, shape=T_world_obj.shape, compression='gzip', chunks=True)
                # subgrp.create_dataset('T_world_wsm', data=T_world_wsm, shape=T_world_wsm.shape, compression='gzip', chunks=True)
                # subgrp.create_dataset('pc_ctr', data=pc_ctr, shape=pc_ctr.shape, compression='gzip', chunks=True)

                if args.visualize:    # debug visualization
                    # scene_dbg = scene.as_trimesh_scene()
                    # mesh_world = deepcopy(mesh).apply_transform(T_obj_world)
                    # pcd1_world = trimesh.points.PointCloud(pc1_world[:, :3], np.array([255, 0, 0, 255]))
                    # pcd2_world = trimesh.points.PointCloud(pc2_world[:, :3], np.array([0, 0, 255]))
                    # pcd3_world = trimesh.points.PointCloud(pc3_world[:, :3], np.array([0, 255, 0, 255]))
                    # pcd4_world = trimesh.points.PointCloud(pc4_world[:, :3], np.array([255, 0, 255, 255]))
                    # scene_dbg.add_geometry(mesh_world)
                    # scene_dbg.add_geometry(pcd1_world)
                    # scene_dbg.add_geometry(pcd2_world)
                    # scene_dbg.add_geometry(pcd3_world)
                    # scene_dbg.add_geometry(pcd4_world)
                    # scene_dbg.add_geometry(
                    #     trimesh.creation.camera_marker(trimesh_camera),
                    #     node_name="camera1",
                    #     transform=T_cam1_world)
                    # scene_dbg.add_geometry(
                    #     trimesh.creation.camera_marker(trimesh_camera),
                    #     node_name="camera2",
                    #     transform=T_cam2_world)
                    # scene_dbg.add_geometry(
                    #     trimesh.creation.camera_marker(trimesh_camera),
                    #     node_name="camera3",
                    #     transform=T_cam3_world)
                    # scene_dbg.add_geometry(
                    #     trimesh.creation.camera_marker(trimesh_camera),
                    #     node_name="camera4",
                    #     transform=T_cam4_world)
                    # scene_dbg.show()

                    # visualize object in world frame
                    pcd_world = trimesh.points.PointCloud(pc_world[:, :3], np.array([0, 255, 0, 255]))
                    # pcd_obj = trimesh.points.PointCloud(pc_obj[:, :3], np.array([0, 0, 255, 255]))
                    trimesh.Scene([pcd_world]).show()

                    # transform to mesh frame
                    pc_obj = (np.linalg.inv(T_world_obj) @ pc_world.T).T
                    pcd_obj = trimesh.points.PointCloud(pc_obj[:, :3], np.array([255, 0, 0, 255]))
                    trimesh.Scene([pcd_obj, pcd_world]).show()


                    # trimesh.Scene([pcd1_world, pcd2_world, pcd3_world, pcd4_world, pcd_obj]).show()

                    # visualize workspace frame 
                    # pcd_wsm = trimesh.points.PointCloud(pc_wsm[:, :3], np.array([255, 255, 0, 255]))
                    # trimesh.Scene([pcd1_world, pcd2_world, pcd3_world, pcd4_world, pcd_wsm]).show()

                    # import IPython; IPython.embed()

                    # import matplotlib.pyplot as plt
                    # plt.imshow(depth1)
                    # plt.show()


                # align_to_cam = False
                # if align_to_cam:
                #     # Get the camera object frame (object center with camera axes)
                #     T_cam2obj = deepcopy(T_cam2objcom)
                #     T_cam2obj[:3, :3] = T_world2cam[:3, :3]
                #     T_obj2cam = np.linalg.inv(T_cam2obj)

                #     pctr_camfrm = p_camfrm[:, :3].mean(axis=0)
                #     p_objfrm = p_camfrm - np.append(pctr_camfrm, 1)
                #     p2_objfrm = p2_camfrm - np.append(pctr_camfrm, 1)

                #     # Get point cloud centroid and object center in object frame
                #     octr_camfrm = T_cam2objcom[:3, 3]
                #     octr_objfrm = octr_camfrm - pctr_camfrm

                #     # Get transform from point cloud centroid to object center
                #     T_octr2pctr = deepcopy(T_cam2objcom)
                #     T_octr2pctr[:3, 3] -= pctr_camfrm
                # else:  # align to object frame
                #     p_objfrm = (T_cam2objcom @ p_camfrm.T).T
                #     p_wfrm = ((np.linalg.inv(T_world2cam) @ T_world2objcom) @ p_camfrm.T).T
                #     p2_objfrm = (T_cam2objcom @ p2_camfrm.T).T
                #     # cam 2 objcom doesn't account for point cloud mismatch?
                #     T_obj2cam = np.linalg.inv(T_cam2objcom)
                #     octr_objfrm = T_cam2objcom[:3, 3]
                #     octr_objfrm = octr_camfrm
                #     T_octr2pctr = np.array([])

                #     # p_wfrm = (T_world2cam @ p_camfrm.T).T
                #     p_wfrm = (np.linalg.inv(T_world2cam) @ p_camfrm.T).T
                #     pcd = trimesh.points.PointCloud(p_camfrm[:, :3], np.array([255, 0, 0, 255]))

                #     p_wfrm = (T_cam2objcom @ p_camfrm.T).T
                #     pcd = trimesh.points.PointCloud(p_wfrm[:, :3], np.array([255, 0, 0, 255]))

                #     mesh = mesh.apply_transform(np.linalg.inv(T_world2objcom))
                #     trimesh.Scene([mesh, pcd]).show()
                #     T_ctr2obj = np.eye(4)
                #     T_ctr2obj[:3, 3] = -mesh.centroid
                #     mesh = mesh.apply_transform(T_ctr2obj)

                # pc_data = {
                #     'cam1_pc': p_objfrm,
                #     'cam2_pc': p2_objfrm
                # }
                # np.save(f'{save_dir}/obj_{mesh_idx}_{obj_name}.npy', pc_data)

                # # Get shape code using pretrained vn-occnets
                # pc = np.concatenate([pc_data['cam1_pc'], pc_data['cam2_pc']], axis=0)[:, :3]
                # pc -= np.mean(pc, axis=0)

                # pc_sampled = regularize_pc_point_count(pc, npoints=1500, use_farthest_point=args.fps)

                # shape_mi = {}
                # shape_pcd = torch.from_numpy(pc_sampled).float().cuda()
                # shape_mi['point_cloud'] = shape_pcd.unsqueeze(0)
                # with torch.no_grad():
                #     z = model.extract_latent(shape_mi).cpu().numpy()  # 1 x 256 x 3

                # if args.visualize:
                #     visualize_scene(scene, target_obj, trimesh_camera, T_world2cam, T_world2cam2)
                #     visualize_pcs(pc_data, octr_objfrm, T_octr2pctr, pc_sampled)
                #     visualize_occ(pc_sampled, shape_mi, model)

                # # Save point clouds in object frame, pc to obj transform, camera pose (in object frame)
                # # Make group for ith point cloud sample
                # subgrp = grp.create_group(f'{i}')
                # subgrp.create_dataset('p_objfrm', data=p_objfrm, shape=p_objfrm.shape, compression='gzip', chunks=True)
                # subgrp.create_dataset('T_octr2pctr', data=T_octr2pctr, shape=T_octr2pctr.shape, compression='gzip', chunks=True)
                # subgrp.create_dataset('T_obj2cam', data=T_obj2cam, shape=T_obj2cam.shape, compression='gzip', chunks=True)
                # subgrp.create_dataset('pc', data=pc, shape=pc.shape, compression='gzip', chunks=True)
                # subgrp.create_dataset('pc_sampled', data=pc_sampled, shape=pc_sampled.shape, compression='gzip', chunks=True)
                # subgrp.create_dataset('shape_code', data=z, shape=z.shape, compression='gzip', chunks=True)

                i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render observations of a randomly generated scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mesh_root", default=".", help="Directory used for loading meshes."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show the scene and camera pose from which observations are rendered.",
    )
    parser.add_argument(
        "--fps",
        action="store_true",
        help="Use furthest-point sampling.",
    )
    parser.add_argument(
        "--n_samples", default=1, type=int,
        help="Number of samples per object"
    )
    parser.add_argument(
        "--target_pc_size", default=1500,
        help="Size of downsampled point cloud"
    )
    args = parser.parse_args()
    main(args)
