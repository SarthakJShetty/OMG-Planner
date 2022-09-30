import os
import argparse
import numpy as np
from omegaconf import OmegaConf

import torch
import pytorch_lightning.utilities.seed as seed_utils


import h5py
from acronym_tools import create_gripper_marker
from manifold_grasping.utils import load_mesh, load_grasps, get_input, scale_logmap, wrist_to_tip, get_plotly_fig

import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import trimesh

# import theseus as th
from pathlib import Path

from manifold_grasping.dataset import GraspDataModule
import time

import plotly
import plotly.graph_objects as go
# import plotly.offline as py
from tqdm import tqdm
from datetime import datetime

from manifold_grasping.generate_grasp_data.collect_dataset_lg import get_closest_idx_cp
from manifold_grasping.dataset import pose_to_pyg, pc_to_pyg, merge_pygs
from manifold_grasping.control_pts import transform_control_points
from manifold_grasping.networks import Decoder
import pandas as pd
from pytorch3d.transforms import rotation_6d_to_matrix

from torch_geometric.data import Data as GData
from torch_geometric.data import Batch as GBatch

import wandb

# for bullet eval
from omg_bullet.get_bullet_labels_mp import run_grasp


def optimize_floating_gripper(net, x, binfo,
    lr=0.001, # TODO move to config 
    loss_thresh=0.0001
    ):
    max_iter = net.max_test_iter
    torch.set_grad_enabled(True)
    # optimize to minimize distance to grasp
    print("optimizing ") # todo change to logger
    x_pose = binfo['pose']
    latent = binfo['latent'] # embedding
    traj = []
    x_pose = x_pose.detach().clone()
    x_pose.requires_grad = True
    opt = torch.optim.Adam([x_pose], lr=lr)
    for i in range(max_iter):
        # Get input for the current timestep
        x = torch.cat([x_pose, latent], axis=1)

        # Predict distance to closest grasp using network
        y_hat = net(x)

        # Save transform and distance for the current timestep
        traj.append((x_pose.detach().clone().squeeze(), y_hat))

        # Backprop to get grad x
        loss = y_hat.mean(axis=1) # cpd
        # loss = torch.abs(loss) # Note this is because networks aren't currently constrained to be positive, will re-train
        if loss < loss_thresh:
            print(f"{i} loss {loss} below threshold {loss_thresh}, terminating early")
            break
        elif i % (max_iter // 100) == 0:
            print(f"{i}: {loss}")

        loss.backward()
        opt.step()
        opt.zero_grad()

        x_pose = opt.param_groups[0]['params'][0]
        
        if net.in_type == 'pq':
            qnorm = torch.linalg.norm(x_pose[:, 3:8])
            div = torch.ones_like(x_pose)
            div[:, 3:8] = qnorm # TODO update for 6D rotation representation
            x_pose = torch.div(x_pose, div)
    return x_pose, traj


# TODO move to utils
# TODO cleanup so that everything is in torch?
def pose_to_T(pose, tip2wrist_T):
    """Convert pose in position+quaternion or position + 6D rotation to transform matrix"""
    if pose.shape[0] == 9: # p6d
        T = torch.eye(4, device=pose.device)
        T[:3, :3] = rotation_6d_to_matrix(pose[3:])
        T[:3, 3] = pose[:3]
        T = (T.detach().cpu().numpy() @ tip2wrist_T)
    elif pose.shape[0] == 7: # pq
        T = (pt.transform_from_pq(pose.detach().cpu().numpy()) @ tip2wrist_T)
    else:
        raise NotImplementedError
    return T


def get_closest_cp_dist(dset_grasp, obj, tip2wrist_T, traj):
    # Get closest grasp to final pose in dataset
    pos_Ts = dset_grasp[obj]['pos_Ts']
    # convert pos Ts to pybullet wrist convention from acronym convention 
    # query / end pose is already in pybullet convention
    T_rotgrasp2grasp = pt.transform_from(pr.matrix_from_axis_angle([0, 0, 1, -np.pi/2]), [0, 0, 0])
    T_rotgrasp2grasp = np.eye(4)
    pos_Ts = pos_Ts @ T_rotgrasp2grasp
    pos_cps = transform_control_points(pos_Ts, batch_size=pos_Ts.shape[0], mode='rt', device='cuda')
    end_pose = traj[-1][0]
    end_T = pose_to_T(end_pose, tip2wrist_T)[np.newaxis, :]
    end_cp = transform_control_points(end_T, batch_size=end_T.shape[0], mode='rt', device='cuda')
    pos_idx_cp, _, cp_dist = get_closest_idx_cp(end_cp, pos_cps)
    closest_T = pos_Ts[pos_idx_cp]
    print(f"Closest distance: {cp_dist:.3f}")
    return end_T, cp_dist, closest_T


def evaluate(net, x, binfo, data_path):
    start_time = time.time()
    dset_grasp = h5py.File(f'{data_path}/dataset.hdf5', 'r')
    if net.use_tip:
        tip2wrist_T = np.linalg.inv(wrist_to_tip(device='cpu')) 
    else:
        tip2wrist_T = np.eye(4) 

    x_pose, traj = optimize_floating_gripper(net, x, binfo)
    obj = binfo['obj'][0]
    end_T, cp_dist, closest_T = get_closest_cp_dist(dset_grasp, obj, tip2wrist_T, traj)
    loss = cp_dist

    if False: # debug visualization
        visualize_grasp_set(data_path, dset_grasp)

    elapsed_time = time.time() - start_time
    print(f"duration: {elapsed_time}")
    return loss, traj, closest_T

def visualize_grasp_set(data_path, dset_grasp):
    mesh, mesh_mctr_T = load_mesh(f'{data_path}/../grasps/{obj}.h5', mesh_root_dir=f'{data_path}/../', load_for_bullet=True)
    # TODO update to follow acronym vs. pybullet wrist convention, but we already have log_visualization so low pri
    pos_Ts = dset_grasp[obj]['pos_Ts']
    grasps = [create_gripper_marker(color=[0, 255, 0], tube_radius=0.003).apply_transform(T) for T in pos_Ts]

    mesh_fig = get_plotly_fig(mesh)
    data = list(mesh_fig.data)
    for grasp in grasps:
        grasp_fig = get_plotly_fig(grasp)
        data += list(grasp_fig.data)
    fig = go.Figure()
    fig.add_traces(data)
    fig.update_layout(coloraxis_showscale=False)
    import plotly.offline as py
    py.iplot(fig)

def evaluate_pointset(net, batch, data_path):
    start_time = time.time()
    dset_grasp = h5py.File(f'{data_path}/dataset.hdf5', 'r')
    
    x_pose, traj = optimize_floating_gripper_pointset(net, batch)
    obj = batch.info['obj'][0]
    tip2wrist_T = np.eye(4)
    end_T, cp_dist, closest_T = get_closest_cp_dist(dset_grasp, obj, tip2wrist_T, traj)
    loss = cp_dist

    if False: # debug visualization
        visualize_grasp_set(data_path, dset_grasp)

    elapsed_time = time.time() - start_time
    print(f"duration: {elapsed_time}")
    return loss, traj, closest_T

def optimize_floating_gripper_pointset(net, batch, 
    lr=0.001, # TODO move to config 
    loss_thresh=0.0001
    ):
    max_iter = net.max_test_iter
    torch.set_grad_enabled(True)

    # Get initial pose
    gripper_pose = batch.info['query_pose'].detach().clone()
    gripper_pose.requires_grad = True

    # Get object point cloud
    obj_pc_idxs = batch.x[:, 1] == 1
    obj_pc = batch.pos[obj_pc_idxs]
    pc_pos, pc_1hot = pc_to_pyg(obj_pc)

    # optimize to minimize distance to grasp
    print("optimizing ") # todo change to logger
    traj = []
    opt = torch.optim.Adam([gripper_pose], lr=lr)
    for i in range(max_iter):
        # Get input for the current timestep
        gripper_pos, gripper_1hot, _ = pose_to_pyg(gripper_pose)
        pos, feat = merge_pygs([gripper_pos, pc_pos], [gripper_1hot, pc_1hot], gripper_pose)
        batch.info['query_pose'] = gripper_pose
        data = GData(x=feat, pos=pos, info=batch.info)
        x = GBatch.from_data_list([data])

        # Predict distance to closest grasp using network
        y_hat = net(x)
        y_hat_masked = feat[:, :1] * y_hat

        # Save transform and distance for the current timestep
        traj.append((gripper_pose.detach().clone().squeeze(), y_hat_masked))

        # Backprop to get grad x
        loss = y_hat_masked.sum() / feat[:, :1].sum() # mean cpd
        if loss < loss_thresh:
            print(f"{i} loss {loss} below threshold {loss_thresh}, terminating early")
            break
        elif i % (max_iter // 100) == 0:
            print(f"{i}: {loss}")

        loss.backward()
        opt.step()
        opt.zero_grad()

        gripper_pose = opt.param_groups[0]['params'][0] # may not be necessary
        
        if net.in_type == 'pq':
            qnorm = torch.linalg.norm(gripper_pose[:, 3:8])
            div = torch.ones_like(gripper_pose)
            div[:, 3:8] = qnorm # TODO update for 6D rotation representation
            gripper_pose = torch.div(gripper_pose, div)
    return gripper_pose, traj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', help='Full path to checkpoint')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--loss_thresh', type=float, default=0.0001)
    parser.add_argument('--max_iter', type=int, default=3000)
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--max_samples', type=int, default=3)
    parser.add_argument('--final_viz_samples', type=int, default=20)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--plot_offline', action='store_true', default=False)
    parser.add_argument('--eval_pybullet', action='store_true', default=False)
    # parser.add_argument('--pybullet_wrist_convention', help='input should be in pybullet wrist convention', action='store_true', default=False)
    args = parser.parse_args()

    # Load config with object names and grasps dataset
    cfg_path = Path(args.ckpt_path).parents[3] / '.hydra' / 'config.yaml'
    cfg = OmegaConf.load(cfg_path)
    cfg.data_root = cfg_path.parents[(len(cfg_path.parents)-1) - 3]
    # cfg.data_root = Path(args.data_path).parents[1]
    # cfg.net.data_path
    # cfg.data_path = args.data_path
    # cfg.net.data_path = cfg.data_path
    cfg.net.max_test_samples = args.max_samples
    cfg.net.max_test_iter = args.max_iter
    cfg.project_root = "/home/thomasweng/projects/manifolds"
    
    # Load model from checkpoint
    net = Decoder.load_from_checkpoint(args.ckpt_path, latent=cfg.net.latent, 
        data_path=cfg.net.data_path,
        max_test_samples=cfg.net.max_test_samples,
        max_test_iter=cfg.net.max_test_iter)
    net.cuda()
    net.eval()

    # Set up dataloader
    cfg.net.batch_size = 1
    seed_utils.seed_everything(0)
    dm = GraspDataModule(cfg=cfg)
    dm.setup_test()
    loader = dm.test_dataloader()
    
    # Set up logging 
    tstamp = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    log_path = Path(cfg_path).parents[1] / 'eval' / tstamp
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    hf = h5py.File(log_path / 'eval.hdf5', 'w')
    for obj in dm.grasps_test.objects:
        obj_grp = hf.create_group(obj)
        cp_dists = obj_grp.create_dataset('cp_dists', shape=(0,), maxshape=(None,))
        end_Ts = obj_grp.create_dataset('end_Ts', shape=(0, 4, 4), maxshape=(None, 4, 4))

    if args.use_wandb:
        wandb_cfg = {
            'args': vars(args),
            'net_cfg': cfg
        }
        wandb.init(
            id=tstamp,
            name=cfg.experiment,
            project="eval-graspfields", 
            entity="cmu-rpad-tweng", 
            config=wandb_cfg,
            dir=str(log_path))

    dset_grasp = h5py.File(f'{cfg.net.data_path}/dataset.hdf5', 'r')
    # tip2wrist_T = np.linalg.inv(wrist_to_tip(device='cpu'))
    tip2wrist_T = np.eye(4)
    for batch_idx, batch in enumerate(loader):
        samples_per_obj = [len(hf[obj]['cp_dists']) for obj in dm.grasps_test.objects]
        if all(samples >= args.max_samples for samples in samples_per_obj):
            print("Max samples for all objects reached")
            break
    
        # get batch
        query_T, grasp_T, info = batch
        batch[0] = batch[0].cuda()
        batch[1] = batch[1].cuda()
        obj = info['obj'][0]
        hf_idx = len(hf[obj]['cp_dists'])
        if hf_idx >= args.max_samples:
            print(f"Skipping {obj}, max samples reached")
            continue
        hf[obj]['cp_dists'].resize(hf_idx+1, axis=0)
        hf[obj]['end_Ts'].resize(hf_idx+1, axis=0)
        grasp_type = info['grasp_type'][0]
        assert grasp_type == 'query'
        print(obj)

        if cfg.net.latent.type == 'embedding':
            batch[2]['pc'] = batch[2]['pc'].cuda()

        x, y, binfo = net.get_batch(batch)

        x_pose, traj = optimize_floating_gripper(net, x, binfo, 
            lr=args.lr, loss_thresh=args.loss_thresh)
        end_T, cp_dist, closest_T = get_closest_cp_dist(dset_grasp, obj, tip2wrist_T, traj)

        # Save data and compute metrics
        hf[obj]['end_Ts'][hf_idx] = end_T
        hf[obj]['cp_dists'][hf_idx] = cp_dist
        dict_obj = dict( [(obj, pd.Series(hf[obj]['cp_dists'])) for obj in hf.keys()] )
        df_obj = pd.DataFrame.from_dict(dict_obj)
        df_obj_stats = pd.DataFrame(df_obj.mean().round(4).astype(str) + " +/- " + df_obj.std().round(3).astype(str))
        all_dists = [v for obj in hf.keys() for v in hf[obj]['cp_dists']]
        df_all = pd.DataFrame(all_dists)
        df_all_stats = pd.DataFrame(df_all.mean().round(4).astype(str) + " +/- " + df_all.std().round(3).astype(str))
        with open(log_path / 'df_obj_stats.md', 'w') as f:
            f.write(df_obj_stats.to_markdown())
        with open(log_path / 'df_all_stats.md', 'w') as f:
            f.write(df_all_stats.to_markdown())
        print(df_obj_stats.to_markdown())
        print(df_all_stats.to_markdown())

        # Visualize trajectory
        if args.visualize:  # debug visualization
            net.log_visualization(obj, traj, closest_T, batch_idx, use_wandb=args.use_wandb, plot_offline=args.plot_offline, fixed_skip=True, rotate_gripper=False)
        # if args.eval_pybullet:
            # run_grasp(obj, closest_T, render=True)
    
    avg_cp_dists = np.array(hf[obj]['cp_dists']).mean()
    std_cp_dists = np.array(hf[obj]['cp_dists']).std()
    print(f"{avg_cp_dists:.3f} +/- {std_cp_dists:.2f}")

    # Visualize end_Ts for each object
    if args.visualize:
        T_rotgrasp2grasp = pt.transform_from(pr.matrix_from_axis_angle([0, 0, 1, -np.pi/2]), [0, 0, 0])  # correct wrist rotation
        T_grasp2rotgrasp = np.linalg.inv(T_rotgrasp2grasp)
        for obj in dm.grasps_test.objects:
            end_Ts = np.array(hf[obj]['end_Ts'])
            end_idxs = np.random.choice(range(len(end_Ts)), size=len(end_Ts) if len(end_Ts) < args.final_viz_samples else args.final_viz_samples)
            end_Ts = end_Ts[end_idxs]
            mesh, mesh_mctr_T = load_mesh(f'{cfg.net.data_path}/../grasps/{obj}.h5', mesh_root_dir=f'{cfg.net.data_path}/../', load_for_bullet=True)
            # note: mctr frame instead of pctr frame
            grippers = [create_gripper_marker(color=[0, 255, 255], tube_radius=0.003, sections=4).apply_transform(end_T @ T_grasp2rotgrasp) for end_T in end_Ts]
            mesh_fig = get_plotly_fig(mesh)
            data = list(mesh_fig.data)
            for gripper in grippers:
                gripper_fig = get_plotly_fig(gripper)
                data += list(gripper_fig.data)
            fig = go.Figure()
            fig.add_traces(data)
            fig.update_layout(coloraxis_showscale=False)
            if args.plot_offline:
                import plotly.offline as py
                py.iplot(fig)
            if args.use_wandb:
                wandb.log({
                    f"3Dfinal_{obj.split('_')[0]}": wandb.Html(plotly.io.to_html(fig)),
                })

            

            # if cfg.net.latent.type == 'embedding':
            #     # Mean-centered pc frame
            #     pc = info['pc'][0].detach().cpu().numpy()  # 1500 x 4
            #     pcd = [trimesh.points.PointCloud(pc[:, :3], colors=np.array([0, 0, 255, 255]))]
            #     # Account for mesh and point cloud center offset
            #     mesh_pctr_T = info['mesh_pctr_T'].squeeze()
            #     mesh = mesh.apply_transform(mctr_mesh_T).apply_transform(mesh_pctr_T)



        # # Get closest grasp to final pose in dataset
        # pos_Ts = dset_grasp[obj]['pos_Ts']
        # pos_cps = transform_control_points(pos_Ts, batch_size=pos_Ts.shape[0], mode='rt', device='cuda')
        # end_pq = traj[-1][0]
        # # TODO do transform from pq in torch
        # end_T = (pt.transform_from_pq(end_pq.detach().cpu().numpy()) @ tip2wrist_T)[np.newaxis, :]
        # end_cp = transform_control_points(end_T, batch_size=end_T.shape[0], mode='rt', device='cuda')
        # pos_idx_cp, _, cp_dist = get_closest_idx_cp(end_cp, pos_cps)
        # closest_T = pos_Ts[pos_idx_cp]
        # print(f"Closest distance: {cp_dist}")

        
        #     query = create_gripper_marker(color=[0, 0, 255], tube_radius=0.002).apply_transform(query_T[0] @ T_grasp2rotgrasp)
        #     # grasp = [create_gripper_marker(color=[0, 255, 0]).apply_transform(grasp_T[0] @ T_grasp2rotgrasp)]
            
        #     query_iters = []
        #     skip = 50 if len(traj) < 500 else len(traj) // 50
        #     traj.reverse()
        #     for i, (pq, dist) in enumerate(traj[::skip]):
        #         T = pt.transform_from_pq(pq.detach().cpu().numpy()) @ tip2wrist_T
        #         if i == 0:
        #             query_iters.append(create_gripper_marker(color=[255, 0, 255], tube_radius=0.003, sections=4).apply_transform(T @ T_grasp2rotgrasp)) # mark final point
        #         else:
        #             query_iters.append(create_gripper_marker(color=[0, 255, 255], sections=3).apply_transform(T @ T_grasp2rotgrasp))

        #     mesh, mesh_mctr_T = load_mesh(f'{cfg.data_path}/../grasps/{obj}.h5', mesh_root_dir=f'{cfg.data_path}/../')
        #     mctr_mesh_T = np.linalg.inv(mesh_mctr_T)

        #     grasp = create_gripper_marker(color=[0, 255, 0], tube_radius=0.003).apply_transform(closest_T @ T_grasp2rotgrasp)

        #     if cfg.net.latent.type == 'embedding':
        #         # Mean-centered pc frame
        #         pc = info['pc'][0].detach().cpu().numpy()  # 1500 x 4
        #         pcd = [trimesh.points.PointCloud(pc[:, :3], colors=np.array([0, 0, 255, 255]))]
        #         # Account for mesh and point cloud center offset
        #         mesh_pctr_T = info['mesh_pctr_T'].squeeze()
        #         mesh = mesh.apply_transform(mctr_mesh_T).apply_transform(mesh_pctr_T)
        #     else:
        #         pcd = []

        #     # plotly
        #     print("process meshes for plotly")
        #     mesh_fig = get_plotly_fig(mesh)
        #     data = list(mesh_fig.data)
        #     query_fig = get_plotly_fig(query)
        #     data += list(query_fig.data)
        #     grasp_fig = get_plotly_fig(grasp)
        #     data += list(grasp_fig.data)
        #     # if cfg.net.latent.type == 'embedding':
        #     #     pc_fig = go.Figure(data=[go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
        #     #                         mode='markers')])
        #     #     data += list(pc_fig.data)

        #     query_figs = []
        #     for query_iter in tqdm(query_iters):
        #         query_fig = get_plotly_fig(query_iter)
        #         data += list(query_fig.data)
        #     fig = go.Figure()
        #     fig.add_traces(data)
        #     fig.update_layout(coloraxis_showscale=False)

        #     # py.iplot(fig)
        #     if args.use_wandb:
        #         wandb.log({
        #             f"3Dplot_{obj.split('_')[0]}_{hf_idx}": wandb.Html(plotly.io.to_html(fig)),
        #         })
        #         for obj in dm.grasps_test.objects:
        #             wandb.run.summary[f"avg_dist_{obj.split('_')[0]}"] = df_obj[obj].mean()
            
