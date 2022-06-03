import argparse
import trimesh
from acronym_tools import load_mesh, create_gripper_marker
import pickle
import yaml
import os
import numpy as np
from .util import *
from easydict import EasyDict as edict

def trajT_to_grasppredT(T):
    offset_T = np.array(rotZ(np.pi / 2)) # transform to correct wrist rotation 
    return T @ offset_T
    
def grasppredT_to_trajT(T):
    offset_T = np.array(rotZ(np.pi / 2)) # transform to correct wrist rotation 
    return T @ np.linalg.inv(offset_T)

def visualize_predicted_grasp(iter, cfg, T_obj2ee, T_obj2goal, T_objfrm2obj, show=False, rotate_wrist=False):
    grasp_root = f"/checkpoint/thomasweng/acronym/grasps"
    obj_name = 'Book_5e90bf1bb411069c115aef9ae267d6b7_0.0268818133810836'
    obj_mesh = load_mesh(f"{grasp_root}/{obj_name}.h5", mesh_root_dir=cfg.acronym_dir)

    obj_mesh = obj_mesh.apply_transform(np.linalg.inv(T_objfrm2obj))

    if rotate_wrist:
        T_obj2ee = trajT_to_grasppredT(T_obj2ee)
        T_obj2goal = trajT_to_grasppredT(T_obj2goal)

    ee_pose = [create_gripper_marker(color=[0, 0, 255]).apply_transform(T_obj2ee)]
    goal_pose = [create_gripper_marker(color=[255, 0, 0]).apply_transform(T_obj2goal)]

    scene = trimesh.Scene([obj_mesh] + ee_pose + goal_pose)
    if not show:
        if not os.path.exists(f'{cfg.exp_dir}/{cfg.exp_name}/{cfg.scene_file}/pred_pkls'):
            os.mkdir(f'{cfg.exp_dir}/{cfg.exp_name}/{cfg.scene_file}/pred_pkls')
        fname = f'{cfg.exp_dir}/{cfg.exp_name}/{cfg.scene_file}/pred_pkls/pred_{iter}.pkl'
        export = trimesh.exchange.export.export_scene(scene, file_obj=None, file_type='dict')
        pickle.dump(export, open(fname, 'wb'))
    else:
        scene.show()
    # import IPython; IPython.embed()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Directory containing pred_pkls and data.npy", type=str)
    args, _ = parser.parse_known_args()

    with open(f'{args.dir}/../args.yml') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    cfg = edict(cfg)

    dir_items = os.listdir(args.dir)
    txt_files = [x for x in dir_items if '.txt' in x]
    assert len(txt_files) == 1
    traj_idx = int(txt_files[0].replace('.txt', ''))

    rew, info, plan = np.load(f'{args.dir}/data.npy', allow_pickle=True)[()]
    T_bot2ee, T_bot2objfrm, T_objfrm2obj, T_obj2goal, T_obj2ee = info[traj_idx]['transforms']

    visualize_predicted_grasp(traj_idx, cfg, T_obj2ee, T_obj2goal, T_objfrm2obj, show=True, rotate_wrist=True)

