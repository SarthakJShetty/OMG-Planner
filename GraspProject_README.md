# Grasp Project README

https://docs.google.com/document/d/1x1PZMdiyG4Ub5JhEP6zYPAnVxBQYJYjMYePLq-Qki9U/edit#

## Installation
* Install OMG-Planner (tweng-dev branch), follow readme
* Install contact-graspnet 
    * sip 4.19.3 from omg planner vs. 4.19.8 from contact-graspnet: run `conda upgrade sip` if `sip -V` gives 4.19.3 or you get an error
    * Navigate to contact_graspnet directory
    * `conda env update --file contact_graspnet_env_tf25.yml --name gm_pipeline`
    * `pip install -e .`
* Install manifold-grasping-torch
    * `pip install -e .`
    * to add: pytorch_lightning==1.4.1
* Install [modified pytransform3d](https://github.com/thomasweng15/pytransform3d) to get around approximation error for logmap
    * `pip install -e .`
* Install acronym (helpful for debugging)
    * `pip install -e .`
* install hydra-core, trimesh, shapely, torch_geometric, liegroups

## Evaluation
* Run grasping using 100-scene test set from OMG-Planner in PyBullet
    * Fixed
        `python -m bullet.panda_scene --method=knowngrasps_Fixed --exp_dir=/data/manifolds/pybullet_eval/knowngrasps_Fixed_dbg/ --acronym_dir=/checkpoint/thomasweng/acronym --experiment --write_video --save_all_trajectories`

    * Contact-Graspnet: 
        `python -m bullet.panda_scene -gi contact_graspnet -gs Fixed`
    * Implicit Grasp Network:
* Run book object only 
    * Contact-Graspnet:
        `python -m bullet.panda_scene -gi contact_graspnet -gs Fixed --scenes acronym_book --output_dir /checkpoint/thomasweng/grasp_manifolds/pybullet_eval/contact_graspnet/ -r`
    * Implicit Grasp Network:
        `python -m bullet.panda_scene -gi ours_outputpose -gs Fixed -w --scenes acronym_book --output_dir /checkpoint/thomasweng/grasp_manifolds/pybullet_eval/ours_outputpose/ -r`
    * Our method with ground-truth grasps:
        `python -m bullet.panda_scene -gi ours_knowngrasps -gs Fixed -w --scenes acronym_book --output_dir /checkpoint/thomasweng/grasp_manifolds/pybullet_eval/ours_knowngrasps/ -r`
* Others (run level set visualization, etc.)

## Training

### Grasp Dataset Setup 
* Download ACRONYM dataset
* Convert acronym meshes so they can be used in pybullet
    * Convert meshes using procedure described in OMG-planner repo https://github.com/liruiw/OMG-Planner#process-new-shapes 
        * Run acronym version of process_shape file (all objects)
           `python -m real_world.process_shape_acronym -a --save_root=/checkpoint/thomasweng/acronym/meshes_omg --mesh_root=/checkpoint/thomasweng/acronym/meshes`
        * Might need to set `ulimit -n 4096` if you get an OSError for too many open files. 
* Get grasps from acronym dataset that are valid in pybullet
    * Book object
        `python -m bullet.get_bullet_labels --mesh_root /checkpoint/thomasweng/acronym_Bookonly_bullet --grasp_root /checkpoint/thomasweng/acronym_Bookonly_bullet/grasps -o Book_5e90bf1bb411069c115aef9ae267d6b7 --out_dir /checkpoint/thomasweng/acronym_Bookonly_bullet/grasps/bullet --overwrite`
    * All objects
        `python -m bullet.get_bullet_labels --mesh_root /checkpoint/thomasweng/acronym --grasp_root /checkpoint/thomasweng/acronym/grasps --out_dir /checkpoint/thomasweng/acronym/grasps_bullet_dbg --overwrite`
        1s shaking `python -m bullet.get_bullet_labels_mp --mesh_root=/checkpoint/thomasweng/acronym --grasp_root=/checkpoint/thomasweng/acronym/grasps --out_dir=/checkpoint/thomasweng/acronym/bullet_grasps --workers=64`
        0.5s shaking `python -m bullet.get_bullet_labels_mp --mesh_root=/checkpoint/thomasweng/acronym --grasp_root=/checkpoint/thomasweng/acronym/grasps --out_dir=/checkpoint/thomasweng/acronym/bullet_grasps --workers=64 --rot_shake_duration=500 --lin_shake_duration=500`

### Training Our Method
* Shape Embedding Network
    * ndf_robot (trained on four views, not a partial point cloud view)
    * point completion network
        * Had to install pyarrow separately
        * for pc_distance make, had to use std=c++14 and create symlink for libtensorflow_framework.so https://github.com/wentaoyuan/pcn/issues/51#issuecomment-652761637
        * pcn was built with tensorflow 1.12 which 3090 does not support. May need to port it to TF 2.5, 1 vs 2 are very different
    <!-- * Generate 30 partial point cloud views for each object (in manifold_grasping_torch)
        `python generate_partial_pc_dataset.py --n_samples=30 --mesh_root=/checkpoint/thomasweng/acronym --objects --save_dir=/data/shape_code/data/acronym` -->
    <!-- * Run training
        `python train.py --multirun hydra/launcher=submitit_slurm hydra.launcher.gpus_per_node=1 hydra.sweep.subdir='_lr${netconf.lr}' datapath=/checkpoint/thomasweng/shape_code/data/acronym/dataset.hdf5 batchsz=32 nworkers=64 netconf.lr=0.005,0.001,0.0005 epochs=50 experiment=allpcs_nobn`
        `python train.py experiment=all_pcs_nobn_aug hydra.run.dir=\'/checkpoint/thomasweng/shape_code/runs/outputs/\${experiment}/\${now:%Y-%m-%d_%H%M%S}\' datapath=/data/shape_code/data/acronym/dataset.hdf5 batchsz=32 nworkers=64 netconf.lr=0.001 epochs=50 val_check_interval=100 netconf.aug_jitter=True netconf.aug_rotate=True netconf.aug_scale=True` -->

* Train Implicit Grasp Distance Network
    * Generate grasp dataset with positive, negative, and free grasps
        * (Book only, unknown distance func) `python data_gen/process_acronym.py --grasp_paths /data/manifolds/acronym/grasps/Book_5e90bf1bb411069c115aef9ae267d6b7_0.0268818133810836.h5 --symmetry --n_free 10000  --bullet_grasps`
        * (Log map, other distances)
    * Run training

    * Evaluate on acronym_book
        `python -m omg.trimesh_viz --dir=/data/manifolds/pybullet_eval/saved_results/outputpose_improvegrasp/predgrasps_grad_goalgrad_bullet_lowestcosttraj_graspsched1.05/2022-03-04-17-19-41_implicitgrasps_outputposegrad/acronym_book_1`

### Testing out VN-OccNets
    * Generate partial point cloud views for each object (in manifold_grasping_torch)
        ```
        python multiple_views_acronym.py --n_samples=1 --mesh_root=/checkpoint/thomasweng/acronym --objects 
        ```

<!-- --save_dir=/data/shape_code/data/acronym -->

### Training Contact-GraspNet
---
* Retrain contact_graspnet
    * Collect scene_contacts and mesh data
        <!-- `contact_graspnet/tools/make_meshes.py` -->
        <!-- `contact_graspnet/tools/prune_grasps.py` -->
        `python tools/create_contact_infos.py /checkpoint/thomasweng/acronym grasps/qualities/bullet/object_in_gripper`
        `python tools/create_table_top_scenes.py /path/to/acronym`
    * (Book only with bullet labels)
        `CUDA_VISIBLE_DEVICES=1 python contact_graspnet/train.py --ckpt_dir checkpoints/bookonly_bullet --data_path /checkpoint/thomasweng/acronym_Bookonly_bullet/ --arg_configs DATA.train_and_test:1`
    * All objects
        `CUDA_VISIBLE_DEVICES=1 python contact_graspnet/train.py --ckpt_dir checkpoints/allpc_bullet --data_path /checkpoint/thomasweng/acronym/`
