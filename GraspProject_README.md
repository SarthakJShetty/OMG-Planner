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

## Evaluation
* Run grasping using 100-scene test set from OMG-Planner in PyBullet
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

### Training Our Method
* Train Shape Embedding Network
    * Generate 30 partial point cloud views for each object
        * `python generate_partial_pc_dataset.py --n_samples=30 --mesh_root=/checkpoint/thomasweng/acronym --objects --save_dir=/data/shape_code/data/acronym`
    * Run training

* Train Implicit Grasp Distance Network
    * Generate grasp dataset with positive, negative, and free grasps
        * (Book only, unknown distance func) `python data_gen/process_acronym.py --grasp_paths /data/manifolds/acronym/grasps/Book_5e90bf1bb411069c115aef9ae267d6b7_0.0268818133810836.h5 --symmetry --n_free 10000  --bullet_grasps`
        * (Log map, other distances)
    * Run training

### Training Contact-GraspNet
---
* Retrain contact_graspnet
    * (Book only with bullet labels)
        `CUDA_VISIBLE_DEVICES=1 python contact_graspnet/train.py --ckpt_dir checkpoints/bookonly_bullet --data_path /checkpoint/thomasweng/acronym_Bookonly_bullet/ --arg_configs DATA.train_and_test:1`
    * All objects
