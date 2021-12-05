Grasp Project README
===

https://docs.google.com/document/d/1x1PZMdiyG4Ub5JhEP6zYPAnVxBQYJYjMYePLq-Qki9U/edit#

Setup 
---

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

## Grasp Dataset Setup 
* Download ACRONYM
* Convert acronym meshes so they can be used in pybullet
    * Convert meshes using procedure described in OMG-planner repo https://github.com/liruiw/OMG-Planner#process-new-shapes 
        * Run acronym version of process_shape file (all objects)
           `python -m real_world.process_shape_acronym -a --save_root=/checkpoint/thomasweng/acronym/meshes_omg --mesh_root=/checkpoint/thomasweng/acronym/meshes`
* Get grasps from acronym dataset that are valid in pybullet
    * Book object
        `python -m bullet.get_bullet_labels --mesh_root /checkpoint/thomasweng/acronym_Bookonly_bullet --grasp_root /checkpoint/thomasweng/acronym_Bookonly_bullet/grasps -o Book_5e90bf1bb411069c115aef9ae267d6b7 --out_dir /checkpoint/thomasweng/acronym_Bookonly_bullet/grasps/bullet --overwrite`
    * All objects
        `python -m bullet.get_bullet_labels --mesh_root /checkpoint/thomasweng/acronym --grasp_root /checkpoint/thomasweng/acronym/grasps --out_dir /checkpoint/thomasweng/acronym/grasps_bullet_dbg --overwrite`

* Generate grasp dataset with positive, negative, and free grasps
* Generate partial point cloud views for each object

Training Implicit Grasp Network
---

Evaluation
---
* Run grasping using 100-scene test set from OMG-Planner in PyBullet
    * Contact-Graspnet: 
        `CUDA_VISIBLE_DEVICES=0 python -m bullet.panda_scene -gi contact_graspnet -gs Fixed`
    * Implicit Grasp Network:
* Others (run level set visualization, etc.)