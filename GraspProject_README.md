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
* Install acronym (helpful for debugging)

## Grasp Dataset Setup 
* Download ACRONYM
* Convert acronym meshes so they can be used in pybullet
    * Convert meshes using procedure described in OMG-planner repo https://github.com/liruiw/OMG-Planner#process-new-shapes 
    * Run acronym version of process_shape file (why?)
* Get grasps from acronym dataset that are valid in pybullet `python -m bullet.panda_testgrasp_scene -w -ow -o Book_5e90bf1bb411069c115aef9ae267d6b7 Mug_10f6e09036350e92b3f21f1137c3c347`
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