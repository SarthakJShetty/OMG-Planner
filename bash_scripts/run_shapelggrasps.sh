#!/bin/bash
# conda activate gmanifolds
# cd ~/projects/manifolds/OMG-Planner

# bash bash_scripts/run.sh dbg 1

EXP_NAME="$1"
N_TRIALS="$2"

# shape large model with re-input min err
for ((i=0;i<$2;i++)) do
    python -m bullet.panda_scene \
        --method=GF_learned_shapelg_minerr --write_video \
        --no-render \
        --eval_type=1obj_float_fixedpose_nograv \
        --smoothness_base_weight=0.1 \
        --base_obstacle_weight=0.7 \
        --base_grasp_weight=12 \
        --base_step_size=0.2 \
        --optim_steps=250 \
        --goal_thresh=0.01 \
        --dset_root='/data/manifolds/acronym_mini_relabel' \
        --pc \
        --ckpt='/data/manifolds/fb_runs/multirun/pq-pq_relabel_shape100fps_rot_dims/2022-05-24_210811/inlayer[4]_augrotz/default_default/70_70/checkpoints/epoch=349-step=419764.ckpt' \
        -o /data/manifolds/pybullet_eval/$EXP_NAME
done

# shape large model with re-input min loss
for ((i=0;i<$2;i++)) do
    python -m bullet.panda_scene \
        --method=GF_learned_shapelg_minloss --write_video \
        --no-render \
        --eval_type=1obj_float_fixedpose_nograv \
        --smoothness_base_weight=0.1 \
        --base_obstacle_weight=0.7 \
        --base_grasp_weight=12 \
        --base_step_size=0.2 \
        --optim_steps=250 \
        --goal_thresh=0.01 \
        --dset_root='/data/manifolds/acronym_mini_relabel' \
        --pc \
        --ckpt='/data/manifolds/fb_runs/multirun/pq-pq_relabel_shape100fps_rot_dims/2022-05-24_210811/inlayer[4]_augrotz/default_default/68_68/checkpoints/epoch=339-step=407104.ckpt' \
        -o /data/manifolds/pybullet_eval/$EXP_NAME
done
