#!/bin/bash
# conda activate gmanifolds
# cd ~/projects/manifolds/OMG-Planner

# bash bash_scripts/run.sh dbg 1

EXP_NAME="$1"
N_TRIALS="$2"

# for ((i=0;i<$2;i++)) do
#     python -m bullet.panda_scene --method=origOMG_known \
#         --write_video --no-render -o=/data/manifolds/pybullet_eval/$EXP_NAME
# done

# for ((i=0;i<$2;i++)) do
#     python -m bullet.panda_scene --method=compOMG_known \
#         --write_video --no-render -o=/data/manifolds/pybullet_eval/$EXP_NAME
# done

# for ((i=0;i<$2;i++)) do
#     python -m bullet.panda_scene --method=GF_known \
#         --write_video --no-render \
#         --smoothness_base_weight=0.1 \
#         --base_obstacle_weight=0.7 \
#         --base_grasp_weight=12 \
#         --base_step_size=0.2 \
#         --optim_steps=250 \
#         --goal_thresh=0.01 \
#         --dset_root='/data/manifolds/acronym_mini_relabel' \
#         --prefix='sm0.1_ob0.7_gr12_st0.2_os250_th0.01_' \
#         -o /data/manifolds/pybullet_eval/$EXP_NAME
# done

for ((i=0;i<$2;i++)) do
    python -m bullet.panda_scene --method=GF_learned \
        --write_video \
        --smoothness_base_weight=0.1 \
        --base_obstacle_weight=0.7 \
        --base_grasp_weight=12 \
        --base_step_size=0.2 \
        --optim_steps=250 \
        --goal_thresh=0.01 \
        --dset_root='/data/manifolds/acronym_mini_relabel' \
        --ckpt='/data/manifolds/fb_runs/multirun/pq-pq_relabel_shape/2022-05-15_204054/lossl1_lr0.0001/default_default/8_8/checkpoints/epoch=479-step=161749.ckpt' \
        --prefix='sm0.1_ob0.7_gr12_st0.2_os250_th0.01_' \
        -o /data/manifolds/pybullet_eval/$EXP_NAME
done

        # --no-render \