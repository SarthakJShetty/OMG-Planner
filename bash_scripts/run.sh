#!/bin/bash
# conda activate gmanifolds
# cd ~/projects/manifolds/OMG-Planner

# bash bash_scripts/run.sh dbg 1

EXP_NAME="$1"
N_TRIALS="$2"

for ((i=0;i<$2;i++)) do
    python -m bullet.panda_scene --method=origOMG_known \
        --write_video \
        --no-render \
        --dset_root='/data/manifolds/acronym_mini_relabel' \
        -o=/data/manifolds/pybullet_eval/$EXP_NAME
done

for ((i=0;i<$2;i++)) do
    python -m bullet.panda_scene --method=compOMG_known \
        --write_video --no-render \
        --dset_root='/data/manifolds/acronym_mini_relabel' \
        -o=/data/manifolds/pybullet_eval/$EXP_NAME
done

for ((i=0;i<$2;i++)) do
    python -m bullet.panda_scene --method=GF_known \
        --write_video --no-render \
        --smoothness_base_weight=0.1 \
        --base_obstacle_weight=0.7 \
        --base_grasp_weight=12 \
        --base_step_size=0.2 \
        --optim_steps=250 \
        --goal_thresh=0.01 \
        --dset_root='/data/manifolds/acronym_mini_relabel' \
        --prefix='sm0.1_ob0.7_gr12_st0.2_os250_th0.01_' \
        -o /data/manifolds/pybullet_eval/$EXP_NAME
done

for ((i=0;i<$2;i++)) do
    python -m bullet.panda_scene --method=GF_learned \
        --write_video \
        --no-render \
        --smoothness_base_weight=0.1 \
        --base_obstacle_weight=0.7 \
        --base_grasp_weight=12 \
        --base_step_size=0.2 \
        --optim_steps=250 \
        --goal_thresh=0.01 \
        --ckpt='/data/manifolds/fb_runs/multirun/pq-pq_mini_relabel/2022-05-16_175103/lossl1_lr0.0001/default_default/4_4/checkpoints/epoch=429-step=145529.ckpt' \
        --dset_root='/data/manifolds/acronym_mini_relabel' \
        --prefix='minerr_sm0.1_ob0.7_gr12_st0.2_os250_th0.01_' \
        -o /data/manifolds/pybullet_eval/$EXP_NAME
done

# for ((i=0;i<$2;i++)) do
#     python -m bullet.panda_scene --method=GF_learned \
#         --write_video \
#         --no-render \
#         --smoothness_base_weight=0.1 \
#         --base_obstacle_weight=0.7 \
#         --base_grasp_weight=12 \
#         --base_step_size=0.2 \
#         --optim_steps=250 \
#         --goal_thresh=0.01 \
#         --ckpt='/data/manifolds/fb_runs/multirun/pq-pq_mini_relabel/2022-05-16_175103/lossl1_lr0.0001/default_default/2_2/checkpoints/epoch=259-step=88211.ckpt' \
#         --dset_root='/data/manifolds/acronym_mini_relabel' \
#         --prefix='minloss_sm0.1_ob0.7_gr12_st0.2_os250_th0.01_' \
#         -o /data/manifolds/pybullet_eval/$EXP_NAME
# done

# for ((i=0;i<$2;i++)) do
#     python -m bullet.panda_scene --method=GF_learned \
#         --write_video \
#         --no-render \
#         --smoothness_base_weight=0.1 \
#         --base_obstacle_weight=0.7 \
#         --base_grasp_weight=12 \
#         --base_step_size=0.2 \
#         --optim_steps=250 \
#         --goal_thresh=0.01 \
#         --ckpt='/data/manifolds/fb_runs/multirun/pq-pq_relabel_shape/2022-05-15_204054/lossl1_lr0.0001/default_default/8_8/checkpoints/epoch=479-step=161749.ckpt' \
#         --dset_root='/data/manifolds/acronym_mini_relabel' \
#         --prefix='shapeminerr_sm0.1_ob0.7_gr12_st0.2_os250_th0.01_' \
#         -o /data/manifolds/pybullet_eval/$EXP_NAME
# done
