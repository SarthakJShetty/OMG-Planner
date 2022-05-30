#!/bin/bash
# conda activate gmanifolds
# cd ~/projects/manifolds/OMG-Planner

# bash bash_scripts/run.sh dbg 1

EXP_NAME="$1"
N_TRIALS="$2"

# 1hot min error
for ((i=0;i<$2;i++)) do
    python -m bullet.panda_scene \
    	--method=GF_learned_1hot_minerr --write_video \
        --no-render \
        --eval_type=1obj_float_fixedpose_nograv \
        --smoothness_base_weight=0.1 \
        --base_obstacle_weight=0.7 \
        --base_grasp_weight=12 \
        --base_step_size=0.2 \
        --optim_steps=250 \
        --goal_thresh=0.01 \
        --dset_root='/data/manifolds/acronym_mini_relabel' \
        --ckpt='/data/manifolds/fb_runs/multirun/pq-pq_mini_relabel/2022-05-27_161440/0/default_default/15_15/checkpoints/epoch=675-step=909345.ckpt' \
        -o /data/manifolds/pybullet_eval/$EXP_NAME
done

# 1hot min loss
for ((i=0;i<$2;i++)) do
    python -m bullet.panda_scene \
    	--method=GF_learned_1hot_minloss --write_video \
        --no-render \
        --eval_type=1obj_float_fixedpose_nograv \
        --smoothness_base_weight=0.1 \
        --base_obstacle_weight=0.7 \
        --base_grasp_weight=12 \
        --base_step_size=0.2 \
        --optim_steps=250 \
        --goal_thresh=0.01 \
        --dset_root='/data/manifolds/acronym_mini_relabel' \
        --ckpt='/data/manifolds/fb_runs/multirun/pq-pq_mini_relabel/2022-05-27_161440/0/default_default/4_4/checkpoints/epoch=167-step=226269.ckpt' \
        -o /data/manifolds/pybullet_eval/$EXP_NAME
done

        # --ckpt='/data/manifolds/fb_runs/multirun/pq-pq_mini/2022-05-10_221352/lossl1_lr0.0001/default_default/1_1/checkpoints/epoch=109-step=37605.ckpt' \
    	# --prefix=sm{smooth_weight}_ob{obstacle_weight}_gr{grasp_weight}_st{step_size}_os{optim_steps}_th{goal_thresh}_ \
        # --prefix=sm{smooth_weight}_ob{obstacle_weight}_gr{grasp_weight}_st{step_size}_os{optim_steps}_th{goal_thresh}_ \


# # shape code 
# python -m bullet.panda_scene \
# 	--method=GF_learned --write_video --render \
# 	--smoothness_base_weight=0.1 --base_obstacle_weight=0.7 --base_step_size=0.2 \
# 	--base_grasp_weight=10.0 \
# 	--optim_steps=250 --goal_thresh=0.01 \
# 	--prefix=sm{smooth_weight}_ob{obstacle_weight}_gr{grasp_weight}_st{step_size}_os{optim_steps}_th{goal_thresh}_ \
# 	-o /data/manifolds/pybullet_eval/dbg --pc \
# 	--ckpt /data/manifolds/fb_runs/multirun/pq-pq_relabel_shape100fps/2022-05-21_224607/0/hpc_ckpt_44.ckpt
# python -m bullet.panda_scene \
# 	--method=GF_learned --write_video --render \
# 	--smoothness_base_weight=0.1 --base_obstacle_weight=0.7 --base_step_size=0.2 \
# 	--base_grasp_weight=10.0 \
# 	--optim_steps=250 --goal_thresh=0.01 \
# 	--prefix=sm{smooth_weight}_ob{obstacle_weight}_gr{grasp_weight}_st{step_size}_os{optim_steps}_th{goal_thresh}_ \
# 	-o /data/manifolds/pybullet_eval/dbg

# # shape code 
# python -m bullet.panda_scene \
# 	--method=GF_learned --write_video --render \
# 	--smoothness_base_weight=0.1 --base_obstacle_weight=0.7 --base_step_size=0.2 \
# 	--base_grasp_weight=10.0 \
# 	--optim_steps=250 --goal_thresh=0.01 \
# 	--prefix=sm{smooth_weight}_ob{obstacle_weight}_gr{grasp_weight}_st{step_size}_os{optim_steps}_th{goal_thresh}_ \
# 	-o /data/manifolds/pybullet_eval/dbg --pc \
# 	--ckpt /data/manifolds/fb_runs/multirun/pq-pq_relabel_shape100fps/2022-05-21_224607/0/hpc_ckpt_44.ckpt

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
