python -m bullet.panda_scene \
    --method=GF_learned_singleshape_minerr --write_video \
    --no-render \
    --eval_type=1obj_float_fixedpose_nograv \
    --smoothness_base_weight=0.1 \
    --base_obstacle_weight=0.7 \
    --base_grasp_weight=12 \
    --base_step_size=0.2 \
    --optim_steps=250 \
    --goal_thresh=0.01 \
    --dset_root='/data/manifolds/acronym_mini_relabel' \
    --ckpt='/data/manifolds/fb_runs/multirun/pq-pq_relabel_shape/2022-05-15_204054/lossl1_lr0.0001/default_default/8_8/checkpoints/epoch=479-step=161749.ckpt' \
    -o /data/manifolds/pybullet_eval/$EXP_NAME

# python -m bullet.panda_scene \
#     --method=GF_learned_shapelg_minerr --write_video \
#     --render \
#     --eval_type=1obj_float_fixedpose_nograv \
#     --smoothness_base_weight=0.1 \
#     --base_obstacle_weight=0.7 \
#     --base_grasp_weight=12 \
#     --base_step_size=0.2 \
#     --optim_steps=250 \
#     --goal_thresh=0.01 \
#     --dset_root='/data/manifolds/acronym_mini_relabel' \
#     --pc \
#     --ckpt='/data/manifolds/fb_runs/multirun/pq-pq_relabel_shape100fps/2022-05-22_143051/0/default_default/51_51/checkpoints/epoch=253-step=306425.ckpt' \
#     -o /data/manifolds/pybullet_eval/$EXP_NAME

# python -m bullet.panda_scene \
#     --method=GF_known \
#     --eval_type=1obj_float_fixedpose_nograv \
#     --write_video --render \
#     --smoothness_base_weight=0.1 \
#     --base_obstacle_weight=0.7 \
#     --base_grasp_weight=12 \
#     --base_step_size=0.2 \
#     --optim_steps=250 \
#     --goal_thresh=0.01 \
#     --dset_root='/data/manifolds/acronym_mini_relabel' \
#     --prefix='sm0.1_ob0.7_gr12_st0.2_os250_th0.01_' \
#     -o /data/manifolds/pybullet_eval/dbg