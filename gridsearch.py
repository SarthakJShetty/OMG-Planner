import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", help="which weights to use for our method", default='/data/manifolds/fb_runs/multirun/pq-pq_mini/2022-05-10_221352/lossl1_lr0.0001/default_default/1_1/checkpoints/epoch=109-step=37605.ckpt')
    parser.add_argument("--exp_name", help="directory name to save experiment to", type=str, default="dbg")

    # # cfg-specific command line args
    # parser.add_argument("--smoothness_base_weight", type=float)
    # parser.add_argument("--base_obstacle_weight", type=float)
    # parser.add_argument("--base_grasp_weight", type=float)
    # parser.add_argument("--base_step_size", type=float)
    # parser.add_argument("--optim_steps", type=float)

    args = parser.parse_args()

    gpu = 1
    for optim_steps in [250]:
        for smooth_weight in [0.05, 0.1]:
            for obstacle_weight in [0.7, 2.0, 5.0]:
                for grasp_weight in [10, 15, 20]:
                    for step_size in [0.1, 0.2]:
                        for goal_thresh in [0.01]:
                            os.system(f"CUDA_VISIBLE_DEVICES={gpu} EGL_GPU={gpu} "
                                      f"python -m bullet.panda_scene "
                                      f"--method=GF_learned_shape_minerr  "
                                      f"--write_video "
                                      f"--no-render "
                                      f"--eval_type=1obj_float_fixedpose_nograv "
                                      f"--smoothness_base_weight={smooth_weight} "
                                      f"--base_obstacle_weight={obstacle_weight} "
                                      f"--base_grasp_weight={grasp_weight} "
                                      f"--base_step_size={step_size} "
                                      f"--optim_steps={optim_steps} "
                                      f"--goal_thresh={goal_thresh} "
                                      f"--dset_root='/data/manifolds/acronym_mini_relabel' "
                                      f"--prefix=sm{smooth_weight}_ob{obstacle_weight}_gr{grasp_weight}_st{step_size}_os{optim_steps}_th{goal_thresh}_ "
                                      f"--pc "
                                      f"--ckpt={args.ckpt} "
                                      f"-o=/data/manifolds/pybullet_eval/{args.exp_name}")
                            # import sys 
                            # sys.exit(1)

                            # os.system(f"CUDA_VISIBLE_DEVICES={gpu} EGL_GPU={gpu} "
                            #           f"python -m bullet.panda_scene "
                            #           f"--method=GF_known "
                            #           f"--write_video "
                            #           f"--no-render "
                            #           f"--smoothness_base_weight={smooth_weight} "
                            #           f"--base_obstacle_weight={obstacle_weight} "
                            #           f"--base_grasp_weight={grasp_weight} "
                            #           f"--base_step_size={step_size} "
                            #           f"--optim_steps={optim_steps} "
                            #           f"--goal_thresh={goal_thresh} "
                            #           f"--prefix=sm{smooth_weight}_ob{obstacle_weight}_gr{grasp_weight}_st{step_size}_os{optim_steps}_th{goal_thresh} "
                            #           f"-o=/data/manifolds/pybullet_eval/{args.exp_name}_known")

    # python -m bullet.panda_scene \
    #     --method=GF_learned_shape_minerr --write_video \
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
    #     --ckpt='/data/manifolds/fb_runs/multirun/pq-pq_relabel_shape100fps/2022-05-22_143051/0/default_default/94_94/checkpoints/epoch=465-step=555572.ckpt' \
    #     -o /data/manifolds/pybullet_eval/$EXP_NAME