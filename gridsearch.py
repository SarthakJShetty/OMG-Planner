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
    for optim_steps in [200, 250, 300]:
        for smooth_weight in [0.1, 0.3]:
            for obstacle_weight in [0.3, 0.5, 0.7]:
                for grasp_weight in [3, 5, 7]:
                    for step_size in [0.1, 0.2, 0.3]:
                        os.system(f"CUDA_VISIBLE_DEVICES={gpu} EGL_GPU={gpu} "
                                  f"python -m bullet.panda_scene "
                                  f"--method=GF_learned "
                                  f"--write_video "
                                  f"--no-render "
                                  f"--smoothness_base_weight={smooth_weight} "
                                  f"--base_obstacle_weight={obstacle_weight} "
                                  f"--base_grasp_weight={grasp_weight} "
                                  f"--base_step_size={step_size} "
                                  f"--optim_steps={optim_steps} "
                                  f"--prefix=sm{smooth_weight}_ob{obstacle_weight}_gr{grasp_weight}_st{step_size}_os{optim_steps}_ "
                                  f"-o=/data/manifolds/pybullet_eval/{args.exp_name}")