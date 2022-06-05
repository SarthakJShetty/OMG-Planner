import os
import argparse
from omegaconf import OmegaConf
import multiprocessing as m


def run_trials(template):
    os.system(template)


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", help="experiment folder name", type=str)
parser.add_argument("--trials", help="number of trials", type=int, default=5)
parser.add_argument("--render", help="render", action='store_true')
parser.add_argument("--workers", help="number of workers", type=int, default=4)
parser.add_argument("--data_root", help="root of data dir", default='/data/manifolds')
args = parser.parse_args()

if not os.path.exists(f"{args.data_root}/pybullet_eval/{args.exp_name}"):
    os.mkdir(f"{args.data_root}/pybullet_eval/{args.exp_name}")

parent_dir = os.path.dirname(os.path.abspath(__file__))
exp_cfg = OmegaConf.load(f'{parent_dir}/eval_cfg.yaml')

# parameters for multiprocessing
trial_templates = []

for var_cfg in exp_cfg.variants:
    render_flag = "--render" if args.render else "--no-render"
    template = (
        f"python -m bullet.panda_scene "
        f"--write_video "
        f"{render_flag} "
        f"--eval_type=1obj_float_fixedpose_nograv "
        f"--run_scenes "
        f"--dset_root='{args.data_root}/acronym_mini_relabel' "
        f"-o {args.data_root}/pybullet_eval/{args.exp_name} "
    )

    if 'OMG' in var_cfg.method:
        pass # no additional params
    elif 'GF' in var_cfg.method:
        template += (
            f"--smoothness_base_weight=0.1 "
            f"--base_obstacle_weight=0.7 "
            f"--base_grasp_weight=12 "
            f"--base_step_size=0.2 "
            f"--optim_steps=250 "
            f"--goal_thresh=0.01 "
            f"--use_min_cost_traj={var_cfg.use_min_cost_traj} "
            f"--ckpt={var_cfg.ckpt} "
        )
        if var_cfg.use_pc:
            template += '--pc '
        if var_cfg.use_initial_ik:
            template += '--use_initial_ik'

    # check if there are existing dirs for this method for this experiment
    # only run up to max n_trials accounting for existing trials
    trials = os.listdir(f"{args.data_root}/pybullet_eval/{args.exp_name}")
    var_trials = len([x for x in trials if var_cfg.method in x])
    for trial_idx in range(var_trials, args.trials):
        template += f"--method={var_cfg.method}_trial{trial_idx} "
        trial_templates.append(template)

with m.Pool(args.workers) as pool:
    pool.map(run_trials, trial_templates)



# EXP_NAME="$1"
# N_TRIALS="$2"

# # # 1hot min error
# # for ((i=0;i<$2;i++)) do
#     python -m bullet.panda_scene \
#     	--method=GF_learned_1hot_minerr --write_video \
#         --no-render \
#         --eval_type=1obj_float_fixedpose_nograv \
#         --smoothness_base_weight=0.1 \
#         --base_obstacle_weight=0.7 \
#         --base_grasp_weight=12 \
#         --base_step_size=0.2 \
#         --optim_steps=250 \
#         --goal_thresh=0.01 \
#         --run_scenes \
#         --dset_root='{args.data_root}/acronym_mini_relabel' \
#         --ckpt='{args.data_root}/fb_runs/multirun/pq-pq_mini_relabel/2022-05-27_161440/0/default_default/15_15/checkpoints/epoch=675-step=909345.ckpt' \
#         -o {args.data_root}/pybullet_eval/$EXP_NAME
# # done

# # 1hot min error
# for ((i=0;i<$2;i++)) do
#     python -m bullet.panda_scene \
#     	--method=GF_learned_1hot_minerr --write_video \
#         --no-render \
#         --eval_type=1obj_float_fixedpose_nograv \
#         --smoothness_base_weight=0.1 \
#         --base_obstacle_weight=0.7 \
#         --base_grasp_weight=12 \
#         --base_step_size=0.2 \
#         --optim_steps=250 \
#         --goal_thresh=0.01 \
#         --run_scenes \
#         --dset_root='{args.data_root}/acronym_mini_relabel' \
#         --ckpt='{args.data_root}/fb_runs/multirun/pq-pq_mini_relabel/2022-05-27_161440/0/default_default/15_15/checkpoints/epoch=675-step=909345.ckpt' \
#         --use_min_cost_traj=1 \
#         -o {args.data_root}/pybullet_eval/${EXP_NAME}_mincosttraj
# done

# # 1hot min loss
# for ((i=0;i<$2;i++)) do
#     python -m bullet.panda_scene \
#     	--method=GF_learned_1hot_minloss --write_video \
#         --no-render \
#         --eval_type=1obj_float_fixedpose_nograv \
#         --smoothness_base_weight=0.1 \
#         --base_obstacle_weight=0.7 \
#         --base_grasp_weight=12 \
#         --base_step_size=0.2 \
#         --optim_steps=250 \
#         --goal_thresh=0.01 \
#         --run_scenes \
#         --dset_root='{args.data_root}/acronym_mini_relabel' \
#         --ckpt='{args.data_root}/fb_runs/multirun/pq-pq_mini_relabel/2022-05-27_161440/0/default_default/4_4/checkpoints/epoch=167-step=226269.ckpt' \
#         -o {args.data_root}/pybullet_eval/$EXP_NAME
# done

# # 1hot min loss
# for ((i=0;i<$2;i++)) do
#     python -m bullet.panda_scene \
#     	--method=GF_learned_1hot_minloss --write_video \
#         --no-render \
#         --eval_type=1obj_float_fixedpose_nograv \
#         --smoothness_base_weight=0.1 \
#         --base_obstacle_weight=0.7 \
#         --base_grasp_weight=12 \
#         --base_step_size=0.2 \
#         --optim_steps=250 \
#         --goal_thresh=0.01 \
#         --run_scenes \
#         --dset_root='{args.data_root}/acronym_mini_relabel' \
#         --ckpt='{args.data_root}/fb_runs/multirun/pq-pq_mini_relabel/2022-05-27_161440/0/default_default/4_4/checkpoints/epoch=167-step=226269.ckpt' \
#         --use_min_cost_traj=1 \
#         -o {args.data_root}/pybullet_eval/${EXP_NAME}_mincosttraj
# done
#         # --ckpt='{args.data_root}/fb_runs/multirun/pq-pq_mini/2022-05-10_221352/lossl1_lr0.0001/default_default/1_1/checkpoints/epoch=109-step=37605.ckpt' \
#     	# --prefix=sm{smooth_weight}_ob{obstacle_weight}_gr{grasp_weight}_st{step_size}_os{optim_steps}_th{goal_thresh}_ \
#         # --prefix=sm{smooth_weight}_ob{obstacle_weight}_gr{grasp_weight}_st{step_size}_os{optim_steps}_th{goal_thresh}_ \


# # # shape code 
# # python -m bullet.panda_scene \
# # 	--method=GF_learned --write_video --render \
# # 	--smoothness_base_weight=0.1 --base_obstacle_weight=0.7 --base_step_size=0.2 \
# # 	--base_grasp_weight=10.0 \
# # 	--optim_steps=250 --goal_thresh=0.01 \
# # 	--prefix=sm{smooth_weight}_ob{obstacle_weight}_gr{grasp_weight}_st{step_size}_os{optim_steps}_th{goal_thresh}_ \
# # 	-o {args.data_root}/pybullet_eval/dbg --pc \
# # 	--ckpt {args.data_root}/fb_runs/multirun/pq-pq_relabel_shape100fps/2022-05-21_224607/0/hpc_ckpt_44.ckpt
# # python -m bullet.panda_scene \
# # 	--method=GF_learned --write_video --render \
# # 	--smoothness_base_weight=0.1 --base_obstacle_weight=0.7 --base_step_size=0.2 \
# # 	--base_grasp_weight=10.0 \
# # 	--optim_steps=250 --goal_thresh=0.01 \
# # 	--prefix=sm{smooth_weight}_ob{obstacle_weight}_gr{grasp_weight}_st{step_size}_os{optim_steps}_th{goal_thresh}_ \
# # 	-o {args.data_root}/pybullet_eval/dbg

# # # shape code 
# # python -m bullet.panda_scene \
# # 	--method=GF_learned --write_video --render \
# # 	--smoothness_base_weight=0.1 --base_obstacle_weight=0.7 --base_step_size=0.2 \
# # 	--base_grasp_weight=10.0 \
# # 	--optim_steps=250 --goal_thresh=0.01 \
# # 	--prefix=sm{smooth_weight}_ob{obstacle_weight}_gr{grasp_weight}_st{step_size}_os{optim_steps}_th{goal_thresh}_ \
# # 	-o {args.data_root}/pybullet_eval/dbg --pc \
# # 	--ckpt {args.data_root}/fb_runs/multirun/pq-pq_relabel_shape100fps/2022-05-21_224607/0/hpc_ckpt_44.ckpt

# # for ((i=0;i<$2;i++)) do
# #     python -m bullet.panda_scene --method=GF_learned \
# #         --write_video \
# #         --no-render \
# #         --smoothness_base_weight=0.1 \
# #         --base_obstacle_weight=0.7 \
# #         --base_grasp_weight=12 \
# #         --base_step_size=0.2 \
# #         --optim_steps=250 \
# #         --goal_thresh=0.01 \
# #         --ckpt='{args.data_root}/fb_runs/multirun/pq-pq_mini_relabel/2022-05-16_175103/lossl1_lr0.0001/default_default/2_2/checkpoints/epoch=259-step=88211.ckpt' \
# #         --dset_root='{args.data_root}/acronym_mini_relabel' \
# #         --prefix='minloss_sm0.1_ob0.7_gr12_st0.2_os250_th0.01_' \
# #         -o {args.data_root}/pybullet_eval/$EXP_NAME
# # done

# # for ((i=0;i<$2;i++)) do
# #     python -m bullet.panda_scene --method=GF_learned \
# #         --write_video \
# #         --no-render \
# #         --smoothness_base_weight=0.1 \
# #         --base_obstacle_weight=0.7 \
# #         --base_grasp_weight=12 \
# #         --base_step_size=0.2 \
# #         --optim_steps=250 \
# #         --goal_thresh=0.01 \
# #         --ckpt='{args.data_root}/fb_runs/multirun/pq-pq_relabel_shape/2022-05-15_204054/lossl1_lr0.0001/default_default/8_8/checkpoints/epoch=479-step=161749.ckpt' \
# #         --dset_root='{args.data_root}/acronym_mini_relabel' \
# #         --prefix='shapeminerr_sm0.1_ob0.7_gr12_st0.2_os250_th0.01_' \
# #         -o {args.data_root}/pybullet_eval/$EXP_NAME
# # done
