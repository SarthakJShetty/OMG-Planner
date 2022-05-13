#!/bin/bash
# conda activate gmanifolds
# cd ~/projects/manifolds/OMG-Planner

# bash bash_scripts/run.sh dbg 1

EXP_NAME="$1"
N_TRIALS="$2"

# for ((i=0;i<=$2;i++)) do
#     python -m bullet.panda_scene --method=origOMG_known --write_video -o=/data/manifolds/pybullet_eval/$EXP_NAME
# done

# for ((i=0;i<=$2;i++)) do
#     python -m bullet.panda_scene --method=compOMG_known --write_video -o=/data/manifolds/pybullet_eval/$EXP_NAME
# done

for ((i=0;i<$2;i++)) do
    python -m bullet.panda_scene --method=GF_learned --write_video -o=/data/manifolds/pybullet_eval/$EXP_NAME
done