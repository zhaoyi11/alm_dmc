#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --constraint='volta'

#SBATCH --output=/scratch/work/%u/pets-pytorch/triton_log/%A_%a.%j.out
#SBATCH --array=0-7

case $SLURM_ARRAY_TASK_ID in
    0) ENV="acrobot-swingup";;
    1) ENV="cheetah-run" ;;
    2) ENV="fish-swim";;
    3) ENV="dog-run" ;;
    4) ENV="quadruped-walk" ;;
    5) ENV="walker-walk" ;;
    6) ENV="humanoid-walk" ;;
    7) ENV="dog-walk" ;;
esac

source /home/zhaoy13/.bashrc

conda activate minlps

export MUJOCO_GL="egl"
export LC_ALL=en_US.UTF-8

srun python3 train.py id=$ENV 
