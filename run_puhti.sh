#!/bin/bash
#SBATCH --job-name=alm
#SBATCH --account=project_2003582
#SBATCH --output=alm_out_%A_%a.txt
#SBATCH --error=alm_err_%A_%a.txt
#SBATCH --time=72:00:00
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
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

export PROJID=project_2003582
export PROJAPPL=/projappl/${PROJID}
export SCRATCH=/scratch/${PROJID}

export LC_ALL=en_US.UTF-8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK # set the number of threads based on --cpus-per-task

srun python3 train.py id=$ENV
