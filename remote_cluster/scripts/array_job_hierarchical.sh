#!/bin/bash
#SBATCH --job-name=test_full_analysis
#SBATCH --account=sh30
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4096
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/rrag0004/sh30/users/rragonnet/tb_hierarchical/remote_cluster/logs/%A_%a.out 
#SBATCH --mail-user=romain.ragonnet@monash.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --array=1

export PYTENSOR_FLAGS=compiledir=$HOME/.pytensor/$SLURM_JOB_ID

cd /projects/sh30/users/rragonnet/tb_hierarchical

pixi run python remote_cluster/scripts/massiverun.py $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID