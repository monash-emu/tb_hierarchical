#!/bin/bash
#SBATCH --job-name=multi_config
#SBATCH --account=sh30
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4096
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/rrag0004/sh30/users/rragonnet/tb_hierarchical/remote_cluster/logs/%A_%a.out 

# To receive an email when job completes or fails
#SBATCH --mail-user=romain.ragonnet@monash.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --array=1-2

export PYTENSOR_FLAGS=compiledir=$HOME/.pytensor/$SLURM_JOB_ID

cd /projects/sh30/users/rragonnet/tb_hierarchical

pixi run python remote_cluster/scripts/massiverun.py $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID