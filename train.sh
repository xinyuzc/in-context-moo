#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu-h100-80g,gpu-a100-80g
#SBATCH --time=01:00:00
#SBATCH --output=outputs/train/%x_%j.output
#SBATCH --error=outputs/train/%x_%j.err

module load mamba
source activate tamo
python --version

expid=DX1_DY1
resume=false 

num_total_epochs=10000
num_burnin_epochs=5000

CUDA_LAUNCH_BLOCKING=1 python train.py --config-name=train \
experiment.expid="${expid}" \
experiment.resume=${resume} \
data.x_dim_list=[1] \
data.y_dim_list=[1] \
train.num_total_epochs=${num_total_epochs} \
train.num_burnin_epochs=${num_burnin_epochs}


