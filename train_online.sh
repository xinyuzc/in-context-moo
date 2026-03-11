#!/bin/bash -l
#SBATCH --job-name=DX12_DY123
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h200-141g-ellis,gpu-h200-141g-short,gpu-b300-288g-ellis,gpu-b300-288g-short
#SBATCH --time=72:00:00
#SBATCH --output=outputs/train_online/%x_%j.output
#SBATCH --error=outputs/train_online/%x_%j.err

module load mamba
source activate tamo
python --version

expid=DX12_DY123_260311_TRAIN_ONLINE
resume=false

num_total_epochs=200000
num_burnin_epochs=190000

x_dim_list="[1,2]"
y_dim_list="[1,2,3]"


python train_online.py --config-name=train \
experiment.expid="${expid}" \
experiment.resume=${resume} \
data.x_dim_list="${x_dim_list}" \
data.y_dim_list="${y_dim_list}" \
train.num_total_epochs=${num_total_epochs} \
train.num_burnin_epochs=${num_burnin_epochs} 


