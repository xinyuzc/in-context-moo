#!/bin/bash -l
#SBATCH --job-name=DX12_DY123
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g
#SBATCH --time=72:00:00
#SBATCH --output=outputs/train/%x_%j.output
#SBATCH --error=outputs/train/%x_%j.err

module load mamba
source activate tamo
python --version

expid=DX12_DY123
resume=false

num_total_epochs=400000
num_burnin_epochs=392000

x_dim_list="[1,2]"
y_dim_list="[1,2,3]"


python train.py --config-name=train \
experiment.expid="${expid}" \
experiment.resume=${resume} \
data.x_dim_list="${x_dim_list}" \
data.y_dim_list="${y_dim_list}" \
train.num_total_epochs=${num_total_epochs} \
train.num_burnin_epochs=${num_burnin_epochs}


