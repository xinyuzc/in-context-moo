#!/bin/bash -l
#SBATCH --job-name=DX12345_DY123
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100-80g
#SBATCH --time=96:00:00
#SBATCH --output=outputs/train_online/%x_%j.output
#SBATCH --error=outputs/train_online/%x_%j.err

module load mamba
source activate tamo
python --version

expid=DX12345_DY123_TRAIN_ONLINE
resume=false

num_total_epochs=200000
num_burnin_epochs=190000

x_dim_list="[1,2,3,4,5]"
y_dim_list="[1,2,3]"

max_x_dim=5
batch_size=64

python train_online.py --config-name=train \
experiment.expid="${expid}" \
experiment.resume=${resume} \
data.x_dim_list="${x_dim_list}" \
data.y_dim_list="${y_dim_list}" \
train.num_total_epochs=${num_total_epochs} \
train.num_burnin_epochs=${num_burnin_epochs} \
prediction.batch_size=${batch_size} \
model.max_x_dim=${max_x_dim} \
data.max_x_dim=${max_x_dim}


