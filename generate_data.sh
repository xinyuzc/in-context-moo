#!/bin/bash -l
#SBATCH --job-name=gp
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-14
#SBATCH --output=log/data/%x_%j.output
#SBATCH --error log/data/%x_%j.err

# environment
module load mamba
source activate tamo
python --version

# ====GP====
CUDA_LAUNCH_BLOCKING=1 python generate_data.py  --config-name=generate_data \
    experiment.mode=train \
    generate.filename="gp_${SLURM_ARRAY_TASK_ID}" \
    generate.x_dim=1 \
    generate.y_dim=1 \
    generate.num_datasets=100000 

# ====GP-based function with global optimum structure====
# CUDA_LAUNCH_BLOCKING=1 python generate_data.py  --config-name=generate_data \
#     experiment.mode=train \
#     data.data_id=opt \
#     generate.sampler_type=opt \
#     generate.filename="gp_${SLURM_ARRAY_TASK_ID}" \
#     generate.x_dim=2 \
#     generate.y_dim=2 \
#     generate.num_datasets=100000 

## ====Test data====
# CUDA_LAUNCH_BLOCKING=1 python generate_data.py  --config-name=generate_data \
#     experiment.mode=test \
#     data.data_id=opt \
#     generate.filename="gp_test" \
#     generate.sampler_type=opt \
#     generate.x_dim=2 \
#     generate.y_dim=2 \
#     generate.num_datasets=1000 
    
