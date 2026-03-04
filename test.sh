#!/bin/bash -l
#SBATCH --job-name=DX12_DY123
#SBATCH --mem=2G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --array=0-29
#SBATCH --output=outputs/test/%x_%j.output
#SBATCH --error=outputs/test/%x_%j.err

module load mamba
source activate tamo
python --version

## ============================================= ##
##              Default Settings                 ##
## ============================================= ##

# Logging
plot_enabled=True

# Experiment
task=optimization
override=True
CKPT_NAMES=("ckpt.tar")
suffix_segment=null

# Optimization
T=100
regret_type="ratio"
num_query_points=2048

# Cache
opt_read_cache=True
opt_write_cache=True
pred_read_cache=True

# Decoupled / Cost mode / fantasy
cost=1.0
cost_mode=False
dim_mask_gen_mode=full
single_obs_x_dim=null
single_obs_y_dim=null
fantasy=False
q=1

# Data
scene=null
data_id=null

## ============================================= ##
##           Experiment Configurations           ##
## ============================================= ##

FUNCTIONS=("dx2_dy2" "AckleyRosenbrock" "AckleyRastrigin" "BraninCurrin")

# Set expid to the TAMO experiment you want to evaluate
expid=DX12_DY123

# --- HPO 3DGS ---
# scene=ship  # "lego", "materials", "mic", "ship"
# FUNCTIONS=("NERF_synthetic_fnum_3")
# max_x_dim=5
# T=30
# suffix_segment=${scene}

# --- Single-objective functions ---
# regret_type="simple"
# FUNCTIONS=("Ackley" "Rastrigin" "Forrester" "Branin" "EggHolder" "dx2_dy1")

# --- Batch size with fantasy ---
# fantasy=True
# q=10
# suffix_segment=batch_q10
# opt_read_cache=False
# opt_write_cache=False
# pred_read_cache=False

# --- Decoupled with cost ---
# suffix_segment="decoupled"
# cost_mode=True
# dim_mask_gen_mode=alternate

## ============================================= ##
##                   Run Test                    ##
## ============================================= ##


for ckpt_name in "${CKPT_NAMES[@]}"; do
    for function_name in "${FUNCTIONS[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python test.py --config-name=test \
            experiment.seed=${SLURM_ARRAY_TASK_ID} \
            data.function_name="${function_name}" \
            data.data_id=${data_id} \
            data.scene=${scene} \
            experiment.expid="${expid}" \
            experiment.task="${task}" \
            experiment.override=${override} \
            optimization.T=${T} \
            optimization.dim_mask_gen_mode=${dim_mask_gen_mode} \
            optimization.single_obs_x_dim=${single_obs_x_dim} \
            optimization.single_obs_y_dim=${single_obs_y_dim} \
            optimization.read_cache=${opt_read_cache} \
            optimization.write_cache=${opt_write_cache} \
            optimization.regret_type=${regret_type} \
            optimization.num_query_points=${num_query_points} \
            optimization.cost=${cost} \
            optimization.cost_mode=${cost_mode} \
            optimization.q=${q} \
            optimization.fantasy=${fantasy} \
            prediction.read_cache=${pred_read_cache} \
            log.plot_enabled=${plot_enabled} \
            extra.ckpt_name=${ckpt_name} \
            extra.suffix_segment=${suffix_segment} 
    done
done