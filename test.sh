#!/bin/bash -l
#SBATCH --job-name=bc
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
plot_per_n_steps=25

# Model 
max_x_dim=4
max_y_dim=3

# Experiment
task=optimization
override=True
seed=${SLURM_ARRAY_TASK_ID}
CKPT_NAMES=("ckpt.tar")
suffix_segment=null

# Optimization
T=100
regret_type="ratio"
use_grid_sampling=True
use_fixed_query_set=True
d=2048

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

FUNCTIONS=("BraninCurrin")
#  "AckleyRosenbrock" "AckleyRastrigin" "BraninCurrin")

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
            data.function_name="${function_name}" \
            experiment.expid="${expid}" \
            experiment.task="${task}" \
            experiment.override=${override} \
            optimization.T=${T} \
            experiment.seed=${seed} \
            optimization.use_grid_sampling=${use_grid_sampling} \
            optimization.use_fixed_query_set=${use_fixed_query_set} \
            optimization.dim_mask_gen_mode=${dim_mask_gen_mode} \
            optimization.single_obs_x_dim=${single_obs_x_dim} \
            optimization.single_obs_y_dim=${single_obs_y_dim} \
            optimization.read_cache=${opt_read_cache} \
            optimization.write_cache=${opt_write_cache} \
            optimization.regret_type=${regret_type} \
            prediction.read_cache=${pred_read_cache} \
            model.max_x_dim=${max_x_dim} \
            model.max_y_dim=${max_y_dim} \
            log.plot_nc_list="[10,50,100]" \
            log.plot_enabled=${plot_enabled} \
            log.plot_per_n_steps=${plot_per_n_steps} \
            extra.ckpt_name=${ckpt_name} \
            optimization.num_query_points=${d} \
            optimization.cost=${cost} \
            extra.suffix_segment=${suffix_segment} \
            optimization.cost_mode=${cost_mode} \
            optimization.q=${q} \
            optimization.fantasy=${fantasy} \
            data.data_id=${data_id} \
            data.scene=${scene}
    done
done