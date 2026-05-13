#!/bin/bash -l
#SBATCH --job-name=DX12345_DY123_Q256
#SBATCH --mem=2G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-4

#SBATCH --output=outputs/test/%x/%j.output
#SBATCH --error=outputs/test/%x/%j.err



module load mamba
source activate tamo
python --version

## ============================================= ##
##              Default Settings                 ##
## ============================================= ##

# Logging
plot_enabled=True

# Experiment
CKPT_NAMES=("ckpt.tar")
suffix_segment=null

# Optimization
regret_type="ratio"
num_query_points=2048 
use_fixed_query_set=true
use_logit_mask=true 

# Reweighted query set
sampling_mode=full
num_reweighted_samples=2048

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

# functions to test 
FUNCTIONS=("BraninCurrin" "AckleyRastrigin" "AckleyRosenbrock" "LaserPlasma")

## ============================================= ##
##           Experiment Configurations           ##
## ============================================= ##

max_x_dim=5
expid=DX12345_DY123

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
# fantasy=False
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

NUM=6  # number of seeds per job

for ckpt_name in "${CKPT_NAMES[@]}"; do
    for function_name in "${FUNCTIONS[@]}"; do
        for i in $(seq 0 $((NUM - 1))); do
            seed=$((SLURM_ARRAY_TASK_ID * NUM + i))
            CUDA_LAUNCH_BLOCKING=1 python test.py --config-name=test \
            experiment.seed=${seed} \
            model.max_x_dim=${max_x_dim} \
            data.max_x_dim=${max_x_dim} \
            data.function_name="${function_name}" \
            data.data_id=${data_id} \
            data.scene=${scene} \
            experiment.expid="${expid}" \
            optimization.T=${T} \
            optimization.dim_mask_gen_mode=${dim_mask_gen_mode} \
            optimization.single_obs_x_dim=${single_obs_x_dim} \
            optimization.single_obs_y_dim=${single_obs_y_dim} \
            optimization.read_cache=${opt_read_cache} \
            optimization.write_cache=${opt_write_cache} \
            optimization.regret_type=${regret_type} \
            optimization.num_query_points=${num_query_points} \
            optimization.use_fixed_query_set=${use_fixed_query_set} \
            optimization.use_logit_mask=${use_logit_mask} \
            optimization.cost=${cost} \
            optimization.cost_mode=${cost_mode} \
            optimization.q=${q} \
            optimization.fantasy=${fantasy} \
            optimization.sampling_mode=${sampling_mode} \
            optimization.num_reweighted_samples=${num_reweighted_samples} \
            prediction.read_cache=${pred_read_cache} \
            log.plot_enabled=${plot_enabled} \
            extra.ckpt_name=${ckpt_name} \
            extra.suffix_segment=${suffix_segment}
        done  # seed loop
    done  # function loop
done  # ckpt loop