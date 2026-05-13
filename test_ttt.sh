#!/bin/bash -l
#SBATCH --job-name=TEST_TIME_TRAINING
#SBATCH --mem=3G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-32g
##SBATCH --partition=gpu-h100-80g,gpu-a100-80g,gpu-h200-141g-short,gpu-b300-288g-short,gpu-b300-288g-ellis,gpu-b300-288g-short,gpu-h200-141g-ellis,gpu-h200-141g-short
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=outputs/test_time_training/%x/%j.output
#SBATCH --error=outputs/test_time_training/%x/%j.err


python --version

## ============================================= ##
##              Experiment Identity              ##
## ============================================= ##

max_x_dim=5
expid=DX12345_DY123
FUNCTIONS=("BraninCurrin")
suffix_segment=debugging

## ============================================= ##
##             Non-default Overrides             ##
## ============================================= ##

# TTT
optimize_parameters=decoder        # default: policy
use_independent_sampler=false      # default: false (false = NUTS posterior over hyperparams; true = prior-sampled hyperparams)

# Adaptive query set
use_adaptive_query_set=false        # default: false
adaptive_topk_temperature=0.5      # default: 0.0

## ============================================= ##
##                   Run Test                    ##
## ============================================= ##

NUM=2  # seeds per array job

for function_name in "${FUNCTIONS[@]}"; do
    for i in $(seq 0 $((NUM - 1))); do
        seed=$((SLURM_ARRAY_TASK_ID * NUM + i))
        python test_time_training.py --config-name=test_ttt \
            experiment.seed=${seed} \
            experiment.expid="${expid}" \
            extra.suffix_segment=${suffix_segment} \
            model.max_x_dim=${max_x_dim} \
            data.max_x_dim=${max_x_dim} \
            data.function_name="${function_name}" \
            data.scene=null \
            test_time.optimize_parameters=${optimize_parameters} \
            test_time.use_independent_sampler=${use_independent_sampler} \
            optimization.use_adaptive_query_set=${use_adaptive_query_set} \
            optimization.adaptive_topk_temperature=${adaptive_topk_temperature}
    done
done