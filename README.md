# In-Context Multi-Objective Optimization 

This repository contains the official implementation of the ICLR 2026 paper: [In-Context Multi-Objective Optimization](https://openreview.net/forum?id=odmeUlWta8).

## Installation

1. Clone the repository
2. Create a conda environment (`tamo`) and install dependencies:
   ```
    conda env create --file environment.yml
    conda activate tamo
   ``` 

## Structure
- `model/`: TAMO architecture
- `data/`: Environment and data relevant utilities
- `configs/`: Configuration yaml files
- `utils/`: Utilities
- `results/`: experiment results
- `datasets/`: synthetic dataset for training or evaluation

## Examples 
Check `notebooks/` for examples of use and visualizations.

## How to start
### Generate synthetic data before training
TAMO is trained on pre-generated datasets for efficiency. To generate data:
```bash
python generate_data.py  --config-name=generate_data \
    experiment.mode=train \
    experiment.device=cuda \
    generate.filename="gp_0" \
    generate.x_dim=1 \
    generate.y_dim=1 \
    generate.sampler_type=gp \
    generate.num_datasets=100000 \
    generate.num_datapoints=300 
```
Check all available arguments in `configs/generate_data.yaml`. 

Dataset will be saved at `datasets/train/x_dim_1/y_dim_1/gp_0.hdf5`. Note that dataset can be alternatively saved under `datasets/{data.data_id}/train/x_dim_1/y_dim_1/gp_0.hdf5` by assigning valid values to `data.data_id`.

### Evaluation 
To run test script (`test.py`): 
```
python test.py --config-name=test
```
Check all available arguments in `configs/test.yaml`.

#### Test functions
`data/function.py` implements test function class, where: 
- `SyntheticFunction`: environment based on botorch function.
- `IntepolatorFunction`: environment based on the linear interpolation of dataset. 

These environments enable: a) evaluating the function at certain inputs, b) sampling from the function, c) computing hypervolume, d) computing regrets. Some core functions: 
- `init()`: Sample initial observations.
- `sample()`: Sample datapoints from the underlying function.
- `transform_outputs()`: Transform function values for TAMO.
- `step()`: update observations and compute hypervolume and regrets so far.

#### Results 
Metrics and plots will be saved under `results/data` and `results/plots` respectively.

## Citation 
```
@misc{zhang2025incontextmultiobjectiveoptimization,
      title={In-Context Multi-Objective Optimization}, 
      author={Xinyu Zhang and Conor Hassan and Julien Martinelli and Daolang Huang and Samuel Kaski},
      year={2025},
      eprint={2512.11114},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.11114}, 
}
```