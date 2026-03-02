"""Generate synthetic function dataset."""

from dataclasses import dataclass, asdict
import os
import os.path as osp
from typing import Optional

import hydra
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from data.sampler import gp_sampler
from data.sampler_global_opt import OptimizationSampler
from data.base.preprocessing import make_range_tensor
from data.function_sampling import generate_sobol_samples
from data.dataset import (
    get_dx_dy_subfolder,
    get_datadir,
    save_dataset,
    MultiFileHDF5Dataset,
)
from utils.seed import set_all_seeds
from utils.dataclasses import DataConfig
from utils.log import get_log_fn, get_log_filename

CLEAN_CACHE_EVERY = 20


@dataclass
class DatasetConfig:
    x_dim: int
    y_dim: int
    sampler_type: str = "gp"
    grid: bool = False
    num_datapoints: int = 300
    num_datasets: int = 100000
    filename: Optional[str] = None

    def __post_init__(self):
        assert self.sampler_type in ["gp", "opt"]
        if self.filename is None:
            self.filename = f"gp_x{self.x_dim}_y{self.y_dim}"

    def to_dict(self):
        return asdict(self)


@hydra.main(version_base=None, config_path="configs", config_name="generate_data.yaml")
def main(config: DictConfig):
    seed = config.experiment.seed
    mode = config.experiment.mode
    expid = config.experiment.expid
    device = config.experiment.device
    resume = config.experiment.resume

    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)

    # ==== Configurations ====
    data_config = DataConfig(**config.data)
    dataset_config = DatasetConfig(**config.generate)

    # ==== Setup logging: data/expid/timestamp.log ====
    log = get_log_fn(get_log_filename(model_name="data", expid=expid))

    # ==== Setup path to save datasets ====
    path = osp.join(
        get_datadir(mode, data_config.data_id),
        get_dx_dy_subfolder(dataset_config.x_dim, dataset_config.y_dim),
    )

    generate_data(
        path=path,
        seed=seed,
        device=device,
        resume=resume,
        data_config=data_config,
        static_data_config=dataset_config,
        log=log,
    )


def generate_data(
    path: str,
    seed: int,
    device: str,
    resume: bool,
    data_config: DataConfig,
    static_data_config: DatasetConfig,
    log: callable = print,
):
    """Generate datasets and save to HDF5 file."""
    datapath = osp.join(path, f"{static_data_config.filename}.hdf5")
    if osp.exists(datapath):
        if not resume:
            raise FileExistsError(
                path, f"File {static_data_config.filename}.hdf5 already exists."
            )
        else:
            epoch = MultiFileHDF5Dataset.group_ele_count(datapath)
    else:
        epoch = 0
        os.makedirs(path, exist_ok=True)
    log(
        f"Generating data at datapath:\t{datapath}, resume:\t{resume}, starting epoch:\t{epoch}"
    )
    log(
        f"    seed:\t{seed}\n"
        f"    {static_data_config.to_dict()}\n"
        f"    {data_config.to_dict()}"
    )

    # Set random seeds
    set_all_seeds(seed)

    x_range_t = make_range_tensor(
        data_config.x_range,
        num_dim=static_data_config.x_dim,
    ).to(device=device)

    if static_data_config.sampler_type == "opt":
        sampler = OptimizationSampler(
            data_kernel_type_list=data_config.data_kernel_type_list,
            sample_kernel_weights=data_config.sample_kernel_weights,
            lengthscale_range=data_config.lengthscale_range,
            std_range=data_config.std_range,
            p_iso=data_config.p_iso,
        )

    # Generate datasets
    for i in tqdm(
        range(epoch, epoch + static_data_config.num_datasets),
        desc=f"Generating {static_data_config.num_datasets} datasets...",
        miniters=500,
    ):

        if static_data_config.sampler_type == "gp":
            # Generate input points: [num_datapoints, x_dim]
            x = generate_sobol_samples(
                x_range=x_range_t,
                num_datapoints=static_data_config.num_datapoints,
                grid=static_data_config.grid,
            )

            # Sample function values: [num_datapoints, y_dim]
            y = gp_sampler(
                x=x.unsqueeze(0),
                x_range=x_range_t,
                y_dim=static_data_config.y_dim,
                sampler_list=data_config.sampler_list,
                sampler_weights=data_config.sampler_weights,
                data_kernel_type_list=data_config.data_kernel_type_list,
                sample_kernel_weights=data_config.sample_kernel_weights,
                lengthscale_range=data_config.lengthscale_range,
                std_range=data_config.std_range,
                min_rank=data_config.min_rank,
                max_rank=data_config.max_rank,
                p_iso=data_config.p_iso,
                jitter=data_config.jitter,
                max_tries=data_config.max_tries,
                device=device,
            )
        else:
            # NOTE shared input points for all objectives
            sobol_samples = generate_sobol_samples(
                x_range=x_range_t,
                num_datapoints=static_data_config.num_datapoints,
                grid=static_data_config.grid,
            )

            _y_list = []
            for _ in range(static_data_config.y_dim):
                # Sample single objective: [1, num_datapoints, 1]
                _, _y, _, _ = sampler.sample(
                    batch_size=1,
                    max_num_ctx_points=None,
                    num_total_points=static_data_config.num_datapoints,
                    x_range=x_range_t,
                    grid=static_data_config.grid,
                    x=sobol_samples,
                )
                _y_list.append(_y)

            x = sobol_samples  # [num_datapoints, x_dim]
            y = torch.cat(_y_list, dim=-1)  # [1, num_datapoints, y_dim]

        y = y.squeeze(0)  # [num_datapoints, y_dim]

        save_dataset(
            datapath,
            grp_name=f"dataset_{i}",
            inputs=x,
            targets=y,
        )

        if i % CLEAN_CACHE_EVERY == 0:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
