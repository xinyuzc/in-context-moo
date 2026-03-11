import re
import os
import os.path as osp
from typing import List, Optional, Union, Tuple

import numpy as np
from torch import Tensor
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
import h5py

from data.function import TestFunction, IntepolatorFunction, SyntheticFunction
from data.laser_plasma import LaserPlasmaDataLoader

ROOT = ""
DATASETS_PATH = osp.join(ROOT, "datasets")


def get_datadir(mode: str, data_id: Optional[str] = None) -> List:
    """Get datapaths: `DATASETS_PATH/mode/data_id"""
    folder_name = mode
    if data_id is not None:
        folder_name = osp.join(folder_name, f"{data_id}")

    path = osp.join(DATASETS_PATH, folder_name)
    return path


def get_dx_dy_subfolder(dx: int, dy: int):
    return f"x_dim_{dx}/y_dim_{dy}"


def get_datapaths(
    mode: str, x_dim_list: List, y_dim_list: List, data_id: Optional[str] = None
) -> List:
    """Get datapaths: `DATASETS_PATH/mode/data_id/x_dim_{dx}/y_dim_{dy}/filename.hdf5"""
    path = get_datadir(mode, data_id)

    datapaths = []
    for dx in x_dim_list:
        for dy in y_dim_list:
            subfolders = get_dx_dy_subfolder(dx, dy)
            data_dir = osp.join(path, subfolders)

            if not osp.exists(data_dir):
                raise ValueError(f"Data directory {data_dir} does not exist.")

            for filename in os.listdir(data_dir):
                datapath = osp.join(data_dir, filename)
                if osp.isfile(datapath):
                    datapaths.append(datapath)

    return datapaths


def map_function_to_gp_datapath(
    function_name: str, mode: str, data_id=None
) -> Optional[str]:
    """Map function name `dx{}_dy{}` to local gp dataset paths.

    Returns:
        datapath: DATASETS_PATH/split/[data_id]/x_dim_{}/y_dim_{}/[filename].hdf5
        function_name: dx{}_dy{}
    """
    pattern = r"dx(\d+)_dy(\d+)"
    match = re.fullmatch(pattern, function_name)

    if match:
        dx = int(match.group(1))
        dy = int(match.group(2))

        datapath = get_datapaths(
            mode=mode, x_dim_list=[dx], y_dim_list=[dy], data_id=data_id
        )

        return datapath, function_name
    else:
        print(f"Function name {function_name} does not match the expected pattern.")
        return None, None


def get_function_environment(
    function_name: str,
    mode: str = "test",
    seed: int = 0,
    device: str = "cpu",
    data_id: Optional[str] = None,
    **kwargs,
) -> TestFunction:
    """Create function environment based on function name.

    Args:
        function_name: LaserPlasma, dx{}_dy{}, or supported botorch benchmark.
        ...

    Returns: test function instance.
    """
    if SyntheticFunction.get_function_constructor(function_name) is None:
        # Interpolation function
        if function_name == "LaserPlasma":
            # LaserPlasma dataset
            data_loader = LaserPlasmaDataLoader(device=device, negate=True)
            x, y, x_bounds, y_bounds = data_loader.get_data()
        else:
            # GP dataset
            hdf5_paths, _ = map_function_to_gp_datapath(
                function_name=function_name, mode=mode, data_id=data_id
            )
            if hdf5_paths is None:
                raise ValueError(f"Unsupported function name: {function_name}")
            if len(hdf5_paths) != 1:
                raise ValueError(
                    f"Expected exactly 1 dataset file for {function_name}, "
                    f"but found {len(hdf5_paths)}: {hdf5_paths}"
                )
            hdf5_path = hdf5_paths[0]

            def _get_bounds(data):
                mins = data.min(dim=0).values  # [dim]
                maxs = data.max(dim=0).values
                return torch.stack([mins, maxs], dim=-1).to(data.device)  # [dim, 2]

            x, y, _, _ = load_dataset(
                grp_name=f"dataset_{seed}",
                hdf5_path=hdf5_path,
                device=device,
            )
            x_bounds = _get_bounds(x)
            y_bounds = _get_bounds(y)

        return IntepolatorFunction(
            function_name=function_name,
            train_x=x,
            train_y=y,
            train_x_bounds=x_bounds,
            train_y_bounds=y_bounds,
        )
    else:
        return SyntheticFunction(function_name=function_name, **kwargs)


def load_dataset(
    hdf5_path: str, grp_name: str, device: str
) -> Tuple[Tensor, Tensor, int, int]:
    """Returns dataset with `grp_name` from an HDF5 file, or None if it doesn't exist."""
    with h5py.File(hdf5_path, "r") as f:
        if grp_name not in f:
            return None

        inputs = torch.tensor(f[grp_name]["inputs"][:], device=device)
        targets = torch.tensor(f[grp_name]["targets"][:], device=device)

        input_dim = f[grp_name].attrs["input_dim"]
        output_dim = f[grp_name].attrs["output_dim"]

    return inputs, targets, input_dim, output_dim


def save_dataset(hdf5_path: str, grp_name: str, inputs: Tensor, targets: Tensor):
    """Add new dataset as a subgroup given by `grp_name` to an HDF5 file."""
    with h5py.File(hdf5_path, "a") as f:  # "a" mode to append datasets
        inputs_np = inputs.float().detach().cpu().numpy()
        targets_np = targets.float().detach().cpu().numpy()

        if grp_name in f:
            # if group already exists, append
            grp = f.get(grp_name)
        else:
            # Otherwise create a new group
            grp = f.create_group(grp_name)

        grp.create_dataset("inputs", data=inputs_np)
        grp.create_dataset("targets", data=targets_np)

        # Store dimension metadata
        grp.attrs["input_dim"] = inputs_np.shape[1] if len(inputs_np.shape) > 1 else 1
        grp.attrs["output_dim"] = (
            targets_np.shape[1] if len(targets_np.shape) > 1 else 1
        )


class MultiFileHDF5Dataset(Dataset):
    """Create a dataset from HDF5 files.

    Args:
        file_paths: List of paths to HDF5 files.
        max_x_dim: Maximum dimension for input data.
        max_y_dim: Maximum dimension for target data.
        zero_mean: Whether to center function values.
        standardize: Whether to min-max scale function values to `range_scale`.
        range_scale: function value ranges if `standardize` is True.
    """

    def __init__(
        self,
        file_paths: List[str],
        max_x_dim: int,
        max_y_dim: int,
        range_scale: Optional[List] = None,
        zero_mean: bool = True,
        standardize: bool = True,
    ):
        if standardize:
            assert range_scale is not None

        self.file_paths = file_paths
        self.max_x_dim = max_x_dim
        self.max_y_dim = max_y_dim
        self.zero_mean = zero_mean
        self.standardize = standardize
        self.range_scale = range_scale

        self.data_per_file = [self.get_data_size(path) for path in file_paths]
        self.cumulative_sizes = [0] + list(
            torch.cumsum(torch.tensor(self.data_per_file), dim=0)
        )
        self.total_size = self.cumulative_sizes[-1]

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        """Get item from the dataset at index `idx`."""
        file_idx = torch.searchsorted(
            torch.tensor(self.cumulative_sizes), idx, right=True
        ).item()
        file_idx -= 1
        local_idx = idx - self.cumulative_sizes[file_idx]
        return self.load_from_file(self.file_paths[file_idx], local_idx)

    @staticmethod
    def group_ele_count(hdf5_path: str, grp_name: Optional[str] = None):
        """Count number of objects in a group.

        Args:
            grp_name (Optional): If None, count the entire HDF5 file.
            hdf5_path (str): Path to the HDF5 file.
            grp_name (Optional[str]): Name of the group to count objects in.

        Returns: number of files under the group.
        """
        try:
            with h5py.File(hdf5_path, "r") as f:
                if grp_name is not None:
                    if grp_name in f:
                        grp = f.get(grp_name)
                        count = len(grp)
                    else:
                        count = 0
                else:
                    # Otherwise the entire file
                    count = len(f)

        except FileNotFoundError:
            print(f"File not found: {hdf5_path}")
        return count

    def get_data_size(self, file_path):
        return self.group_ele_count(file_path)

    @staticmethod
    def _standardize(
        range_scale: Union[np.ndarray, List[float]], yvals, eps: float = 1e-8
    ) -> np.ndarray:
        """Scale yvals to a given range.

        Args:
            range_scale: [min, max]
            yvals: [num_samples, y_dims]

        Returns: scaled yvals
        """
        mins = yvals.min(axis=0, keepdims=True)
        maxs = yvals.max(axis=0, keepdims=True)
        norm_term = maxs - mins + eps

        scale_factor = range_scale[1] - range_scale[0]
        yvals = (yvals - mins) / norm_term
        yvals = range_scale[0] + yvals * scale_factor

        return yvals

    @staticmethod
    def _zero_mean(yvals: np.ndarray) -> np.ndarray:
        means = np.mean(yvals, axis=0, keepdims=True)
        return yvals - means

    def load_from_file(self, file_path, local_idx):
        """Load dataset from group named `dataset_{local_idx}`."""
        try:
            with h5py.File(file_path, "r", swmr=True) as f:
                grp_name = f"dataset_{local_idx}"
                group = f[grp_name]

                xvals = group["inputs"][:]
                yvals = group["targets"][:]

                valid_x_counts = group.attrs["input_dim"]
                valid_y_counts = group.attrs["output_dim"]

            if self.standardize:
                yvals = self._standardize(self.range_scale, yvals)
            elif self.zero_mean:
                yvals = self._zero_mean(yvals)

            # Pad xvals and yvals to fixed max dim counts to enable batching
            pad_x = self.max_x_dim - valid_x_counts
            pad_y = self.max_y_dim - valid_y_counts

            # assert pad_x >= 0, f"{self.max_x_dim}-{valid_x_counts}<0"
            # assert pad_y >= 0, f"{self.max_y_dim}-{valid_y_counts}<0"
            
            xvals = np.pad(xvals, ((0, 0), (0, pad_x)), "constant")
            yvals = np.pad(yvals, ((0, 0), (0, pad_y)), "constant")

        except FileNotFoundError:
            print(f"File not found: {file_path}")

        return xvals, yvals, valid_x_counts, valid_y_counts


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    split: str,
    device: str,
    num_workers: int = 0,
    prefetch_factor: Optional[int] = None,
) -> StatefulDataLoader:
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        generator=torch.Generator(device="cpu"),
        pin_memory=(device != "cpu"),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
    )

    return dataloader
