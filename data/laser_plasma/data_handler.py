"""LaserPlasmaDataLoader for laser plasma data; call class function `get_data()` to get processed laser plasma data."""

import torch
from torch import Tensor
import numpy as np
import os.path as osp
from pathlib import Path

current_file_path = Path(__file__)
current_module_path = current_file_path.parent


def read_npy_file(filepath: str) -> dict:
    data = np.load(filepath, allow_pickle=True)
    return data


class LaserPlasmaDataLoader:
    """Dataloader for laser plasma data."""

    def __init__(
        self,
        filepath=f"{current_module_path}/DataSet/MultiObjective",
        x_filename: str = "train_xMO.npy",
        y_filename: str = "train_objMO.npy",
        function_name: str = "LaserPlasma",
        device: str = "cpu",
        negate: bool = True,
    ):
        self.function_name = function_name
        self.filepath = filepath
        self.device = device

        x_filepath = osp.join(filepath, x_filename)
        y_filepath = osp.join(filepath, y_filename)
        raw_x = read_npy_file(x_filepath)[:, :-1]  # [n, 4]
        raw_y = read_npy_file(y_filepath)  # [n, 3]
        if negate:
            raw_y = -raw_y

        # into Tensor
        self.x = self._npy_to_tensor(raw_x)
        self.y = self._npy_to_tensor(raw_y)

        self.num_samples = self.x.shape[0]

        # get bounds
        self.x_bounds = self._get_bounds(self.x)
        self.y_bounds = self._get_bounds(self.y)

    def _get_bounds(self, raw_data):
        mins = torch.min(raw_data, 0)[0]
        maxs = torch.max(raw_data, 0)[0]
        bounds = torch.stack([mins, maxs], dim=1).to(self.device)

        return bounds  # [d, 2]

    def _npy_to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.from_numpy(array).float().to(self.device)

    def get_data(self):
        """Returns: x of shape (n_samples, 4), y of shape (n_samples, 3), x bounds of shape (4, 2), y bounds of shape (3, 2)."""
        return self.x, self.y, self.x_bounds, self.y_bounds


if __name__ == "__main__":
    data_loader = LaserPlasmaDataLoader()
    x, y, x_bounds, y_bounds = data_loader.get_data()
    print(x.shape, y.shape)
    print(x_bounds)
    print(y_bounds)
