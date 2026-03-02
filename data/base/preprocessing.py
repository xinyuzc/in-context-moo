"""Preprocessing utilities for data transformation and interpolation."""

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import numpy as np
from scipy.optimize import differential_evolution
from torch import Tensor
import torch
from utils.types import FloatListOrNested, NestedFloatList, FloatListOrNestedOrTensor
from typing import Tuple, Optional, List


_sigma = 0.0


def set_noise_level(level: float) -> None:
    global _sigma
    _sigma = level


def get_noise_level() -> float:
    global _sigma
    return _sigma


def tuple_list_to_nested_list(list_of_tuples: List[tuple]) -> NestedFloatList:
    """N x [tuple] -> N x [list]."""
    return [list(t) if not isinstance(t, list) else t for t in list_of_tuples]


def make_range_nested_list(
    range_list: FloatListOrNested, num_dim: int
) -> NestedFloatList:
    """max_dim x [min, max] or [min, max] -> num_dim x [min, max]

    Args:
        range_list: max_dim x [min, max] or [min, max]
        num_dim: number of dimensions to create ranges for

    Returns: num_dim x [min, max]
    """
    if isinstance(range_list[0], (int, float)):
        # Single list: repeat `num_dim` times
        range_list = [range_list for _ in range(num_dim)]
    else:
        # Nested list: first `num_dim`
        if len(range_list) < num_dim:
            raise ValueError(
                f"Expected at least {num_dim} ranges, got {len(range_list)}."
            )
        range_list = range_list[:num_dim]

    return range_list


def make_range_tensor(range_list: FloatListOrNestedOrTensor, num_dim: int) -> Tensor:
    """[min, max] | num_dim x [min, max] | Tensor of shape [num_dim, 2] -> [num_dim, 2]."""
    if isinstance(range_list, Tensor):
        assert range_list.shape == (num_dim, 2)
    else:
        range_list = make_range_nested_list(range_list, num_dim)
        range_list = torch.tensor(range_list)
    return range_list.float()


def add_gaussian_noise(data: Tensor) -> Tensor:
    """Add Gaussian noise to data.
    noise level _sigma is set in config.
    """
    sigma = get_noise_level()
    return data + torch.randn_like(data) * sigma


def transform(
    data: Tensor,
    inp_bounds: Tensor,
    out_bounds: Optional[Tensor] = None,
    transform_method: str = "min_max",
    eps: float = 1e-8,
):
    """Transform data.
    Args:
        data: [B, N, D] or [N, D]
        inp_bounds: [B, D, 2] or [D, 2]
        out_bounds: [B, D, 2] or [D, 2]
        transform_method: (str)
            "min_max": (data - min_in) / (max_in - min_in) * (max_out - min_out) + min_out
            "normalize": (data - min) / (max - min)
        eps (float): a small value to prevent division by zero

    Returns:
        data_transformed: [..., D]
    """
    # Bounds shape for broadcast: [B, D, 2] -> [B, 1, D, 2]
    if inp_bounds.ndim == 3:
        inp_bounds = inp_bounds.unsqueeze(1)
    if out_bounds is not None and out_bounds.ndim == 3:
        out_bounds = out_bounds.unsqueeze(1)

    if transform_method == "normalize":
        # [B, 1, D] or [D]
        denominator = inp_bounds[..., 1] - inp_bounds[..., 0]
        denominator = denominator + eps
        data_transformed = (data - inp_bounds[..., 0]) / denominator
    elif transform_method == "min_max":
        assert out_bounds is not None, f"out_bounds must be provided."
        if torch.allclose(inp_bounds, out_bounds):
            return data

        # Compute scale and shift
        denominator = inp_bounds[..., 1] - inp_bounds[..., 0]  # [..., D]
        denominator = denominator + eps
        scale = out_bounds[..., 1] - out_bounds[..., 0]  # [..., D]

        # Apply transformation
        data_transformed = (data - inp_bounds[..., 0]) / denominator
        data_transformed *= scale
        data_transformed += out_bounds[..., 0]
    else:
        raise NotImplementedError(
            f"Transform method {transform_method} not implemented."
        )

    # Shape check
    assert data_transformed.shape == data.shape
    return data_transformed


class OneDInterpolatorExt:
    """Interpolator on 1-dimensional data points.

    Args:
        points (np.ndarray): shape [N, 1]
        values (np.ndarray): shape [N, ...]
    """

    def __init__(self, points: np.ndarray, values: np.ndarray):
        assert points.shape[-1] == 1, "1-dimensional points expected."

        # Points should be strictly increasing, otherwise the results would be meaningless
        points = points.flatten()  # [N, ]
        sort_indices = np.argsort(points)
        self.points = points[sort_indices]
        self.values = values[sort_indices]

        assert np.all(np.diff(self.points) > 0), "Points must be strictly increasing."

        self.points_min = np.min(self.points).item()
        self.points_max = np.max(self.points).item()

    def __call__(self, x_new: np.ndarray) -> np.ndarray:
        """Interpolate at inputs.

        Args:
            inputs: shape [N,] or [N, 1]
        Returns:
            t: Interpolated values, shape [N, DY]
        """
        y_dim = self.values.shape[1]
        values_interp = []
        for d in range(y_dim):
            vd_interp = np.interp(
                x_new,
                self.points,
                self.values[:, d],
                left=self.values[0, d],
                right=self.values[-1, d],
            ).flatten()
            values_interp.append(vd_interp)

        values_interp = np.stack(values_interp, axis=-1)  # [N, DY]
        return values_interp


class NDInterpolatorExt:
    """interpolator on N-D data points.
    ref: https://github.com/NYCU-RL-Bandits-Lab/BOFormer/blob/main/Environment/benchmark_functions.py

    Args:
        points (np.ndarray): shape [N, D]
        values (np.ndarray): shape [N, ...]
    """

    def __init__(self, points: np.ndarray, values: np.ndarray):
        assert points.shape[-1] > 1, "Points must have at least 2 dimensions."

        self.funcinterp = LinearNDInterpolator(points, values)
        self.funcnearest = NearestNDInterpolator(points, values)

    def __call__(self, *args) -> np.ndarray:
        """Interpolate at inputs.

        Args:
            inputs:DX coordinates or `[N, DX]` array for N points.
        Returns:
            t (np.ndarray): Interpolated values, shape [N, DY]
        """
        t = self.funcinterp(*args)

        if np.isscalar(t):
            if np.isnan(t):
                return self.funcnearest(*args)
            else:
                return t
        else:
            if np.any(np.isnan(t)):
                return self.funcnearest(*args)
            else:
                return t


class NPInterpolatorExtTensor:
    """Wrapper for NDInterpolatorExt for PyTorch tensors.

    Args:
        points (np.ndarray): shape [N, D]
        values (np.ndarray): shape [N, ...]
    """

    def __init__(self, points: Tensor, values: Tensor):
        points_np = points.double().detach().cpu().numpy()
        values_np = values.double().detach().cpu().numpy()

        if points_np.shape[-1] == 1:
            self.interpolator = OneDInterpolatorExt(points_np, values_np)
        else:
            self.interpolator = NDInterpolatorExt(points_np, values_np)

    def __call__(self, x: Tensor) -> Tensor:
        """Batch call for interpolation.

        Args:
            x (Tensor): Input points, shape [DX] or [N, DX]

        Returns:
            y (Tensor): Interpolated values, shape [N, DY]
        """
        x_np = x.double().detach().cpu().numpy()

        if x_np.ndim <= 2:
            # No batch dimension
            y_np = self.interpolator(x_np)
            assert y_np.ndim == 2
        elif x_np.ndim == 3:
            # # Flatten batch dim, interpolate, then reshape
            B, N, DX = x_np.shape
            assert B == 1, f"Batch size > 1 not supported yet, got {B}."
            x_flat = x_np.reshape(-1, DX)
            y_flat = self.interpolator(x_flat)
            y_np = y_flat.reshape(B, N, -1)
        else:
            raise ValueError(f"Invalid input shape {x_np.shape}")

        y = torch.from_numpy(y_np).to(x)
        return y


def _scipy_adapted_function(function, x_np: np.ndarray) -> np.ndarray:
    """Adapt torch-based function for np input / output for use with SciPy optimization."""
    x_tensor = torch.tensor(x_np, dtype=torch.float64)
    result = function(x_tensor)
    return result.detach().cpu().numpy()


def _optimize_with_differential_evolution(
    func, bounds: NestedFloatList, minimize: bool
) -> Tuple[float, List[float]]:
    """Optimize a bounded function with differential evolution.

    Args:
        func: Torch-based function
        bounds: Input bounds for the function, DX x [[x_min, x_max]]
        minimize: If True, minimize the function; if False, maximize it

    Returns:
        min_value / max_value: Optimum value
        min_point / max_point: Optimum location
    """

    def _minimize_with_differential_evolution(
        function, bounds: NestedFloatList
    ) -> Tuple[float, List[float]]:
        """Minimize a bounded function with differential evolution."""
        wrapped_function = lambda x: _scipy_adapted_function(function, x)

        # Find global minimum
        result = differential_evolution(wrapped_function, bounds=bounds)

        min_value = result.fun
        min_point = result.x

        return min_value, min_point

    if minimize:
        return _minimize_with_differential_evolution(func, bounds)
    else:
        # Find minimum of the negated function
        func_negated = lambda x: -func(x)

        max_value_negated, max_point = _minimize_with_differential_evolution(
            func_negated, bounds
        )

        # Flip the sign back for maximum of the original function
        max_value = -max_value_negated

        return max_value, max_point


def estimate_objective_bounds(
    func: callable, num_objectives: int, x_bounds: NestedFloatList
) -> NestedFloatList:
    """Estimate output bounds for each objective function over the input bounds.

    Args:
        func: x -> objectives
        num_objectives: Number of objectives in the function
        x_bounds: Input bounds for the function, DX x [[x_min, x_max]]

    Returns:
        y_bounds: DY x [[y_min, y_max]]
    """
    y_bounds = []

    for i in range(num_objectives):
        # Extract i-th objective
        func_i = lambda x, i=i: func(x).view(-1, num_objectives)[..., [i]]

        # Find min and max
        min_i, _ = _optimize_with_differential_evolution(
            func=func_i, bounds=x_bounds, minimize=True
        )
        max_i, _ = _optimize_with_differential_evolution(
            func=func_i, bounds=x_bounds, minimize=False
        )

        y_bounds.append([min_i, max_i])

    return y_bounds


def has_nan_or_inf(tensor: Tensor, name: str, log: callable = print) -> bool:
    """Check if a tensor contains nan or inf values. Logs the tensor if it does."""
    if torch.isnan(tensor).any():
        log(f"{name} contains NaNs:\n{tensor}")
        return True

    if torch.isinf(tensor).any():
        log(f"{name} contains Infs:\n{tensor}")
        return True

    return False
