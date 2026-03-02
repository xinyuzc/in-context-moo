"""Samplers for GP-based functions with global optimum structure."""

import math
from typing import Tuple, List, Optional
import random

import torch
from torch.distributions.bernoulli import Bernoulli
from torch import Tensor
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.settings import fast_pred_var, cholesky_jitter, cholesky_max_tries
import numpy as np
import scipy.stats as sps

from data.sampler import _get_data_kernel
from data.function_sampling import generate_sobol_samples
from data.base.gpytorch_utils import ExactGPModel


MAXIMIZE = False
DATA_KERNEL_TYPE_LIST = ["rbf", "matern32", "matern52"]
SAMPLE_KERNEL_WEIGHTS = [1, 1, 1]
L_RANGE = [0.1, 2.0]
STD_RANGE = [0.1, 1.0]
P_ISO = 0.5
JITTER = 1e-3
N_SAMPLES = 1
MAX_TRIES = 10


class SimpleGPSampler:
    def __init__(self, kernel_name: str):
        self.kernel_name = kernel_name

    def sample(
        self,
        test_X,
        train_X,
        train_y,
        length_scale,
        std,
        n_samples=N_SAMPLES,
        correlated=True,
        max_tries=MAX_TRIES,
        jitter=JITTER,
        **kwargs,
    ):
        """Sample function values at test_X conditioned on (train_X, train_y) from a GP.

        Args:
            test_X, (num_points, d_x): input locations to sample function values.
            train_X, (num_train_points, d_x): training input locations.
            train_y, (num_train_points, 1): training function values.
            length_scale, (d_x) or (1): lengthscale(s) of the kernel.
            std, (1): standard deviation of the kernel.
            n_samples, int: number of samples to draw.
            correlated, bool: whether to sample correlated points.
            max_tries, int: maximum number of Cholesky attempts.
            jitter, float: jitter for numerical stability.

        Returns:
            samples, (num_points, n_samples): sampled function values at test_X.
        """
        likelihood = GaussianLikelihood().to(test_X)
        likelihood.noise = 1e-4  # small noise for numerical stability

        kernel = _get_data_kernel(kernel_type=self.kernel_name, x_dim=train_X.shape[-1])

        model = ExactGPModel(kernel, likelihood, train_X, train_y.squeeze()).to(test_X)

        # 2. Set hyperparameters manually (no training)
        model.covar_module.base_kernel.lengthscale = length_scale
        model.covar_module.outputscale = std.pow(2)

        model.eval()
        likelihood.eval()

        model.to(train_X)
        likelihood.to(train_X)

        with torch.no_grad(), fast_pred_var(), cholesky_jitter(
            jitter
        ), cholesky_max_tries(max_tries):
            dist = likelihood(model(test_X))

            if correlated:
                # Sample correlated points
                samples = dist.rsample(torch.Size([n_samples]))
                return (
                    samples.permute(1, 0).unsqueeze(-1) if n_samples > 1 else samples.T
                )
            else:
                # Sample each point independently
                mu = dist.mean
                sigma = dist.stddev
                eps = torch.randn(n_samples, len(mu), device=mu.device)
                samples = mu + sigma * eps
                return samples.T


class OptimizationSampler(object):
    def __init__(
        self,
        data_kernel_type_list: list = DATA_KERNEL_TYPE_LIST,
        sample_kernel_weights: list = SAMPLE_KERNEL_WEIGHTS,
        lengthscale_range: list = L_RANGE,
        std_range: list = STD_RANGE,
        p_iso: float = P_ISO,
        maximize: bool = MAXIMIZE,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        """Sampler for functions with global optimum structure based on GPs.

        Args:
            data_kernel_type_list, list: list of kernel function names.
            sample_kernel_weights, list: sampling weights for each kernel function.
            lengthscale_range, list: range of lengthscale.
            std_range, list: range of standard deviation.
            p_iso, float: probability of isotropic kernel.
            maximize, bool: whether to create functions with global maximum (True) or minimum (False).
            device, str: device to run the sampler.
        """
        assert len(data_kernel_type_list) == len(sample_kernel_weights)
        assert len(lengthscale_range) == 2
        assert len(std_range) == 2

        self.data_kernel_type_list = data_kernel_type_list
        self.sample_kernel_weights = sample_kernel_weights
        self.lengthscale_range = lengthscale_range
        self.std_range = std_range
        self.p_iso = p_iso

        self.maximize = maximize
        self.device = device

    @staticmethod
    def transform_with_global_optimum(
        x_range: Tensor,
        x: Tensor,
        y: Tensor,
        xopt: Tensor,
        f_offset: Optional[Tensor] = None,
        maximize: bool = True,
    ):
        """transform function values to ensure a global optima at xopt.
            If maximize=True, the function has global maximum at (xopt, -f_offset).
            If maximize=False, the function has global minimum at (xopt, f_offset).

        Args:
            x_range, (d_x, 2): data range of each input dimension.
            x, (num_points, d_x): datapoints with d_x features.
            y, (num_points, 1): function values.
            xopt, (1, d_x): global minimum location.
            f_offset, (1): function offset.
            maximize, bool: whether to create a function with global maximum (True) or minimum (False).

        Returns:
            f: transformed function values.
            yopt: the global optimum.
        """
        # Compute quadratic bowl factor: smaller range, steeper bowl; smaller variation, smoother bowl
        quad_lengthscale_squared = torch.mean(
            (x_range[:, 1].float() - x_range[:, 0].float()) ** 2
        )
        quadratic_factor = 1.0 / quad_lengthscale_squared
        quadratic_factor *= torch.max(torch.abs(y))

        # set function offset
        if f_offset is None:
            f_offset = torch.zeros((1, 1), device=x.device)

        # transform the GP samples by taking the absolute value, adding a quadratic bowl and an offset
        f = (
            torch.abs(y)
            + quadratic_factor
            * torch.sum(
                (x - xopt) ** 2, dim=-1, keepdim=True
            )  # (B, num_test_points, 1)
            + f_offset
        )
        y = f
        yopt = f_offset

        # flip the function
        if maximize:
            f = -f
            yopt = -f_offset

        return f, yopt

    def _get_max_num_ctx_points(self, x_dim: int) -> int:
        _intervals = [1, 4, 6]
        _max_nc_list = [50, 100, 200]

        if x_dim == _intervals[0]:
            return _max_nc_list[0]
        elif x_dim < _intervals[1]:
            return _max_nc_list[1]
        elif x_dim < _intervals[2]:
            return _max_nc_list[2]
        else:
            raise ValueError("Input dimension too high.")

    def sample(
        self,
        batch_size: int = 64,
        max_num_ctx_points: int = 128,
        num_total_points: int = 256,
        x_range: List[List[float]] = [[-5.0, 5.0]],
        grid: bool = True,
        x: Optional[Tensor] = None,
        xopt: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """sample a batch of datapoints from GP-based functions with global optimum structure.

        Args:
            batch_size, int: the size of data batch.
            max_num_ctx_points, int: the maximal number of context points.
            num_total_points, int: the total number of context and target points.
            x_range, list: input range.
            grid: whether to sample from a grid or randomly.

        Returns:
            x, (B, num_total_points, d_x): input location.
            y, (B, num_total_points, 1): corresponding function values.
            xopt, (B, 1, d_x): : location of optimum.
            yopt, (B, 1, 1): optimal function value.
        """
        (_X, _Y, _XOPT, _YOPT) = ([], [], [], [])
        if not isinstance(x_range, Tensor):
            x_range = torch.tensor(x_range)  # (d_x, 2)

        if max_num_ctx_points is None:
            max_num_ctx_points = self._get_max_num_ctx_points(len(x_range))

        for _ in range(batch_size):
            _x, _y, _xopt, _yopt = self.sample_a_function(
                max_num_ctx_points=max_num_ctx_points,
                num_total_points=num_total_points,
                x_range=x_range,
                grid=grid,
                x=x,
                xopt=xopt,
            )

            _X.append(_x)
            _Y.append(_y)
            _XOPT.append(_xopt)
            _YOPT.append(_yopt)

        return (
            torch.stack(_X, dim=0),
            torch.stack(_Y, dim=0),
            torch.stack(_XOPT, dim=0),
            torch.stack(_YOPT, dim=0),
        )

    def sample_length_scale(self) -> Tuple[Tensor, Tensor]:
        """sample lengthscale using truncated log-normal distribution and standard deviation using uniform distribution.

        Returns:
            length, (d_x): lengthscale for each input dimension.
            std (1): standard deviation.
        """
        mu = np.log(2 / 3)
        sigma = 0.5
        # mu, sigma = np.log(1 / 3), 0.75
        a = (np.log(self.lengthscale_range[0]) - mu) / sigma
        b = (np.log(self.lengthscale_range[1]) - mu) / sigma
        rv = sps.truncnorm(a, b, loc=mu, scale=sigma)
        length = torch.tensor(
            np.exp(rv.rvs(size=(self.d_x))), device=self.device, dtype=torch.float32
        )
        # scale along with input dimension
        length *= math.sqrt(self.d_x)

        # whether the process is isotropic
        if self.d_x > 1:
            is_iso = Bernoulli(self.p_iso).sample()  # (1)
            if is_iso:
                # adjust lengthscale if isotropic
                length[:] = length[0]  # (d_x)

        std = (
            torch.rand(1, device=self.device) * (self.std_range[1] - self.std_range[0])
            + self.std_range[0]
        )
        return length, std

    def sample_a_function(
        self,
        x_range: Tensor,
        max_num_ctx_points: int = 50,
        num_total_points: int = 100,
        grid: bool = False,
        x: Optional[Tensor] = None,
        xopt: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """create and sample from a function.

        Args:
            x_range (Tensor): input range, [d_x, 2].
            max_num_ctx_points (int): the maximal number of context points.
            num_total_points (int): the total number of context and target points.
            grid (bool): whether to sample from a grid or randomly.
            x (Tensor): optional input locations, [num_total_points, d_x].
            xopt (Tensor): optional optimum location, [1, d_x].

        Returns:
            x (Tensor): input location, [num_total_points, d_x].
            y (Tensor): function values, [num_total_points, 1].
            xopt (Tensor): optimum location, [1, d_x].
            yopt (Tensor): optimal function value, [1, 1].
        """
        # Sample context size
        self.d_x = x_range.shape[0]
        if max_num_ctx_points is None:
            max_num_ctx_points = self._get_max_num_ctx_points(self.d_x)
        num_ctx_points = random.randint(3, max_num_ctx_points)

        # Sample kernel hyperparameters
        length_scale, sigma_f = self.sample_length_scale()

        # Sample function mean according to input range and lengthscale: [1]
        n_temp = int(
            torch.ceil(
                torch.prod(x_range[:, 1] - x_range[:, 0]) / torch.prod(length_scale)
            )
            .int()
            .item()
        )
        temp_mean = torch.zeros(size=(n_temp,))
        temp_stds = torch.full(size=(n_temp,), fill_value=sigma_f.item())
        temp_samples = torch.abs(torch.normal(temp_mean, temp_stds))  # [n_temp]
        mean_f = temp_samples.max()

        # Create a sharp optimum with a minor probability by increasing the mean function value
        p_rare, rare_tau = 0.1, 1.0
        if torch.rand(1) < p_rare:
            mean_f = mean_f + torch.exp(torch.tensor(rare_tau))

        # Generate input locations
        if x is None:
            x = generate_sobol_samples(
                x_range=x_range, num_datapoints=num_total_points, grid=grid
            )
        else:
            if len(x) != num_total_points:
                raise ValueError(f"Input x should have {num_total_points} points.")

        # Sample the optimum location: (xopt, 0)
        xopt_idx = torch.randint(1, num_ctx_points, (1,)).item()
        xopt = x[xopt_idx : xopt_idx + 1]
        yopt = torch.zeros(size=(1, 1), dtype=torch.float32, device=self.device)

        # other input locations
        x_rest = torch.cat([x[:xopt_idx], x[xopt_idx + 1 :]], dim=0)

        # Sample kernel type
        kernel_name = random.choices(
            population=self.data_kernel_type_list,
            weights=self.sample_kernel_weights,
            k=1,
        )[0]

        # Sample at all other locations conditioned on the optimum
        gp_sampler = SimpleGPSampler(kernel_name=kernel_name)
        y_rest = gp_sampler.sample(
            test_X=x_rest,
            train_X=xopt,
            train_y=yopt,
            length_scale=length_scale,
            std=sigma_f,
            correlated=True,
        )

        x_full = torch.cat([x_rest[:xopt_idx], xopt, x_rest[xopt_idx:]], dim=0)
        y_full = torch.cat([y_rest[:xopt_idx], yopt, y_rest[xopt_idx:]], dim=0)

        # create function with global optimum structure by transforming the GP samples
        y_full, yopt = self.transform_with_global_optimum(
            x_range=x_range,
            x=x_full,
            y=y_full,
            xopt=xopt,
            maximize=self.maximize,
        )

        return x_full, y_full, xopt, yopt
