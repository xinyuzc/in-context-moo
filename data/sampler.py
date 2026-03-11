"""GP samplers."""

from typing import List, Tuple, Optional, Union
import random

import torch
from torch import Tensor
from gpytorch.kernels import RBFKernel, MaternKernel, Kernel
from gpytorch.models import IndependentModelList
from gpytorch.likelihoods import (
    MultitaskGaussianLikelihood,
    GaussianLikelihood,
    LikelihoodList,
)
from gpytorch.settings import fast_pred_var, cholesky_jitter, cholesky_max_tries
import numpy as np
import scipy.stats as sps
from torch.distributions.bernoulli import Bernoulli

from data.base.gpytorch_utils import (
    MultitaskGPModel,
    ExactGPModel,
    RotatedARDKernel,
    sample_orthonormal_matrix,
)
from data.base.preprocessing import make_range_tensor
from data.function_sampling import generate_sobol_samples
from utils.types import FloatListOrNestedOrTensor

DATA_KERNEL_TYPE_LIST = ["rbf", "matern32", "matern52"]
SAMPLE_KERNEL_WEIGHTS = [1, 1, 1]
L_RANGE = [0.1, 2.0]
STD_RANGE = [0.1, 1.0]
MIN_RANK = 1
P_ISO = 0.5
JITTER = 1e-6
MAX_TRIES = 6


def _get_data_kernel(kernel_type: str, x_dim: int) -> Kernel:
    if kernel_type == "rbf":
        kernel = RBFKernel(ard_num_dims=x_dim)
    elif kernel_type == "matern12":
        kernel = MaternKernel(nu=0.5, ard_num_dims=x_dim)
    elif kernel_type == "matern32":
        kernel = MaternKernel(nu=1.5, ard_num_dims=x_dim)
    elif kernel_type == "matern52":
        kernel = MaternKernel(nu=2.5, ard_num_dims=x_dim)
    elif kernel_type == "rotated_ard":
        kernel = RotatedARDKernel(num_dims=x_dim)
    else:
        raise NotImplementedError(f"Unsupported data kernel type: {kernel_type}")

    return kernel


def _sample_lengthscale(
    x_dim: int,
    lengthscale_range: List,
    p_iso: float,
    mu=np.log(2 / 3),
    sigma=0.5,
) -> Tensor:
    """Sample lengthscales.

    Args:
        x_dim (int): x dimensions
        lengthscale_range: (x_dim)
        p_iso: probability of using same lengthscale for each dimension
        mu: truncated normal distribution mean
        sigma: truncated normal distribution sigma
    """
    # Sample from truncated normal distribution
    a = (np.log(lengthscale_range[0]) - mu) / sigma
    b = (np.log(lengthscale_range[1]) - mu) / sigma

    rv = sps.truncnorm(a, b, loc=mu, scale=sigma)
    lengthscale = Tensor(np.exp(rv.rvs(size=(x_dim))))

    # Scale lengthscale with x dims: http://arxiv.org/abs/2402.02229
    lengthscale *= torch.tensor(x_dim, device=lengthscale.device).sqrt()

    if x_dim > 1:
        # isotropic kernel with same lengthscale for each dimension
        is_iso = Bernoulli(p_iso).sample()  # (1)
        if is_iso:
            lengthscale[:] = lengthscale[0]

    return lengthscale


def _check_samples(
    x: Tensor,
    y: Tensor,
    kernel_type_list: List[str],
    lengthscale_list: List[str],
    std_list: List[Tensor],
    covar_list: List[Tensor],
    max_condition_number: float = 1e6,
):
    """Check if the samples contain NaNs, Infs, or ill-conditioned covariance matrices."""
    if not torch.all(torch.isfinite(x)):
        return False

    if not torch.all(torch.isfinite(y)):
        return False
    
    return True


def multi_task_gp_prior_sampler(
    x_range: Union[List, Tensor],
    x_dim: int,
    num_datapoints: int,
    num_tasks: int,
    data_kernel_type_list: List = DATA_KERNEL_TYPE_LIST,
    sample_kernel_weights: List = SAMPLE_KERNEL_WEIGHTS,
    lengthscale_range: List = L_RANGE,
    std_range: List = STD_RANGE,
    min_rank: int = MIN_RANK,  # Lower rank, higher task correlation
    max_rank: Optional[int] = None,
    p_iso: float = P_ISO,  # Probability of using isotropic kernel
    grid: bool = False,
    device: str = "cuda",
    x: Optional[Tensor] = None,
    jitter: float = JITTER,
    max_tries: int = MAX_TRIES,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """Sample from multi-task gp priors with a single data kernel and a task kernel."""
    assert std_range[0] > 0 and std_range[1] > std_range[0]
    assert lengthscale_range[0] > 0 and lengthscale_range[1] > lengthscale_range[0]

    x_range = make_range_tensor(x_range, x_dim).to(device)

    # Sample inputs: [num_datapoints, x_dim]
    if x is None:
        x = generate_sobol_samples(
            x_range=x_range, num_datapoints=num_datapoints, grid=grid
        )

    # Sample data kernel
    data_kernel_type = random.choices(
        population=data_kernel_type_list, weights=sample_kernel_weights, k=1
    )[0]
    data_kernel = _get_data_kernel(kernel_type=data_kernel_type, x_dim=x_dim)

    # Sample lengthscales for data kernel: [x_dim]
    lengthscale = _sample_lengthscale(
        x_dim=x_dim, lengthscale_range=lengthscale_range, p_iso=p_iso
    ).to(device)

    # Sample std for task kernel: [num_tasks]
    std = torch.rand(num_tasks, device=device)
    std = std * (std_range[1] - std_range[0]) + std_range[0]

    # Sample rank for task kernel
    if max_rank is not None:
        assert max_rank <= num_tasks, "`max_rank` should be no more than `num_tasks`."
    else:
        max_rank = num_tasks
    rank = random.randint(min_rank, max_rank)

    # Setup likelihood and model
    likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)
    model = MultitaskGPModel(
        train_x=None,
        train_y=None,
        likelihood=likelihood,
        kernel=data_kernel,
        num_tasks=num_tasks,
        rank=rank,
    )
    if data_kernel_type == "rotated_ard":
        sampled_R = sample_orthonormal_matrix(x_dim).to(device)
        model.covar_module.data_covar_module.raw_lengthscales.data = lengthscale
        model.covar_module.data_covar_module.R.data = sampled_R

    else:
        # Set lengthscales for the data kernel
        model.covar_module.data_covar_module.lengthscale = lengthscale

    # Set different variance for for different tasks by adjusting `v` vector in IndexKernel
    # `var` would be element-wise SoftPlus of passed values.
    model.covar_module.task_covar_module.var = std**2

    # Set up model and likelihood to evaluation mode and move to the correct dtype
    model.eval()
    likelihood.eval()

    model.to(x)
    likelihood.to(x)

    # sample from the prior distribution
    with torch.no_grad(), fast_pred_var(), cholesky_jitter(jitter), cholesky_max_tries(
        max_tries
    ):
        prior_dist = model(x)
        y = prior_dist.sample(torch.Size([1])).squeeze(0)  # [num_datapoints, num_task]

    is_valid = _check_samples(
        x=x,
        y=y,
        kernel_type_list=[data_kernel_type],
        lengthscale_list=[lengthscale],
        std_list=[std],
        covar_list=[model.covar_module(x).evaluate()],
    )

    # Free up memory
    model = model.cpu()
    likelihood = likelihood.cpu()
    model.eval()
    del model, likelihood

    if not is_valid:
        return None, None

    return x, y


def multi_output_gp_prior_sampler(
    x_range: Union[List, Tensor],
    x_dim: int,
    num_tasks: int,
    num_datapoints: int,
    data_kernel_type_list: List = DATA_KERNEL_TYPE_LIST,
    sample_kernel_weights: List = SAMPLE_KERNEL_WEIGHTS,
    lengthscale_range: List = L_RANGE,
    std_range: List = STD_RANGE,
    p_iso: float = P_ISO,
    grid: bool = False,
    device: str = "cuda",
    x: Optional[Tensor] = None,
    jitter: float = JITTER,
    max_tries: int = MAX_TRIES,
    **kwargs,
):
    """Sample from multi-output gp prior, with independent outputs, different kernels for each."""
    assert lengthscale_range[0] > 0 and lengthscale_range[1] > lengthscale_range[0]
    assert std_range[0] > 0 and std_range[1] > std_range[0]

    x_range = make_range_tensor(x_range, x_dim).to(device)

    # Sample inputs: [num_datapoints, x_dim]
    if x is None:
        x = generate_sobol_samples(
            x_range=x_range, num_datapoints=num_datapoints, grid=grid
        )

    # Sample data kernel for each task
    data_kernel_type = random.choices(
        population=data_kernel_type_list, weights=sample_kernel_weights, k=num_tasks
    )

    models = []
    likelihoods = []

    # Sample data kernel for each task
    for _, kernel_type in enumerate(data_kernel_type):
        # Sample lengthscale: [x_dim]
        lengthscale = _sample_lengthscale(
            x_dim=x_dim, lengthscale_range=lengthscale_range, p_iso=p_iso
        ).to(device)

        # Sample std: [1]
        std = torch.rand(1, device=device)
        std = std * (std_range[1] - std_range[0]) + std_range[0]

        # Sample data kernel
        data_kernel = _get_data_kernel(kernel_type=kernel_type, x_dim=x_dim)
        if kernel_type == "rotated_ard":
            sampled_R = sample_orthonormal_matrix(x_dim).to(device)
            data_kernel.raw_lengthscales.data = lengthscale
            data_kernel.R.data = sampled_R
        else:
            data_kernel.lengthscale = lengthscale

        # Setup likelihood and model
        likelihood = GaussianLikelihood()
        likelihoods.append(likelihood)
        model = ExactGPModel(
            kernel=data_kernel, likelihood=likelihood, train_x=None, train_y=None
        )
        model.covar_module.outputscale = std**2
        models.append(model)

    # Setup likelihood and model lists
    model = IndependentModelList(*models)
    likelihood = LikelihoodList(*likelihoods)

    model.eval()
    likelihood.eval()

    model.to(x)
    likelihood.to(x)

    # Sample from the prior distribution
    with torch.no_grad(), fast_pred_var(), cholesky_jitter(jitter), cholesky_max_tries(
        max_tries
    ):
        # This contains predictions for all models' outcomes as a list
        prior_dist_list = model(*[x for _ in range(num_tasks)])

        # num_tasks x [(num_datapoints)] -> [num_datapoints, num_tasks]
        ys = [
            prior_dist.sample(torch.Size([1])).squeeze(0)
            for prior_dist in prior_dist_list
        ]
        y = torch.stack(ys, dim=-1)

    # Check if the sampled outputs contain NaNs or Infs
    is_valid = _check_samples(
        x=x,
        y=y,
        kernel_type_list=[
            submodel.covar_module.base_kernel for submodel in model.models
        ],
        lengthscale_list=[
            submodel.covar_module.base_kernel.lengthscale for submodel in model.models
        ],
        std_list=[submodel.covar_module.outputscale for submodel in model.models],
        covar_list=[submodel.covar_module(x).evaluate() for submodel in model.models],
    )

    # Free up memory
    model = model.cpu()
    likelihood = likelihood.cpu()
    model.eval()
    del model, likelihood

    if not is_valid:
        return None, None

    return x, y


def sample_nc(x_dim: int, min_nc: int = 2, max_nc: int = 50, warmup: bool = False):
    """Sample context size based on number of dimensions:
        If warmup: use `scale_factor * max_nc` for stable training
        Otherwise: sample from `[min_nc, scale_factor * max_nc]`

    Examples:
        warmup:
            num_dim=1: 50
            num_dim=2: 100
            num_dim=3: 100
            num_dim=4: 200

        Otherwise:
            num_dim=1: [2, 50]
            num_dim=2: [2, 100]
            num_dim=3: [2, 100]
            num_dim=4: [2, 200]
    """
    scale_factor = 1
    if 1 < x_dim <= 3:
        scale_factor = 2
    elif x_dim > 3:
        scale_factor = 4

    max_nc_scaled = int(max_nc * scale_factor)
    if warmup:
        nc = max_nc_scaled
    else:
        nc = random.randint(min_nc, max_nc_scaled)

    return nc


_sampler_func_dict = {
    "multi_task_gp_prior_sampler": multi_task_gp_prior_sampler,
    "multi_output_gp_prior_sampler": multi_output_gp_prior_sampler,
}


def gp_sampler(
    x: Tensor,
    x_range: FloatListOrNestedOrTensor,
    y_dim: int,
    sampler_list: list,
    sampler_weights: list,
    data_kernel_type_list: list = DATA_KERNEL_TYPE_LIST,
    sample_kernel_weights: list = SAMPLE_KERNEL_WEIGHTS,
    lengthscale_range: tuple = L_RANGE,
    std_range: tuple = STD_RANGE,
    min_rank: int = MIN_RANK,
    max_rank: Optional[int] = None,
    p_iso: float = P_ISO,
    jitter: float = JITTER,
    max_tries: int = MAX_TRIES,
    device: str = "cuda",
) -> Tensor:
    """Sample batches from multi-output or multi-task gp priors.

    Args:
        x: [B, N, x_dim], input locations
        x_range (tensor, list, or nested list): input ranges
        y_dim (int): number of tasks
        ...

    Returns: function values of shape [B, N, y_dim]
    """
    B, N, x_dim = x.shape

    x_range_t = make_range_tensor(x_range, x_dim).to(device)
    assert x_range_t.shape == (x_dim, 2)

    y_list = []
    for b in range(B):
        # Sample a sampler function
        sampler = random.choices(population=sampler_list, weights=sampler_weights, k=1)[
            0
        ]
        sampler_func = _sampler_func_dict.get(sampler, None)
        assert sampler_func is not None, f"Sampler {sampler} is not defined."

        # Sample until y values are valid (no NaNs or Infs): [M, y_dim]
        while True:
            try:
                _, y = sampler_func(
                    x=x[b],
                    x_range=x_range_t,
                    x_dim=x_dim,
                    num_datapoints=N,
                    num_tasks=y_dim,
                    data_kernel_type_list=data_kernel_type_list,
                    sample_kernel_weights=sample_kernel_weights,
                    lengthscale_range=lengthscale_range,
                    std_range=std_range,
                    min_rank=min_rank,
                    max_rank=max_rank,
                    p_iso=p_iso,
                    jitter=jitter,
                    max_tries=max_tries,
                    device=device,
                )
            except Exception:
                print("Exception when sampling from gps.")
                continue

            if y is not None and not torch.isnan(y).any() and not torch.isinf(y).any():
                break

        y_list.append(y)

    y_stacked = torch.stack(y_list, dim=0)  # [B, M, y_dim]

    return y_stacked
