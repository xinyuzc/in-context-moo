"""Create *discrete* environment based on batched GP samples.

Key functions:
    1. init(): Sample initial observations
    2. step(): Evaluate function at new inputs, update context, compute hypervolume and regret
"""

import random
from typing import Optional, List, Tuple, Union

from einops import repeat
import torch
from torch import Tensor
import numpy as np

from gpytorch.mlls import SumMarginalLogLikelihood

from gpytorch.models import IndependentModelList
from gpytorch.likelihoods import GaussianLikelihood, LikelihoodList
from gpytorch.settings import fast_pred_var, cholesky_jitter, cholesky_max_tries

from data.sampler import (
    gp_sampler,
    sample_nc,
    _get_data_kernel,
    _sample_lengthscale,
    _check_samples,
    DATA_KERNEL_TYPE_LIST,
    SAMPLE_KERNEL_WEIGHTS,
    L_RANGE,
    STD_RANGE,
    P_ISO,
    JITTER,
    MAX_TRIES,
)
from data.base.gpytorch_utils import ExactGPModel, sample_orthonormal_matrix
from data.base.preprocessing import transform, make_range_tensor
from data.function_sampling import sample_domain, generate_sobol_samples
from data.base.masking import generate_dim_mask, restore_by_mask, gather_by_indices
from data.moo import MOO
from data.sampler_global_opt import OptimizationSampler
from data.gp_nuts_sampler import multi_output_gp_posterior_sampler_nuts
from utils.dataclasses import DataConfig
from utils.config import get_train_y_range

NUM_SAMPLES = 1


def multi_output_gp_posterior_sampler(
    x_ctx: Tensor,                                          # [n_ctx, x_dim]
    y_ctx: Tensor,                                          # [n_ctx, num_tasks]
    x_range: Union[List, Tensor],
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
    optimize_hyperparams: bool = False,
    **kwargs,
):
    """Sample from independent multi-output GP posteriors, one per task.

    Mirrors `multi_output_gp_prior_sampler` in data/sampler.py
    
    Args:
        x_ctx: [n_ctx, x_dim] real context inputs (shared across tasks).
        y_ctx: [n_ctx, num_tasks] real context outputs.
        x_range, num_datapoints, grid: passed to `generate_sobol_samples` to
            build the test grid `x` if `x` is None.
        x: [num_datapoints, x_dim] optional precomputed test inputs.
        Other args: same role as in `multi_output_gp_prior_sampler`.

    Returns:
        (x, y) with shapes [num_datapoints, x_dim] and [num_datapoints, num_tasks],
        or (None, None) if the sample fails the validity check.
    """
    assert lengthscale_range[0] > 0 and lengthscale_range[1] > lengthscale_range[0]
    assert std_range[0] > 0 and std_range[1] > std_range[0]
    assert x_ctx.dim() == 2 and y_ctx.dim() == 2
    assert x_ctx.shape[0] == y_ctx.shape[0], "x_ctx / y_ctx must share n_ctx"

    x_dim = x_ctx.shape[-1]
    num_tasks = y_ctx.shape[-1]

    x_range = make_range_tensor(x_range, x_dim).to(device)
    x_ctx = x_ctx.to(device)
    y_ctx = y_ctx.to(device)

    # Test inputs
    if x is None:
        x = generate_sobol_samples(
            x_range=x_range, num_datapoints=num_datapoints, grid=grid
        )
    x = x.to(device)

    # Sample data kernel per task
    data_kernel_type = random.choices(
        population=data_kernel_type_list, weights=sample_kernel_weights, k=num_tasks
    )

    models = []
    likelihoods = []

    for t, kernel_type in enumerate(data_kernel_type):
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

        # Conditioned model for task t
        likelihood = GaussianLikelihood()
        likelihoods.append(likelihood)
        model = ExactGPModel(
            kernel=data_kernel,
            likelihood=likelihood,
            train_x=x_ctx,
            train_y=y_ctx[:, t],
        )
        model.covar_module.outputscale = std**2
        models.append(model)

    model = IndependentModelList(*models)
    likelihood = LikelihoodList(*likelihoods)

    model.to(x)
    likelihood.to(x)

    if optimize_hyperparams:
        # Joint MLE across all tasks (SumMarginalLogLikelihood)
        model.train()
        likelihood.train()
        mll = SumMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        train_inputs = [m.train_inputs[0] for m in model.models]
        train_targets = [m.train_targets for m in model.models]
        n_opt_steps = 50
        with cholesky_jitter(jitter), cholesky_max_tries(max_tries):
            for _ in range(n_opt_steps):
                optimizer.zero_grad()
                outputs = model(*train_inputs)
                loss = -mll(outputs, train_targets)
                loss.backward()
                optimizer.step()

    model.eval()
    likelihood.eval()

    # Sample from the posterior at x
    with torch.no_grad(), fast_pred_var(), cholesky_jitter(jitter), cholesky_max_tries(
        max_tries
    ):
        post_dist_list = model(*[x for _ in range(num_tasks)])

        ys = [
            post_dist.sample(torch.Size([1])).squeeze(0)
            for post_dist in post_dist_list
        ]
        y = torch.stack(ys, dim=-1)  # [num_datapoints, num_tasks]

    # Validity check (matches prior sampler)
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


def prepare_prediction_batches(
    x: Tensor,
    y: Tensor,
    valid_x_counts: Union[Tensor, int],
    valid_y_counts: Union[Tensor, int],
    dim_scatter_mode: str,
    min_nc: int,
    max_nc: int,
    nc_fixed: Optional[int] = None,
    warmup: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    """Prepare prediction batch from full x, y tensors.

    Args:
        x: [B, N, max_x_dim]
        y: [B, N, max_y_dim]
        valid_x_counts: [B] | int
        valid_y_counts: [B] | int
        dim_scatter_mode: ["random_k", "top_k"]
        min_nc: Minimum number of contexts
        max_nc: Maximum number of contexts
        nc_fixed: Optional fixed number of contexts, if any
        warmup: Whether to use warmup when sampling nc

    Returns:
        x: (Rearranged) x, [B, N, max_x_dim]
        y: (Rearranged) y, [B, N, max_y_dim]
        x_mask: [B, max_x_dim] | [max_x_dim]
        y_mask: [B, max_y_dim] | [max_y_dim]
        nc: Context size (int)
    """
    max_x_dim = x.shape[-1]
    max_y_dim = y.shape[-1]

    # Generate dimension masks
    x_mask, x_indices = generate_dim_mask(
        max_dim=max_x_dim,
        device=x.device,
        k=valid_x_counts,
        dim_scatter_mode=dim_scatter_mode,
    )

    y_mask, y_indices = generate_dim_mask(
        max_dim=max_y_dim,
        device=y.device,
        k=valid_y_counts,
        dim_scatter_mode=dim_scatter_mode,
    )

    # Rearrange tensor by indices
    x = gather_by_indices(x, x_indices)
    y = gather_by_indices(y, y_indices)

    if nc_fixed is None:
        max_vcount = torch.max(valid_x_counts)
        nc = sample_nc(min_nc=min_nc, max_nc=max_nc, x_dim=max_vcount, warmup=warmup)
    else:
        nc = nc_fixed

    # Randomly split into context and target
    perm = torch.randperm(x.shape[1], device=x.device)
    idx1, idx2 = perm[:nc], perm[nc:]
    xc = x[:, idx1]
    yc = y[:, idx1]
    xt = x[:, idx2]
    yt = y[:, idx2]

    return xc, yc, xt, yt, x_mask, y_mask


class GPSampleFunction:
    """Discrete Function environment built on batched GP samples.
    NOTE Only support sampling on joint design space.
    """

    def __init__(
        self,
        data_config: DataConfig,
        batch_size: int,
        d: int,
        use_grid_sampling: bool,
        use_factorized_policy: bool,
        x_dim: Optional[int] = None,
        y_dim: Optional[int] = None,
        num_samples: int = NUM_SAMPLES,
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        restore_full_dim_later: bool = True,
        x_ctx: Optional[Tensor] = None,
        y_ctx: Optional[Tensor] = None,
        kernel_types: Optional[List[str]] = None,
        num_gps: int = 1,
        num_fns: int = 1,
        use_independent_sampler: bool = False,
        optimize_hyperparams: bool = False,
        nuts_warmup: int = 128,
        **kwargs,
    ):
        assert not use_factorized_policy, "use_factorize_policy=True is not supported."
        # Randomly sample x and y dim if not specified
        x_dim = random.choice(data_config.x_dim_list) if x_dim is None else x_dim
        y_dim = random.choice(data_config.y_dim_list) if y_dim is None else y_dim

        self.max_x_dim = data_config.max_x_dim
        self.max_y_dim = data_config.max_y_dim

        x_ctx_tiled, y_ctx_tiled = None, None
        if x_ctx is not None and y_ctx is not None:
            # y_ctx is already on the model's training scale, so the GP posterior
            # samples inherit that scale. Re-standardising per draw destroys the
            # GP's variance structure and makes y_ctx and the synthetic targets
            # disagree about the function value at the conditioning points.
            x, y, chunks_, chunk_mask_, x_ctx_tiled, y_ctx_tiled = (
                self.sample_from_post(
                    x_ctx=x_ctx,
                    y_ctx=y_ctx,
                    num_test_points=d,
                    num_fns=num_fns,
                    num_gps=num_gps,
                    kernel_types=kernel_types or data_config.data_kernel_type_list,
                    x_range=data_config.x_range,
                    use_grid_sampling=use_grid_sampling,
                    device=device,
                    use_independent_sampler=use_independent_sampler,
                    optimize_hyperparams=optimize_hyperparams,
                    nuts_warmup=nuts_warmup,
                    lengthscale_range=data_config.lengthscale_range,
                    std_range=data_config.std_range,
                    p_iso=data_config.p_iso,
                    jitter=data_config.jitter,
                    max_tries=data_config.max_tries,
                    sample_kernel_weights=data_config.sample_kernel_weights,
                )
            )
        else:
            x, y, chunks_, chunk_mask_ = self.sample_from_syn(
                x_dim=x_dim,
                y_dim=y_dim,
                batch_size=batch_size,
                d=d,
                use_grid_sampling=use_grid_sampling,
                use_factorized_policy=use_factorized_policy,
                x_range=data_config.x_range,
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
                standardize=data_config.standardize,
                device=device,
                sampler_type=data_config.function_name,
            )

        # Generate valid dim masks: [max_x_dim], [max_y_dim]
        self.x_mask, _ = generate_dim_mask(
            k=x_dim,
            max_dim=data_config.max_x_dim,
            dim_scatter_mode=data_config.dim_scatter_mode,
            device=device,
        )
        self.y_mask, _ = generate_dim_mask(
            k=y_dim,
            max_dim=data_config.max_y_dim,
            dim_scatter_mode=data_config.dim_scatter_mode,
            device=device,
        )

        # Restore full dimensions later
        if restore_full_dim_later:
            self.chunks_, self.chunk_mask_ = chunks_, chunk_mask_
        else:
            x, self.chunks_, self.chunk_mask_ = self.prepare_full_dimensions(
                mask=self.x_mask, tensors=[x, chunks_, chunk_mask_]
            )
            y = self.prepare_full_dimensions(mask=self.y_mask, tensors=[y])[0]

        self.restore_full_dim_later = restore_full_dim_later

        self.x_dim = x_dim
        self.y_dim = y_dim
        self._num_samples = num_samples
        self.num_points = x.shape[1]
        self.batch_size = x.shape[0] * num_samples

        self._x = self.repeat_along_batch(x, self._num_samples)
        self._y = self.repeat_along_batch(y, self._num_samples)

        if x_ctx_tiled is not None:
            x_ctx_tiled = self.repeat_along_batch(x_ctx_tiled, self._num_samples)
            y_ctx_tiled = self.repeat_along_batch(y_ctx_tiled, self._num_samples)
            self.x_ctx_padded = restore_by_mask(x_ctx_tiled, self.x_mask, dim=-1)
            self.y_ctx_padded = restore_by_mask(y_ctx_tiled, self.y_mask, dim=-1)
        else:
            self.x_ctx_padded = None
            self.y_ctx_padded = None


        # Pre-compute max hypervolume for regret calculation over *both context and candidate points*
        # [B]
        y_pool = self._y
        if y_ctx_tiled is not None:
            y_ctx_for_pool = (
                y_ctx_tiled
                if restore_full_dim_later
                else restore_by_mask(y_ctx_tiled, self.y_mask, dim=-1)
            )
            y_pool = torch.cat([y_pool, y_ctx_for_pool.to(y_pool)], dim=1)

        self.max_hv, _, _ = MOO.compute_hv(
            solutions=y_pool,
            minimum=torch.min(y_pool, dim=1).values,
            maximum=torch.max(y_pool, dim=1).values,
            y_mask=None if restore_full_dim_later else self.y_mask,
            normalize=False,
        )
        self.max_hv_norm, _, _ = MOO.compute_hv(
            solutions=y_pool,
            minimum=torch.min(y_pool, dim=1).values,
            maximum=torch.max(y_pool, dim=1).values,
            y_mask=None if restore_full_dim_later else self.y_mask,
            normalize=True,
        )

        # Pre-compute min and max for each objective
        y_mins_ = torch.min(y_pool, dim=1).values  # [B, dim]
        y_maxs_ = torch.max(y_pool, dim=1).values

        # [B, max_y_dim], [B, max_x_dim]
        self.y_mins = restore_by_mask(data=y_mins_, mask=self.y_mask, dim=-1)
        self.y_maxs = restore_by_mask(data=y_maxs_, mask=self.y_mask, dim=-1)

    @property
    def chunks(self):
        """[d, max_x_dim]"""
        return restore_by_mask(data=self.chunks_, mask=self.x_mask, dim=-1)

    @property
    def chunk_mask(self):
        """[num_chunks, max_x_dim]"""
        return restore_by_mask(data=self.chunk_mask_, mask=self.x_mask, dim=-1)

    @staticmethod
    def sample_from_syn(
        x_dim: int,
        y_dim: int,
        batch_size: int,
        d: int,
        use_grid_sampling: bool,
        use_factorized_policy: bool,
        device: str,
        x_range,
        sampler_list,
        sampler_weights,
        data_kernel_type_list,
        sample_kernel_weights,
        lengthscale_range,
        std_range,
        min_rank,
        max_rank,
        p_iso,
        jitter,
        max_tries,
        standardize: bool = True,
        seed: int = 0,
        range_scale: List[float] = get_train_y_range(),
        sampler_type: str = "gp",
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample batched synthetic function from GPs.

        Returns:
            x: Input points, [b, m, dx]
            y: Output points, [b, m, dy]
            chunks: Chunks of input points, [d, dx]
            chunk_mask: Mask for valid chunks, [num_chunks, dx]
        """
        x_range_tensor = make_range_tensor(range_list=x_range, num_dim=x_dim).to(
            device=device
        )

        # Sample input locations shared in the batch
        if use_factorized_policy:
            x, chunks, chunk_mask = sample_domain(
                d=d,
                max_x_dim=x_dim,
                device=device,
                input_bounds=x_range,
                use_grid_sampling=use_grid_sampling,
                use_factorized_policy=use_factorized_policy,
                seed=seed,
            )
        else:
            x = generate_sobol_samples(
                x_range=x_range_tensor, num_datapoints=d, grid=use_grid_sampling
            )
            chunks = x.clone()
            chunk_mask = torch.ones(
                x.shape[-1], device=x.device, dtype=torch.bool
            ).unsqueeze(0)

        if sampler_type == "gp":
            x = x.unsqueeze(0).expand(batch_size, -1, -1)  # [b, m, dx]

            # Sample on GPs
            y = gp_sampler(
                x=x,
                x_range=x_range,
                y_dim=y_dim,
                sampler_list=sampler_list,
                sampler_weights=sampler_weights,
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
        elif sampler_type == "opt":
            sampler = OptimizationSampler(
                data_kernel_type_list=data_kernel_type_list,
                sample_kernel_weights=sample_kernel_weights,
                lengthscale_range=lengthscale_range,
                std_range=std_range,
                p_iso=p_iso,
                device=device,
            )

            _y_list = []
            for _ in range(y_dim):
                y = sampler.sample(
                    batch_size=batch_size,
                    max_num_ctx_points=None,
                    num_total_points=x.shape[0],
                    x_range=x_range_tensor,
                    grid=use_grid_sampling,
                    x=x,
                )[1]
                _y_list.append(y)

            x = x.unsqueeze(0).expand(batch_size, -1, -1)  # [b, m, dx]
            y = torch.cat(_y_list, dim=-1)  # [b, m, dy]
            del _y_list
        else:
            raise ValueError(sampler_type)

        if standardize:
            mins = torch.min(y, dim=1).values
            maxs = torch.max(y, dim=1).values

            # bounds: [b, dy, 2]
            inp_bounds = torch.stack([mins, maxs], dim=-1)
            out_bounds = make_range_tensor(range_list=range_scale, num_dim=y_dim)
            out_bounds = out_bounds.to(device=device)

            y = transform(
                data=y,
                inp_bounds=inp_bounds,
                out_bounds=out_bounds,
                transform_method="min_max",
            )
        return x, y, chunks, chunk_mask

    @staticmethod
    def sample_from_post(
        x_ctx: Tensor,
        y_ctx: Tensor,
        num_test_points: int,
        num_fns: int,
        num_gps: int,
        kernel_types: List[str],
        x_range,
        use_grid_sampling: bool = False,
        device: str = "cuda",
        seed: int = 0,
        use_independent_sampler: bool = False,
        optimize_hyperparams: bool = False,
        nuts_warmup: int = 128,
        lengthscale_range: List = L_RANGE,
        std_range: List = STD_RANGE,
        p_iso: float = P_ISO,
        jitter: float = JITTER,
        max_tries: int = MAX_TRIES,
        sample_kernel_weights: Optional[List] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample functions from a GP posterior conditioned on (x_ctx, y_ctx).

        x_ctx: [B, N, dx], y_ctx: [B, N, dy].

        Sampling methods:
            * Default (fully-Bayesian, NUTS): per (b, task), run NUTS to draw
              num_gps*num_fns kernel hyperparameter samples from
              p(theta | x_ctx[b], y_ctx[b, :, t]); for each theta, condition on
              the data and draw one function value at x_test. Diversity comes
              from theta variability, not just predictive noise.
            * `use_independent_sampler=True`: per-task ExactGPs with
              hyperparams sampled from priors, conditioned on (x_ctx[b],
              y_ctx[b]). If `optimize_hyperparams=True`, MLE-optimizes the
              sampled hyperparams via a SumMarginalLogLikelihood across tasks
              before drawing.

        Returns:
            x, y of shape [num_gps*num_fns*B, num_test_points, *].
            chunks, chunk_mask.
            x_ctx_tiled, y_ctx_tiled of shape [num_gps*num_fns*B, N, *],
                aligned with the batch order of x / y so callers can pair each
                synthetic function with the real observations its GP was fit on.
        """
        B_outer = x_ctx.shape[0]
        dx, dy = x_ctx.shape[-1], y_ctx.shape[-1]
        n_ctx = x_ctx.shape[-2]

        # Sample test points
        x_range_t = make_range_tensor(range_list=x_range, num_dim=dx).to(device=device)
        x_test = generate_sobol_samples(
            x_range=x_range_t,
            num_datapoints=num_test_points,
            grid=use_grid_sampling,
            seed=seed,
        )
        out_dtype = x_test.dtype

        num_draws = num_gps * num_fns
        weights = (
            sample_kernel_weights
            if sample_kernel_weights is not None
            else [1] * len(kernel_types)
        )

        if use_independent_sampler:
            # Per-task ExactGP path. Each (b, draw) call resamples hyperparams
            # (used as init when optimize_hyperparams=True) and conditions on
            # x_ctx[b], y_ctx[b], producing one joint draw at x_test.
            ys = []
            for b in range(B_outer):
                for _ in range(num_draws):
                    while True:
                        try:
                            _, y_bi = multi_output_gp_posterior_sampler(
                                x_ctx=x_ctx[b],
                                y_ctx=y_ctx[b],
                                x_range=x_range,
                                num_datapoints=num_test_points,
                                data_kernel_type_list=kernel_types,
                                sample_kernel_weights=weights,
                                lengthscale_range=lengthscale_range,
                                std_range=std_range,
                                p_iso=p_iso,
                                grid=use_grid_sampling,
                                device=device,
                                x=x_test,
                                jitter=jitter,
                                max_tries=max_tries,
                                optimize_hyperparams=optimize_hyperparams,
                            )
                        except Exception:
                            print("Exception when sampling from posterior.")
                            continue
                        if (
                            y_bi is not None
                            and not torch.isnan(y_bi).any()
                            and not torch.isinf(y_bi).any()
                        ):
                            break
                    ys.append(y_bi)
            # Order is (b, draw) to match downstream reshape.
            y = torch.stack(ys, dim=0).to(out_dtype)
        else:
            # Default: fully-Bayesian path, vectorized over (b, task)
            y_full = multi_output_gp_posterior_sampler_nuts(
                x_ctx=x_ctx,
                y_ctx=y_ctx,
                x_test=x_test,
                num_draws=num_draws,
                kernel_types=kernel_types,
                sample_kernel_weights=weights,
                std_range=std_range,
                nuts_warmup=nuts_warmup,
                jitter=jitter,
                max_tries=max_tries,
                device=device,
            )  # [B_outer, num_draws, T, dy]
            # Flatten (b, draw) -> (B_outer * num_draws) to match downstream order.
            y = y_full.reshape(-1, num_test_points, dy).to(out_dtype)

        x = x_test.unsqueeze(0).expand(y.shape[0], -1, -1)

        chunks = x_test.clone()
        chunk_mask = torch.ones(dx, device=x_test.device, dtype=torch.bool).unsqueeze(0)

        # Tile real observations to match y's batch order: y[i] comes from gp
        # (b, g) sample f with i = (b*num_gps + g)*num_fns + f, so each x_ctx[b]
        # must appear num_gps*num_fns times consecutively.
        x_ctx_tiled = x_ctx.repeat_interleave(num_gps * num_fns, dim=0)
        y_ctx_tiled = y_ctx.repeat_interleave(num_gps * num_fns, dim=0)

        return x, y, chunks, chunk_mask, x_ctx_tiled, y_ctx_tiled

    @staticmethod
    def repeat_along_batch(tensor: Tensor, num_repeat: int):
        """Repeat each element in the batch dimension `num_repeat` times."""
        if tensor.ndim < 2:
            raise ValueError("Expected at least 2 dimensions: (batch + features).")
        if num_repeat == 1:
            return tensor

        expanded = tensor.unsqueeze(1).expand(-1, num_repeat, *tensor.shape[1:])
        reshaped = expanded.reshape(-1, *tensor.shape[1:])
        return reshaped

    @staticmethod
    def prepare_full_dimensions(mask: Tensor, tensors: List[Tensor]) -> List[Tensor]:
        full_tensors = [
            restore_by_mask(data=tensor, mask=mask, dim=-1) for tensor in tensors
        ]
        return full_tensors

    @staticmethod
    def update_context(new: Tensor, old: Optional[Tensor]) -> Tensor:
        """Update context with new observations.
        - If old is None, return new [B, num_new, DY]
        - Else concatenate old and new, return [B, num_old + num_new, DY]
        """
        if old is None:
            return new
        else:
            batch_size_old, _, dim_old = old.shape
            batch_size_new, _, dim_new = new.shape

            assert batch_size_new == batch_size_old
            assert dim_new == dim_old

            return torch.cat((old, new), dim=1)

    def compute_hv(self, solutions: Tensor, normalize: bool = False) -> np.ndarray:
        """Compute hypervolume.

        Args:
            solutions: [B, N, dy_max]
            normalize: Whether to normalize sols and ref_points before computing hv

        Returns:
            reward: shape [B]
            sols_tfm: (Optionally transformed) solutions, [B, N, dy_max]
            reward_ref_points: reference points, [B, dy_max]
        """
        reward, sols_tfm, ref_points = MOO.compute_hv(
            solutions=solutions,
            minimum=self.y_mins,
            maximum=self.y_maxs,
            y_mask=self.y_mask,
            normalize=normalize,
        )

        return reward, sols_tfm, ref_points

    def compute_regret(
        self, solutions: Tensor, regret_type: str = "ratio"
    ) -> np.ndarray:
        """Compute regret.

        Args:
            solutions: [B, N, dy_max]
            regret_type: in ["value", "ratio", "norm_ratio"]

        Returns:
            regret: [B]
        """
        regret_np = MOO.compute_regret(
            solutions=solutions,
            minimum=self.y_mins,  # [B, dy_max]
            maximum=self.y_maxs,
            regret_type=regret_type,
            y_mask=self.y_mask,
            max_hv=self.max_hv,
            max_hv_norm=self.max_hv_norm,
        )
        return regret_np

    @staticmethod
    def batch_gather(tensor, dim, index, full_dim_mask=None) -> Tensor:
        """Gather tensor along dim by index, optionally restore full data dimensions."""
        index_expanded = index.expand(-1, -1, tensor.shape[-1])
        tensor_gathered = torch.gather(tensor, dim=dim, index=index_expanded)
        if full_dim_mask is not None:
            tensor_gathered = restore_by_mask(
                data=tensor_gathered, mask=full_dim_mask, dim=-1
            )
        return tensor_gathered

    def step(
        self,
        index_new: Tensor,
        x_ctx: Optional[Tensor] = None,
        y_ctx: Optional[Tensor] = None,
        compute_hv: bool = True,
        compute_regret: bool = True,
        regret_type: str = "ratio",
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[np.ndarray], Optional[np.ndarray]]:
        """Evaluate function input `x_new`, update context, optionally compute reward / regret.

        Args:
            index_new: Index of new datapoints in training data, [B, num_new, 1]
            x_ctx: Optional context input points, shape [B, num_ctx, dx_max]
            y_ctx: Optional context output points, shape [B, num_ctx, dy_max]
            compute_reward: compute reward from choosing `x_new` if True
            compute_regret: compute regret from choosing `x_new` if True
            regret_type: Type of regret to compute, defaults to "ratio"

        Returns:
            x_ctx: Updated context input points, shape [B, num_ctx + num_new, dx_max]
            y_ctx: Updated context output points, shape [B, num_ctx + num_new, dy_max]
            reward (np.ndarray): Reward from choosing `x_new`, shape [B] or None if not computed
            regret (np.ndarray): Regret from choosing `x_new`, shape [B] or None if not computed
        """
        x_new = self.batch_gather(
            tensor=self._x,
            dim=1,
            index=index_new,
            full_dim_mask=self.x_mask if self.restore_full_dim_later else None,
        )
        y_new = self.batch_gather(
            tensor=self._y,
            dim=1,
            index=index_new,
            full_dim_mask=self.y_mask if self.restore_full_dim_later else None,
        )

        x_ctx = GPSampleFunction.update_context(new=x_new, old=x_ctx)
        y_ctx = GPSampleFunction.update_context(new=y_new, old=y_ctx)

        reward, regret = None, None
        if compute_hv:
            reward = self.compute_hv(y_ctx)[0]
        if compute_regret:
            regret = self.compute_regret(y_ctx, regret_type)

        return x_ctx, y_ctx, reward, regret

    def init(
        self,
        num_initial_points: int,
        regret_type: str,
        compute_hv: bool = True,
        compute_regret: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Sample initial points and evaluate function at them.

        Returns:
            x_ctx: [B, num_initial_points, dx_max]
            y_ctx: [B, num_initial_points, dy_max]
            reward (np.ndarray): [B] or None
            regret (np.ndarray): [B] or None
        """
        # Randomly choose initial points from the pool
        indices = torch.randperm(self.num_points, device=device)[:num_initial_points]
        indices = repeat(indices, "n -> b n 1", b=self.batch_size)

        # Evaluate at initial points
        x_ctx, y_ctx, reward, regret = self.step(
            index_new=indices,
            x_ctx=None,
            y_ctx=None,
            compute_hv=compute_hv,
            compute_regret=compute_regret,
            regret_type=regret_type,
        )

        return x_ctx, y_ctx, reward, regret


if __name__ == "__main__":
    device = "cpu"

    # ── Test 1: repeat_along_batch ──────────────────────────────────
    print("Test 1: repeat_along_batch")
    t = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  # [2, 2, 2]
    out = GPSampleFunction.repeat_along_batch(t, num_repeat=3)
    assert out.shape == (6, 2, 2), f"Expected (6,2,2), got {out.shape}"
    # Batch 0 repeats 3 times, then batch 1 repeats 3 times
    assert torch.equal(out[0], out[1]) and torch.equal(out[1], out[2])
    assert torch.equal(out[3], out[4]) and torch.equal(out[4], out[5])
    assert torch.equal(out[0], t[0]) and torch.equal(out[3], t[1])
    # num_repeat=1 returns same tensor
    out1 = GPSampleFunction.repeat_along_batch(t, num_repeat=1)
    assert torch.equal(out1, t)
    # 1D tensor should raise
    try:
        GPSampleFunction.repeat_along_batch(torch.tensor([1.0, 2.0]), 2)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  PASSED")

    # ── Test 2: update_context ──────────────────────────────────────
    print("Test 2: update_context")
    new = torch.randn(4, 3, 5)
    # old=None → return new
    result = GPSampleFunction.update_context(new=new, old=None)
    assert torch.equal(result, new)
    # old provided → concatenate along dim 1
    old = torch.randn(4, 2, 5)
    result = GPSampleFunction.update_context(new=new, old=old)
    assert result.shape == (4, 5, 5), f"Expected (4,5,5), got {result.shape}"
    assert torch.equal(result[:, :2], old)
    assert torch.equal(result[:, 2:], new)
    print("  PASSED")

    # ── Test 3: batch_gather ────────────────────────────────────────
    print("Test 3: batch_gather")
    B, N, D = 2, 10, 3
    tensor = torch.arange(B * N * D, dtype=torch.float).reshape(B, N, D)
    index = torch.tensor([[[0], [2]], [[1], [3]]])  # [2, 2, 1]
    gathered = GPSampleFunction.batch_gather(tensor, dim=1, index=index)
    assert gathered.shape == (2, 2, 3), f"Expected (2,2,3), got {gathered.shape}"
    assert torch.equal(gathered[0, 0], tensor[0, 0])
    assert torch.equal(gathered[0, 1], tensor[0, 2])
    assert torch.equal(gathered[1, 0], tensor[1, 1])
    assert torch.equal(gathered[1, 1], tensor[1, 3])
    # With mask restoration
    mask = torch.tensor([True, False, True, False, True])  # 3 valid → 5 dims
    gathered_masked = GPSampleFunction.batch_gather(
        tensor, dim=1, index=index, full_dim_mask=mask
    )
    assert gathered_masked.shape == (
        2,
        2,
        5,
    ), f"Expected (2,2,5), got {gathered_masked.shape}"
    # Masked-out dims should be zero
    assert (gathered_masked[:, :, 1] == 0).all()
    assert (gathered_masked[:, :, 3] == 0).all()
    print("  PASSED")

    # ── Test 4: prepare_full_dimensions ─────────────────────────────
    print("Test 4: prepare_full_dimensions")
    mask = torch.tensor([True, False, True])  # valid_dim=2, max_dim=3
    data = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [1, 2, 2]
    result = GPSampleFunction.prepare_full_dimensions(mask, [data])
    assert len(result) == 1
    assert result[0].shape == (1, 2, 3)
    assert result[0][0, 0, 0] == 1.0
    assert result[0][0, 0, 1] == 0.0  # masked out
    assert result[0][0, 0, 2] == 2.0
    print("  PASSED")

    # ── Test 5: prepare_prediction_batches ──────────────────────────
    print("Test 5: prepare_prediction_batches")
    B, N, max_x_dim, max_y_dim = 4, 20, 5, 3
    x = torch.randn(B, N, max_x_dim)
    y = torch.randn(B, N, max_y_dim)
    xc, yc, xt, yt, x_mask, y_mask = prepare_prediction_batches(
        x=x,
        y=y,
        valid_x_counts=3,
        valid_y_counts=2,
        dim_scatter_mode="top_k",
        min_nc=2,
        max_nc=10,
        nc_fixed=5,
    )
    assert xc.shape == (B, 5, max_x_dim), f"xc shape: {xc.shape}"
    assert yc.shape == (B, 5, max_y_dim), f"yc shape: {yc.shape}"
    assert xt.shape == (B, N - 5, max_x_dim), f"xt shape: {xt.shape}"
    assert yt.shape == (B, N - 5, max_y_dim), f"yt shape: {yt.shape}"
    # x_mask should have exactly 3 True values
    assert x_mask.sum().item() == 3
    assert y_mask.sum().item() == 2
    print("  PASSED")

    # ── Test 6: GPSampleFunction end-to-end ─────────────────────────
    print("Test 6: GPSampleFunction end-to-end")
    data_config = DataConfig(
        function_name="gp",
        max_x_dim=4,
        max_y_dim=3,
        x_dim_list=[2],
        y_dim_list=[2],
        dim_scatter_mode="top_k",
    )
    batch_size = 2
    env = GPSampleFunction(
        data_config=data_config,
        batch_size=batch_size,
        d=8,
        use_grid_sampling=True,
        use_factorized_policy=False,
        x_dim=2,
        y_dim=2,
        num_samples=1,
        device=device,
    )
    assert env.batch_size == batch_size
    assert env._x.shape[0] == batch_size
    assert env._x.shape[2] == 2  # x_dim (before full-dim restore)
    assert env._y.shape[2] == 2  # y_dim
    assert env.max_hv.shape == (batch_size,)
    print(f"  num_points={env.num_points}, max_hv={env.max_hv}")

    # Test init
    x_ctx, y_ctx, reward, regret = env.init(
        num_initial_points=2,
        regret_type="ratio",
        device=device,
    )
    assert x_ctx.shape == (batch_size, 2, data_config.max_x_dim)
    assert y_ctx.shape == (batch_size, 2, data_config.max_y_dim)
    assert reward.shape == (batch_size,)
    assert regret.shape == (batch_size,)
    print(f"  init: reward={reward}, regret={regret}")

    # Test step
    idx = torch.randint(0, env.num_points, (batch_size, 1, 1))
    x_ctx2, y_ctx2, reward2, regret2 = env.step(
        index_new=idx,
        x_ctx=x_ctx,
        y_ctx=y_ctx,
        regret_type="ratio",
    )
    assert x_ctx2.shape == (batch_size, 3, data_config.max_x_dim)
    assert y_ctx2.shape == (batch_size, 3, data_config.max_y_dim)
    print(f"  step: reward={reward2}, regret={regret2}")
    print("  PASSED")

    print("\nAll tests passed!")
