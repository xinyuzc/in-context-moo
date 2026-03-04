"""Evaluation script for optimization and prediction.

Functions:
    evaluate_optimization(): Evaluate optimization on a test function
    evaluate_prediction(): Evaluate prediction on datasets
"""

import gc
import math
import os.path as osp
from typing import List, Optional, Tuple, Dict, Any

import torch
from torch import Tensor
import wandb
from tqdm import tqdm
from einops import repeat

from model import TAMO

from utils.config import build_dataloader, get_train_x_range, get_train_y_range
from utils.seed import set_all_seeds
from utils.dataclasses import (
    ExConfig,
    DataConfig,
    PredictionConfig,
    OptimizationConfig,
    LogConfig,
)
from utils.log import Averager, MetricTracker
from utils.plot import plot_1d, plot_acq_values, plot_prediction_batch
from utils.types import NestedFloatList
from utils.save import save_data, save_fig

from data.obs_tracker import ObservationTracker
from data.dataset import MultiFileHDF5Dataset
from data.base.preprocessing import make_range_nested_list, has_nan_or_inf
from data.function import TestFunction
from data.function_sampling import get_num_subspace_points
from data.gp_sample_function import prepare_prediction_batches

from forwards import select_next_query_wrapper, prediction_forward

PLACEHOLDER_NLL_VALUE = -1.0


class OptimizationLogger:
    """Handles structured logging for optimization process."""

    def __init__(self, log_fn: callable = print, use_wandb: bool = False):
        self.log = log_fn
        self.use_wandb = use_wandb

    def log_step(self, step: int, observation_tracker: ObservationTracker, metrics: MetricTracker):
        """Log information for current optimization step."""
        latest = metrics.get_latest_values()

        log_line = f"\n{'='*60}\n"
        log_line += f"Step {step}\n"
        log_line += f"{observation_tracker.__repr__()}\n"
        log_line += f"{'-'*60}\n"

        if latest["hv"] is not None:
            log_line += f"  Hypervolume:  {latest['hv']}\n"
        if latest["regret"] is not None:
            log_line += f"  Regret:       {latest['regret']}\n"
        if latest["entropy"] is not None:
            log_line += f"  Entropy:      {latest['entropy']}\n"

        if latest["x_query"] is not None and latest["y_query"] is not None:
            log_line += f"  Latest Query Points:\n"
            log_line += f"    x_next: {latest['x_query']}\n"
            log_line += f"    y_next: {latest['y_query']}\n"

        log_line += f"{'='*60}"
        self.log(log_line)

    def log_prediction_step(
        self,
        step: int,
        nll_t: Tensor,
        mse_t: Tensor,
    ):
        """Log prediction metrics for current step."""
        log_line = f"  [Prediction @ Step {step}]\n"
        log_line += f"    NLL Target:  {nll_t.mean().item():.4f}\n"
        log_line += f"    MSE Target:  {mse_t.mean().item():.6f}\n"
        self.log(log_line)

        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log(
                {
                    f"opt/nll_target": nll_t.mean().item(),
                    f"opt/mse_target_mean": mse_t.mean().item(),
                    f"opt/step": step,
                }
            )

    def log_summary(self, metrics: MetricTracker, test_function: TestFunction):
        """Log final summary statistics."""
        stats = metrics.get_statistics()

        log_line = f"\n{'='*60}\n"
        log_line += f"OPTIMIZATION SUMMARY\n"
        log_line += f"{'='*60}\n"
        log_line += f"Function: {test_function.function_name}\n"
        log_line += f"Dimensions: dx={test_function.x_dim}, dy={test_function.y_dim}\n"
        log_line += f"Max HV: {test_function.max_hv:.4f}\n"
        log_line += f"{'-'*60}\n"

        if "hv_final_mean" in stats:
            log_line += f"Final Hypervolume:  {stats['hv_final_mean']:.4f} ± {stats['hv_final_std']:.4f}\n"
            log_line += f"Max Hypervolume:    {stats['hv_max']:.4f}\n"

        if "regret_final_mean" in stats:
            log_line += f"Final Regret:       {stats['regret_final_mean']:.4f} ± {stats['regret_final_std']:.4f}\n"
            log_line += f"Min Regret:         {stats['regret_min']:.4f}\n"

        if "total_time" in stats:
            log_line += f"Total Time:         {stats['total_time']:.2f}s\n"
            log_line += f"Avg Time/Step:      {stats['avg_time_per_step']:.4f}s ± {stats['std_time_per_step']:.4f}s\n"

        if "nll_t_final" in stats:
            log_line += f"{'-'*60}\n"
            log_line += f"Prediction Metrics:\n"
            log_line += f"  Final NLL Target: {stats['nll_t_final']:.4f}\n"
            log_line += f"  Mean NLL Target:  {stats['nll_t_mean']:.4f} ± {stats['nll_t_std']:.4f}\n"

        log_line += f"{'='*60}"
        self.log(log_line)

        # Log summary to wandb
        if self.use_wandb:
            wandb.log({f"summary/{k}": v for k, v in stats.items()})


def _save_all_data(
    metrics: MetricTracker,
    x_ctx: Tensor,
    y_ctx: Tensor,
    data_save_path: str,
    opt_cfg: OptimizationConfig,
    exp_cfg: ExConfig,
    log: callable,
):
    """Save all metrics in organized manner."""
    # Get stacked metrics
    stacked = metrics.get_stacked_metrics()

    # Add final context
    stacked["x_ctx"] = x_ctx.detach().cpu()
    stacked["y_ctx"] = y_ctx.detach().cpu()

    # Save main metrics
    for key, val in stacked.items():
        save_data(
            data=val,
            path=data_save_path,
            config=opt_cfg,
            filename=key,
            override=exp_cfg.override,
            log=log,
        )


def _save_all_plots(
    metrics: MetricTracker,
    opt_cfg: OptimizationConfig,
    exp_cfg: ExConfig,
    plot_save_path: str,
    log: callable,
):
    """Generate and save all optimization plots."""
    stacked = metrics.get_stacked_metrics()
    batch_size = opt_cfg.batch_size

    for b in range(batch_size):
        batch_prefix = f"b{b}_" if batch_size > 1 else ""

        plots = {
            f"{batch_prefix}hv": plot_1d(
                y_vals=stacked["hv"][b],
                title="Hypervolume over Iterations",
                ylabel="Hypervolume",
            ),
            f"{batch_prefix}regret": plot_1d(
                y_vals=stacked["regret"][b],
                title="Regret over Iterations",
                ylabel="Regret",
            ),
            f"{batch_prefix}entropy": plot_1d(
                y_vals=stacked["entropy"][b],
                title="Entropy over Iterations",
                ylabel="Entropy",
            ),
            f"{batch_prefix}time": plot_1d(
                y_vals=stacked["time"],
                title="Inference Time over Iterations",
                ylabel="Time (s)",
            ),
        }

        # Add prediction plots if available
        if "nll_t" in stacked:
            plots.update(
                {
                    "nll_t": plot_1d(
                        y_vals=stacked["nll_t"],
                        title="NLL Target over Iterations",
                        ylabel="NLL Target",
                    ),
                }
            )

        # Save all plots
        for name, fig in plots.items():
            save_fig(
                fig=fig,
                path=plot_save_path,
                config=opt_cfg,
                filename=name,
                override=exp_cfg.override,
                log=log,
                log_to_wandb=exp_cfg.log_to_wandb,
            )


def evaluate_optimization(
    model: TAMO,
    plot_save_path: str,
    data_save_path: str,
    test_function: TestFunction,
    exp_cfg: ExConfig,
    opt_cfg: OptimizationConfig,
    data_cfg: DataConfig,
    log_cfg: LogConfig,
    pred_cfg: Optional[PredictionConfig] = None,
    log: callable = print,
    **kwargs,
):
    """Evaluate optimization performance on a test function.
    Args:
        model: TAMO model
        plot_save_path: Directory to save plots
        data_save_path: Directory to save data
        test_function: Test function environment
        exp_cfg: Experiment configuration
        opt_cfg: Optimization configuration
        data_cfg: Data configuration
        pred_cfg: Prediction configuration
        log: Logging function
    """
    # Set random seed; separate save path for each seed
    set_all_seeds(exp_cfg.seed)

    plot_save_path = osp.join(plot_save_path, str(exp_cfg.seed))
    data_save_path = osp.join(data_save_path, str(exp_cfg.seed))

    log(
        f"""--- Testing details ---
        Function name:\t{test_function.function_name}
        Sigma:\t{data_cfg.sigma}
        dx:\t{test_function.x_dim}
        dy:\t{test_function.y_dim}
        x bounds:\t{test_function.x_bounds}
        y bounds:\t{test_function.y_bounds}
        Max hv:\t{test_function.max_hv:.4f}
        Seed:\t{exp_cfg.seed}"""
    )

    model_x_range = make_range_nested_list(get_train_x_range(), test_function.x_dim)
    model_y_range = make_range_nested_list(get_train_y_range(), test_function.y_dim)

    log(
        f"""--- Pre-trained ranges details ---
        Model pre-trained on x_range: {model_x_range}
        Model pre-trained on y_range: {model_y_range}"""
    )

    run_optimization(
        model=model,
        test_function=test_function,
        model_x_range=model_x_range,
        model_y_range=model_y_range,
        exp_cfg=exp_cfg,
        opt_cfg=opt_cfg,
        data_cfg=data_cfg,
        pred_cfg=pred_cfg,
        log_cfg=log_cfg,
        plot_save_path=plot_save_path,
        data_save_path=data_save_path,
        log=log,
    )


def run_prediction_on_test_function(
    test_function: TestFunction,
    model: TAMO,
    x_ctx: Tensor,
    y_ctx: Tensor,
    x_mask: Tensor,
    y_mask: Tensor,
    y_mask_tar: Tensor,
    train_x_range: NestedFloatList,
    train_y_range: NestedFloatList,
    batch_size: int,
    read_cache: bool,
    num_subspace_points: int = 500,
    sigma: float = 0.0,
    plot_enabled: bool = False,
    y_mask_history: Optional[Tensor] = None,
    seed: int = 0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Dict[str, Any]]]:
    """Evaluate model predictions at a single optimization step.

    Samples target points from the test function, evaluates predictive performance
    on context and target sets, and optionally generates visualization plots.

    Args:
        test_function: Test function to sample from
        model: TAMO model
        x_ctx: Context inputs [batch, num_ctx, x_dim]
        y_ctx: Context outputs [batch, num_ctx, y_dim]
        x_mask: Input dimension mask [x_dim]
        y_mask: Observed output dimension mask [y_dim]
        y_mask_tar: Target output dimension mask [y_dim]
        train_x_range: Input bounds used during training
        train_y_range: Output bounds used during training
        batch_size: Number of parallel evaluations
        read_cache: Whether to read from prediction cache
        num_subspace_points: Points to sample in subspace (default: 500)
        sigma: Observation noise (unused, kept for API compatibility)
        plot_enabled: Whether to generate prediction plots
        y_mask_history: History of observed masks [num_steps, y_dim], for plotting
        seed: Random seed for reproducible sampling

    Returns:
        (nll_t, mse_t, figs)
    """
    device = x_mask.device

    # Step 1: Sample target points from test function
    x_tar, y_tar, _, _ = test_function.sample(
        input_bounds=train_x_range,
        batch_size=batch_size,
        num_subspace_points=num_subspace_points,
        use_grid_sampling=True,
        use_factorized_policy=False,
        device=device,
        x_mask=x_mask,
        y_mask=y_mask_tar,
        seed=seed,
    )

    # Step 2: Scale function values to model training range
    y_ctx_scaled = test_function.transform_outputs(
        outputs=y_ctx, output_bounds=train_y_range
    )
    y_tar_scaled = test_function.transform_outputs(
        outputs=y_tar, output_bounds=train_y_range
    )

    # Step 3: Expand masks to match batch size
    x_mask_exp = repeat(x_mask, "d -> b d", b=batch_size)
    y_mask_exp = repeat(y_mask, "d -> b d", b=batch_size)
    y_mask_tar_exp = repeat(y_mask_tar, "d -> b d", b=batch_size)

    # Step 4: Compute prediction metrics on target points
    nll_t, mse_t, _ = prediction_forward(
        model=model,
        x_ctx=x_ctx,
        x_tar=x_tar,
        y_ctx=y_ctx_scaled,
        y_tar=y_tar_scaled,
        x_mask=x_mask_exp,
        y_mask=y_mask_exp,
        y_mask_tar=y_mask_tar_exp,
        read_cache=read_cache,
    )

    # Step 5: Generate plots if requested
    figs = None
    if plot_enabled:
        figs = _generate_visualizations(
            model=model,
            x_ctx=x_ctx,
            y_ctx_scaled=y_ctx_scaled,
            x_tar=x_tar,
            y_tar_scaled=y_tar_scaled,
            x_mask_exp=x_mask_exp,
            y_mask_exp=y_mask_exp,
            y_mask_tar_exp=y_mask_tar_exp,
            y_mask_history=y_mask_history,
            read_cache=read_cache,
        )

    return nll_t, mse_t, figs


def _generate_visualizations(
    model: TAMO,
    x_ctx: Tensor,
    y_ctx_scaled: Tensor,
    x_tar: Tensor,
    y_tar_scaled: Tensor,
    x_mask_exp: Tensor,
    y_mask_exp: Tensor,
    y_mask_tar_exp: Tensor,
    read_cache: bool,
    y_mask_history: Optional[Tensor] = None,
) -> Dict[str, Any]:
    """Generate prediction visualizations.

    Args:
        model: TAMO model
        x_ctx: Context inputs [batch_size, num_ctx, x_dim]
        y_ctx_scaled: Scaled context outputs [batch_size, num_ctx, y_dim]
        x_tar: Target inputs [batch_size, num_tar, x_dim]
        y_tar_scaled: Scaled target outputs [batch_size, num_tar, y_dim]
        x_mask_exp: Expanded input mask [batch_size, x_dim]
        y_mask_exp: Expanded output mask [batch_size, y_dim]
        y_mask_tar_exp: Expanded target mask [batch_size, y_dim]
        y_mask_history: History of observation masks
        test_function: Test function for plotting
        read_cache: Whether to use cached predictions
        write_cache: Whether to cache predictions

    Returns:
        Dictionary with keys 'mean' and 'std' containing matplotlib figures
    """
    num_context = x_ctx.shape[1]
    figs = {}

    # Generate both mean and standard deviation plots
    for plot_mean in [True, False]:
        fig = plot_prediction_batch(
            model=model,
            nc=num_context,
            xc=x_ctx,
            yc=y_ctx_scaled,
            x=x_tar,
            y=y_tar_scaled,
            x_mask=x_mask_exp,
            y_mask=y_mask_exp,
            y_mask_tar=y_mask_tar_exp,
            read_cache=read_cache,
            y_mask_history=y_mask_history,
            plot_mean=plot_mean,
            plot_order=True,
        )
        plot_type = "mean" if plot_mean else "std"
        figs[plot_type] = fig

    return figs


def _should_plot(
    cost_used: float,
    cost_total: float,
    plot_per_n_unit_cost: int,
    plot_enabled: bool,
    init_cost: float = 1.0,
) -> bool:
    """Determine whether to plot at current optimization step.

    Args:
        cost_used: Current accumulated cost
        cost_total: Total cost budget
        plot_per_n_unit_cost: Plot frequency in cost units (<=0 disables plotting)
        plot_enabled: Whether plotting is enabled
        init_cost: Initial cost (typically after initialization)

    Returns:
        True if plotting should occur at this step
    """
    if not plot_enabled or plot_per_n_unit_cost <= 0:
        return False

    # Plot at: start, end, or periodic intervals
    # Cast to int to avoid float comparison
    cost_used = int(round(cost_used))
    cost_total = int(round(cost_total))
    init_cost = int(round(init_cost))

    is_start_step = cost_used == init_cost
    is_final_step = cost_used == cost_total
    is_periodic_step = (
        cost_used < cost_total and (cost_used - init_cost) % plot_per_n_unit_cost == 0
    )

    return is_start_step or is_final_step or is_periodic_step


def _save_prediction_plots(
    figs: Dict[str, Any],
    observation_tracker: ObservationTracker,
    x_ctx: Tensor,
    nll_t: Tensor,
    T: int,
    plot_save_path: str,
    opt_cfg: OptimizationConfig,
    exp_cfg: ExConfig,
    log: callable,
) -> None:
    """Save prediction plots with descriptive filenames.

    Generates filenames that encode information about the current optimization
    state including observed dimensions, context size, cost, and prediction quality.

    Args:
        figs: Dictionary of matplotlib figures to save
        observation_tracker: Observed mask and cost tracker
        x_ctx: Context inputs for determining context size
        nll_t: Target negative log-likelihood for filename
        T: Total optimization budget
        plot_save_path: Directory to save plots
        opt_cfg: Optimization configuration
        exp_cfg: Experiment configuration
        log: Logging function
    """
    # Extract observed dimension indices as strings
    observed_x_dims = observation_tracker.x_mask.nonzero(as_tuple=False)[:, 0].tolist()
    observed_y_dims = observation_tracker.y_mask.nonzero(as_tuple=False)[:, 0].tolist()

    x_dims_str = "".join(map(str, observed_x_dims))
    y_dims_str = "".join(map(str, observed_y_dims))

    # Compute mean NLL values for filename
    nll_t_mean = nll_t.detach().mean().item()

    # Generate descriptive filename prefix
    # Format: context_dx{dims}dy{dims}_nc{count}_t{current}T{total}_nllc{value}nllt{value}
    count = x_ctx.shape[1]
    current = observation_tracker.get_cost_used()
    filename_prefix = (
        f"context_dx{x_dims_str}dy{y_dims_str}_"
        f"nc{count}_"
        f"t{current}T{T}_"
        f"nll{nll_t_mean}"
    )

    # Save each figure with the prefix
    for plot_type, fig in figs.items():
        filename = f"{plot_type}_{filename_prefix}"
        save_fig(
            fig=fig,
            path=plot_save_path,
            config=opt_cfg,
            filename=filename,
            override=exp_cfg.override,
            log=log,
            log_to_wandb=exp_cfg.log_to_wandb,
        )


def run_optimization(
    model: TAMO,
    test_function: TestFunction,
    model_x_range: NestedFloatList,
    model_y_range: NestedFloatList,
    plot_save_path: str,
    data_save_path: str,
    exp_cfg: ExConfig,
    opt_cfg: OptimizationConfig,
    data_cfg: DataConfig,
    log_cfg: LogConfig,
    pred_cfg: Optional[PredictionConfig] = None,
    log: callable = print,
    predict: bool = True,
):
    if predict:
        assert pred_cfg is not None, "`pred_cfg` must be provided if perform prediction"

    # ------------------------------------------------------------------
    # Logging: metric tracker and logger
    # ------------------------------------------------------------------
    metrics = MetricTracker()
    logger = OptimizationLogger(log_fn=log, use_wandb=exp_cfg.log_to_wandb)

    # ------------------------------------------------------------------
    # Dimension and cost state tracker
    # ------------------------------------------------------------------
    observation_tracker = ObservationTracker(
        x_dim=test_function.x_dim,
        y_dim=test_function.y_dim,
        dim_mask_gen_mode=opt_cfg.dim_mask_gen_mode,
        single_obs_y_dim=opt_cfg.single_obs_y_dim,
        device=exp_cfg.device,
        num_initial_points=opt_cfg.num_initial_points,
        cost_mode=opt_cfg.cost_mode,
        cost=opt_cfg.cost,
    )

    # ------------------------------------------------------------------
    # Initial observations
    # ------------------------------------------------------------------
    x_ctx, y_ctx, hv, regret = test_function.init(
        input_bounds=model_x_range,
        batch_size=opt_cfg.batch_size,
        num_initial_points=opt_cfg.num_initial_points,
        regret_type=opt_cfg.regret_type,
        compute_hv=True,
        compute_regret=True,
        device=exp_cfg.device,
        seed=exp_cfg.seed,
    )

    # Record and log first step
    metrics.add_optimization_step(
        hv=hv,
        hv_query=hv.clone(),
        regret=regret,
        entropy=torch.zeros((opt_cfg.batch_size,), device=exp_cfg.device),
        time=[0.0] * opt_cfg.num_initial_points,
        x_query=x_ctx,
        y_query=y_ctx,
    )
    logger.log_step(step=0, observation_tracker=observation_tracker, metrics=metrics)

    q_chunk, q_chunk_mask, logit_mask = None, None, None

    # Number of query points and optimization horizon
    d = get_num_subspace_points(
        x_dim=test_function.x_dim,
        use_factorized_policy=opt_cfg.use_factorized_policy,
        d=opt_cfg.num_query_points,
    )
    T = opt_cfg.sample_T()
    log(f"Number of subspace points (d): {d}, Total cost budget (T): {T}")

    model = model.to(exp_cfg.device)
    model.eval()
    with torch.no_grad():
        while observation_tracker.get_cost_used() <= T:
            # Perform prediction evaluation
            if predict:
                should_plot_now = _should_plot(
                    cost_used=observation_tracker.get_cost_used(),
                    cost_total=T,
                    plot_per_n_unit_cost=log_cfg.plot_per_n_steps,
                    plot_enabled=log_cfg.plot_enabled,
                    init_cost=observation_tracker.initial_cost,
                )

                nll_t, mse_t, figs = run_prediction_on_test_function(
                    test_function=test_function,
                    model=model,
                    x_ctx=x_ctx,
                    y_ctx=y_ctx,
                    x_mask=observation_tracker.x_mask,
                    y_mask=observation_tracker.y_mask,
                    y_mask_tar=observation_tracker.y_mask_target,
                    train_x_range=model_x_range,
                    train_y_range=model_y_range,
                    batch_size=opt_cfg.batch_size,
                    read_cache=pred_cfg.read_cache,
                    sigma=data_cfg.sigma,
                    plot_enabled=should_plot_now,
                    y_mask_history=observation_tracker.y_mask_observed,
                    seed=exp_cfg.seed,
                )

                nll_t = nll_t.detach()
                mse_t = mse_t.detach()

                # Update prediction metrics
                metrics.add_prediction_step(nll_t=nll_t, mse_t=mse_t)

                # Log prediction results
                logger.log_prediction_step(
                    step=observation_tracker.get_cost_used(),
                    nll_t=nll_t,
                    mse_t=mse_t,
                )

                # Save plots with descriptive filenames
                if figs is not None:
                    _save_prediction_plots(
                        figs=figs,
                        observation_tracker=observation_tracker,
                        x_ctx=x_ctx,
                        nll_t=nll_t,
                        T=T,
                        plot_save_path=plot_save_path,
                        opt_cfg=opt_cfg,
                        exp_cfg=exp_cfg,
                        log=log,
                    )
                    del figs

            # Transform function values to model training range
            y_ctx_scaled = test_function.transform_outputs(
                outputs=y_ctx, output_bounds=model_y_range
            )

            # Select next **batch** of query points
            batch_x_next_list = []
            batch_entr_list = []
            batch_infer_time_list = []

            batch_x_ctx = x_ctx.clone()
            batch_y_ctx = y_ctx_scaled.clone()

            for qi in range(opt_cfg.q):
                # select next query point
                # If fantasy is True, context will be updated inside the function
                batch_x_ctx, batch_y_ctx, action_res = select_next_query_wrapper(
                    fantasy=opt_cfg.fantasy,
                    x_ctx=batch_x_ctx,
                    y_ctx=batch_y_ctx,
                    model=model,
                    observation_tracker=observation_tracker,
                    model_x_range=model_x_range,
                    opt_config=opt_cfg,
                    pred_config=pred_cfg,
                    d=d,
                    T=T,
                    query_chunks=q_chunk,
                    query_x_mask=q_chunk_mask,
                    logit_mask=logit_mask,
                )

                # Update batch records
                x_next = action_res.next_x
                acq_values = action_res.logits  # [B, n, d]
                entropy = action_res.entropy
                q_chunk = action_res.q_chunk
                q_chunk_mask = action_res.q_chunk_mask
                infer_time = action_res.infer_time
                logit_mask = action_res.logit_mask

                # [Tensor | scalar] x q
                batch_x_next_list.append(x_next)
                batch_entr_list.append(entropy)
                batch_infer_time_list.append(infer_time)

                # update mask only after last query in the batch
                observation_tracker.step(update_mask=(qi == opt_cfg.q - 1))

            # Concatenate batch query points
            batch_x_next = torch.cat(batch_x_next_list, dim=1)  # [B, q, max_x_dim]
            batch_entropy = torch.stack(batch_entr_list, dim=1)  # [B, q]

            # Evaluate at batch queries
            x_ctx, y_ctx, hv, regret = test_function.step(
                input_bounds=model_x_range,
                x_new=batch_x_next,
                x_ctx=x_ctx,
                y_ctx=y_ctx,
                compute_hv=True,
                compute_regret=True,
                regret_type=opt_cfg.regret_type,
            )
            batch_y_next = y_ctx[:, -opt_cfg.q :]  # [B, q, dy]

            hv_next = test_function.compute_hv(
                solutions=batch_y_next, y_mask=observation_tracker.y_mask_target
            )

            # Add optimization step metrics
            # avg_time = sum(batch_infer_time_list) / len(batch_infer_time_list) if batch_infer_time_list else 0.0
            metrics.add_optimization_step(
                hv=hv,
                hv_query=hv_next,
                regret=regret,
                entropy=batch_entropy,
                time=batch_infer_time_list,
                x_query=batch_x_next,
                y_query=batch_y_next,
            )

            # Log current step
            logger.log_step(step=observation_tracker.get_cost_used(), observation_tracker=observation_tracker, metrics=metrics)

            if acq_values is not None and _should_plot(
                observation_tracker.get_cost_used(),
                T,
                log_cfg.plot_per_n_steps,
                log_cfg.plot_enabled,
                observation_tracker.initial_cost,
            ):
                acq_fig = plot_acq_values(q_chunk=q_chunk, acq_values=acq_values)
                save_fig(
                    acq_fig,
                    plot_save_path,
                    config=opt_cfg,
                    filename=f"acq_heatmap_t{observation_tracker.get_cost_used()}_T{T}",
                    override=exp_cfg.override,
                    log=log,
                    log_to_wandb=exp_cfg.log_to_wandb,
                )

    # Log summary statistics
    logger.log_summary(metrics=metrics, test_function=test_function)

    # Save all metrics to disk
    _save_all_data(
        metrics=metrics,
        x_ctx=x_ctx,
        y_ctx=y_ctx,
        data_save_path=data_save_path,
        opt_cfg=opt_cfg,
        exp_cfg=exp_cfg,
        log=log,
    )

    # Generate and save all plots
    if log_cfg.plot_enabled:
        _save_all_plots(
            metrics=metrics,
            opt_cfg=opt_cfg,
            exp_cfg=exp_cfg,
            plot_save_path=plot_save_path,
            log=log,
        )

    del x_ctx, y_ctx
    del q_chunk, q_chunk_mask, logit_mask


def evaluate_prediction(
    model: TAMO,
    datapaths: List[str],
    data_save_path: str,
    plot_save_path: str,
    exp_cfg: ExConfig,
    pred_cfg: PredictionConfig,
    data_cfg: DataConfig,
    log_cfg: LogConfig,
    num_workers: int = 0,
    prefetch_factor: Optional[int] = None,
    log: callable = print,
    **kwargs,
):
    """Evaluate tamo predictions on a single dataset."""
    set_all_seeds(exp_cfg.seed)

    plot_save_path = osp.join(plot_save_path, str(exp_cfg.seed))
    data_save_path = osp.join(data_save_path, str(exp_cfg.seed))

    for datapath in tqdm(
        datapaths, desc="Running prediction on a dataset", unit="dataset"
    ):
        dataset = MultiFileHDF5Dataset(
            file_paths=[datapath],
            max_x_dim=data_cfg.max_x_dim,
            max_y_dim=data_cfg.max_y_dim,
            standardize=True,
            range_scale=get_train_y_range(),
        )
        log(f"Evaluating prediction on data from:\n{dataset.file_paths}\n\n")

        run_prediction(
            model=model,
            dataset=dataset,
            plot_save_path=plot_save_path,
            exp_cfg=exp_cfg,
            pred_cfg=pred_cfg,
            data_cfg=data_cfg,
            log_cfg=log_cfg,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            log=log,
        )


def run_prediction(
    model: TAMO,
    dataset: MultiFileHDF5Dataset,
    plot_save_path: str,
    exp_cfg: ExConfig,
    pred_cfg: PredictionConfig,
    data_cfg: DataConfig,
    log_cfg: LogConfig,
    num_workers: int = 0,
    prefetch_factor: Optional[int] = None,
    log: callable = print,
    **kwargs,
) -> None:
    """Evaluate prediction on dataset with seed."""
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=pred_cfg.batch_size,
        split=exp_cfg.mode,
        device=exp_cfg.device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    model = model.to(exp_cfg.device)
    model.eval()

    ravg = Averager()
    with torch.no_grad():
        for epoch, (x, y, valid_x_counts, valid_y_counts) in enumerate(dataloader):
            if has_nan_or_inf(x, "x") or has_nan_or_inf(y, "y"):
                continue

            # Prepare batches
            x = x.to(exp_cfg.device)
            y = y.to(exp_cfg.device)
            valid_x_counts = valid_x_counts.to(exp_cfg.device)
            valid_y_counts = valid_y_counts.to(exp_cfg.device)

            x, y, x_mask, y_mask, nc = prepare_prediction_batches(
                x=x,
                y=y,
                valid_x_counts=valid_x_counts,
                valid_y_counts=valid_y_counts,
                dim_scatter_mode=data_cfg.dim_scatter_mode,
                min_nc=pred_cfg.min_nc,
                max_nc=pred_cfg.max_nc,
                nc_fixed=pred_cfg.nc,
            )

            # Predict on context
            nll_c, mse_c, _ = prediction_forward(
                model=model,
                x_ctx=x[:, :nc],
                y_ctx=y[:, :nc],
                x_tar=x[:, :nc],
                y_tar=y[:, :nc],
                x_mask=x_mask,
                y_mask=y_mask,
                read_cache=pred_cfg.read_cache,
            )

            # Predict on target
            nll_t, mse_t, _ = prediction_forward(
                model=model,
                x_ctx=x[:, :nc],
                y_ctx=y[:, :nc],
                x_tar=x[:, nc:],
                y_tar=y[:, nc:],
                x_mask=x_mask,
                y_mask=y_mask,
                read_cache=pred_cfg.read_cache,
            )

            # Log metrics
            log_dict = {
                "nll_context": nll_c.detach().item(),
                "nll_target": nll_t.detach().item(),
            }

            for j, (mse_c_val, mse_t_val) in enumerate(zip(mse_c, mse_t)):
                mse_c_val = mse_c_val.detach().cpu().item()
                mse_t_val = mse_t_val.detach().cpu().item()

                log_dict[f"mse_context_{j}"] = mse_c_val
                log_dict[f"mse_target_{j}"] = mse_t_val

                log_dict[f"rmse_context_{j}"] = math.sqrt(mse_c_val)
                log_dict[f"rmse_target_{j}"] = math.sqrt(mse_t_val)

            ravg.batch_update(log_dict)

            if log_cfg.plot_enabled and epoch == 0:
                plot_nc_list = log_cfg.plot_nc_list or [nc]
                for pnc in plot_nc_list:
                    fig = plot_prediction_batch(
                        model=model,
                        nc=pnc,
                        x=x,
                        y=y,
                        x_mask=x_mask,
                        y_mask=y_mask,
                        y_mask_tar=y_mask,  # always full observation case
                        read_cache=pred_cfg.read_cache,
                    )

                    # Save figure
                    save_fig(
                        fig=fig,
                        path=plot_save_path,
                        config=pred_cfg,
                        filename=f"nc{pnc}",
                        override=exp_cfg.override,
                        log=log,
                        log_to_wandb=exp_cfg.log_to_wandb,
                    )

    # Log results
    line = f"[results, seed={exp_cfg.seed}]\n" f"{ravg.info()}"
    log(line)

    if exp_cfg.log_to_wandb:
        wandb.log(
            {
                "eval/nll_context": ravg.get("nll_context"),
                "eval/nll_target": ravg.get("nll_target"),
            }
        )
