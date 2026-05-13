"""Evaluation for optimization and prediction.

Public API:
    evaluate_optimization()  -- run optimization loop on a test function
    evaluate_prediction()    -- run prediction evaluation on HDF5 datasets
"""

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
from utils.dataclasses import ExConfig, DataConfig, PredictionConfig, OptimizationConfig, LogConfig
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


# ==============================================================================
# Logging
# ==============================================================================

_SEP  = "=" * 60
_DASH = "-" * 60


class OptimizationLogger:
    """Structured logging for the optimization loop."""

    def __init__(self, log_fn: callable = print, use_wandb: bool = False):
        self.log = log_fn
        self.use_wandb = use_wandb

    def log_step(self, step: int, observation_tracker: ObservationTracker, metrics: MetricTracker):
        latest = metrics.get_latest_values()
        lines = [_SEP, f"Step {step}", str(observation_tracker), _DASH]
        for key, label in [("hv", "Hypervolume"), ("regret", "Regret"), ("entropy", "Entropy")]:
            if latest[key] is not None:
                lines.append(f"  {label:<14}{latest[key]}")
        if latest["x_query"] is not None and latest["y_query"] is not None:
            lines += ["  Latest Query Points:",
                      f"    x_next: {latest['x_query']}",
                      f"    y_next: {latest['y_query']}"]
        lines.append(_SEP)
        self.log("\n".join(lines))

    def log_prediction_step(self, step: int, nll_t: Tensor, mse_t: Tensor):
        self.log(
            f"  [Prediction @ Step {step}]\n"
            f"    NLL Target:  {nll_t.mean().item():.4f}\n"
            f"    MSE Target:  {mse_t.mean().item():.6f}"
        )
        if self.use_wandb:
            wandb.log({"opt/nll_target": nll_t.mean().item(),
                       "opt/mse_target_mean": mse_t.mean().item(),
                       "opt/step": step})

    def log_summary(self, metrics: MetricTracker, test_function: TestFunction):
        stats = metrics.get_statistics()
        lines = [
            _SEP, "OPTIMIZATION SUMMARY", _SEP,
            f"Function: {test_function.function_name}",
            f"Dimensions: dx={test_function.x_dim}, dy={test_function.y_dim}",
            f"Max HV: {test_function.max_hv:.4f}",
            _DASH,
        ]
        if "hv_final_mean" in stats:
            lines += [f"Final Hypervolume:  {stats['hv_final_mean']:.4f} ± {stats['hv_final_std']:.4f}",
                      f"Max Hypervolume:    {stats['hv_max']:.4f}"]
        if "regret_final_mean" in stats:
            lines += [f"Final Regret:       {stats['regret_final_mean']:.4f} ± {stats['regret_final_std']:.4f}",
                      f"Min Regret:         {stats['regret_min']:.4f}"]
        if "total_time" in stats:
            lines += [f"Total Time:         {stats['total_time']:.2f}s",
                      f"Avg Time/Step:      {stats['avg_time_per_step']:.4f}s ± {stats['std_time_per_step']:.4f}s"]
        if "nll_t_final" in stats:
            lines += [_DASH, "Prediction Metrics:",
                      f"  Final NLL Target: {stats['nll_t_final']:.4f}",
                      f"  Mean NLL Target:  {stats['nll_t_mean']:.4f} ± {stats['nll_t_std']:.4f}"]
        lines.append(_SEP)
        self.log("\n".join(lines))
        if self.use_wandb:
            wandb.log({f"summary/{k}": v for k, v in stats.items()})


# ==============================================================================
# Save helpers
# ==============================================================================

def _save_all_data(metrics, x_ctx, y_ctx, data_save_path, opt_cfg, exp_cfg, log):
    stacked = metrics.get_stacked_metrics()
    stacked["x_ctx"] = x_ctx.detach().cpu()
    stacked["y_ctx"] = y_ctx.detach().cpu()
    for key, val in stacked.items():
        save_data(data=val, path=data_save_path, config=opt_cfg,
                  filename=key, override=exp_cfg.override, log=log)


def _save_all_plots(metrics, opt_cfg, exp_cfg, plot_save_path, log):
    stacked = metrics.get_stacked_metrics()
    for b in range(opt_cfg.batch_size):
        prefix = f"b{b}_" if opt_cfg.batch_size > 1 else ""
        plots = {
            f"{prefix}hv":      plot_1d("Hypervolume over Iterations",     stacked["hv"][b],      ylabel="Hypervolume"),
            f"{prefix}regret":  plot_1d("Regret over Iterations",          stacked["regret"][b],  ylabel="Regret"),
            f"{prefix}entropy": plot_1d("Entropy over Iterations",         stacked["entropy"][b], ylabel="Entropy"),
            f"{prefix}time":    plot_1d("Inference Time over Iterations",  stacked["time"],        ylabel="Time (s)"),
        }
        if "nll_t" in stacked:
            plots["nll_t"] = plot_1d("NLL Target over Iterations", stacked["nll_t"], ylabel="NLL Target")
        for name, fig in plots.items():
            save_fig(fig=fig, path=plot_save_path, config=opt_cfg, filename=name,
                     override=exp_cfg.override, log=log, log_to_wandb=exp_cfg.log_to_wandb)


# ==============================================================================
# Optimization evaluation
# ==============================================================================

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
    """Evaluate optimization on a test function."""
    set_all_seeds(exp_cfg.seed)
    plot_save_path = osp.join(plot_save_path, str(exp_cfg.seed))
    data_save_path = osp.join(data_save_path, str(exp_cfg.seed))

    model_x_range = make_range_nested_list(get_train_x_range(), test_function.x_dim)
    model_y_range = make_range_nested_list(get_train_y_range(), test_function.y_dim)

    log(
        f"--- Test function ---\n"
        f"  name={test_function.function_name}, dx={test_function.x_dim}, dy={test_function.y_dim}\n"
        f"  sigma={data_cfg.sigma}, max_hv={test_function.max_hv:.4f}, seed={exp_cfg.seed}\n"
        f"  x_bounds={test_function.x_bounds}, y_bounds={test_function.y_bounds}\n"
        f"--- Pre-trained ranges ---\n"
        f"  x_range={model_x_range}, y_range={model_y_range}"
    )

    run_optimization(
        model=model, test_function=test_function,
        model_x_range=model_x_range, model_y_range=model_y_range,
        exp_cfg=exp_cfg, opt_cfg=opt_cfg, data_cfg=data_cfg,
        pred_cfg=pred_cfg, log_cfg=log_cfg,
        plot_save_path=plot_save_path, data_save_path=data_save_path, log=log,
    )


def _should_plot(cost_used, cost_total, plot_per_n_unit_cost, plot_enabled, init_cost=1.0) -> bool:
    """True if a plot should be produced at the current cost step."""
    if not plot_enabled or plot_per_n_unit_cost <= 0:
        return False
    t, T, t0 = int(round(cost_used)), int(round(cost_total)), int(round(init_cost))
    return t == t0 or t == T or (t < T and (t - t0) % plot_per_n_unit_cost == 0)


def _generate_visualizations(
    model, x_ctx, y_ctx_scaled, x_tar, y_tar_scaled,
    x_mask_exp, y_mask_exp, y_mask_tar_exp, read_cache, y_mask_history=None,
    overlay=None,
) -> Dict[str, Any]:
    """Return {'mean': fig, 'std': fig} prediction plots."""
    nc = x_ctx.shape[1]
    return {
        ("mean" if plot_mean else "std"): plot_prediction_batch(
            model=model, nc=nc, xc=x_ctx, yc=y_ctx_scaled, x=x_tar, y=y_tar_scaled,
            x_mask=x_mask_exp, y_mask=y_mask_exp, y_mask_tar=y_mask_tar_exp,
            read_cache=read_cache, y_mask_history=y_mask_history,
            plot_mean=plot_mean, plot_order=True, overlay=overlay,
        )
        for plot_mean in [True, False]
    }


def run_prediction_on_test_function(
    test_function: TestFunction,
    model: TAMO,
    x_ctx: Tensor, y_ctx: Tensor,
    x_mask: Tensor, y_mask: Tensor, y_mask_tar: Tensor,
    train_x_range: NestedFloatList, train_y_range: NestedFloatList,
    batch_size: int, read_cache: bool,
    num_subspace_points: int = 500, sigma: float = 0.0,
    plot_enabled: bool = False,
    y_mask_history: Optional[Tensor] = None,
    seed: int = 0,
    overlay: Optional[Dict[str, Any]] = None,
) -> Tuple[Tensor, Tensor, Optional[Dict[str, Any]]]:
    """Evaluate model predictions at a single optimization step.

    Samples target points from test_function, computes NLL/MSE on context and
    target sets, and optionally returns visualization plots.

    Args:
        x_ctx:               [B, num_ctx, x_dim]
        y_ctx:               [B, num_ctx, y_dim]
        x_mask / y_mask:     [x_dim] / [y_dim]  active input/output dimensions
        y_mask_tar:          [y_dim]  target output dimensions
        y_mask_history:      [t, y_dim]  history of observed masks
        sigma:               unused, kept for API compatibility

    Returns:
        (nll_t, mse_t, figs)  where figs is None unless plot_enabled
    """
    device = x_mask.device

    x_tar, y_tar, _, _ = test_function.sample(
        input_bounds=train_x_range, batch_size=batch_size,
        num_subspace_points=num_subspace_points, use_grid_sampling=True,
        use_factorized_policy=False, device=device,
        x_mask=x_mask, y_mask=y_mask_tar, seed=seed,
    )

    y_ctx_scaled = test_function.transform_outputs(outputs=y_ctx, output_bounds=train_y_range)
    y_tar_scaled = test_function.transform_outputs(outputs=y_tar, output_bounds=train_y_range)

    x_mask_exp     = repeat(x_mask,    "d -> b d", b=batch_size)
    y_mask_exp     = repeat(y_mask,    "d -> b d", b=batch_size)
    y_mask_tar_exp = repeat(y_mask_tar, "d -> b d", b=batch_size)

    nll_t, mse_t, _ = prediction_forward(
        model=model, x_ctx=x_ctx, x_tar=x_tar, y_ctx=y_ctx_scaled, y_tar=y_tar_scaled,
        x_mask=x_mask_exp, y_mask=y_mask_exp, y_mask_tar=y_mask_tar_exp, read_cache=read_cache,
    )

    figs = _generate_visualizations(
        model=model, x_ctx=x_ctx, y_ctx_scaled=y_ctx_scaled,
        x_tar=x_tar, y_tar_scaled=y_tar_scaled,
        x_mask_exp=x_mask_exp, y_mask_exp=y_mask_exp, y_mask_tar_exp=y_mask_tar_exp,
        read_cache=read_cache, y_mask_history=y_mask_history,
        overlay=overlay,
    ) if plot_enabled else None

    return nll_t, mse_t, figs


def _save_prediction_plots(figs, observation_tracker, x_ctx, nll_t, T, plot_save_path, opt_cfg, exp_cfg, log):
    """Save prediction plots; filenames encode observed dims, context size, cost, and NLL."""
    x_dims = "".join(map(str, observation_tracker.x_mask.nonzero(as_tuple=False)[:, 0].tolist()))
    y_dims = "".join(map(str, observation_tracker.y_mask.nonzero(as_tuple=False)[:, 0].tolist()))
    t = observation_tracker.get_cost_used()
    prefix = f"context_dx{x_dims}dy{y_dims}_nc{x_ctx.shape[1]}_t{t}T{T}_nll{nll_t.detach().mean().item()}"
    for plot_type, fig in figs.items():
        save_fig(fig=fig, path=plot_save_path, config=opt_cfg, filename=f"{plot_type}_{prefix}",
                 override=exp_cfg.override, log=log, log_to_wandb=exp_cfg.log_to_wandb)


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
    """Run the optimization loop on test_function and save results."""
    if predict:
        assert pred_cfg is not None, "`pred_cfg` must be provided if perform prediction"

    metrics = MetricTracker()
    logger  = OptimizationLogger(log_fn=log, use_wandb=exp_cfg.log_to_wandb)

    observation_tracker = ObservationTracker(
        x_dim=test_function.x_dim, y_dim=test_function.y_dim,
        dim_mask_gen_mode=opt_cfg.dim_mask_gen_mode,
        single_obs_y_dim=opt_cfg.single_obs_y_dim,
        device=exp_cfg.device, num_initial_points=opt_cfg.num_initial_points,
        cost_mode=opt_cfg.cost_mode, cost=opt_cfg.cost,
    )

    # Initial observations
    x_ctx, y_ctx, hv, regret = test_function.init(
        input_bounds=model_x_range, batch_size=opt_cfg.batch_size,
        num_initial_points=opt_cfg.num_initial_points,
        regret_type=opt_cfg.regret_type, compute_hv=True, compute_regret=True,
        device=exp_cfg.device, seed=exp_cfg.seed,
    )
    metrics.add_optimization_step(
        hv=hv, hv_query=hv.clone(), regret=regret,
        entropy=torch.zeros((opt_cfg.batch_size,), device=exp_cfg.device),
        time=[0.0] * opt_cfg.num_initial_points, x_query=x_ctx, y_query=y_ctx,
    )
    logger.log_step(step=0, observation_tracker=observation_tracker, metrics=metrics)

    d = get_num_subspace_points(x_dim=test_function.x_dim,
                                use_factorized_policy=opt_cfg.use_factorized_policy,
                                d=opt_cfg.num_query_points)
    T = opt_cfg.sample_T()
    log(f"Subspace points d={d}, cost budget T={T}")

    q_chunk = None
    q_chunk_mask = None 
    logit_mask = None

    model = model.to(exp_cfg.device)
    model.eval()
    with torch.no_grad():
        while observation_tracker.get_cost_used() <= T:
            t = observation_tracker.get_cost_used()
            should_plot = _should_plot(t, T, log_cfg.plot_per_n_steps, log_cfg.plot_enabled,
                                       observation_tracker.initial_cost)

            # ---- Prediction evaluation ----
            if predict:
                nll_t, mse_t, figs = run_prediction_on_test_function(
                    test_function=test_function, model=model,
                    x_ctx=x_ctx, y_ctx=y_ctx,
                    x_mask=observation_tracker.x_mask,
                    y_mask=observation_tracker.y_mask,
                    y_mask_tar=observation_tracker.y_mask_target,
                    train_x_range=model_x_range, train_y_range=model_y_range,
                    batch_size=opt_cfg.batch_size, read_cache=pred_cfg.read_cache,
                    sigma=data_cfg.sigma, plot_enabled=should_plot,
                    y_mask_history=observation_tracker.y_mask_observed, seed=exp_cfg.seed,
                )
                nll_t, mse_t = nll_t.detach(), mse_t.detach()
                metrics.add_prediction_step(nll_t=nll_t, mse_t=mse_t)
                logger.log_prediction_step(step=t, nll_t=nll_t, mse_t=mse_t)
                if figs is not None:
                    _save_prediction_plots(figs=figs, observation_tracker=observation_tracker,
                                           x_ctx=x_ctx, nll_t=nll_t, T=T,
                                           plot_save_path=plot_save_path,
                                           opt_cfg=opt_cfg, exp_cfg=exp_cfg, log=log)
                    del figs

            # ---- Batch query selection ----
            y_ctx_scaled = test_function.transform_outputs(outputs=y_ctx, output_bounds=model_y_range)
            batch_x_ctx, batch_y_ctx = x_ctx.clone(), y_ctx_scaled.clone()
            batch_x_next_list, batch_entr_list, batch_infer_time_list = [], [], []
            acq_values = None

            for qi in range(opt_cfg.q):
                batch_x_ctx, batch_y_ctx, action_res = select_next_query_wrapper(
                    fantasy=opt_cfg.fantasy, x_ctx=batch_x_ctx, y_ctx=batch_y_ctx,
                    model=model, observation_tracker=observation_tracker,
                    model_x_range=model_x_range, opt_config=opt_cfg, pred_config=pred_cfg,
                    d=d, T=T, query_chunks=q_chunk, query_x_mask=q_chunk_mask, logit_mask=logit_mask,
                )
                acq_values   = action_res.logits
                q_chunk      = action_res.q_chunk
                q_chunk_mask = action_res.q_chunk_mask
                logit_mask   = action_res.logit_mask
                batch_x_next_list.append(action_res.next_x)
                batch_entr_list.append(action_res.entropy)
                batch_infer_time_list.append(action_res.infer_time)
                observation_tracker.step(update_mask=(qi == opt_cfg.q - 1))

            batch_x_next  = torch.cat(batch_x_next_list, dim=1)   # [B, q, max_x_dim]
            batch_entropy = torch.stack(batch_entr_list, dim=1)   # [B, q]

            x_ctx, y_ctx, hv, regret = test_function.step(
                input_bounds=model_x_range, x_new=batch_x_next, x_ctx=x_ctx, y_ctx=y_ctx,
                compute_hv=True, compute_regret=True, regret_type=opt_cfg.regret_type,
            )
            batch_y_next = y_ctx[:, -opt_cfg.q:]
            hv_next = test_function.compute_hv(solutions=batch_y_next,
                                               y_mask=observation_tracker.y_mask_target)

            metrics.add_optimization_step(
                hv=hv, hv_query=hv_next, regret=regret, entropy=batch_entropy,
                time=batch_infer_time_list, x_query=batch_x_next, y_query=batch_y_next,
            )
            logger.log_step(step=observation_tracker.get_cost_used(),
                            observation_tracker=observation_tracker, metrics=metrics)

            # Acquisition heatmap (re-evaluate plot condition after stepping)
            if q_chunk is not None and acq_values is not None and _should_plot(
                observation_tracker.get_cost_used(), T,
                log_cfg.plot_per_n_steps, log_cfg.plot_enabled, observation_tracker.initial_cost,
            ):
                save_fig(
                    plot_acq_values(q_chunk=q_chunk, acq_values=acq_values),
                    plot_save_path, config=opt_cfg,
                    filename=f"acq_heatmap_t{observation_tracker.get_cost_used()}_T{T}",
                    override=exp_cfg.override, log=log, log_to_wandb=exp_cfg.log_to_wandb,
                )

    logger.log_summary(metrics=metrics, test_function=test_function)
    _save_all_data(metrics=metrics, x_ctx=x_ctx, y_ctx=y_ctx,
                   data_save_path=data_save_path, opt_cfg=opt_cfg, exp_cfg=exp_cfg, log=log)
    if log_cfg.plot_enabled:
        _save_all_plots(metrics=metrics, opt_cfg=opt_cfg, exp_cfg=exp_cfg,
                        plot_save_path=plot_save_path, log=log)

    del x_ctx, y_ctx, q_chunk, q_chunk_mask, logit_mask


# ==============================================================================
# Prediction evaluation
# ==============================================================================

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
    """Evaluate model predictions on a list of HDF5 dataset files."""
    set_all_seeds(exp_cfg.seed)
    plot_save_path = osp.join(plot_save_path, str(exp_cfg.seed))
    data_save_path = osp.join(data_save_path, str(exp_cfg.seed))

    for datapath in tqdm(datapaths, desc="Running prediction on a dataset", unit="dataset"):
        dataset = MultiFileHDF5Dataset(
            file_paths=[datapath], max_x_dim=data_cfg.max_x_dim, max_y_dim=data_cfg.max_y_dim,
            standardize=True, range_scale=get_train_y_range(),
        )
        log(f"Evaluating prediction on:\n{dataset.file_paths}\n")
        run_prediction(
            model=model, dataset=dataset, plot_save_path=plot_save_path,
            exp_cfg=exp_cfg, pred_cfg=pred_cfg, data_cfg=data_cfg, log_cfg=log_cfg,
            num_workers=num_workers, prefetch_factor=prefetch_factor, log=log,
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
    """Evaluate prediction on a dataset."""
    dataloader = build_dataloader(
        dataset=dataset, batch_size=pred_cfg.batch_size, split=exp_cfg.mode,
        device=exp_cfg.device, num_workers=num_workers, prefetch_factor=prefetch_factor,
    )

    model = model.to(exp_cfg.device)
    model.eval()
    ravg = Averager()

    with torch.no_grad():
        for epoch, (x, y, valid_x_counts, valid_y_counts) in enumerate(dataloader):
            if has_nan_or_inf(x, "x") or has_nan_or_inf(y, "y"):
                continue

            x, y = x.to(exp_cfg.device), y.to(exp_cfg.device)
            valid_x_counts = valid_x_counts.to(exp_cfg.device)
            valid_y_counts = valid_y_counts.to(exp_cfg.device)

            x, y, x_mask, y_mask, nc = prepare_prediction_batches(
                x=x, y=y, valid_x_counts=valid_x_counts, valid_y_counts=valid_y_counts,
                dim_scatter_mode=data_cfg.dim_scatter_mode,
                min_nc=pred_cfg.min_nc, max_nc=pred_cfg.max_nc, nc_fixed=pred_cfg.nc,
            )

            ctx_kw = dict(model=model, x_ctx=x[:, :nc], y_ctx=y[:, :nc],
                          x_mask=x_mask, y_mask=y_mask, read_cache=pred_cfg.read_cache)
            nll_c, mse_c, _ = prediction_forward(**ctx_kw, x_tar=x[:, :nc], y_tar=y[:, :nc])
            nll_t, mse_t, _ = prediction_forward(**ctx_kw, x_tar=x[:, nc:], y_tar=y[:, nc:])

            log_dict = {"nll_context": nll_c.detach().item(), "nll_target": nll_t.detach().item()}
            for j, (mc, mt) in enumerate(zip(mse_c, mse_t)):
                mc, mt = mc.detach().cpu().item(), mt.detach().cpu().item()
                log_dict |= {f"mse_context_{j}": mc, f"mse_target_{j}": mt,
                              f"rmse_context_{j}": math.sqrt(mc), f"rmse_target_{j}": math.sqrt(mt)}
            ravg.batch_update(log_dict)

            if log_cfg.plot_enabled and epoch == 0:
                for pnc in (log_cfg.plot_nc_list or [nc]):
                    fig = plot_prediction_batch(
                        model=model, nc=pnc, x=x, y=y, x_mask=x_mask, y_mask=y_mask,
                        y_mask_tar=y_mask, read_cache=pred_cfg.read_cache,
                    )
                    save_fig(fig=fig, path=plot_save_path, config=pred_cfg, filename=f"nc{pnc}",
                             override=exp_cfg.override, log=log, log_to_wandb=exp_cfg.log_to_wandb)

    log(f"[results, seed={exp_cfg.seed}]\n{ravg.info()}")
    if exp_cfg.log_to_wandb:
        wandb.log({"eval/nll_context": ravg.get("nll_context"), "eval/nll_target": ravg.get("nll_target")})