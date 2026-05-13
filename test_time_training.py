"""Test-time training scripts."""

import time
from dataclasses import dataclass, field, replace
import os.path as osp
import os
from typing import List, Optional, Tuple

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra
from tqdm import tqdm

from utils.wandb_wrapper import init as wandb_init
from model import TAMO
from utils.log import get_log_filename, get_log_fn
from utils.config import (
    get_train_x_range,
    get_train_y_range,
    load_checkpoint,
    build_tamo,
)
from utils.seed import set_all_seeds
from utils.paths import (
    get_exp_path,
    get_result_plot_path,
    get_result_data_path,
    get_filename_base,
)
from utils.dataclasses import (
    ExConfig,
    DataConfig,
    PredictionConfig,
    OptimizationConfig,
    LogConfig,
    LossConfig,
)
from utils.log import MetricTracker
from utils.plot import plot_acq_values
from utils.types import NestedFloatList
from utils.save import save_fig

from data.dataset import map_function_to_gp_datapath, get_function_environment
from data.obs_tracker import ObservationTracker
from data.base.preprocessing import make_range_nested_list, make_range_tensor, transform
from data.gp_sample_function import GPSampleFunction
from data.function import TestFunction
from data.function_sampling import get_num_subspace_points, generate_sobol_samples
from data.moo import MOO

from evaluate import (
    OptimizationLogger,
    _save_all_data,
    _save_all_plots,
    run_prediction_on_test_function,
    _save_prediction_plots,
    _should_plot,
)
from forwards import (
    select_next_query_wrapper,
    select_next_query,
    compute_policy_loss,
    _get_cumulative_rewards,
)
from model.layers import GMMPredictionHead

# ==============================================================================
# Optimization evaluation
# ==============================================================================


@dataclass
class TestTimeConfig:
    enabled: bool = True
    mode: str = "policy"

    num_start_ttt: int = 20
    num_obs: int = 20
    data_kernel_type_list: List[str] = field(
        default_factory=lambda: ["rbf", "matern32", "matern52"]
    )
    num_gps: int = 4
    num_fns: int = 4
    num_test_points: int = 256
    # If True, sample kernel hyperparameters from priors (per-task ExactGP path)
    # instead of the default fully-Bayesian NUTS posterior over p(theta | data).
    use_independent_sampler: bool = False
    # If True, replace the GP posterior pool with the *actual* test function
    # values at a fixed Sobol grid. Oracle sanity check (upper bound on TTT
    # gains); not deployable since real evaluation budgets don't expose truth.
    use_true_function: bool = False

    num_epoch: int = 10 
    optimizer_type: str = "adam"
    optimize_parameters: str = "decoder"
    lr: float = 1e-3
    weight_decay: float = 1e-2

    # Predictor-mode only: random context size per epoch in [min_nc, max_nc]
    min_nc: int = 2
    max_nc: int = 50

    prompt_K: int = 8
    prompt_init: str = "random"

    # Softmax-reweighted REINFORCE (entropic objective; see
    # compute_policy_loss_softmax). When True, replaces the standardized
    # per-step REINFORCE in forwards.compute_policy_loss with a per-
    # trajectory softmax(beta * R) weighting over the batch. Policy mode
    # only.
    use_softmax_reweighted_loss: bool = False
    softmax_beta: float = 1.0
    # "final" | "sum" | "discounted_sum"  → per-trajectory weight, no within-
    # rollout credit assignment. "per_step" → per-step softmax over the
    # discounted return-to-go G_{b,t}, with weights centered by 1/B (so β=0
    # gives a zero-gradient loss, not entropy minimization). per_step is the
    # principled fix when the trajectory-level loss collapses to entropy
    # minimization at small β.
    softmax_reduction: str = "discounted_sum"


class TTTSoftPrompt(nn.Module):
    """K virtual (x, y) context pairs in input-value space."""

    def __init__(
        self, K, dx_max, dy_max, x_lo, x_hi, y_lo, y_hi, x_init=None, y_init=None
    ):
        super().__init__()

        self.K = K
        # Unconstrained; mapped to bounded ranges via tanh below.
        # Use x_init and y_init if provided, so step 0 is rounghly equivalent to duplicate real points
        self.x_raw = nn.Parameter(
            torch.zeros(K, dx_max)
            if x_init is None
            else self._inverse_map(x_init, x_lo, x_hi)
        )
        self.y_raw = nn.Parameter(
            torch.zeros(K, dy_max)
            if y_init is None
            else self._inverse_map(y_init, y_lo, y_hi)
        )
        self.register_buffer("x_lo", x_lo)
        self.register_buffer("x_hi", x_hi)
        self.register_buffer("y_lo", y_lo)
        self.register_buffer("y_hi", y_hi)

    def expand(self, B):
        # from [-1, 1] to [0, 1] to [x_lo, x_hi]
        x = self.x_lo + (self.x_hi - self.x_lo) * 0.5 * (torch.tanh(self.x_raw) + 1)
        y = self.y_lo + (self.y_hi - self.y_lo) * 0.5 * (torch.tanh(self.y_raw) + 1)

        # [B, K, dx_max] to [B, K, dy_max]
        return x.unsqueeze(0).expand(B, -1, -1), y.unsqueeze(0).expand(B, -1, -1)

    @staticmethod
    def _inverse_map(x, lo, hi, eps=1e-6):
        raise NotImplementedError(f"batch x is not considered.")
        # x in [lo, hi]  ->  raw in R such that expand(raw) == x
        u = (x - lo) / (hi - lo)  # (0, 1)
        u = u.clamp(eps, 1.0 - eps)  # avoid atanh(±1) = ±inf
        return torch.atanh(2.0 * u - 1.0)


def _plot_test_and_gp_samples(
    test_function: TestFunction,
    gp_sample_function: GPSampleFunction,
    input_bounds: NestedFloatList,
    save_path: str,
    num_grid_points: int = 200,
):
    """2 x valid_y_counts: row 0 test function, row 1 GP posterior sample.

    Supports x_dim in {1, 2}: line plots for 1-d, contour plots for 2-d.
    Saves one figure per group-leader index `i` where i % (num_fns * batch_size) == 0,
    appending `_g{group_idx}` to the stem of `save_path`.
    """
    valid_x_count = test_function.x_dim
    if valid_x_count not in (1, 2):
        return

    valid_y_counts = test_function.y_dim
    device = gp_sample_function._x.device
    bounds = make_range_nested_list(input_bounds, valid_x_count)

    if valid_x_count == 1:
        x_lo, x_hi = float(bounds[0][0]), float(bounds[0][1])
        x_grid = torch.linspace(x_lo, x_hi, num_grid_points, device=device).unsqueeze(
            -1
        )
        with torch.no_grad():
            y_grid = test_function(x=x_grid, input_bounds=input_bounds)
        x_grid_np = x_grid.cpu().squeeze(-1).numpy()
        y_grid_np = y_grid.cpu().numpy()
    else:
        n = num_grid_points
        g0 = torch.linspace(float(bounds[0][0]), float(bounds[0][1]), n, device=device)
        g1 = torch.linspace(float(bounds[1][0]), float(bounds[1][1]), n, device=device)
        X0, X1 = torch.meshgrid(g0, g1, indexing="xy")
        x_grid = torch.stack([X0.reshape(-1), X1.reshape(-1)], dim=-1)
        with torch.no_grad():
            y_grid = test_function(x=x_grid, input_bounds=input_bounds)
        y_grid_np = y_grid.reshape(n, n, valid_y_counts).cpu().numpy()
        X0n, X1n = X0.cpu().numpy(), X1.cpu().numpy()

    base, ext = osp.splitext(save_path)
    B_total = gp_sample_function._x.shape[0]

    for g, i in enumerate(range(0, B_total)):
        fig, axes = plt.subplots(
            2,
            valid_y_counts,
            figsize=(5 * valid_y_counts, 8),
            squeeze=False,
        )
        if valid_x_count == 1:
            xs_i = gp_sample_function._x[i].detach().cpu().squeeze(-1)
            ys_i = gp_sample_function._y[i].detach().cpu()
            order = xs_i.argsort()
            xs_i = xs_i[order].numpy()
            ys_i = ys_i[order].numpy()
            for j in range(valid_y_counts):
                axes[0, j].plot(x_grid_np, y_grid_np[:, j], color="k")
                axes[0, j].set_title(f"test fn, output {j}")
                axes[1, j].plot(xs_i, ys_i[:, j])
                axes[1, j].set_title(f"gp sample [{i}], output {j}")
        else:
            x_post = gp_sample_function._x[i].detach().cpu().numpy()
            y_post = gp_sample_function._y[i].detach().cpu().numpy()
            for j in range(valid_y_counts):
                cs0 = axes[0, j].contourf(X0n, X1n, y_grid_np[..., j], levels=20)
                fig.colorbar(cs0, ax=axes[0, j])
                axes[0, j].set_title(f"test fn, output {j}")
                cs1 = axes[1, j].tricontourf(
                    x_post[:, 0],
                    x_post[:, 1],
                    y_post[:, j],
                    levels=20,
                )
                fig.colorbar(cs1, ax=axes[1, j])
                axes[1, j].set_title(f"gp sample [{i}], output {j}")

        plt.tight_layout()
        plt.savefig(f"{base}_g{g}{ext}", dpi=120)
        plt.close(fig)


def _make_optimizer(params, cfg: TestTimeConfig) -> torch.optim.Optimizer:
    if cfg.optimizer_type == "adam":
        return torch.optim.Adam(params, lr=cfg.lr)
    if cfg.optimizer_type == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise NotImplementedError(f"Unknown optimizer_type: {cfg.optimizer_type}")


def _build_gp_sample_function(
    *,
    test_function: TestFunction,
    data_cfg: DataConfig,
    opt_config: OptimizationConfig,
    test_time_config: TestTimeConfig,
    device: str,
    observed_x,
    observed_y_scaled,
) -> GPSampleFunction:
    return GPSampleFunction(
        data_config=data_cfg,
        batch_size=1,
        d=test_time_config.num_test_points,
        use_grid_sampling=False,
        use_factorized_policy=False,
        x_dim=test_function.x_dim,
        y_dim=test_function.y_dim,
        device=device,
        x_ctx=observed_x,
        y_ctx=observed_y_scaled,
        kernel_types=test_time_config.data_kernel_type_list,
        num_gps=test_time_config.num_gps,
        num_fns=test_time_config.num_fns,
        num_samples=opt_config.num_samples,
        use_independent_sampler=test_time_config.use_independent_sampler,
    )


class _TrueFunctionSampler:
    """Oracle counterpart to GPSampleFunction.

    Replaces the GP-posterior pool with the *actual* test-function values at a
    fixed Sobol grid. Used as a TTT sanity check (upper bound on what TTT can
    extract); not deployable — real evaluations don't expose the truth.

    Mirrors the GPSampleFunction surface consumed by _run_policy_ttt /
    _run_predictor_ttt: _x / _y, x_ctx_padded / y_ctx_padded, x_mask / y_mask,
    chunks / chunk_mask, batch_size / num_points, restore_full_dim_later,
    step, and the batch_gather staticmethod (reused). HV / regret are
    delegated to test_function in raw y space (matches run_optimization's
    outer loop); the rollout's scaled y_ctx is inverse-transformed on demand
    so callers stay in scaled space.
    """

    restore_full_dim_later = True
    batch_gather = staticmethod(GPSampleFunction.batch_gather)

    def __init__(
        self,
        *,
        test_function: TestFunction,
        data_cfg: DataConfig,
        opt_config: OptimizationConfig,
        test_time_config: TestTimeConfig,
        device: str,
        observed_x: torch.Tensor,
        observed_y_scaled: torch.Tensor,
    ):
        x_dim = test_function.x_dim
        y_dim = test_function.y_dim
        # Match GP path's effective batch: num_gps * num_fns replicates * num_samples,
        # all sharing the same oracle function. Action-sampling stochasticity gives
        # per-replicate variance for REINFORCE.
        B = test_time_config.num_gps * test_time_config.num_fns * opt_config.num_samples

        self._test_function = test_function
        self._input_bounds = data_cfg.x_range
        self._model_y_range = get_train_y_range()

        x_range_t = make_range_tensor(self._input_bounds, num_dim=x_dim).to(device=device)
        x_test = generate_sobol_samples(
            x_range=x_range_t,
            num_datapoints=test_time_config.num_test_points,
            grid=False,
            seed=0,
        )  # [num_test_points, x_dim]

        # Pre-evaluate the oracle on the Sobol grid; scale outputs so the model
        # sees the same y range it saw during training.
        with torch.no_grad():
            y_test_raw = test_function.evaluate(x=x_test, input_bounds=self._input_bounds)
            y_test_scaled = test_function.transform_outputs(
                outputs=y_test_raw, output_bounds=self._model_y_range
            )

        self._x = x_test.unsqueeze(0).expand(B, -1, -1).contiguous()
        self._y = y_test_scaled.unsqueeze(0).expand(B, -1, -1).contiguous()

        # All-True dim masks: at test time max_*_dim == valid dim (run_optimization
        # patches data_cfg.max_*_dim from the test function's dims).
        self.x_mask = torch.ones(x_dim, dtype=torch.bool, device=device)
        self.y_mask = torch.ones(y_dim, dtype=torch.bool, device=device)

        # Candidate query set is the same Sobol grid the pool was evaluated on
        # → step(index_new) gathers a real (x, y_scaled) pair.
        self.chunks_ = x_test
        self.chunk_mask_ = torch.ones(1, x_dim, dtype=torch.bool, device=device)

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.num_points = x_test.shape[0]
        self.batch_size = B

        self.x_ctx_padded = observed_x.expand(B, -1, -1).contiguous()
        self.y_ctx_padded = observed_y_scaled.expand(B, -1, -1).contiguous()

        # Cached scaled-space bounds for the on-the-fly inverse transform in step().
        self._scaled_y_bounds = make_range_tensor(
            self._model_y_range, num_dim=y_dim
        ).to(device=device)

        # Cache regret normalizers. max_hv comes from test_function directly
        # (true global optimum on the function); max_hv_norm is computed once
        # on (raw pool + raw observations), mirroring GPSampleFunction so
        # regret_type='norm_ratio' has the cached value MOO.compute_regret needs.
        # ref_point comes from test_function (may be tighter than y_bounds[:, 1],
        # e.g., BraninCurrin uses [18, 6] not [309, 14]); using it consistently
        # matches test_function.compute_hv / .max_hv.
        self._y_bounds = test_function.y_bounds.to(device=device)
        self._ref_point = test_function.ref_point.to(device=device)
        observed_y_raw = self._to_raw(observed_y_scaled)
        y_pool_raw = torch.cat(
            [y_test_raw.unsqueeze(0), observed_y_raw.to(y_test_raw)], dim=1
        )  # [1, N + n_real, y_dim]
        self._max_hv = test_function.max_hv  # scalar; MOO broadcasts to [B]
        self._max_hv_norm, _, _ = MOO.compute_hv(
            solutions=y_pool_raw,
            ref_point=self._ref_point,
            minimum=self._y_bounds[:, 0],
            maximum=self._y_bounds[:, 1],
            normalize=True,
            y_mask=self.y_mask,
        )

    @property
    def chunks(self) -> torch.Tensor:
        return self.chunks_

    @property
    def chunk_mask(self) -> torch.Tensor:
        return self.chunk_mask_

    def _to_raw(self, y_scaled: torch.Tensor) -> torch.Tensor:
        return transform(
            data=y_scaled,
            inp_bounds=self._scaled_y_bounds.to(device=y_scaled.device, dtype=y_scaled.dtype),
            out_bounds=self._test_function.y_bounds.to(device=y_scaled.device, dtype=y_scaled.dtype),
            transform_method="min_max",
        )

    def step(
        self,
        index_new: torch.Tensor,
        x_ctx: Optional[torch.Tensor] = None,
        y_ctx: Optional[torch.Tensor] = None,
        compute_hv: bool = True,
        compute_regret: bool = True,
        regret_type: str = "ratio",
    ):
        x_new = GPSampleFunction.batch_gather(
            tensor=self._x, dim=1, index=index_new, full_dim_mask=None,
        )
        y_new_scaled = GPSampleFunction.batch_gather(
            tensor=self._y, dim=1, index=index_new, full_dim_mask=None,
        )
        x_ctx = GPSampleFunction.update_context(new=x_new, old=x_ctx)
        y_ctx = GPSampleFunction.update_context(new=y_new_scaled, old=y_ctx)

        reward = regret = None
        if compute_hv or compute_regret:
            # Invert min-max to raw scale; HV / regret are in raw y space,
            # matching the outer optimization loop's test_function.compute_*.
            y_ctx_raw = self._to_raw(y_ctx)
            if compute_hv:
                reward = self._test_function.compute_hv(
                    solutions=y_ctx_raw, y_mask=self.y_mask
                ).detach().cpu().numpy()
            if compute_regret:
                # Call MOO directly with cached max_hv / max_hv_norm so
                # norm_ratio works without re-deriving normalizers each step.
                # ref_point matches test_function.compute_hv for consistency.
                regret = MOO.compute_regret(
                    solutions=y_ctx_raw,
                    ref_point=self._ref_point,
                    minimum=self._y_bounds[:, 0],
                    maximum=self._y_bounds[:, 1],
                    regret_type=regret_type,
                    y_mask=self.y_mask,
                    max_hv=self._max_hv,
                    max_hv_norm=self._max_hv_norm,
                )
        return x_ctx, y_ctx, reward, regret


def _build_true_function_sampler(
    *,
    test_function: TestFunction,
    data_cfg: DataConfig,
    opt_config: OptimizationConfig,
    test_time_config: TestTimeConfig,
    device: str,
    observed_x,
    observed_y_scaled,
) -> _TrueFunctionSampler:
    return _TrueFunctionSampler(
        test_function=test_function,
        data_cfg=data_cfg,
        opt_config=opt_config,
        test_time_config=test_time_config,
        device=device,
        observed_x=observed_x,
        observed_y_scaled=observed_y_scaled,
    )


def _log_epoch_status(*, log, epoch, params, loss_name, loss_val, grad_norm):
    if any(p.grad is not None for p in params):
        log(
            f"[Epoch {epoch}] {loss_name}: {loss_val:.4f} | "
            f"Total Grad Norm: {grad_norm:.4f}"
        )
    else:
        log(
            f"[Epoch {epoch}] WARNING: Gradients are completely missing! "
            f"Check your graph."
        )


def _save_ttt_curves(
    *,
    plot_save_path: str,
    tag: str,
    step_reward_mean: List[float],
    final_step_reward_mean: List[float],
    nll: List[float],
):
    suffix = f"_{tag}" if tag else ""

    if step_reward_mean:
        os.makedirs(plot_save_path, exist_ok=True)
        epochs = range(1, len(step_reward_mean) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, step_reward_mean, marker="o")
        axes[0].set(xlabel="epoch", ylabel="mean step reward", title="step_rewards.mean()")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs, final_step_reward_mean, marker="o", color="C1")
        axes[1].set(xlabel="epoch", ylabel="mean final-step reward", title="step_rewards[:, -1].mean()")
        axes[1].grid(True, alpha=0.3)
        if tag:
            fig.suptitle(f"TTT reward curves ({tag})")
        fig.tight_layout()
        fig.savefig(osp.join(plot_save_path, f"ttt_reward{suffix}.png"), dpi=120)
        plt.close(fig)

    if nll:
        os.makedirs(plot_save_path, exist_ok=True)
        epochs = range(1, len(nll) + 1)
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.plot(epochs, nll, marker="o", color="C2")
        ax.set(xlabel="epoch", ylabel="mean NLL", title="predictor NLL")
        ax.grid(True, alpha=0.3)
        if tag:
            fig.suptitle(f"TTT NLL ({tag})")
        fig.tight_layout()
        fig.savefig(osp.join(plot_save_path, f"ttt_nll{suffix}.png"), dpi=120)
        plt.close(fig)


def compute_policy_loss_softmax(
    step_rewards: torch.Tensor,
    log_probs: torch.Tensor,
    beta: float = 1.0,
    reduction: str = "discounted_sum",
    discount_factor: float = 0.99,
    clip_rewards: bool = True,
    batch_first: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax-reweighted REINFORCE (entropic objective from TTT-Discover).

    Optimizes log E_a[exp(beta * R(a))] in place of E[R(a)]. The gradient
    reduces to standard policy gradient reweighted by w_b = softmax_b(beta R),
    so high-return rollouts dominate. beta -> 0 recovers uniform REINFORCE;
    beta -> inf concentrates on the argmax rollout. Useful for BO/MOO where
    best-found, not average regret, is the figure of merit.

    Derivation: d/dtheta log E[exp(beta R)] = E_a[w(a) * d log pi(a)/dtheta]
    where w(a) = exp(beta R(a)) / Z. The batch estimator therefore takes
    weights = softmax_b(beta R_b) over the rollouts in one TTT epoch. All
    rollouts share the same real-observation prefix (one "state"), so the
    softmax is over the full batch dimension.

    Args:
        step_rewards:     [B, H] (or [H, B] if batch_first=False), e.g. neg
                          regret per step. Detached internally.
        log_probs:        [B, H] (or [H, B]), per-step log-prob of the
                          sampled query. Carries gradient.
        beta:             temperature; higher = more peaked on top rollouts.
        reduction:        how to collapse per-step rewards into the
                          softmax-weighting scalar.
                          "final": per-trajectory R_b = r_{b, H-1}.
                          "sum": per-trajectory R_b = sum_t r_{b, t}.
                          "discounted_sum": per-trajectory R_b = sum_t
                            gamma^t r_{b, t}.
                          "per_step": per-step return-to-go
                            G_{b, t} = sum_{t' >= t} gamma^{t'-t} r_{b, t'};
                            weights are softmax_b(beta * G_{:, t}) and the
                            loss is mean_t sum_b (1/B - w_{b, t}) * logp_{b, t}.
                            Restores temporal credit assignment AND centers
                            on the uniform baseline so beta=0 -> zero loss
                            (no entropy collapse, unlike the trajectory
                            variants).
        discount_factor:  gamma for "discounted_sum".
        clip_rewards:     mirror of forwards.compute_policy_loss: zero out
                          steps that don't push best-so-far before reducing.
        batch_first:      input layout flag, same convention as the existing
                          loss in forwards.py.

    Returns:
        loss:             scalar; minimize to maximize log E[exp(beta R)].
        step_rewards:     [B, H], the (possibly clipped) immediate rewards
                          passed through, for logging parity with the
                          existing compute_policy_loss return.

    Scale note: per-step log-probs are mean-reduced over H (not summed) so
    the loss magnitude matches forwards.compute_policy_loss's mean-over-
    (B, H) scale. The optimum is unchanged — only the effective lr is.
    The summed-trajectory variant from the paper corresponds to lr * H.
    """
    if not batch_first:
        step_rewards = step_rewards.transpose(0, 1)
        log_probs = log_probs.transpose(0, 1)

    B, H = step_rewards.shape
    assert log_probs.shape == (B, H), f"{log_probs.shape}"
    assert B > 1, "softmax-reweighted REINFORCE needs B > 1 to define weights"

    step_rewards = step_rewards.detach()

    if clip_rewards:
        step_rewards_cummax = torch.cummax(step_rewards, dim=-1).values
        is_info = step_rewards == step_rewards_cummax
        step_rewards = step_rewards * is_info.float()

    if reduction == "per_step":
        # Per-step credit assignment + per-step softmax reweighting.
        # G_{b, t} = sum_{t' >= t} gamma^{t' - t} * r_{b, t'}  (return-to-go)
        # w_{b, t} = softmax_b(beta * G_{:, t})                (over batch, per step)
        # loss    = -mean_t sum_b (w_{b, t} - 1/B) * logp_{b, t}
        # Centering by 1/B is the variance-reduction baseline for the
        # weighted policy gradient: at beta=0, w = 1/B exactly, so centered
        # weights are zero and the loss is identically zero — no entropy
        # minimization. The scale matches standard REINFORCE (mean over
        # time, sum over batch with uniform baseline subtracted).
        G = _get_cumulative_rewards(step_rewards, discount_factor=discount_factor)
        weights = torch.softmax(beta * G, dim=0).detach()       # [B, H]
        centered = weights - (1.0 / B)
        loss = -(centered * log_probs).sum(dim=0).mean()
        return loss, step_rewards

    if reduction == "final":
        R = step_rewards[:, -1]
    elif reduction == "sum":
        R = step_rewards.sum(dim=-1)
    elif reduction == "discounted_sum":
        discounts = discount_factor ** torch.arange(
            H, device=step_rewards.device, dtype=step_rewards.dtype
        )
        R = (step_rewards * discounts).sum(dim=-1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    # softmax is shift-invariant, so no baseline subtraction is needed; but
    # subtract the max for numerical stability before exponentiating.
    weights = torch.softmax(beta * R, dim=0).detach()

    logp_traj = log_probs.mean(dim=-1)
    loss = -(weights * logp_traj).sum()

    return loss, step_rewards


def _run_policy_ttt(
    *,
    model: TAMO,
    gp_sample_function: GPSampleFunction,
    optimizer: torch.optim.Optimizer,
    opt_config: OptimizationConfig,
    loss_config: LossConfig,
    data_cfg: DataConfig,
    test_time_config: TestTimeConfig,
    params: List[torch.nn.Parameter],
    prompt: Optional[TTTSoftPrompt],
    T: int,
    device: str,
    log,
    step_reward_mean_history: List[float],
    final_step_reward_mean_history: List[float],
):
    assert gp_sample_function.x_ctx_padded is not None, \
        "TTT policy mode requires real observations to condition on."

    x_ctx_real = gp_sample_function.x_ctx_padded
    y_ctx_real = gp_sample_function.y_ctx_padded
    B = x_ctx_real.shape[0]
    n_real = x_ctx_real.shape[1]
    assert n_real < T, f"TTT called with no remaining budget: n_real={n_real}, T={T}"
    remaining = T - n_real
    use_prompt = test_time_config.optimize_parameters == "prompt"

    for ttt_epoch in tqdm(
        range(test_time_config.num_epoch), desc="test-time training (policy)"
    ):
        neg_regrets_list: List[torch.Tensor] = []
        log_probs_list: List[torch.Tensor] = []

        # Real-only buffer; the prompt is re-prepended each step so its
        # autograd path to every step's logp survives the no_grad step()
        # below (which would otherwise detach the prompt rows).
        x_real = x_ctx_real.clone()
        y_real = y_ctx_real.clone()
        logit_mask = None

        for step in range(remaining + 1):
            if use_prompt:
                x_p, y_p = prompt.expand(B)
                x_ctx = torch.cat([x_p, x_real], dim=1)
                y_ctx = torch.cat([y_p, y_real], dim=1)
            else:
                x_ctx, y_ctx = x_real, y_real

            # bf16 autocast halves activation memory on CUDA; no-op on CPU.
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_available(),
            ):
                query_results = select_next_query(
                    model=model,
                    x_ctx=x_ctx,
                    y_ctx=y_ctx,
                    x_mask=gp_sample_function.x_mask,
                    y_mask=gp_sample_function.y_mask,
                    input_bounds=data_cfg.x_range,
                    opt_config=opt_config,
                    d=opt_config.num_query_points,
                    t=n_real + step,
                    T=T,
                    query_chunks=gp_sample_function.chunks,
                    query_x_mask=gp_sample_function.chunk_mask,
                    logit_mask=logit_mask,
                )
            indices = query_results.indices
            logp = query_results.log_probs
            logit_mask = query_results.logit_mask
            del query_results, x_ctx, y_ctx  # trim local refs (graph still held)

            # Append under no_grad: GP lookup needs no autograd, and the
            # prompt must not get baked into a detached x_real downstream.
            with torch.no_grad():
                x_real, y_real, _, regret = gp_sample_function.step(
                    index_new=indices.unsqueeze(-1).unsqueeze(-1),
                    x_ctx=x_real,
                    y_ctx=y_real,
                    compute_hv=False,
                    compute_regret=True,
                    regret_type=opt_config.regret_type,
                )

            neg_regrets_list.append(
                -torch.tensor(regret, device=device, dtype=torch.float32)
            )
            log_probs_list.append(logp)

        neg_regrets = torch.stack(neg_regrets_list, dim=0)
        log_probs = torch.stack(log_probs_list, dim=0)

        if test_time_config.use_softmax_reweighted_loss:
            loss_acq, step_rewards = compute_policy_loss_softmax(
                step_rewards=neg_regrets,
                log_probs=log_probs,
                beta=test_time_config.softmax_beta,
                reduction=test_time_config.softmax_reduction,
                discount_factor=loss_config.discount_factor,
                clip_rewards=loss_config.clip_rewards,
                batch_first=False,
            )
        else:
            loss_acq, step_rewards = compute_policy_loss(
                step_rewards=neg_regrets,
                log_probs=log_probs,
                use_cumulative_r=loss_config.use_cumulative_rewards,
                discount_factor=loss_config.discount_factor,
                batch_standardize=loss_config.batch_standardize,
                clip_rewards=loss_config.clip_rewards,
                batch_first=False,
            )

        step_reward_mean_history.append(step_rewards.mean().detach().item())
        final_step_reward_mean_history.append(
            step_rewards[:, -1].mean().detach().item()
        )
        loss_val = loss_acq.detach().item()

        log(
            f"logp mean/std: {log_probs.mean().item():.4f}/{log_probs.std().item():.4f} | "
            f"reward mean/std: {step_rewards.mean().item():.4f}/{step_rewards.std().item():.4f} | "
            f"final-step reward mean/std: "
            f"{step_rewards[:, -1].mean().item():.4f}/{step_rewards[:, -1].std().item():.4f}"
        )

        optimizer.zero_grad()
        loss_acq.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=loss_config.max_norm)
        optimizer.step()
        _log_epoch_status(
            log=log, epoch=ttt_epoch, params=params,
            loss_name="Loss", loss_val=loss_val, grad_norm=grad_norm,
        )
        # Defragment between epochs.
        torch.cuda.empty_cache()


def _run_predictor_ttt(
    *,
    model: TAMO,
    gp_sample_function: GPSampleFunction,
    optimizer: torch.optim.Optimizer,
    loss_config: LossConfig,
    test_time_config: TestTimeConfig,
    params: List[torch.nn.Parameter],
    prompt: Optional[TTTSoftPrompt],
    device: str,
    log,
    nll_history: List[float],
):
    assert gp_sample_function.x_ctx_padded is not None, \
        "TTT predictor mode requires real observations to condition on."

    B = gp_sample_function.batch_size
    N = gp_sample_function.num_points
    x_mask_b = gp_sample_function.x_mask.unsqueeze(0).expand(B, -1)
    y_mask_b = gp_sample_function.y_mask.unsqueeze(0).expand(B, -1)
    full_dim_x = (
        gp_sample_function.x_mask if gp_sample_function.restore_full_dim_later else None
    )
    full_dim_y = (
        gp_sample_function.y_mask if gp_sample_function.restore_full_dim_later else None
    )

    # Detach once: GP pools carry an rsample graph that would be freed
    # after the first backward and break subsequent epochs.
    x_pool = gp_sample_function._x.detach()
    y_pool = gp_sample_function._y.detach()

    # Real observations, already tiled + padded by gp_sample_function.
    x_ctx_real = gp_sample_function.x_ctx_padded
    y_ctx_real = gp_sample_function.y_ctx_padded

    # Target: all synthetic GP samples (precomputed; constant across epochs).
    all_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, -1, -1)
    x_tar = GPSampleFunction.batch_gather(
        tensor=x_pool, dim=1, index=all_idx, full_dim_mask=full_dim_x,
    )
    y_tar = GPSampleFunction.batch_gather(
        tensor=y_pool, dim=1, index=all_idx, full_dim_mask=full_dim_y,
    )
    use_prompt = test_time_config.optimize_parameters == "prompt"

    for ttt_epoch in tqdm(
        range(test_time_config.num_epoch), desc="test-time training (predictor)"
    ):
        # Re-prepend each epoch so the prompt's autograd path is rebuilt.
        if use_prompt:
            assert prompt is not None
            x_p, y_p = prompt.expand(B)
            x_ctx = torch.cat([x_p, x_ctx_real], dim=1)
            y_ctx = torch.cat([y_p, y_ctx_real], dim=1)
        else:
            x_ctx, y_ctx = x_ctx_real, y_ctx_real

        # bf16 autocast halves activation memory on CUDA; no-op on CPU.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            output = model.predict(
                x_ctx=x_ctx,
                y_ctx=y_ctx,
                x_tar=x_tar,
                x_mask=x_mask_b,
                y_mask=y_mask_b,
                read_cache=False,
            )
        # nanmean: GMM stds are NaN on invalid output dims by design
        # (see GMMPredictionHead._process_parameters).
        nll = GMMPredictionHead.nll_loss(output, y_tar, y_mask=y_mask_b)
        loss = torch.nanmean(nll)

        loss_val = loss.detach().item()
        nll_history.append(loss_val)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=loss_config.max_norm)
        optimizer.step()
        del output, nll, loss, x_ctx, y_ctx  # trim per-epoch graph refs

        _log_epoch_status(
            log=log, epoch=ttt_epoch, params=params,
            loss_name="NLL", loss_val=loss_val, grad_norm=grad_norm,
        )
        # Defragment between epochs.
        torch.cuda.empty_cache()


def test_time_training(
    model: TAMO,
    data_cfg: DataConfig,
    opt_config: OptimizationConfig,
    loss_config: LossConfig,
    T: int,
    device: str,
    test_function: TestFunction,
    params: List[torch.nn.Parameter],
    test_time_config: TestTimeConfig = TestTimeConfig(),
    observed_x=None,
    observed_y_scaled=None,
    log=print,
    plot_save_path: str = "results/plot",
    tag: str = "",
    prompt: Optional[TTTSoftPrompt] = None,
):
    log(f"TTT mode: {test_time_config.mode} | parameters: {sum(p.numel() for p in params):,}")

    start = time.time()
    model.train()
    optimizer = _make_optimizer(params, test_time_config)
    sampler_builder = (
        _build_true_function_sampler
        if test_time_config.use_true_function
        else _build_gp_sample_function
    )
    # The default NUTS path needs autograd for its potential-energy gradients,
    # but run_optimization wraps this whole loop in torch.no_grad().
    with torch.enable_grad():
        gp_sample_function = sampler_builder(
            test_function=test_function,
            data_cfg=data_cfg,
            opt_config=opt_config,
            test_time_config=test_time_config,
            device=device,
            observed_x=observed_x,
            observed_y_scaled=observed_y_scaled,
        )
    # _plot_test_and_gp_samples(test_function, gp_sample_function, get_train_x_range(), plot_save_path)

    step_reward_mean_history: List[float] = []
    final_step_reward_mean_history: List[float] = []
    nll_history: List[float] = []

    with torch.enable_grad():
        if test_time_config.mode == "policy":
            _run_policy_ttt(
                model=model,
                gp_sample_function=gp_sample_function,
                optimizer=optimizer,
                opt_config=opt_config,
                loss_config=loss_config,
                data_cfg=data_cfg,
                test_time_config=test_time_config,
                params=params,
                prompt=prompt,
                T=T,
                device=device,
                log=log,
                step_reward_mean_history=step_reward_mean_history,
                final_step_reward_mean_history=final_step_reward_mean_history,
            )
        elif test_time_config.mode == "predictor":
            _run_predictor_ttt(
                model=model,
                gp_sample_function=gp_sample_function,
                optimizer=optimizer,
                loss_config=loss_config,
                test_time_config=test_time_config,
                params=params,
                prompt=prompt,
                device=device,
                log=log,
                nll_history=nll_history,
            )
        else:
            raise ValueError(
                f"Unknown TTT mode: {test_time_config.mode}. Use 'policy' or 'predictor'."
            )

    torch.cuda.empty_cache()
    _save_ttt_curves(
        plot_save_path=plot_save_path,
        tag=tag,
        step_reward_mean=step_reward_mean_history,
        final_step_reward_mean=final_step_reward_mean_history,
        nll=nll_history,
    )

    model.eval()
    ttt_time = time.time() - start
    log(f"Test-time training time: {ttt_time: .4f}s")
    return ttt_time


def _setup_ttt_parameters(
    *,
    model: TAMO,
    test_function: TestFunction,
    test_time_cfg: TestTimeConfig,
    exp_cfg: ExConfig,
):
    """Configure model param `requires_grad` for TTT.

    Returns (ttt_params, prompt, hook_handle, prev_requires_grad). When TTT
    is disabled, returns empty/None placeholders. `prev_requires_grad` is the
    snapshot needed to restore the original state after run_optimization.
    """
    if not test_time_cfg.enabled:
        return [], None, None, []

    prev_requires_grad = [(p, p.requires_grad) for p in model.parameters()]

    valid = {"policy", "decoder", "encoder", "all", "prompt", "global"}
    if test_time_cfg.optimize_parameters not in valid:
        raise ValueError(f"optimize_parameters must be one of {valid}")

    mode = test_time_cfg.optimize_parameters

    if mode == "prompt":
        # Prompt must match the model's padded dims so torch.cat with x_real /
        # y_real (padded to max_*_dim by gp_sample_function) is valid.
        x_bounds = make_range_tensor(get_train_x_range(), test_function.x_dim).to(exp_cfg.device)
        y_bounds = make_range_tensor(get_train_y_range(), test_function.y_dim).to(exp_cfg.device)
        if test_time_cfg.prompt_init == "from_obs":
            raise NotImplementedError
        else:
            x_init, y_init = None, None
        prompt = TTTSoftPrompt(
            K=test_time_cfg.prompt_K,
            dx_max=test_function.x_dim,
            dy_max=test_function.y_dim,
            x_lo=x_bounds[:, 0],
            x_hi=x_bounds[:, 1],
            y_lo=y_bounds[:, 0],
            y_hi=y_bounds[:, 1],
            x_init=x_init,
            y_init=y_init,
        ).to(exp_cfg.device)
        for p in model.parameters():
            p.requires_grad = False
        return list(prompt.parameters()), prompt, None, prev_requires_grad

    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        return list(model.parameters()), None, None, prev_requires_grad

    # Head-based modes: policy / decoder / encoder / global.
    if test_time_cfg.mode == "policy":
        head_module = model.decoder.policy_head
    elif test_time_cfg.mode == "predictor":
        head_module = model.decoder.prediction_head
    else:
        raise ValueError(
            f"Unknown TTT mode: {test_time_cfg.mode}. Use 'policy' or 'predictor'."
        )

    for p in model.parameters():
        p.requires_grad = False

    ttt_params: List[torch.nn.Parameter] = []
    hook_handle = None

    if mode == "global":
        target_idx = 1 if test_time_cfg.mode == "policy" else 0
        mask = torch.zeros_like(model.decoder.task_tokens)
        mask[target_idx] = 1.0
        model.decoder.task_tokens.requires_grad_(True)
        hook_handle = model.decoder.task_tokens.register_hook(lambda g: g * mask)
        ttt_params.append(model.decoder.task_tokens)
    elif mode in ("policy", "decoder"):
        for p in head_module.parameters():
            p.requires_grad = True
            ttt_params.append(p)
        # "decoder" also unfreezes the transformer MLP layers.
        if mode == "decoder":
            for layer in model.decoder.transformer.layers:
                layer.linear1.requires_grad_(True)
                layer.linear2.requires_grad_(True)
                ttt_params.extend(layer.linear1.parameters())
                ttt_params.extend(layer.linear2.parameters())

            # target_idx = 1 if test_time_cfg.mode == "policy" else 0
            # mask = torch.zeros_like(model.decoder.task_tokens)
            # mask[target_idx] = 1.0

            # model.decoder.task_tokens.requires_grad_(True)
            # hook_handle = model.decoder.task_tokens.register_hook(lambda g: g * mask)
            # ttt_params.append(model.decoder.task_tokens)
    else:  # "encoder"
        for layer in model.encoder.transformer.layers:
            layer.linear1.requires_grad_(True)
            layer.linear2.requires_grad_(True)
            ttt_params.extend(layer.linear1.parameters())
            ttt_params.extend(layer.linear2.parameters())

    return ttt_params, None, hook_handle, prev_requires_grad


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
    test_time_cfg=TestTimeConfig(),
    loss_cfg=None,
    train_opt_cfg=None,
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
        test_time_cfg=test_time_cfg,
        loss_cfg=loss_cfg,
        train_opt_cfg=train_opt_cfg,
    )


def _build_adaptive_q1(
    *,
    x_ctx,
    y_ctx,
    model: TAMO,
    observation_tracker: ObservationTracker,
    model_x_range: NestedFloatList,
    opt_config: OptimizationConfig,
    d: int,
    T: int,
    use_adaptive_global_region: bool = False, 
):
    """Run pass-1 (Sobol Q0 → policy logits) and build Q1 (gaussian around top-K).

    Returns a dict with the pass-1 QueryResult plus the constructed candidates,
    or None if Q0/logits could not be obtained (no refinement possible).
    """
    # `select_next_query` ties `return_logits` to `use_logit_mask` (forwards.py
    # line 402). Adaptive refinement *needs* the logits to pick top-K, so force
    # the flag on for this pass and pass `logit_mask=None` to keep a fresh
    # (no-carryover) mask — the original reason for disabling use_logit_mask
    # in adaptive_opt was to avoid stale-shape carryover, not to suppress logits.
    # Also force use_fixed_query_set=True so OptimizationConfig.__post_init__
    # doesn't clamp use_logit_mask back to False when the run uses
    # use_fixed_query_set=False; pass1 passes `query_chunks=None` explicitly so
    # the Q0 sample is fresh in either configuration.
    pass1_opt = replace(opt_config, use_logit_mask=True, use_fixed_query_set=True)
    pass1 = select_next_query(
        model=model,
        x_ctx=x_ctx,
        y_ctx=y_ctx,
        x_mask=observation_tracker.x_mask,
        y_mask=observation_tracker.y_mask,
        input_bounds=model_x_range,
        opt_config=pass1_opt,
        d=d,
        t=observation_tracker.get_cost_used(),
        T=T,
        observed_target_y_mask=observation_tracker.y_mask_target,
        query_chunks=None,
        query_x_mask=None,
        logit_mask=None,
    )

    q0 = pass1.q_chunk            # [d, max_x_dim]
    logits = pass1.logits         # [B, n, d] or None
    if q0 is None or logits is None:
        return None

    # Average across batch then sum across factorized dims (n=1 for joint policy,
    # n=x_dim for factorized — under independence, log P(Q0_j) = Σ_n logits[n, j]).
    # Previously did mean_logits[0], which silently ignored all but the first dim
    # under factorized policy.
    mean_logits = logits.mean(dim=0)
    if mean_logits.dim() == 2:
        mean_logits = mean_logits.sum(dim=0)
    K = min(opt_config.adaptive_top_k, mean_logits.shape[-1])
    temperature = opt_config.adaptive_topk_temperature
    if temperature > 0.0:
        # Gumbel-top-K: sample K without replacement from softmax(logits / T)
        gumbel = -torch.log(-torch.log(
            torch.rand_like(mean_logits).clamp_min(torch.finfo(mean_logits.dtype).tiny)
        ))
        top_idx = (mean_logits / temperature + gumbel).topk(K).indices
    else:
        top_idx = mean_logits.topk(K).indices         # [K]
    top_q = q0[top_idx]                               # [K, max_x_dim]

    x_mask = observation_tracker.x_mask              # [max_x_dim]
    bounds = make_range_tensor(model_x_range, num_dim=x_mask.shape[-1]).to(top_q.device)
    range_width = (bounds[:, 1] - bounds[:, 0])      # [max_x_dim]
    # Scale the cube/Gaussian half-width to Q0's average nearest-neighbour
    # spacing, so refinement actually bridges gaps between Q0 grid points
    # instead of being a fixed fraction of the global range. With this,
    # `adaptive_sigma_frac` is interpreted as a multiple of Q0 spacing
    # (e.g. 3.0 means the cube half-width is ~3x the inter-Q0-point spacing).
    dx_valid = max(int(x_mask.sum().item()), 1)
    d_q0 = q0.shape[0]
    spacing_frac = (1.0 / d_q0) ** (1.0 / dx_valid)
    radius = (opt_config.adaptive_sigma_frac * spacing_frac * range_width).to(
        dtype=top_q.dtype
    )
    m = opt_config.adaptive_samples_per_top
    dx = top_q.shape[-1]

    mode = opt_config.adaptive_trust_region
    if mode == "gaussian":
        noise = torch.randn(K, m, dx, device=top_q.device, dtype=top_q.dtype) * radius
        q1_adaptive = (top_q.unsqueeze(1) + noise).reshape(K * m, -1)
    elif mode == "cube_sobol":
        # Per-seed cube clipped to the global bounds, then Sobol-sampled.
        cube_lo = torch.maximum(top_q - radius, bounds[:, 0])  # [K, dx]
        cube_hi = torch.minimum(top_q + radius, bounds[:, 1])  # [K, dx]
        cube_width = (cube_hi - cube_lo).unsqueeze(1)          # [K, 1, dx]
        engine = torch.quasirandom.SobolEngine(dimension=dx, scramble=True)
        u = engine.draw(K * m).to(device=top_q.device, dtype=top_q.dtype).view(K, m, dx)
        q1_adaptive = (cube_lo.unsqueeze(1) + u * cube_width).reshape(K * m, -1)
    else:
        raise ValueError(
            f"Unknown adaptive_trust_region: {mode!r}. Expected 'gaussian' or 'cube_sobol'."
        )

    q1_adaptive = torch.clamp(q1_adaptive, min=bounds[:, 0], max=bounds[:, 1])
    q1_adaptive = q1_adaptive * x_mask.to(q1_adaptive.dtype)

    if use_adaptive_global_region:
        q1 = torch.cat([q0, q1_adaptive], dim=0)
    else:
        q1 = q1_adaptive

    return {
        "pass1": pass1,
        "top_q": top_q,
        "std": radius,
        "q1": q1,
        "q1_adaptive": q1_adaptive,
    }


def _adaptive_query_select(
    *,
    fantasy: bool,
    x_ctx,
    y_ctx,
    model: TAMO,
    observation_tracker: ObservationTracker,
    model_x_range: NestedFloatList,
    opt_config: OptimizationConfig,
    pred_config: PredictionConfig,
    d: int,
    T: int,
    logit_mask,
    use_adaptive_global_region=False, 
):
    """Phase-1 two-pass adaptive query selection.

    1. Sample Q0 (Sobol via existing pipeline) and score with the policy.
    2. Take top-K of Q0 via batch-mean logits.
    3. Build Q1 by sampling Gaussian (sigma_frac * per-dim range) around each top-K query.
    4. Score Q1 and pick argmax (with fantasy as configured).
    """
    # Q0 is re-sampled every step and Q1 changes shape across passes, so a
    # carried-over logit_mask from the previous step has the wrong size. Force
    # use_logit_mask off in this path (matches the WARN logged at startup).
    adaptive_opt = replace(opt_config, use_logit_mask=False)

    built = _build_adaptive_q1(
        x_ctx=x_ctx, y_ctx=y_ctx, model=model,
        observation_tracker=observation_tracker, model_x_range=model_x_range,
        opt_config=adaptive_opt, d=d, T=T, 
        use_adaptive_global_region=use_adaptive_global_region
    )

    if built is None:
        # Cannot refine — fall back to the (separately re-run) first-pass result
        pass1 = select_next_query(
            model=model, x_ctx=x_ctx, y_ctx=y_ctx,
            x_mask=observation_tracker.x_mask, y_mask=observation_tracker.y_mask,
            input_bounds=model_x_range, opt_config=adaptive_opt, d=d,
            t=observation_tracker.get_cost_used(), T=T,
            observed_target_y_mask=observation_tracker.y_mask_target,
            logit_mask=None,
        )
        x_ctx_out = torch.cat([x_ctx, pass1.next_x], dim=1) if fantasy else x_ctx
        return x_ctx_out, y_ctx, pass1

    q1 = built["q1"]
    pass1 = built["pass1"]

    # Pass 2: use the constructed q1 as a fixed query set. epsilon controls
    # sample-vs-argmax over Q1 (configurable; default 1.0 preserves prior behavior).
    pass2_opt = replace(
        adaptive_opt,
        use_fixed_query_set=True,
        epsilon=opt_config.adaptive_pass2_epsilon,
    )
    x_ctx_out, y_ctx_out, result = select_next_query_wrapper(
        fantasy=fantasy,
        x_ctx=x_ctx,
        y_ctx=y_ctx,
        model=model,
        observation_tracker=observation_tracker,
        model_x_range=model_x_range,
        opt_config=pass2_opt,
        pred_config=pred_config,
        d=q1.shape[0],
        T=T,
        query_chunks=q1,
        query_x_mask=pass1.q_chunk_mask,
        logit_mask=None,
    )
    # Stash the cube-only candidate set so the caller can compute a separate
    # HV that isn't dominated by the (possibly fixed) global Q0.
    result.q_chunk_adaptive_only = built["q1_adaptive"]
    return x_ctx_out, y_ctx_out, result


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
    test_time_cfg: TestTimeConfig = TestTimeConfig(),
    loss_cfg: LossConfig = LossConfig(),
    train_opt_cfg=None,
):
    """Run the optimization loop on test_function and save results."""
    if predict:
        assert pred_cfg is not None, "`pred_cfg` must be provided if perform prediction"

    metrics = MetricTracker()
    logger = OptimizationLogger(log_fn=log, use_wandb=exp_cfg.log_to_wandb)

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

    # Initial observations
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

    d = get_num_subspace_points(
        x_dim=test_function.x_dim,
        use_factorized_policy=opt_cfg.use_factorized_policy,
        d=opt_cfg.num_query_points,
    )
    T = opt_cfg.sample_T()
    log(f"Subspace points d={d}, cost budget T={T}")

    q_chunk = None
    q_chunk_mask = None
    logit_mask = None

    if opt_cfg.use_adaptive_query_set:
        assert opt_cfg.batch_size == 1, (
            "use_adaptive_query_set requires batch_size=1: top-K is shared across "
            "the batch (the chunk pipeline takes an unbatched [d, max_x_dim] tensor), "
            f"got batch_size={opt_cfg.batch_size}."
        )
        if opt_cfg.use_fixed_query_set or opt_cfg.use_logit_mask:
            log(
                "[WARN] use_adaptive_query_set is enabled: a fresh Sobol Q0 is "
                "drawn at every q-step and Q1 candidates change each step, so "
                "`use_fixed_query_set` and `use_logit_mask` are effectively "
                "ignored in the adaptive path."
            )

    model = model.to(exp_cfg.device)

    if not test_time_cfg.enabled:
        log("[INFO] Test-time training is disabled — skipping parameter setup.")
    ttt_params, prompt, hook_handle, prev_requires_grad = _setup_ttt_parameters(
        model=model,
        test_function=test_function,
        test_time_cfg=test_time_cfg,
        exp_cfg=exp_cfg,
    )

    model.eval()
    with torch.no_grad():
        while observation_tracker.get_cost_used() <= T:
            t = observation_tracker.get_cost_used()
            should_plot = _should_plot(
                t,
                T,
                log_cfg.plot_per_n_steps,
                log_cfg.plot_enabled,
                observation_tracker.initial_cost,
            )

            # ---- Build trust-region overlay for prediction plots (adaptive only) ----
            pred_overlay = None
            if predict and should_plot and opt_cfg.use_adaptive_query_set:
                _y_ctx_scaled_preview = test_function.transform_outputs(
                    outputs=y_ctx, output_bounds=model_y_range
                )
                _built = _build_adaptive_q1(
                    x_ctx=x_ctx,
                    y_ctx=_y_ctx_scaled_preview,
                    model=model,
                    observation_tracker=observation_tracker,
                    model_x_range=model_x_range,
                    opt_config=opt_cfg,
                    d=d,
                    T=T,
                )
                if _built is not None:
                    pred_overlay = {
                        "top_q": _built["top_q"].detach(),
                        "std": _built["std"].detach(),
                    }
                    log(f"pred_overlay is created for visualization.")
                else: 
                    log(f"pred_overlay is not created.")
                    

            # ---- Prediction evaluation ----
            if predict:
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
                    plot_enabled=should_plot,
                    y_mask_history=observation_tracker.y_mask_observed,
                    seed=exp_cfg.seed,
                    overlay=pred_overlay,
                )
                nll_t, mse_t = nll_t.detach(), mse_t.detach()
                metrics.add_prediction_step(nll_t=nll_t, mse_t=mse_t)
                logger.log_prediction_step(step=t, nll_t=nll_t, mse_t=mse_t)
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
                # Release the overlay's top_q / std tensors now that the
                # prediction plot has been saved (matplotlib refs are gone).
                pred_overlay = None

            # ---- Batch query selection ----
            y_ctx_scaled = test_function.transform_outputs(
                outputs=y_ctx, output_bounds=model_y_range
            )

            # ---- Test-time training ----
            if (
                test_time_cfg.enabled
                and t >= test_time_cfg.num_start_ttt
                and t < T
                and t % test_time_cfg.num_obs == 0
            ):
                ttt_time = test_time_training(
                    model=model,
                    data_cfg=data_cfg,
                    opt_config=train_opt_cfg,
                    loss_config=loss_cfg,
                    T=T,
                    device=exp_cfg.device,
                    test_function=test_function,
                    params=ttt_params,
                    test_time_config=test_time_cfg,
                    observed_x=x_ctx,
                    observed_y_scaled=y_ctx_scaled,
                    log=log,
                    plot_save_path=plot_save_path,
                    tag=f"t{t}",
                    prompt=prompt,
                )
                # TTT runs many forward+backward passes; release fragments now.
                torch.cuda.empty_cache()
            else:
                ttt_time = 0.0

            batch_x_ctx, batch_y_ctx = x_ctx.clone(), y_ctx_scaled.clone()

            # If optimizing prompt, append prompt to the context
            if test_time_cfg.optimize_parameters == "prompt":
                _b = batch_x_ctx.shape[0]
                x_p, y_p = prompt.expand(_b)  # [B, K, dx_max], [B, K, dy_max]
                batch_x_ctx = torch.cat([x_p, batch_x_ctx], dim=1)
                batch_y_ctx = torch.cat([y_p, batch_y_ctx], dim=1)

            batch_x_next_list, batch_entr_list = [], []
            batch_infer_time_list = []
            acq_values = None

            for qi in range(opt_cfg.q):
                if opt_cfg.use_adaptive_query_set:
                    batch_x_ctx, batch_y_ctx, action_res = _adaptive_query_select(
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
                        logit_mask=logit_mask,
                        use_adaptive_global_region=opt_cfg.use_adaptive_global_region
                    )
                else:
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
                acq_values = action_res.logits
                q_chunk = action_res.q_chunk
                q_chunk_mask = action_res.q_chunk_mask
                q_chunk_adaptive = action_res.q_chunk_adaptive_only
                logit_mask = action_res.logit_mask
                batch_x_next_list.append(action_res.next_x)
                batch_entr_list.append(action_res.entropy)
                batch_infer_time_list.append(action_res.infer_time)
                observation_tracker.step(update_mask=(qi == opt_cfg.q - 1))

            # NOTE Hack to take ttt_time into account
            batch_infer_time_list[0] += ttt_time

            batch_x_next = torch.cat(batch_x_next_list, dim=1)  # [B, q, max_x_dim]
            batch_entropy = torch.stack(batch_entr_list, dim=1)  # [B, q]

            x_ctx, y_ctx, hv, regret = test_function.step(
                input_bounds=model_x_range,
                x_new=batch_x_next,
                x_ctx=x_ctx,
                y_ctx=y_ctx,
                compute_hv=True,
                compute_regret=True,
                regret_type=opt_cfg.regret_type,
            )
            batch_y_next = y_ctx[:, -opt_cfg.q :]
            hv_next = test_function.compute_hv(
                solutions=batch_y_next, y_mask=observation_tracker.y_mask_target
            )


            # NOTE find the maximal hv that can be obtained on q_chunk
            if q_chunk is not None and q_chunk.dim() == 2:
                # Pass q_chunk to test_function, get function values
                # q_chunk: [d, max_x_dim] -> y_chunk: [d, dy_max]
                y_chunk = test_function.evaluate(
                    x=q_chunk, input_bounds=model_x_range
                )

                # Compute normalized hypervolume on the function values
                # q_chunk is shared across the batch, so a single value suffices
                max_hv_chunk = test_function.compute_hv(
                    solutions=y_chunk.unsqueeze(0),
                    y_mask=observation_tracker.y_mask_target,
                )  # [1]
                del y_chunk  # consumed by compute_hv; no longer needed

                # Also compute HV on the adaptive-only candidates (cubes around
                # top-K, no global Q0). When Q0 is concatenated into q_chunk,
                # the dense Q0 frontier dominates max_hv_chunk; this metric
                # exposes whether the trust-region cubes are themselves
                # contributing meaningful Pareto coverage.
                max_hv_chunk_adaptive = None
                if (
                    q_chunk_adaptive is not None
                    and q_chunk_adaptive.dim() == 2
                ):
                    y_chunk_adaptive = test_function.evaluate(
                        x=q_chunk_adaptive, input_bounds=model_x_range
                    )
                    max_hv_chunk_adaptive = test_function.compute_hv(
                        solutions=y_chunk_adaptive.unsqueeze(0),
                        y_mask=observation_tracker.y_mask_target,
                    )
                    del y_chunk_adaptive

                # Log hv / the max hv over q_chunk
                hv_ratio = hv / max_hv_chunk.clamp(min=1e-8)
                log_msg = (
                    f"  hv / max_hv(q_chunk): {hv_ratio.mean().item():.4f} "
                    f"(hv={hv.mean().item():.4f}, "
                    f"max_hv_chunk={max_hv_chunk.item():.4f}"
                )
                if max_hv_chunk_adaptive is not None:
                    log_msg += (
                        f", max_hv_chunk_adaptive_only="
                        f"{max_hv_chunk_adaptive.item():.4f}"
                    )
                log_msg += ")"
                log(log_msg)
                if exp_cfg.log_to_wandb:
                    import wandb
                    wandb_log = {
                        "opt/max_hv_q_chunk": max_hv_chunk.item(),
                        "opt/hv_over_max_hv_q_chunk": hv_ratio.mean().item(),
                        "opt/step": observation_tracker.get_cost_used(),
                    }
                    if max_hv_chunk_adaptive is not None:
                        wandb_log["opt/max_hv_q_chunk_adaptive_only"] = (
                            max_hv_chunk_adaptive.item()
                        )
                    wandb.log(wandb_log)

            metrics.add_optimization_step(
                hv=hv,
                hv_query=hv_next,
                regret=regret,
                entropy=batch_entropy,
                time=batch_infer_time_list,
                x_query=batch_x_next,
                y_query=batch_y_next,
            )
            logger.log_step(
                step=observation_tracker.get_cost_used(),
                observation_tracker=observation_tracker,
                metrics=metrics,
            )

            # Acquisition heatmap (re-evaluate plot condition after stepping)
            if (
                q_chunk is not None
                and acq_values is not None
                and _should_plot(
                    observation_tracker.get_cost_used(),
                    T,
                    log_cfg.plot_per_n_steps,
                    log_cfg.plot_enabled,
                    observation_tracker.initial_cost,
                )
            ):
                save_fig(
                    plot_acq_values(q_chunk=q_chunk, acq_values=acq_values),
                    plot_save_path,
                    config=opt_cfg,
                    filename=f"acq_heatmap_t{observation_tracker.get_cost_used()}_T{T}",
                    override=exp_cfg.override,
                    log=log,
                    log_to_wandb=exp_cfg.log_to_wandb,
                )

            # Defragment between optimization steps; safe no-op on CPU.
            torch.cuda.empty_cache()

    logger.log_summary(metrics=metrics, test_function=test_function)
    _save_all_data(
        metrics=metrics,
        x_ctx=x_ctx,
        y_ctx=y_ctx,
        data_save_path=data_save_path,
        opt_cfg=opt_cfg,
        exp_cfg=exp_cfg,
        log=log,
    )
    if log_cfg.plot_enabled:
        _save_all_plots(
            metrics=metrics,
            opt_cfg=opt_cfg,
            exp_cfg=exp_cfg,
            plot_save_path=plot_save_path,
            log=log,
        )

    # Restore original requires_grad state so caller's no_grad / eval contracts hold.
    for p, flag in prev_requires_grad:
        p.requires_grad = flag

    del x_ctx, y_ctx, q_chunk, q_chunk_mask, logit_mask
    if hook_handle is not None: 
        hook_handle.remove()

@hydra.main(version_base=None, config_path="configs", config_name="test_ttt.yaml")
def main(config: DictConfig):
    assert config.experiment.mode == "test", f"Set mode to 'test'!"

    torch.set_printoptions(threshold=torch.inf)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    set_all_seeds(config.experiment.seed)

    # ------------------------------------------------------------------
    # Config dataclasses, enabling dot access
    # ------------------------------------------------------------------
    cfg_map = {
        "experiment": ExConfig,
        "prediction": PredictionConfig,
        "optimization": OptimizationConfig,
        "data": DataConfig,
        "log": LogConfig,
        "loss": LossConfig,
        "train_optimization": OptimizationConfig,
        "test_time": TestTimeConfig,
    }
    cfgs = {k: v(**config[k]) for k, v in cfg_map.items()}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_filename = get_log_filename(
        model_name=cfgs["experiment"].model_name,
        expid=cfgs["experiment"].expid,
        prefix=cfgs["experiment"].mode,
    )
    log = get_log_fn(filename=log_filename)

    exp_path = get_exp_path(
        model_name=cfgs["experiment"].model_name,
        expid=cfgs["experiment"].expid,
    )

    log(f"""--- Setup logging and experiment path ---
        Logging information is saving to:\t{log_filename}
        Experiment checkpoint will be read from:\t{exp_path}""")

    if cfgs["experiment"].log_to_wandb:
        log(f"--- Setup W&B ---\n{config.wandb}")
        wandb_init(config=config, **config.wandb)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt = load_checkpoint(
        exp_path=exp_path,
        device=cfgs["experiment"].device,
        resume=cfgs["experiment"].resume,
        ckpt_name=config.extra.ckpt_name,
    )
    model_state_dict = ckpt.get("model", {})
    if not model_state_dict:
        raise RuntimeError(
            f"Invalid checkpoint loaded from {exp_path}. "
            "Checkpoint is either empty or missing the 'model' key."
        )

    # ------------------------------------------------------------------
    # Build model and load weights
    # ------------------------------------------------------------------
    model = build_tamo(dict(config.model))
    model = model.to(cfgs["experiment"].device)
    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    if missing:
        log(f"[WARNING] Missing keys after checkpoint load:\n  " + "\n  ".join(missing))
    if unexpected:
        log(f"[WARNING] Unexpected keys in checkpoint:\n  " + "\n  ".join(unexpected))

    log(
        f"--- Model built: TAMO ---\n"
        f"  Config: {model.config}\n"
        f"  Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    # ------------------------------------------------------------------
    # Data / Function env
    # ------------------------------------------------------------------
    if cfgs["experiment"].task == "prediction":
        datapaths, function_name = map_function_to_gp_datapath(
            function_name=cfgs["data"].function_name,
            mode=cfgs["experiment"].mode,
            data_id=cfgs["data"].data_id,
        )
        assert datapaths, f"Unsupported function name: {cfgs['data'].function_name}"
    elif cfgs["experiment"].task == "optimization":
        test_function = get_function_environment(
            function_name=cfgs["data"].function_name,
            seed=cfgs["experiment"].seed,
            device=cfgs["experiment"].device,
            data_id=cfgs["data"].data_id,
            scene=cfgs["data"].scene,
        )
        function_name = cfgs["data"].function_name

        # NOTE hack: at test time, all dimensions are valid, so we adjust the max dimensions in data configurations according to test function dimensions
        # And the learnable prompts would be of the same dimensions as the test function
        cfgs["data"].max_x_dim = test_function.x_dim
        cfgs["data"].max_y_dim = test_function.y_dim
    else:
        raise ValueError(
            f"Unsupported task: {cfgs['experiment'].task}. "
            "Supported tasks are 'prediction' and 'optimization'."
        )

    # ------------------------------------------------------------------
    # Save paths
    # ------------------------------------------------------------------
    _filename_base = get_filename_base(
        function_name=function_name,
        ckpt_name=config.extra.ckpt_name,
        suffix_segment=config.extra.suffix_segment,
    )
    plot_save_path = get_result_plot_path(
        model_name=cfgs["experiment"].model_name,
        expid=cfgs["experiment"].expid,
        task_type=cfgs["experiment"].task,
        filename_base=_filename_base,
    )
    data_save_path = get_result_data_path(
        model_name=cfgs["experiment"].model_name,
        expid=cfgs["experiment"].expid,
        task_type=cfgs["experiment"].task,
        filename_base=_filename_base,
    )

    log(f"""--- Setup saving paths ---
        plot_save_path:\t{plot_save_path}
        data_save_path:\t{data_save_path}""")

    evaluate_optimization(
        model=model,
        test_function=test_function,
        plot_save_path=plot_save_path,
        data_save_path=data_save_path,
        exp_cfg=cfgs["experiment"],
        opt_cfg=cfgs["optimization"],
        data_cfg=cfgs["data"],
        pred_cfg=cfgs["prediction"],
        log_cfg=cfgs["log"],
        loss_cfg=cfgs["loss"],
        train_opt_cfg=cfgs["train_optimization"],
        test_time_cfg=cfgs["test_time"],
        log=log,
    )


if __name__ == "__main__":
    main()
