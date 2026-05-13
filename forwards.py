"""Optimization and prediction forwards."""

from enum import Enum
import random
from dataclasses import dataclass
from typing import Optional, Tuple
import time

from einops import repeat
import numpy as np
from torch import Tensor
import torch
import torch.nn.functional as F

from utils.types import FloatListOrNestedOrTensor, NestedFloatList
from utils.dataclasses import (
    OptimizationConfig,
    PredictionConfig,
    DataConfig,
    LossConfig,
)
from utils.config import get_train_y_range
from data.obs_tracker import ObservationTracker
from data.function_sampling import factorized_to_flat_index, sample_factorized_domain
from data.gp_sample_function import GPSampleFunction
from data.moo import MOO
from model import TAMO
from model.layers import GMMPredictionHead


GAMMA = 0.99


class SamplingMode(str, Enum):
    """Modes for query point sampling.
    
    - FULL:         sample d query points from design space
    - REWEIGHTED:   rewighted sample d query points from a larger set based on predicted scores (negative simple regret)
    - TOPD:         sample d query points from a larger set with top-d highest predicted scores
    """
    FULL = "full"
    REWEIGHTED = "reweighted"
    TOPD = "top_d"


@dataclass
class QueryResult:
    next_x: Tensor  # [B, 1, max_x_dim]
    indices: Tensor  # [B]
    log_probs: Tensor  # [B]
    entropy: Tensor  # [B]
    logits: Optional[Tensor]  # [B, n, d]
    q_chunk: Tensor  # [d, max_x_dim]
    q_chunk_mask: Tensor  # [n, max_x_dim]
    infer_time: float
    logit_mask: Optional[Tensor]  # [B, n, d]
    q_chunk_adaptive_only: Optional[Tensor] = None  # [K*m, max_x_dim], adaptive cubes only (no Q0)

    def _as_tuple(self):
        return (
            self.next_x,
            self.indices,
            self.log_probs,
            self.entropy,
            self.logits,
            self.q_chunk,
            self.q_chunk_mask,
            self.infer_time,
            self.logit_mask,
            self.q_chunk_adaptive_only,
        )

    def __iter__(self):
        return iter(self._as_tuple())

    def __getitem__(self, i):
        return self._as_tuple()[i]


# ===========================
# query proposal 
# ===========================
# == helpers ==
def _mask_out_used_chunks(
    logit_mask: Tensor,  # [B, n, d]
    used_indices: Tensor,  # [B, n]
) -> Tensor:
    """Mask visited indices as `False`.

    Args:
        logit_mask:     [B, n, d], `False` means visited
        used_indices:   [B, n]

    Returns: mask of shape [B, n, d]
    """
    B, n = used_indices.shape
    d = logit_mask.shape[-1]

    # when samples come from factorized spaces, masking out one element in one chunk will also mask out all related designs on full space.
    assert n == 1, f"Only support full policy (n=1) for now"

    logit_mask = logit_mask.bool().reshape(B * n, -1)  # [B * n, d]
    logit_mask[torch.arange(B * n), used_indices.reshape(-1)] = False
    return logit_mask.reshape(B, n, d)


def _reweighted_sample(
    model: TAMO,
    opt_config: OptimizationConfig,
    x_ctx: Tensor,
    y_ctx: Tensor,
    x_mask: Tensor,
    y_mask: Tensor,
    input_bounds: Tensor,
    d: int,
    sampling_mode: str = SamplingMode.REWEIGHTED.value,
    num_samples: int = 4096,
    t: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Reweigted sample the queries by taking the top-d candidates.
    p(y | D_c) =
    integral p(y, x | D_c)dx = integral p(y | x, D_c)p(x)dx approximates
    1/n sum^n_i p(y | x_i, D_c)p(x_i) -> this is exactly what we are doing here
    TODO can we learn p(y | D_c) directly

    Args:
        model:          TAMO
        opt_config:     Optimization config.
        x_ctx:          Context inputs.
        y_ctx:          Context outcomes.
        x_mask:         Mask for valid x dimensions.
        y_mask:         Mask for valid y dimensions.
        input_bounds:   Bounds of inputs.
        d:              Number of queries.
        num_samples:    Number of total samples to be reweighted sampling.

    Returns: query chunk of shape [B, n, d, dx_max],
    query x mask of shape [n, dx_max]
    """
    if opt_config.use_factorized_policy:
        raise NotImplementedError(
            "Reweighted sampling is not implemented for factorized policy."
        )

    B, _, max_x_dim = x_ctx.shape

    # Sample: [ns, dx_max], [n_chunks, dx_max]
    num_samples = max(d, num_samples)
    query_chunks, query_x_mask = sample_factorized_domain(
        d=num_samples,
        max_x_dim=max_x_dim,
        device=x_ctx.device,
        x_mask=x_mask,
        input_bounds=input_bounds,
        use_grid_sampling=opt_config.use_grid_sampling,
        use_factorized_policy=opt_config.use_factorized_policy,
        seed=int(t) * int(num_samples),
    )

    # Expand batch dim
    x_mask_expanded = repeat(x_mask, "d -> B d", B=B)
    y_mask_expanded = repeat(y_mask, "d -> B d", B=B)
    query_chunks_expanded = repeat(query_chunks, "n d -> B n d", B=B)

    # Take a subset of samples
    if num_samples > d:
        # Compute scores based on predicted function values
        out = model.predict(
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_tar=query_chunks_expanded,
            x_mask=x_mask_expanded,
            y_mask=y_mask_expanded,
            read_cache=False,
        )
        y_tar = GMMPredictionHead.expected_value(out)
        y_counts = y_mask.int().sum().item()

        if y_counts == 1:
            # Single objective *minimization*
            scores = -(y_tar * y_mask.int()).sum(dim=-1)  # [B, ns]
        else:
            # Multi-objective minimization
            # Compute hypervolume from taking each solution - expensive
            y_range = get_train_y_range()
            minimum = torch.tensor(y_range[0]).expand(y_tar.shape[-1])
            maximum = torch.tensor(y_range[1]).expand(y_tar.shape[-1])

            # Batch all samples into a single compute_hv call: [B*ns, N_ctx+1, dy_max]
            y_ctx_rep = y_ctx.repeat_interleave(num_samples, dim=0)
            y_tar_flat = y_tar.reshape(B * num_samples, 1, y_tar.shape[-1])
            solutions_all = torch.cat([y_ctx_rep, y_tar_flat], dim=1)
            y_mask_all = y_mask_expanded.repeat_interleave(num_samples, dim=0)

            hv_all = MOO.compute_hv(
                solutions=solutions_all,
                y_mask=y_mask_all,
                minimum=minimum,
                maximum=maximum,
            )[0]

            # [B, ns]
            scores = torch.from_numpy(hv_all).reshape(B, num_samples).to(x_ctx)

        if sampling_mode == SamplingMode.REWEIGHTED.value:
            # Reweighted sampling based on scores and samples
            scores = scores - scores.max(dim=-1, keepdim=True).values  # safe softmax
            probs = F.softmax(scores, dim=-1)  # [B, num_samples]
            sampled_indices = torch.multinomial(probs, num_samples=d, replacement=False)

        elif sampling_mode == SamplingMode.TOPD.value:
            # Top-d
            _, sorted_indices = torch.sort(scores, dim=-1, descending=True)
            sampled_indices = sorted_indices[:, :d]
        else: 
            raise ValueError(sampling_mode)

        # Gather samples: [B, ns]
        query_chunks_expanded = torch.gather(
            query_chunks_expanded,
            dim=1,
            index=sampled_indices.unsqueeze(-1).expand(-1, -1, max_x_dim),
        )

    n = query_x_mask.shape[0]
    query_chunks_expanded = query_chunks_expanded.unsqueeze(1).expand(-1, n, -1, -1)
    return query_chunks_expanded, query_x_mask


def _prepare_query_chunks(
    model,
    opt_config,
    x_ctx,
    y_ctx,
    x_mask,
    y_mask,
    input_bounds,
    d,
    query_chunks,
    q_x_mask,
    t: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns (query_chunks, query_chunks_expanded, query_x_mask).

    `t` (current cost step) is used as a per-step Sobol-skip multiplier so
    fresh-Q0 paths produce non-overlapping Sobol windows step-to-step.
    """
    B, _, dx_max = x_ctx.shape
    if opt_config.sampling_mode == SamplingMode.FULL.value:
        if query_chunks is None or not opt_config.use_fixed_query_set:
            query_chunks, q_x_mask = sample_factorized_domain(
                d=d,
                max_x_dim=dx_max,
                device=x_ctx.device,
                x_mask=x_mask,
                input_bounds=input_bounds,
                use_grid_sampling=opt_config.use_grid_sampling,
                use_factorized_policy=opt_config.use_factorized_policy,
                seed=int(t) * int(d),
            )

        expanded = (
            query_chunks.unsqueeze(0)
            .unsqueeze(0)
            .expand(B, q_x_mask.shape[0], -1, -1)
            .contiguous()
        )
        return query_chunks, expanded, q_x_mask
    else:
        expanded, q_x_mask = _reweighted_sample(
            model=model,
            opt_config=opt_config,
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_mask=x_mask,
            y_mask=y_mask,
            input_bounds=input_bounds,
            d=d,
            sampling_mode=opt_config.sampling_mode,
            num_samples=opt_config.num_reweighted_samples,
            t=t,
        )
        return None, expanded, q_x_mask


def _expand_masks(x_mask, y_mask, query_x_mask, observed_target_y_mask, B):
    """Broadcast all masks to batch dim."""
    return (
        repeat(x_mask, "d -> B d", B=B),
        repeat(y_mask, "d -> B d", B=B),
        repeat(query_x_mask, "n d -> B n d", B=B),
        (
            None
            if observed_target_y_mask is None
            else repeat(observed_target_y_mask, "d -> B d", B=B)
        ),
    )


def _unpack_action_results(results, B, n, d, dx_max):
    """Reshape raw model output into (next_x, chunk_indices, log_probs, entropy, logits)."""
    next_x_raw, indices_raw, logp_raw, entropy_raw = results[:4]
    chunk_indices = indices_raw.reshape(B, n).detach()
    return (
        next_x_raw.reshape(B, 1, dx_max).detach(),
        chunk_indices,
        logp_raw.reshape(B, n).sum(-1),  # log_probs [B]
        entropy_raw.reshape(B, n).sum(-1),  # entropy   [B]
        results[4] if len(results) > 4 else None,
    )


def select_next_query(
    model: TAMO,
    x_ctx: Tensor,
    y_ctx: Tensor,
    x_mask: Tensor,
    y_mask: Tensor,
    input_bounds: FloatListOrNestedOrTensor,
    opt_config: OptimizationConfig,
    d: int,
    t: int,
    T: int,
    observed_target_y_mask: Optional[Tensor] = None,
    query_chunks: Optional[Tensor] = None,
    query_x_mask: Optional[Tensor] = None,
    auto_clear_cache: bool = True,
    logit_mask: Optional[Tensor] = None,
) -> QueryResult:
    """Select the next query point based on current context from the query set.

    Args:
        model:                  TAMO
        x_ctx:                  [B, num_ctx, max_x_dim]
        y_ctx:                  [B, num_ctx, max_y_dim]
        x_mask:                 [max_x_dim], mask for valid x dimensions
        y_mask:                 [max_y_dim], mask for valid y dimensions
        input_bounds:           Bounds for inputs
        opt_config:             Optimization config
        d:                      Number of candidate queries for joint policy, and number of subspace samples for factorized policy
        t:                      Current time step
        T:                      Total budget
        observed_target_y_mask: [max_y_dim], optional mask for observed y dimensions of targets
        query_chunks:           [d, max_x_dim], optional query data
        query_x_mask:           [n, max_x_dim], optional mask for query x dimensions per subspace
        auto_clear_cache:       Whether to clean up cached embedding at t=T
        logit_mask:             [B, n, d], optional mask for queried inputs

    Returns: QueryResult dataclass containing
            proposed query [B, 1, max_x_dim],
            indices of proposed query [B],
            log prob of proposed query (with gradients) [B],
            policy's entropy [B],
            optional logit values [B, n, d],
            query chunk [d, max_x_dim],
            query x mask [n, max_x_dim],
            inference time in seconds,
            optional logit mask [B, n, d]
    """
    B, _, dx_max = x_ctx.shape

    query_chunks, query_chunks_expanded, query_x_mask = _prepare_query_chunks(
        model,
        opt_config,
        x_ctx,
        y_ctx,
        x_mask,
        y_mask,
        input_bounds,
        d,
        query_chunks,
        query_x_mask,
        t=t,
    )

    n = query_x_mask.shape[0]
    x_mask_e, y_mask_e, qx_mask_e, obs_mask_e = _expand_masks(
        x_mask, y_mask, query_x_mask, observed_target_y_mask, B
    )

    # Sanity check: disable logit mask if reweighted sampling, or query set is smaller than budget
    use_logit_mask = False if d < T else opt_config.use_logit_mask

    if use_logit_mask:
        if logit_mask is None:
            logit_mask = torch.ones((B, n, d), device=x_ctx.device, dtype=torch.bool)
    else:
        logit_mask = None

    t0 = time.time()
    results = model.action(
        x_ctx=x_ctx,
        y_ctx=y_ctx,
        x_mask=x_mask_e,
        y_mask=y_mask_e,
        query_chunks=query_chunks_expanded,
        query_x_mask=qx_mask_e,
        observed_target_y_mask=obs_mask_e,
        t=t,
        T=T,
        use_budget=opt_config.use_time_budget,
        return_logits=use_logit_mask,
        read_cache=opt_config.read_cache,
        write_cache=opt_config.write_cache,
        auto_clear_cache=auto_clear_cache,
        logit_mask=logit_mask,
        epsilon=opt_config.epsilon,
    )
    infer_time = time.time() - t0

    next_x, chunk_indices, log_probs, entropy, logits = _unpack_action_results(
        results, B, n, d, dx_max
    )
    indices = factorized_to_flat_index(chunk_indices, n=n, d=d).squeeze(-1)

    if use_logit_mask:
        logit_mask = _mask_out_used_chunks(
            logit_mask=logit_mask, used_indices=chunk_indices
        )

    return QueryResult(
        next_x=next_x,
        indices=indices,
        log_probs=log_probs,
        entropy=entropy,
        logits=logits,
        q_chunk=query_chunks,
        q_chunk_mask=query_x_mask,
        infer_time=infer_time,
        logit_mask=logit_mask,
    )


def select_next_query_wrapper(
    fantasy: bool,
    x_ctx: Tensor,
    y_ctx: Tensor,
    model: TAMO,
    observation_tracker: ObservationTracker,
    model_x_range: NestedFloatList,
    opt_config: OptimizationConfig,
    pred_config: PredictionConfig,
    d: int,
    T: int,
    query_chunks: Optional[Tensor] = None,
    query_x_mask: Optional[Tensor] = None,
    logit_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, QueryResult]:
    """Wrapper around select_next_query for evaluation; handles context fantasization and input rescaling.

    Args:
        fantasy:            Whether to update context with fantasized outcomes
        x_ctx:              [B, N_ctx, x_dim]
        y_ctx:              [B, N_ctx, y_dim]
        model:              TAMO
        observation_tracker: Observed mask and cost tracker
        model_x_range:      Input bounds used during model training
        opt_config:         Optimization configuration
        pred_config:        Prediction configuration
        d:                  Number of subspace points for query selection
        T:                  Total optimization budget
        query_chunks:       [d, max_x_dim], optional precomputed query chunks
        query_x_mask:       [n, max_x_dim], optional masks for query chunks
        logit_mask:         [B, n, d], optional mask for queried inputs

    Returns: updated x_ctx [B, N_ctx+1, x_dim], y_ctx [B, N_ctx+1, y_dim], QueryResult
    """
    if fantasy:
        assert not opt_config.write_cache
        assert not opt_config.read_cache
        assert not pred_config.read_cache

    result = select_next_query(
        model=model,
        x_ctx=x_ctx,
        y_ctx=y_ctx,
        x_mask=observation_tracker.x_mask,
        y_mask=observation_tracker.y_mask,
        input_bounds=model_x_range,
        opt_config=opt_config,
        d=d,
        t=observation_tracker.get_cost_used(),
        T=T,
        observed_target_y_mask=observation_tracker.y_mask_target,
        query_chunks=query_chunks,
        query_x_mask=query_x_mask,
        logit_mask=logit_mask,
    )

    if fantasy:
        # Update context with fantasized outcome
        x_ctx = torch.cat([x_ctx, result.next_x], dim=1)

        # Prepare expanded masks
        b = x_ctx.shape[0]
        x_mask_exp = repeat(observation_tracker.x_mask, "d -> b d", b=b)
        y_mask_exp = repeat(observation_tracker.y_mask, "d -> b d", b=b)
        y_mask_tar_exp = repeat(observation_tracker.y_mask_target, "d -> b d", b=b)

        # Predict fantasized outcome
        out = model.predict(
            x_ctx=x_ctx[:, :-1],
            y_ctx=y_ctx,
            x_tar=x_ctx,
            x_mask=x_mask_exp,
            y_mask=y_mask_exp,
            observed_target_y_mask=y_mask_tar_exp,
            read_cache=False,
        )
        mean = GMMPredictionHead.expected_value(out)[:, -1:, :]
        y_ctx = torch.cat([y_ctx, mean], dim=1)

    return x_ctx, y_ctx, result


# ===========================
# Loss computations, forwards  
# ===========================
# == helpers ==
def _get_cumulative_rewards(reward: Tensor, discount_factor: float = GAMMA) -> Tensor:
    """Compute discount future rewards: R_t = r_t + gamma * R_{t+1}

    Args:
        reward:             [B, H]
        discount_factor:    gamma

    Returns: discounted future rewards of shape [B, H]
    """
    _, H = reward.shape
    cumulative_rewards = torch.zeros_like(reward)

    for t in reversed(range(H)):
        if t == H - 1:
            cumulative_rewards[:, t] = reward[:, t]
        else:
            cumulative_rewards[:, t] = (
                reward[:, t] + discount_factor * cumulative_rewards[:, t + 1]
            )

    return cumulative_rewards


def _standardize(
    step_rewards: Tensor,
    batch_standardize: bool,
    eps=np.finfo(np.float32).eps.item(),
) -> Tensor:
    B, H = step_rewards.shape
    dim = 0 if batch_standardize else -1
    assert (B if batch_standardize else H) > 1
    return (step_rewards - step_rewards.mean(dim=dim, keepdim=True)) / (
        step_rewards.std(dim=dim, keepdim=True) + eps
    )


def compute_policy_loss(
    step_rewards: Tensor,  # [B, H]
    log_probs: Tensor,  # [B, H]
    eps: float = np.finfo(np.float32).eps.item(),
    use_cumulative_r: bool = True,
    discount_factor: float = GAMMA,
    batch_standardize: bool = True,
    clip_rewards: bool = True,
    batch_first: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Compute policy learning loss.

    Args:
        step_rewards:       [B, H], immediate rewards at each step
        log_probs:          [B, H], log-prob of each step's action
        eps:                small value to avoid division by zero
        use_cumulative_r:   whether to use discounted future rewards
        discount_factor:    gamma
        batch_standardize:  whether to standardize rewards over batch dimension
        clip_rewards:       whether to zero out rewards that don't improve best-so-far

    Returns: loss of shape [1], (clipped) immediate rewards of shape [B, H]
    """
    if not batch_first:
        # [H, B] -> [B, H]
        step_rewards = step_rewards.transpose(0, 1)
        log_probs = log_probs.transpose(0, 1)

    B, H = step_rewards.shape
    assert log_probs.shape == (B, H), f"{log_probs.shape}"

    # No gradients from rewards
    step_rewards = step_rewards.detach()

    if clip_rewards:
        # [1, 0, 3, 2, 4] -> [1, 1, 3, 3, 4]
        step_rewards_cummax = torch.cummax(step_rewards, dim=-1).values

        # [1, 1, 3, 3, 4] * [T, F, T, F, T] = [1, 0, 3, 0, 4]
        is_info = step_rewards == step_rewards_cummax
        step_rewards *= is_info.float()

    if use_cumulative_r:
        reward = _get_cumulative_rewards(
            reward=step_rewards, discount_factor=discount_factor
        )
        reward = _standardize(reward, batch_standardize, eps)
    else:
        reward = _standardize(step_rewards, batch_standardize, eps)
        discounts = discount_factor ** torch.arange(H, device=reward.device)
        reward = discounts * reward

    loss = -reward * log_probs

    return torch.mean(loss), step_rewards


def optimization_forward(
    model: TAMO,
    data_cfg: DataConfig,
    opt_config: OptimizationConfig,
    loss_config: LossConfig,
    T: int,
    device: str,
):
    """Optimization forward (model + loss)"""
    # Initialize sampler
    gp_sample_function = GPSampleFunction(
        data_config=data_cfg,
        batch_size=opt_config.batch_size,
        num_samples=opt_config.num_samples,
        d=opt_config.num_query_points,
        use_grid_sampling=opt_config.use_grid_sampling,
        use_factorized_policy=opt_config.use_factorized_policy,
        device=device,
    )

    # Initializations
    if opt_config.random_num_initial:
        num_initial_points = random.randint(1, T - 1)
    else:
        num_initial_points = opt_config.num_initial_points

    x_ctx, y_ctx, _, _ = gp_sample_function.init(
        num_initial_points=num_initial_points,
        regret_type=opt_config.regret_type,
        compute_hv=False,
        compute_regret=False,
        device=device,
    )

    # Preallocate tensors
    B = x_ctx.shape[0]  # num_tasks x num_samples
    neg_regrets = torch.empty((T, B), device=device)
    log_probs = torch.empty((T, B), device=device)

    for t in range(1, T + 1):
        query_results = select_next_query(
            model=model,
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_mask=gp_sample_function.x_mask,
            y_mask=gp_sample_function.y_mask,
            input_bounds=data_cfg.x_range,
            opt_config=opt_config,
            d=opt_config.num_query_points,
            t=t,
            T=T,
            query_chunks=gp_sample_function.chunks,
            query_x_mask=gp_sample_function.chunk_mask,
        )
        indices = query_results.indices
        logp = query_results.log_probs
        last_entropy = query_results.entropy

        # Update context with new query points
        x_ctx, y_ctx, _, regret = gp_sample_function.step(
            index_new=indices.unsqueeze(-1).unsqueeze(-1),
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            compute_hv=False,
            compute_regret=True,
            regret_type=opt_config.regret_type,
        )

        # Update tensors
        neg_regret = -torch.tensor(regret, device=device, dtype=torch.float32)
        neg_regrets[t - 1] = neg_regret
        log_probs[t - 1] = logp
        last_entropy = last_entropy.detach()

    # Compute policy loss over all trajectories
    loss_acq, step_rewards = compute_policy_loss(
        step_rewards=neg_regrets,
        log_probs=log_probs,
        use_cumulative_r=loss_config.use_cumulative_rewards,
        discount_factor=loss_config.discount_factor,
        batch_standardize=loss_config.batch_standardize,
        clip_rewards=loss_config.clip_rewards,
        batch_first=False,
    )

    # Compute statistics
    step_reward_mean = step_rewards.mean().detach().item()
    final_step_reward_mean = step_rewards[:, -1].mean().detach().item()
    final_step_entropy_mean = last_entropy.mean().detach().item()

    del gp_sample_function, x_ctx, y_ctx, query_results

    return (loss_acq, step_reward_mean, final_step_reward_mean, final_step_entropy_mean)


def _reduce(
    tensor: Tensor, dim: int | tuple = None, reduction: str = "nanmean"
) -> Tensor:
    """Reduce a tensor along the specified dimension.

    Args:
        tensor:     Can be of any shape
        dim:        Dimension(s) to reduce. If None, reduces all
        reduction:  ["mean", "sum", "nanmean"]

    Returns: tensor reduced along `dim` according to `reduction` mode.
    """
    if reduction == "nanmean":
        return torch.nanmean(tensor, dim=dim)
    elif reduction == "mean":
        return torch.mean(tensor, dim=dim)
    elif reduction == "sum":
        return torch.sum(tensor, dim=dim)
    else:
        raise ValueError(
            f"Invalid reduction type: {reduction}. Must be one of ['mean', 'sum', 'nanmean']."
        )


def prediction_forward(
    model: TAMO,
    x_ctx: Tensor,
    y_ctx: Tensor,
    x_tar: Tensor,
    y_tar: Tensor,
    x_mask: Tensor,
    y_mask: Tensor,
    y_mask_tar: Optional[Tensor] = None,
    read_cache: bool = False,
):
    """Forward pass for prediction (model + loss).

    Args:
        model:      TAMO
        x_ctx:      [B, nc, max_x_dim]
        y_ctx:      [B, nc, max_y_dim]
        x_tar:      [B, nt, max_x_dim]
        y_tar:      [B, nt, max_y_dim]
        x_mask:     [B, dx_max]
        y_mask:     [B, dy_max]
        read_cache: whether to read context embedding from cache

    Returns:
        nll of shape [1],
        mse of shape [max_y_dim],
        inference time
    """
    # GMMOutput: (means, stds, weights) of shape [B, nt, dy_max, K]
    t1 = time.time()
    output = model.predict(
        x_ctx=x_ctx,
        y_ctx=y_ctx,
        x_tar=x_tar,
        x_mask=x_mask,
        y_mask=y_mask,
        observed_target_y_mask=y_mask_tar,
        read_cache=read_cache,
    )
    inference_time = time.time() - t1

    nll = GMMPredictionHead.nll_loss(output, y_tar)
    nll = _reduce(nll)

    mean = GMMPredictionHead.expected_value(output).detach()
    mse = F.mse_loss(input=mean, target=y_tar, reduction="none")
    mse = _reduce(mse, dim=(0, 1))

    return nll, mse, inference_time


def _get_opt_curriculum(
    num_cur,
    num_total,
    intervals=(0.0, 0.25, 0.5, 0.75, 1.0),
    points=(32, 64, 128, 256),
    batchsizes=(32, 8, 4, 4),
    horizons=(25, 50, 75, 100),
):
    assert num_cur <= num_total, f"{num_cur} > {num_total}"

    frac = num_cur / num_total

    # [0,1,2,3]
    stage = sum(frac > thres for thres in intervals[1:])
    return points[stage], horizons[stage], batchsizes[stage]
