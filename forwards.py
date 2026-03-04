"""Optimization and prediction forwards."""

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
from utils.dataclasses import OptimizationConfig, PredictionConfig, DataConfig, LossConfig
from data.states import ObservationTracker
from data.function_sampling import factorized_to_flat_index, sample_factorized_domain
from data.gp_sample_function import GPSampleFunction
from model import TAMO
from model.layers import GMMPredictionHead


@dataclass
class QueryResult:
    next_x: Tensor  # [B, 1, max_x_dim] - selected query point
    indices: Tensor  # [B] - flat indices of selected points
    log_probs: Tensor  # [B] - log probabilities (gradients preserved)
    entropy: Tensor  # [B] - entropy of the policy
    logits: Optional[Tensor]  # [B, n, d] - raw logits if evaluate=True
    q_chunk: Tensor  # [d, max_x_dim] - query chunks
    q_chunk_mask: Tensor  # [n, max_x_dim] - chunk masks
    infer_time: float  # inference time in seconds
    logit_mask: Optional[Tensor]  # [B, n, d] - updated logit mask

    def __iter__(self):
        """Enable tuple unpacking for backward compatibility."""
        return iter(
            (
                self.next_x,
                self.indices,
                self.log_probs,
                self.entropy,
                self.logits,
                self.q_chunk,
                self.q_chunk_mask,
                self.infer_time,
                self.logit_mask,
            )
        )

    def __getitem__(self, index: int):
        """Enable index access for backward compatibility."""
        fields = (
            self.next_x,
            self.indices,
            self.log_probs,
            self.entropy,
            self.logits,
            self.q_chunk,
            self.q_chunk_mask,
            self.infer_time,
            self.logit_mask,
        )
        return fields[index]


GAMMA = 0.99


def _get_cumulative_rewards(reward: Tensor, discount_factor: float = 0.98) -> Tensor:
    """Compute discount future return.

    Args:
        reward: [B, H]
        discount_factor (float): discount factor on future rewards

    Returns:
        cumulative_rewards of shape [B, H]
    """
    _, H = reward.shape
    cumulative_rewards = torch.zeros_like(reward)

    for t in reversed(range(H)):
        # t = H - 1: cumulative_rewards[:, H - 1] = reward[:, H - 1]
        # t = H - 2: cumulative_rewards[:, H - 2] = reward[:, H - 2] + discount_factor * cumulative_rewards[:, H - 1]
        if t == H - 1:
            # the last step: R_t = r_t
            cumulative_rewards[:, t] = reward[:, t]
        else:
            # other steps: R_t = r_t + gamma * R_{t+1}
            cumulative_rewards[:, t] = (
                reward[:, t] + discount_factor * cumulative_rewards[:, t + 1]
            )

    return cumulative_rewards


def _standardize(
    B: int,
    H: int,
    step_rewards: Tensor,
    batch_standardize: bool,
    eps=np.finfo(np.float32).eps.item(),
) -> Tensor:
    """Reward standardization along batch dim or horizon dim."""
    if batch_standardize:
        assert B > 1
        rewards = (step_rewards - step_rewards.mean(dim=0, keepdim=True)) / (
            step_rewards.std(dim=0, keepdim=True) + eps
        )
    else:
        assert H > 1
        rewards = (step_rewards - step_rewards.mean(dim=-1, keepdim=True)) / (
            step_rewards.std(dim=-1, keepdim=True) + eps
        )

    # print(f"reward after standardization: \n{rewards}")
    return rewards


def compute_policy_loss(
    step_rewards: Tensor,  # [B, H]
    log_probs: Tensor,  # [B, H]
    eps: float = np.finfo(np.float32).eps.item(),
    use_cumulative_r: bool = True,
    discount_factor: float = GAMMA,
    batch_standardize: bool = True,
    clip_rewards: bool = True,
    batch_first: bool = True,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """Compute policy learning loss.

    Args:
        step_rewards (Tensor): immediate rewards of each step's action, [B, H]
        log_probs (Tensor): log probabilities of each step's action, [B, H]
        eps (float): small value to avoid division by zero when standardizing rewards
        use_cumulative_r (bool): whether to use discount future rewards or immediate rewards
        discount_factor (float): i.e., gamma.
        batch_standardize: whether to standardize rewards over batch dimension or horizon dimension
        clip_rewards: whether to clip rewards to zero if they are not informative

    Returns: loss of shape [1], (clipped) step rewards of shape [B, H]
    """
    if not batch_first:
        # [H, B] -> [B, H]
        step_rewards = step_rewards.transpose(0, 1)
        log_probs = log_probs.transpose(0, 1)

    B, H = step_rewards.shape
    assert log_probs.shape == (B, H), f"{log_probs.shape}"

    # No gradients from rewards
    step_rewards = step_rewards.detach()

    # Credit assignment: Zero out rewards that don't improve best-so-far
    # Only steps that achieve new maxima get credit; plateaus are zeroed out
    # print(f"step_rewards before clipping: \n{step_rewards}")
    if clip_rewards:
        # [1, 0, 3, 2, 4] -> [1, 1, 3, 3, 4]
        step_rewards_cummax = torch.cummax(step_rewards, dim=-1).values

        # e.g. [1, 1, 3, 3, 4] * [T, F, T, F, T] = [1, 0, 3, 0, 4]
        is_info = step_rewards == step_rewards_cummax
        step_rewards *= (is_info).float()

    # Compute cumulative or discounted immediate rewards
    if use_cumulative_r:
        # print(f"step_rewards before cumulative: \n{step_rewards}")
        reward = _get_cumulative_rewards(
            reward=step_rewards, discount_factor=discount_factor
        )
        # print(f"cumulative rewards: \n{reward}")
        reward = _standardize(B, H, reward, batch_standardize, eps)
        # print(f"Cumulative reward: \n{reward}")
    else:
        reward = _standardize(B, H, step_rewards, batch_standardize, eps)
        discounts = discount_factor ** torch.arange(H, device=reward.device)
        reward = discounts * reward
        # print(f"Immediate reward: \n{reward}")

    loss = -reward * log_probs

    return torch.mean(loss), step_rewards


def _mask_out_used_chunks(
    logit_mask: Tensor,  # [B, n, d]
    used_indices: Tensor,  # [B, n]
) -> Tensor:
    B, n = used_indices.shape
    d = logit_mask.shape[-1]

    # With current implementation of combining chunks into full space designs,
    # masking out an element in a chunk will also mask out all related designs
    # TODO
    assert n == 1, f"Only support full policy (n=1) for now"

    logit_mask = logit_mask.bool().view(B * n, -1)  # [B * n, d]
    logit_mask[torch.arange(B * n), used_indices.view(-1)] = False
    return logit_mask.view(B, n, d)


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
    use_logit_mask: bool = False,
    logit_mask: Optional[Tensor] = None,
) -> QueryResult:
    """Select the next query point based on current context and query set.

    Args:
        model: TAMO model
        x_ctx: [B, num_ctx, max_x_dim]
        y_ctx: [B, num_ctx, max_y_dim]
        x_mask: [max_x_dim]
        y_mask: [max_y_dim]
        input_bounds: list / nested list / tensor
        opt_cfg: Optimization config (policy flags, caching, epsilon, etc.)
        d: Number of candidate points per subspace
        t: Current time step
        T: Total time steps (budget)
        observed_target_y_mask (Optional): [max_y_dim]
        query_chunks (Optional): [d, max_x_dim]
        query_x_mask (Optional): [n, max_x_dim]
        auto_clear_cache: Clear cache at t=T
        use_logit_mask: Whether to mask out queried points
        logit_mask: [B, n, d]

    Returns:
        QueryResult dataclass containing:
            next_x: Selected query point [B, 1, max_x_dim]
            indices: Flat indices of selected points [B]
            log_probs: Log probabilities (with gradients) [B]
            entropy: Policy entropy [B]
            logits: Raw logits if evaluate=True [B, n, d]
            q_chunk: Query chunks [d, max_x_dim]
            q_chunk_mask: Chunk masks [n, max_x_dim]
            infer_time: Inference time in seconds
            logit_mask: Updated logit mask [B, n, d]
    """
    B, _, dx_max = x_ctx.shape
    device = x_ctx.device

    # Generate query set if not provided or not fixed
    if query_chunks is None or not opt_config.use_fixed_query_set:
        query_chunks, query_x_mask = sample_factorized_domain(
            d=d,
            max_x_dim=dx_max,
            device=device,
            x_mask=x_mask,
            input_bounds=input_bounds,
            use_grid_sampling=opt_config.use_grid_sampling,
            use_factorized_policy=opt_config.use_factorized_policy,
        )

    # Get dimensions
    n = query_x_mask.shape[0]

    # Expand tensors for batch processing
    query_x_mask_expanded = query_x_mask.unsqueeze(0).expand(B, -1, -1)
    query_chunks_expanded = (
        query_chunks.unsqueeze(0).unsqueeze(0).expand(B, n, -1, -1).contiguous()
    )
    x_mask_expanded = x_mask.unsqueeze(0).expand(B, -1)
    y_mask_expanded = y_mask.unsqueeze(0).expand(B, -1)
    observed_target_y_mask_expanded = (
        None
        if observed_target_y_mask is None
        else observed_target_y_mask.unsqueeze(0).expand(B, -1)
    )

    # Create logit mask
    if use_logit_mask:
        if logit_mask is None:
            logit_mask = torch.ones((B, n, d), device=device, dtype=torch.bool)
    else:
        logit_mask = None

    # Run model inference
    t0 = time.time()
    results = model.action(
        x_ctx=x_ctx,
        y_ctx=y_ctx,
        x_mask=x_mask_expanded,
        y_mask=y_mask_expanded,
        query_chunks=query_chunks_expanded,
        query_x_mask=query_x_mask_expanded,
        observed_target_y_mask=observed_target_y_mask_expanded,
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

    # Unpack results
    next_x_raw, indices_raw, logp_raw, entropy_raw = results[:4]
    logits = results[4] if len(results) > 4 else None

    # Reshape to expected dimensions
    next_x = next_x_raw.view(B, 1, dx_max).detach()
    chunk_indices = indices_raw.view(B, n).detach()
    chunk_logp = logp_raw.view(B, n)  # Keep gradients
    chunk_entropy = entropy_raw.view(B, n)  # Keep gradients

    # Collapse factorized indices to flat indices: [B, n] -> [B]
    indices = factorized_to_flat_index(chunk_indices=chunk_indices, n=n, d=d)
    indices = indices.squeeze(-1)
    log_probs = chunk_logp.sum(dim=-1)
    entropy = chunk_entropy.sum(dim=-1)

    # Update logit mask to exclude used points
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
    """Optimization step wrapper to handle batch setup with fantasized outcomes.
    Now only used in evaluation... TODO Could be used for training script for decoupled / cost-aware training.

    Args:
        fantasy (bool): Whether to update context with fantasized outcomes
        x_ctx (Tensor): Current context inputs [B, N_ctx, x_dim]
        y_ctx (Tensor): Current context outputs [B, N_ctx, y_dim]
        model (TAMO): model for query selection
        observation_tracker (ObservationTracker): Observed mask and cost tracker
        model_x_range (NestedFloatList): Input bounds used during model training
        opt_cfg (OptimizationConfig): Optimization configuration
        pred_cfg (PredictionConfig): Prediction configuration
        d (int): Number of subspace points for query selection
        T (int): Total optimization budget
        q_chunk (Optional[Tensor]): Precomputed query chunks for batch selection
        q_chunk_mask (Optional[Tensor]): Masks for query chunks
        logit_mask (Optional[Tensor]): Logit masks for query selection

    Returns:
        Tuple containing updated x_ctx, y_ctx, and QueryResult
    """

    def _validate_inputs(fantasy, opt_cfg, pred_cfg):
        if not fantasy:
            return

        assert (
            opt_cfg.write_cache is False
        ), "Fantasy with write_cache=True not supported"
        assert opt_cfg.read_cache is False, "Fantasy with read_cache=True not supported"
        assert (
            pred_cfg.read_cache is False
        ), "Fantasy with read_cache=True not supported"

    _validate_inputs(fantasy, opt_config, pred_config)

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
        use_logit_mask=True,
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
            use_logit_mask=False,
        )
        indices = query_results.indices
        logp = query_results.log_probs
        entropy = query_results.entropy

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
        neg_regret.requires_grad_(False)
        neg_regrets[t - 1] = neg_regret
        log_probs[t - 1] = logp
        entropy = entropy.detach()

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
    final_step_entropy_mean = entropy.mean().detach().item()

    del gp_sample_function, x_ctx, y_ctx, query_results

    return (loss_acq, step_reward_mean, final_step_reward_mean, final_step_entropy_mean)


def _reduce(
    tensor: Tensor, dim: int | tuple = None, reduction: str = "nanmean"
) -> Tensor:
    """Reduce a tensor along the specified dimension.

    Args:
        tensor: Can be of any shape.
        dim: Dimension(s) to reduce. If None, reduces all.
        reduction: ["mean", "sum", "nanmean"].

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
        model: MPALE model
        x_ctx: context inputs, [B, nc, max_x_dim]
        y_ctx: context function values, [B, nc, max_y_dim]
        x_tar: target locations, [B, nt, max_x_dim]
        y_tar: ground truth target function values, [B, nt, max_y_dim]
        x_mask: [B, dx_max]
        y_mask: [B, dy_max]
        read_cache: whether to read embedded context from cache

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
    intervals=[0.0, 0.25, 0.5, 0.75, 1.0],
    points=[32, 64, 128, 256],
    batchsizes=[32, 8, 4, 4],
    horizons=[25, 50, 75, 100],
):
    assert num_cur <= num_total, f"{num_cur} > {num_total}"

    scale_factor = num_cur / num_total
    for i, thres in enumerate(intervals):
        if scale_factor <= thres:
            scale_factor = i
            break
    scale_factor = max(0, scale_factor - 1)  # 0, 1, 2

    return points[scale_factor], horizons[scale_factor], batchsizes[scale_factor]
