"""Hypervolume utilities for multi-objective minimization."""

from enum import Enum
from typing import Optional, Tuple

import numpy as np
import torch
from einops import repeat
from pymoo.indicators.hv import HV
from torch import Tensor

from data.base.masking import compact_by_mask
from data.base.preprocessing import transform


class RegretType(str, Enum):
    """Regret modes for optimization evaluation.

    - SIMPLE:     true_min - current_min  (single-objective)
    - VALUE:      -current_hv
    - RATIO:      -current_hv / max_hv
    - NORM_RATIO: normalize ys → RATIO
    """

    SIMPLE = "simple"
    VALUE = "value"
    RATIO = "ratio"
    NORM_RATIO = "norm_ratio"


def _tnp(x: Tensor | np.ndarray) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, Tensor) else x


class MOO:
    """Hypervolume and regret utilities for multi-objective minimization."""

    @staticmethod
    def compute_hv(
        solutions: Tensor,
        minimum: Tensor,
        maximum: Tensor,
        ref_point: Optional[Tensor] = None,
        normalize: bool = False,
        y_mask: Optional[Tensor] = None,
    ) -> Tuple[np.ndarray, Tensor, Tensor]:
        """Compute hypervolume.

        Args:
            solutions:  [B, N, dy_max]
            minimum:    [dy_max] or [B, dy_max]
            maximum:    [dy_max] or [B, dy_max]
            ref_point:  [dy_max]
            normalize:  normalize y values to [0, 1] before computing hv
            y_mask:     [dy_max]

        Returns: hv [B], solutions [B, N, dy_max], ref_point [B, 1, dy_max]
        """
        return compute_normalized_hv_batch(
            solutions=solutions,
            minimum=minimum,
            maximum=maximum,
            ref_point=ref_point,
            y_mask=y_mask,
            normalize=normalize,
        )

    @staticmethod
    def _reward(
        do_single_objective: bool,
        minimum: Tensor,
        maximum: Tensor,
        solutions: Tensor,
        ref_point: Optional[Tensor] = None,
        normalize_y: bool = False,
        mask: Optional[Tensor] = None,
    ) -> np.ndarray | float:
        """Compute reward (negative min or hypervolume) over solutions.

        Args:
            do_single_objective:    use min instead of HV
            minimum:                [dy_max] or [B, dy_max]
            maximum:                [dy_max] or [B, dy_max]
            solutions:              [B, N, dy_max]
            ref_point:              [dy_max]
            normalize_y:            normalize y values when computing HV
            mask:                   [dy_max]

        Returns: [B]
        """
        if do_single_objective:
            mins = solutions.min(dim=1).values  # [B, dy_max]
            return (
                -compact_by_mask(data=mins, mask=mask, dim=-1).squeeze(-1).cpu().numpy()
            )
        return MOO.compute_hv(
            solutions=solutions,
            ref_point=ref_point,
            minimum=minimum,
            maximum=maximum,
            normalize=normalize_y,
            y_mask=mask,
        )[0]

    @staticmethod
    def compute_regret(
        solutions: Tensor,
        minimum: Tensor,
        maximum: Tensor,
        regret_type: str,
        candidates: Optional[Tensor] = None,
        ref_point: Optional[Tensor] = None,
        y_mask: Optional[Tensor] = None,
        optimal_value: Optional[np.ndarray | float] = None,
        max_hv: Optional[np.ndarray | float] = None,
        max_hv_norm: Optional[np.ndarray | float] = None,
    ) -> np.ndarray:
        """Compute regret for minimization problem.

        Args:
            solutions:      [B, N, D]
            minimum:        [D] or [B, D]
            maximum:        [D] or [B, D]
            regret_type:    'simple' | 'value' | 'ratio' | 'norm_ratio'
            candidates:     [B, M, D], required for norm_ratio if max_hv_norm not given
            ref_point:      [D] or [B, D]
            y_mask:         [D] or [B, D]
            optimal_value:  [B] or float, required for simple regret
            max_hv:         [B] or float, pre-computed max HV for ratio regret
            max_hv_norm:    [B] or float, pre-computed max normalized HV for norm_ratio

        Returns: regret [B]
            simple:     mins - best_mins  (single-objective only)
            value:      -hv
            ratio:      -hv / max_hv
            norm_ratio: -hv_norm / max_hv_norm  (over candidate set)
        """
        assert regret_type in ["simple", "value", "ratio", "norm_ratio"]
        dy_valid = (
            solutions.shape[-1] if y_mask is None else y_mask.sum(dim=-1).max().item()
        )
        assert regret_type != "simple" or dy_valid == 1

        do_single_objective = regret_type == "simple"
        normalize_y = regret_type == "norm_ratio"

        current_reward = MOO._reward(
            do_single_objective=do_single_objective,
            minimum=minimum,
            maximum=maximum,
            ref_point=ref_point,
            solutions=solutions,
            normalize_y=normalize_y,
            mask=y_mask,
        )
        optimal_reward = MOO._compute_optimal_reward(
            regret_type=regret_type,
            do_single_objective=do_single_objective,
            minimum=minimum,
            maximum=maximum,
            ref_point=ref_point,
            candidates=candidates,
            normalize_y=normalize_y,
            y_mask=y_mask,
            optimal_value=optimal_value,
            max_hv=max_hv,
            max_hv_norm=max_hv_norm,
        )
        return MOO._compute_regret_from_rewards(
            regret_type, current_reward, optimal_reward
        )

    @staticmethod
    def _compute_optimal_reward(
        regret_type: str,
        do_single_objective: bool,
        minimum: Tensor,
        maximum: Tensor,
        ref_point: Optional[Tensor],
        candidates: Optional[Tensor],
        normalize_y: bool,
        y_mask: Optional[Tensor],
        optimal_value: Optional[np.ndarray | float],
        max_hv: Optional[np.ndarray | float],
        max_hv_norm: Optional[np.ndarray | float],
    ) -> np.ndarray | float:
        """Baseline reward for regret: optimal value, 0, or max HV over candidates."""
        if regret_type == RegretType.SIMPLE.value:
            return _tnp(-minimum if optimal_value is None else -optimal_value)

        if regret_type == RegretType.VALUE.value:
            return 0.0

        # RATIO / NORM_RATIO: use cached value or compute from candidates
        cached = max_hv if regret_type == RegretType.RATIO.value else max_hv_norm
        if cached is None:
            assert (
                candidates is not None
            ), "candidates required when max_hv[_norm] not provided"
            cached = MOO._reward(
                do_single_objective=do_single_objective,
                minimum=minimum,
                maximum=maximum,
                ref_point=ref_point,
                solutions=candidates,
                normalize_y=normalize_y,
                mask=y_mask,
            )
        return cached

    @staticmethod
    def _compute_regret_from_rewards(
        regret_type: str,
        current_reward: np.ndarray | float,
        optimal_reward: np.ndarray | float,
    ) -> np.ndarray:
        if regret_type in [RegretType.SIMPLE.value, RegretType.VALUE.value]:
            return optimal_reward - current_reward
        return _norm(-current_reward, optimal_reward)


def compute_normalized_hv_batch(
    solutions: Tensor,
    minimum: Tensor,
    maximum: Tensor,
    ref_point: Optional[Tensor] = None,
    y_mask: Optional[Tensor] = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, Tensor, Tensor]:
    """Compute (normalized) hypervolume for solution batches.

    Args:
        solutions:  [B, N, max_y_dim]
        minimum:    [max_y_dim] or [B, max_y_dim]
        maximum:    [max_y_dim] or [B, max_y_dim]
        ref_point:  [max_y_dim] or [B, max_y_dim]
        y_mask:     [max_y_dim] or [B, max_y_dim]
        normalize: normalize solutions and ref_point to [0, 1]

    Returns: hv [B], solutions [B, N, max_y_dim], ref_point [B, 1, max_y_dim]
    """
    solutions, ref_point = _get_sols_n_ref_points(
        solutions=solutions,
        minimum=minimum,
        maximum=maximum,
        ref_point=ref_point,
        normalize=normalize,
    )
    hv = _compute_hv_batch(ref_point=ref_point, solutions=solutions, y_mask=y_mask)
    return hv, solutions, ref_point


def _compute_hv_batch(
    ref_point: Tensor, solutions: Tensor, y_mask: Optional[Tensor] = None
) -> np.ndarray:
    """Compute hypervolume for solution batches.

    Args:
        ref_point:  [B, 1, max_y_dim]
        solutions:  [B, N, max_y_dim]
        y_mask:     [B, max_y_dim]

    Returns: hvs [B]
    """
    B, _, max_y_dim = solutions.shape
    assert ref_point.shape == (B, 1, max_y_dim)

    if y_mask is not None:
        mask = y_mask.reshape(-1, max_y_dim).expand(B, -1)  # [B, max_y_dim]

    hvs = np.empty(B, dtype=np.float64)
    for b in range(B):
        m = mask[b] if y_mask is not None else slice(None)
        hvs[b] = _compute_hv(ref_point[b, 0, m], solutions[b, :, m])
    return hvs


def _compute_hv(
    ref_point: np.ndarray | Tensor, solutions: np.ndarray | Tensor
) -> float:
    """Compute hypervolume for a single solution set.

    Args:
        ref_point: [y_dim]
        solutions: [N, y_dim]

    Returns: hv (float)
    """
    ref_point = _tnp(ref_point)
    solutions = _tnp(solutions)

    # pymoo's HV can segfault on 1D data; handle directly
    if ref_point.shape[0] == 1:
        dominated = solutions[solutions[:, 0] <= ref_point[0]]
        return (
            0.0 if len(dominated) == 0 else float(ref_point[0] - dominated[:, 0].min())
        )

    return HV(ref_point=ref_point)(solutions)


def _get_sols_n_ref_points(
    solutions: Tensor,
    minimum: Tensor,
    maximum: Tensor,
    ref_point: Optional[Tensor] = None,
    normalize: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Prepare solutions and reference points for HV computation.

    Sets ref_point to maximum if not provided. Optionally normalizes to [0, 1]
    to remove bias towards larger-scaled objectives.

    Args:
        solutions:  [B, N, D]
        minimum:    [B, D] or [D]
        maximum:    [B, D] or [D]
        ref_point:  [B, D] or [D]
        normalize:  normalize solutions and ref_point to [0, 1]

    Returns: solutions [B, N, D], ref_point [B, 1, D]
    """
    minimum = minimum.to(solutions.device)
    maximum = maximum.to(solutions.device)

    B, _, D = solutions.shape
    if minimum.ndim == 1:
        minimum = repeat(minimum, "d -> b d", b=B)
    if maximum.ndim == 1:
        maximum = repeat(maximum, "d -> b d", b=B)

    if ref_point is None:
        ref_point = maximum.clone()
    elif ref_point.ndim == 1:
        ref_point = repeat(ref_point, "d -> b d", b=B)

    assert ref_point.shape == (
        B,
        D,
    ), f"ref_point.shape={ref_point.shape}, (B, D)=({B}, {D})"
    ref_point = ref_point.unsqueeze(1)  # [B, 1, D]

    if normalize:
        input_bounds = torch.stack([minimum, maximum], dim=-1)
        solutions = transform(
            data=solutions, inp_bounds=input_bounds, transform_method="normalize"
        )
        ref_point = transform(
            data=ref_point, inp_bounds=input_bounds, transform_method="normalize"
        )

    return solutions, ref_point.to(solutions.device)


def _norm(data: np.ndarray, div: np.ndarray) -> np.ndarray:
    """data / div, replacing div == 0 with 1.0 to avoid division by zero."""
    return data / np.where(div == 0, 1.0, div)
