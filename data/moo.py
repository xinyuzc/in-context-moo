"""Compute hypervolume when doing multi-objective minimization.

Functions:
    compute_hv: compute (optionally normalized) hypervolume
    compute_regret: compute regret for single- or multi-objective minimization problems
"""

from enum import Enum

from pymoo.indicators.hv import HV
import numpy as np
import torch
from torch import Tensor

from data.base.preprocessing import transform
from data.base.masking import compact_by_mask
from einops import repeat
from typing import Optional, Tuple


class RegretType(str, Enum):
    """Regret computation modes for optimization evaluation.

    - SIMPLE: Absolute regret for single-objective (optimal_value - current_min)
    - VALUE: Negative hypervolume (0 - current_hv)
    - RATIO: Divide by max HV (-current_hv / max_hv)
    - NORM_RATIO: Normalize function values before computing `RATIO`
    """

    SIMPLE = "simple"
    VALUE = "value"
    RATIO = "ratio"
    NORM_RATIO = "norm_ratio"


def _tnp(x: Tensor | np.ndarray) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, Tensor) else x


class MOO:
    """Minimization class.

    Functions:
        compute_hv: compute (optionally normalized) hypervolume
        compute_regret: compute regret for single- or multi-objective minimization problems
    """

    @staticmethod
    def compute_hv(
        solutions: Tensor,
        minimum: Tensor,  # [B, dy_max]  or [dy_max,]
        maximum: Tensor,  # [B, dy_max]  or [dy_max,]
        ref_point: Optional[Tensor] = None,
        normalize: bool = False,
        y_mask: Optional[Tensor] = None,
    ) -> Tuple[np.ndarray, Tensor, Tensor]:
        """Compute hypervolume for multi-objective minimization.

        Args:
            solutions: [B, N, dy_max]
            minimum: [dy_max] or [B, dy_max]
            maximum: [dy_max] or [B, dy_max]
            ref_point: [dy_max]
            normalize: Wether to normalize y values when computing hv
            y_mask: [dy_max]

        Returns:
            reward: [B]
            solutions_normalized: (Optionally normalized) solutions used for computing hv, [B, N, dy_max]
            ref_point_normalized: (Optionally normalized) reference points used for computing, [B, dy_max]
        """
        assert solutions is not None

        # Compute (normalized) hv
        reward, solutions_normalized, ref_point_normalized = (
            compute_normalized_hv_batch(
                solutions=solutions,
                minimum=minimum,
                maximum=maximum,
                ref_point=ref_point,
                y_mask=y_mask,
                normalize=normalize,
            )
        )

        return reward, solutions_normalized, ref_point_normalized

    @staticmethod
    def _reward(
        do_single_objective_optimization: bool,
        minimum: Tensor,
        maximum: Tensor,
        solutions: Tensor,
        ref_point: Optional[Tensor] = None,
        normalize_y: bool = False,
        mask: Optional[Tensor] = None,
    ) -> np.ndarray | float:
        """Compute reward for multi-objective minimization over solutions.

        Args:
            do_single_objective_optimization: whether doing single-objective optimization.
            If True, compute (-best_min) as reward. Otherwise, compute hypervolume as reward.
            minimum: [dy_max] or [B, dy_max]
            maximum: [dy_max] or [B, dy_max]
            solutions: [B, N, dy_max]
            ref_point: [dy_max]
            normalize_y: whether to normalize y values if computing hypervolume
            mask: [dy_max]

        Returns: reward: [B,] or 0.0, where:
            If use_simple_regret is True:
                reward is -best_min defined on solutions
            Otherwise:
                reward is hypervolume computed on solutions
        """
        if do_single_objective_optimization:
            mins = solutions.min(dim=1).values  # [B, dy_max]
            reward = -compact_by_mask(data=mins, mask=mask, dim=-1).squeeze(-1)  # [B,]
            reward = reward.cpu().numpy()
        else:
            reward = MOO.compute_hv(
                solutions=solutions,
                ref_point=ref_point,
                minimum=minimum,
                maximum=maximum,
                normalize=normalize_y,
                y_mask=mask,
            )[0]

        return reward

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
            solutions (Tensor): solution batch, [B, N, D]
            minimum (Tensor): minimum bounds, [B, D] or [D]
            maximum (Tensor): maximum bounds, [B, D] or [D]
            regret_type (str): 'simple' or 'value' or 'ratio' or 'norm_ratio'
            candidates (Tensor): optional candidate batch, [B, M, D]
            ref_point (Tensor): optional reference point for hypervolume computation, [B, D] or [D]
            y_mask (Tensor): optional mask for valid objectives, [B, D]
            optimal_value (): optimal value for single-objective functions, float or np.ndarray [B,] or None
            max_hv (): maximum hypervolume over candidates, float or np.ndarray [B,] or None
            max_hv_norm (): maximum normalized hypervolume over candidates, float or np.ndarray [B,] or None
            y_mask (Tensor): [B, D]

        Returns:
            regret_np (np.ndarray): [B], where:
                    simple regret: [-best_mins - (-mins)] = mins - best_mins, only for single-objective functions
                    value regret: -hv
                    ratio regret: -hv / max hv
                    norm_ratio regret (on discrete set): - normalized hv / max normalized hv on candidate set

        """
        # Value and shape checks
        assert regret_type in ["simple", "value", "ratio", "norm_ratio"]

        dy_max = solutions.shape[-1]
        dy_valid = dy_max if y_mask is None else y_mask.sum(dim=-1).max().item()
        assert regret_type != "simple" or dy_valid == 1

        do_single_objective = regret_type == "simple"
        normalize_y = regret_type == "norm_ratio"

        # Compute current reward over solutions: [B,]
        current_reward = MOO._reward(
            do_single_objective_optimization=do_single_objective,
            minimum=minimum,
            maximum=maximum,
            ref_point=ref_point,
            solutions=solutions,
            normalize_y=normalize_y,
            mask=y_mask,
        )

        # Compute optimal reward and regret based on regret_type
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

        # Compute regret
        regret_np = MOO._compute_regret_from_rewards(
            regret_type=regret_type,
            current_reward=current_reward,
            optimal_reward=optimal_reward,
        )

        return regret_np

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
        """Compute optimal reward based on regret type.

        Args:
            regret_type: Type of regret computation
            do_single_objective: Whether single-objective optimization
            minimum, maximum, ref_point: Bounds and reference point
            candidates: Candidate solutions for computing max HV
            normalize_y: Whether to normalize y values
            y_mask: Mask for valid objectives
            optimal_value: Pre-computed optimal value (for simple regret)
            max_hv: Pre-computed max hypervolume (for ratio regret)
            max_hv_norm: Pre-computed normalized max HV (for norm_ratio regret)

        Returns:
            Optimal reward value(s)
        """
        if regret_type == RegretType.SIMPLE.value:
            if optimal_value is None:
                optimal_reward = -minimum
            else:
                optimal_reward = -optimal_value
            return _tnp(optimal_reward)

        elif regret_type == RegretType.VALUE.value:
            return 0.0

        elif regret_type == RegretType.RATIO.value:
            if max_hv is None:
                assert (
                    candidates is not None
                ), "candidates required when max_hv not provided"
                max_hv = MOO._reward(
                    do_single_objective_optimization=do_single_objective,
                    minimum=minimum,
                    maximum=maximum,
                    ref_point=ref_point,
                    solutions=candidates,
                    normalize_y=normalize_y,
                    mask=y_mask,
                )
            return max_hv

        else:  # RegretType.NORM_RATIO
            if max_hv_norm is None:
                assert (
                    candidates is not None
                ), "candidates required when max_hv_norm not provided"
                max_hv_norm = MOO._reward(
                    do_single_objective_optimization=do_single_objective,
                    minimum=minimum,
                    maximum=maximum,
                    ref_point=ref_point,
                    solutions=candidates,
                    normalize_y=normalize_y,
                    mask=y_mask,
                )
            return max_hv_norm

    @staticmethod
    def _compute_regret_from_rewards(
        regret_type: str,
        current_reward: np.ndarray | float,
        optimal_reward: np.ndarray | float,
    ) -> np.ndarray:
        """Compute regret from current and optimal rewards.

        Args:
            regret_type: Type of regret computation
            current_reward: Current reward from solutions
            optimal_reward: Optimal/baseline reward

        Returns:
            Regret values
        """
        if regret_type in [RegretType.SIMPLE.value, RegretType.VALUE.value]:
            # Absolute regret: optimal - current
            return optimal_reward - current_reward
        else:  # RegretType.RATIO or RegretType.NORM_RATIO
            # Normalized regret: -current / optimal
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

        1. Prepare sols and ref points
        2. Optionally normalize sols and ref points to [0, 1]
        3. Compute hypervolume

    Args:
        solutions: [B, N, max_y_dim]
        minimum, maximum, ref_point, y_mask: [B, max_y_dim] or [max_y_dim]
        normalize: Normalize `solutions` and `ref_point` to [0, 1] if True

    Returns:
        hv: hypervolume, [B]
        solutions: (optionally normalized) solutions, [B, N, max_y_dim]
        ref_point: (optionally normalized) reference points, [B, 1, max_y_dim]
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
        ref_point: [B, 1, max_y_dim]
        solutions: [B, N, max_y_dim]
        y_mask: [B, max_y_dim]

    Returns: hvs [B]
    """
    B, _, max_y_dim = solutions.shape
    assert ref_point.shape == (B, 1, max_y_dim)

    hvs = np.empty(B, dtype=np.float64)

    if y_mask is None:
        for b in range(B):
            refs_b = ref_point[b, 0]  # [dy]
            sols_b = solutions[b]  # [N, dy]
            hv = _compute_hv(ref_point=refs_b, solutions=sols_b)
            hvs[b] = hv
    else:
        # Compute hv only on valid objectives
        assert y_mask.shape[-1] == max_y_dim
        mask_expanded = y_mask.view(-1, max_y_dim).expand(B, -1)  # [B, max_y_dim]

        for b in range(B):
            refs_b = ref_point[b, 0, mask_expanded[b]]  # [dy]
            sols_b = solutions[b, :, mask_expanded[b]]  # [N, dy]
            hv = _compute_hv(ref_point=refs_b, solutions=sols_b)
            hvs[b] = hv

    return hvs


def _compute_hv(
    ref_point: np.ndarray | Tensor, solutions: np.ndarray | Tensor
) -> float:
    """Compute hypervolume for a solution batch.

    Args:
        ref_point: [y_dim]
        solutions: [num_solutions, y_dim]

    Returns: hv (float)
    """
    ref_point = _tnp(ref_point)
    solutions = _tnp(solutions)

    hv_indicator = HV(ref_point=ref_point)
    hv = hv_indicator(solutions)

    return hv


def _get_sols_n_ref_points(
    solutions: Tensor,
    minimum: Tensor,
    maximum: Tensor,
    ref_point: Optional[Tensor] = None,
    normalize: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Get solutions and reference points for hypervolume computation when doing minimization.

        If `ref_point` is not provided: set ref_point to maximum
        If `normalize` is True: `solutions` and `ref_point` are normalized to `[0, 1]`

    Normalizing function values can remove bias towards larger objectives.

    Args:
        solutions: [B, N, D]
        minimum, maximum, ref_point: [B, D] or [D]
        normalize: Whether to normalize `solutions` and `ref_point` to [0, 1]

    Returns:
        solutions: [B, N, D]
        ref_point: [B, 1, D]
    """
    minimum = minimum.to(solutions.device)
    maximum = maximum.to(solutions.device)

    B, _, D = solutions.shape

    if minimum.ndim == 1:
        minimum = repeat(minimum, "d -> b d", b=B)
    if maximum.ndim == 1:
        maximum = repeat(maximum, "d -> b d", b=B)

    # Prepare value for reference points
    if ref_point is None:
        ref_point = maximum.clone()  # [B, D]
    elif ref_point.ndim == 1:
        ref_point = repeat(ref_point, "d -> b d", b=B)

    assert ref_point.shape == (B, D)
    ref_point = ref_point.unsqueeze(1)  # [B, 1, D]

    # Normalize solutions and reference points if requested
    if normalize:
        # Get input bounds: [..., D, 2]
        input_bounds = torch.stack([minimum, maximum], dim=-1)
        solutions = transform(
            data=solutions, inp_bounds=input_bounds, transform_method="normalize"
        )
        ref_point = transform(
            data=ref_point, inp_bounds=input_bounds, transform_method="normalize"
        )
        # solutions = min_max_normalize(solutions, minimum, maximum)
        # ref_point = min_max_normalize(ref_point, minimum, maximum)

    ref_point = ref_point.to(solutions.device)
    return solutions, ref_point


def _norm(data: np.ndarray, div: np.ndarray) -> np.ndarray:
    """Returns data / div [B,], with div replaced by 1.0 where div == 0 to avoid division by zero."""
    div = np.where(div == 0, 1.0, div)
    return data / div
