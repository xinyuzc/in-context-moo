from typing import Optional

from torch import Tensor
import torch


class ObservationTracker:
    """Manages dimension masks and cost tracking during multi-objective optimization.

    NOTE only involves in test time computing now; to be introduced to training time.

    This class tracks which objectives are observed at each optimization step and
    accumulates costs associated with observations. It supports various observation
    strategies through different mask generation modes.

    Args:
        x_dim: Dimension of input space
        y_dim: Dimension of output/objective space
        dim_mask_gen_mode: Strategy for generating dimension masks. Options:
            - "full": Observe all objectives at each step
            - "single": Observe only one fixed objective
            - "random": Randomly select one objective per step
            - "alternate": Cycle through objectives in order
        single_obs_y_dim: Index of objective to observe when mode is "single"
        device: PyTorch device for tensor storage (default: "cuda")
        num_initial_points: Number of initial random observations before optimization
        cost_mode: If True, cost per observed objective; if False, unit cost per step
        cost: Cost value per observed objective dimension

    Attributes:
        x_mask: Boolean mask for input dimensions [x_dim]
        y_mask_target: Boolean mask for target objectives [y_dim]
        y_mask_observed: History of observed objective masks [num_steps, y_dim]
        cost_used: Accumulated cost per objective dimension [y_dim]
        initial_cost: Total cost after initial observations

    Example:
        >>> observation_tracker = ObservationTracker(x_dim=5, y_dim=3, dim_mask_gen_mode="alternate")
        >>> mask = observation_tracker.step()  # Get mask for next observation
        >>> total_cost = observation_tracker.get_cost_used()
    """

    # Valid mask generation modes
    VALID_MODES = ["full", "single", "random", "alternate"]

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        dim_mask_gen_mode: str = "full",
        single_obs_y_dim: Optional[int] = None,
        device: str = "cuda",
        num_initial_points: int = 1,
        cost_mode: bool = True,
        cost: float = 1.0,
    ):
        # Validate inputs
        self._validate_inputs(
            x_dim, y_dim, dim_mask_gen_mode, single_obs_y_dim, num_initial_points
        )

        # Store configuration
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.y_observed_mode = dim_mask_gen_mode
        self.single_obs_y_dim = single_obs_y_dim
        self.device = device
        self.cost_mode = cost_mode

        # Initialize masks
        self.x_mask = torch.ones((x_dim,), dtype=torch.bool, device=device)
        self.y_mask_target = torch.ones((y_dim,), dtype=torch.bool, device=device)
        self.y_mask_observed = torch.empty(0, y_dim, device=device, dtype=torch.bool)

        # Initialize costs
        self.cost_each = torch.full(
            (y_dim,), fill_value=cost, dtype=torch.float32, device=device
        )
        self.cost_used = torch.zeros(y_dim, dtype=torch.float32, device=device)

        # Perform initial observations
        for _ in range(num_initial_points):
            self.step(update_mask=True)

        self.initial_cost = self.get_cost_used()

    @staticmethod
    def _validate_inputs(
        x_dim: int,
        y_dim: int,
        dim_mask_gen_mode: str,
        single_obs_y_dim: Optional[int],
        num_initial_points: int,
    ) -> None:
        """Validate initialization parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        if x_dim < 1:
            raise ValueError(f"x_dim must be >= 1, got {x_dim}")
        if y_dim < 1:
            raise ValueError(f"y_dim must be >= 1, got {y_dim}")
        if num_initial_points < 1:
            raise ValueError(
                f"num_initial_points must be >= 1, got {num_initial_points}"
            )
        if dim_mask_gen_mode not in ObservationTracker.VALID_MODES:
            raise ValueError(
                f"dim_mask_gen_mode must be one of {ObservationTracker.VALID_MODES}, "
                f"got '{dim_mask_gen_mode}'"
            )
        if dim_mask_gen_mode == "single":
            if single_obs_y_dim is None:
                raise ValueError(
                    "single_obs_y_dim must be provided when dim_mask_gen_mode='single'"
                )
            if not (0 <= single_obs_y_dim < y_dim):
                raise ValueError(
                    f"single_obs_y_dim must be in [0, {y_dim}), got {single_obs_y_dim}"
                )

    def get_cost_used(self) -> float:
        """Get total accumulated cost across all objectives.

        Returns:
            Total cost as a float scalar
        """
        return self.cost_used.sum().item()

    @property
    def y_mask(self) -> Tensor:
        """Get the most recent observed objective mask.

        Returns:
            Boolean tensor of shape [y_dim] indicating which objectives were
            observed in the last step

        Raises:
            IndexError: If no observations have been made yet
        """
        if len(self.y_mask_observed) == 0:
            raise IndexError("No observations recorded yet. Call step() first.")
        return self.y_mask_observed[-1]

    @property
    def num_observations(self) -> int:
        """Get the total number of observations made so far.

        Returns:
            Number of observations (steps taken)
        """
        return self.y_mask_observed.shape[0]

    def step(self, update_mask: bool = True) -> Tensor:
        """Execute one optimization step, updating masks and costs.

        Args:
            update_mask: If True, generate a new mask according to the mode.
                        If False, reuse the previous mask.

        Returns:
            Boolean tensor of shape [y_dim] representing the new observation mask

        Raises:
            RuntimeError: If update_mask=False but no previous mask exists
        """
        # Get previous mask if it exists
        prev_mask = self.y_mask_observed[-1] if self.num_observations > 0 else None

        # Generate or reuse mask
        if update_mask:
            mask = self._generate_dim_mask(prev_mask)
        else:
            if prev_mask is None:
                raise RuntimeError(
                    "Cannot reuse mask: no previous observations exist. "
                    "Set update_mask=True for the first step."
                )
            mask = prev_mask.clone()

        # Append to observation history
        self.y_mask_observed = torch.cat(
            [self.y_mask_observed, mask.unsqueeze(0)], dim=0
        )

        # Update accumulated costs
        self._update_cost(mask)

        return mask

    def _update_cost(self, mask: Tensor) -> None:
        """Update cost accumulator based on newly observed dimensions.

        Args:
            mask: Boolean tensor indicating which objectives were observed
        """
        if self.cost_mode:
            # Cost per observed objective
            step_cost = self.cost_each * mask.float()
            self.cost_used += step_cost
        else:
            # Unit cost per step (store in first element)
            self.cost_used[0] += 1.0

    def _generate_dim_mask(self, prev_mask: Optional[Tensor]) -> Tensor:
        """Generate dimension mask based on the configured mode.

        Args:
            prev_mask: Previous mask, required for "alternate" mode

        Returns:
            Boolean tensor of shape [y_dim] indicating which objectives to observe
        """
        return self._get_dim_mask(
            dim=self.y_dim,
            device=self.device,
            prev_mask=prev_mask,
            mode=self.y_observed_mode,
            single_obs_dim=self.single_obs_y_dim,
        )

    @staticmethod
    def _get_dim_mask(
        dim: int,
        device: str,
        prev_mask: Optional[Tensor] = None,
        mode: str = "full",
        single_obs_dim: Optional[int] = None,
    ) -> Tensor:
        """Generate dimension mask according to specified mode.

        Args:
            dim: Number of dimensions
            device: PyTorch device for tensor creation
            prev_mask: Previous mask of shape [dim], used in "alternate" mode
            mode: Mask generation strategy
            single_obs_dim: Fixed dimension index for "single" mode

        Returns:
            Boolean tensor of shape [dim] where True indicates observed dimensions

        Raises:
            ValueError: If mode is invalid or required parameters are missing
        """
        if mode == "full":
            return torch.ones((dim,), dtype=torch.bool, device=device)

        if mode not in ["single", "random", "alternate"]:
            raise ValueError(f"Unknown mask generation mode: {mode}")

        # For single-dimension observation modes
        if mode == "single":
            if single_obs_dim is None:
                raise ValueError("single_obs_dim required for mode='single'")
            obs_dim = single_obs_dim

        elif mode == "random":
            obs_dim = torch.randint(0, dim, (1,), device=device).item()

        else:  # mode == "alternate"
            # Handle edge case: only one dimension
            if dim == 1:
                return torch.ones((dim,), dtype=torch.bool, device=device)

            # Determine next dimension in sequence
            if prev_mask is None:
                obs_dim = 0
            else:
                # Find currently observed dimension
                prev_indices = prev_mask.int().nonzero(as_tuple=False)[:, 0]
                if len(prev_indices) != 1:
                    raise ValueError(
                        f"Expected exactly one observed dimension in prev_mask, "
                        f"found {len(prev_indices)}"
                    )
                prev_dim = prev_indices[0].item()
                # Cycle to next dimension
                obs_dim = (prev_dim + 1) % dim

        # Create sparse mask with single True value
        mask = torch.zeros(dim, dtype=torch.bool, device=device)
        mask[obs_dim] = True
        return mask

    def reset(self) -> None:
        """Reset observation history and costs to initial state."""
        self.y_mask_observed = torch.empty(
            0, self.y_dim, device=self.device, dtype=torch.bool
        )
        self.cost_used = torch.zeros(
            self.y_dim, dtype=torch.float32, device=self.device
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ObservationTracker(x_dim={self.x_dim}, y_dim={self.y_dim}, "
            f"mode={self.y_observed_mode}, observations={self.num_observations}, "
            f"cost={self.get_cost_used():.2f})"
        )
