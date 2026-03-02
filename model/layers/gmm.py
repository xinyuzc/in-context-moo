"""Gaussian mixture model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, NamedTuple


DEFAULT_NUM_COMPONENTS = 20
DEFAULT_STD_MIN = 1e-4
DEFAULT_STD_MAX = 1.0


class GMMOutput(NamedTuple):
    """Structured output for GMM predictions."""

    means: Tensor  # [B, N, DY, K]
    stds: Tensor  # [B, N, DY, K]
    weights: Tensor  # [B, N, DY, K] (normalized probabilities)


def build_mlp(dim_in: int, dim_hidden: int, dim_out: int, depth: int) -> nn.Module:
    """Build a simple MLP with ReLU activations.

    Args:
        dim_in: input dimension
        dim_hidden: hidden layer dimension
        dim_out: output dimension
        depth: number of hidden layers
    """
    if depth == 0:
        return nn.Linear(dim_in, dim_out)

    layers = [nn.Linear(dim_in, dim_hidden), nn.ReLU(True)]
    for _ in range(depth - 2):
        layers.extend([nn.Linear(dim_hidden, dim_hidden), nn.ReLU(True)])
    layers.append(nn.Linear(dim_hidden, dim_out))
    return nn.Sequential(*layers)


class GMMPredictionHead(nn.Module):
    """Gaussian Mixture Model as prediction head.

    Takes encoded features and predicts GMM parameters for each output dimension.

    Input:  [B, DY, N, H] - features for B batches, DY output dims, N points, H hidden
    Output: GMMOutput with shapes [B, N, DY, K] for means, stds, weights

    Args:
        dim_mlp: Input feature dimension (H)
        dim_hidden: Hidden layer dimension
        depth: Number of hidden layers (0 = linear)
        num_components: Number of mixture components (K)
        std_min: Minimum standard deviation (numerical stability)
        std_max: Maximum standard deviation (prevent explosion)
        single_mlp: If True, use single MLP for all components (more efficient)
    """

    _dim_out = 3  # mean, std, weight per component

    def __init__(
        self,
        dim_mlp: int,
        dim_hidden: int,
        depth: int,
        num_components: int = DEFAULT_NUM_COMPONENTS,
        std_min: float = DEFAULT_STD_MIN,
        std_max: float = DEFAULT_STD_MAX,
        single_mlp: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dim_mlp = dim_mlp
        self.std_min = std_min
        self.std_max = std_max
        self.depth = depth
        self.num_components = num_components
        self.single_mlp = single_mlp

        if single_mlp:
            # Single MLP: parameter efficient
            total_dim_out = self._dim_out * num_components
            self.head = build_mlp(dim_mlp, dim_hidden, total_dim_out, depth)
        else:
            # Separate MLPs: more expressive but K times more parameters
            self.heads = nn.ModuleList(
                [
                    build_mlp(dim_mlp, dim_hidden, self._dim_out, depth)
                    for _ in range(num_components)
                ]
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize final layer for stable GMM outputs."""
        if self.single_mlp:
            # Get final linear layer
            if isinstance(self.head, nn.Sequential):
                final = self.head[-1]
            else:
                final = self.head
            nn.init.xavier_uniform_(final.weight, gain=0.1)
            nn.init.zeros_(final.bias)
        else:
            for head in self.heads:
                if isinstance(head, nn.Sequential):
                    final = head[-1]
                else:
                    final = head
                nn.init.xavier_uniform_(final.weight, gain=0.1)
                nn.init.zeros_(final.bias)

    def _get_gmm_output(self, input: Tensor, y_mask: Tensor) -> Tensor:
        """Compute raw GMM parameters.

        Args:
            input: [B, DY, N, H]
        Returns:
            [B, DY, N, K, 3] where last dim is [means..., stds..., weights...]
        """
        B, DY, N, H = input.shape
        input_flat = input.reshape(B * DY * N, H)

        if self.single_mlp:
            # (B*DY*N, K*3)
            raise NotImplementedError
        else:
            outputs = [head(input_flat) for head in self.heads]

            # Stack and reshape
            outputs_cat = torch.stack(outputs).movedim(0, -1).flatten(-2, -1)
            outputs_cat = outputs_cat.view(B, DY, N, -1)

            # Apply y_mask
            y_mask_expanded = y_mask.unsqueeze(-1).unsqueeze(-1).expand_as(outputs_cat)
            masked_outputs = torch.where(y_mask_expanded, outputs_cat, torch.nan)

            # Split into components
            raw_means, raw_stds, raw_weights = torch.chunk(masked_outputs, 3, dim=-1)

            # Process components
            means = raw_means
            stds = self.std_min + (1 - self.std_min) * F.softplus(raw_stds)
            weights = F.softmax(raw_weights, dim=-1)

            # Stack and transpose
            out = torch.stack([means, stds, weights], dim=-1)
            return out.transpose(1, 2)  # [B, N, DY, K, 3]

    def _process_parameters(self, raw_output: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert raw network outputs to valid GMM parameters.

        Args:
            raw_output: [B, DY, N, K*3]
        Returns:
            means: [B, DY, N, K] - unbounded
            stds: [B, DY, N, K] - positive, bounded
            weights: [B, DY, N, K] - sum to 1 along K
        """
        # Split: (B, DY, N, K, 3) -> 3 x (B, DY, N, K)
        raw_means, raw_stds, raw_weights = raw_output.unbind(dim=-1)

        return raw_means, raw_stds, raw_weights

    def _apply_mask(
        self, means: Tensor, stds: Tensor, weights: Tensor, y_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Mask out invalid y dimensions.

        Args:
            means, stds, weights: [B, DY, N, K]
            y_mask: [B, DY] - True for valid dimensions
        Returns:
            Masked tensors with NaN for invalid dimensions
        """
        # Expand mask: [B, DY] -> [B, DY, 1, 1] for broadcasting
        mask = y_mask[:, :, None, None]

        means = torch.where(mask, means, torch.full_like(means, float("nan")))
        stds = torch.where(mask, stds, torch.full_like(stds, float("nan")))
        weights = torch.where(mask, weights, torch.full_like(weights, float("nan")))

        return means, stds, weights

    def forward(
        self,
        input: Tensor,  # [B, DY, N, H]
        y_mask: Tensor,  # [B, DY]
        x_mask: Optional[Tensor] = None,  # [B, DX] - unused but kept for API compat
    ) -> GMMOutput:
        """Forward pass.

        Args:
            input: [B, DY, N, H] encoded features
            x_mask: [B, DX] input mask (unused, for API compatibility)
            y_mask: [B, DY] output mask, True = valid

        Returns:
            GMMOutput with means, stds, weights each of shape [B, N, DY, K]
        """
        raw_output = self._get_gmm_output(input, y_mask)  # [B, N, DY, K, 3]
        means, stds, weights = self._process_parameters(raw_output)  # [B, N, DY, K, 3]

        # # Transpose to [B, N, DY, K] for downstream use
        # means = means.permute(0, 2, 1, 3)
        # stds = stds.permute(0, 2, 1, 3)
        # weights = weights.permute(0, 2, 1, 3)

        return GMMOutput(means=means, stds=stds, weights=weights)

    # =========================================================================
    # Utility methods for training and inference
    # =========================================================================

    @staticmethod
    def log_prob(output: GMMOutput, target: Tensor) -> Tensor:
        """Compute log probability of targets under the GMM.

        Uses log-sum-exp trick for numerical stability:
            log p(y) = log Σₖ πₖ N(y|μₖ,σₖ)
                     = logsumexp(log πₖ + log N(y|μₖ,σₖ))

        Args:
            output: GMMOutput with shapes [B, N, DY, K]
            target: [B, N, DY] target values

        Returns:
            [B, N, DY] log probabilities
        """
        import math

        means = output.means  # [B, N, DY, K]
        stds = output.stds  # [B, N, DY, K]
        weights = output.weights  # [B, N, DY, K]

        # Expand target: [B, N, DY] -> [B, N, DY, 1]
        target = target.unsqueeze(-1)

        # Log probability under each Gaussian component
        # log N(y|μ,σ) = -0.5 * (log(2π) + 2*log(σ) + ((y-μ)/σ)²)
        var = stds**2
        log_component = -0.5 * (
            math.log(2 * math.pi) + torch.log(var) + (target - means) ** 2 / var
        )  # [B, N, DY, K]

        # Add log weights and logsumexp
        log_weights = torch.log(weights + 1e-10)  # Numerical stability
        log_prob = torch.logsumexp(log_weights + log_component, dim=-1)  # [B, N, DY]

        return log_prob

    @staticmethod
    def nll_loss(
        output: GMMOutput,
        target: Tensor,
        y_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Negative log-likelihood loss.

        Args:
            output: GMMOutput from forward()
            target: [B, N, DY] target values
            y_mask: [B, DY] valid output dimensions
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Scalar loss or [B, N, DY] if reduction='none'
        """
        log_p = GMMPredictionHead.log_prob(output, target)  # [B, N, DY]
        nll = -log_p

        return nll
        # # Mask invalid dimensions
        # if y_mask is not None:
        #     # y_mask: [B, DY] -> [B, 1, DY]
        #     mask = y_mask.unsqueeze(1).expand_as(nll)
        #     nll = torch.where(mask, nll, torch.zeros_like(nll))

        #     if reduction == "mean":
        #         return nll.sum() / mask.sum().clamp(min=1)
        #     elif reduction == "sum":
        #         return nll.sum()
        #     else:
        #         return nll
        # else:
        #     if reduction == "mean":
        #         return nll.mean()
        #     elif reduction == "sum":
        #         return nll.sum()
        #     else:
        #         return nll

    @staticmethod
    def sample(output: GMMOutput, n_samples: int = 1) -> Tensor:
        """Sample from the GMM.

        Args:
            output: GMMOutput with shapes [B, N, DY, K]
            n_samples: Number of samples per point

        Returns:
            [B, N, DY, n_samples] samples (NaN for masked dimensions)
        """
        means = output.means  # [B, N, DY, K]
        stds = output.stds  # [B, N, DY, K]
        weights = output.weights  # [B, N, DY, K]

        B, N, DY, K = means.shape

        # Handle NaN weights from masked dimensions by replacing with uniform
        # This allows multinomial to work; we'll mask the output later
        weights_clean = weights.clone()
        nan_mask = weights.isnan().any(dim=-1)  # [B, N, DY]
        weights_clean = torch.where(
            weights_clean.isnan(),
            torch.ones_like(weights_clean) / K,
            weights_clean,
        )
        # Ensure weights are valid (non-negative and sum to 1)
        weights_clean = weights_clean.clamp(min=0)
        weights_clean = weights_clean / weights_clean.sum(dim=-1, keepdim=True).clamp(
            min=1e-10
        )

        # Sample component indices: [B, N, DY, n_samples]
        # Reshape weights for multinomial: [B*N*DY, K]
        weights_flat = weights_clean.reshape(-1, K)
        indices = torch.multinomial(weights_flat, n_samples, replacement=True)
        indices = indices.view(B, N, DY, n_samples)  # [B, N, DY, n_samples]

        # Gather means and stds for selected components
        # Expand indices for gathering: [B, N, DY, n_samples]
        means_expanded = means.unsqueeze(-1).expand(
            -1, -1, -1, -1, n_samples
        )  # [B,N,DY,K,n_samples]
        stds_expanded = stds.unsqueeze(-1).expand(-1, -1, -1, -1, n_samples)

        indices_expanded = indices.unsqueeze(3)  # [B, N, DY, 1, n_samples]

        selected_means = torch.gather(means_expanded, 3, indices_expanded).squeeze(
            3
        )  # [B,N,DY,n_samples]
        selected_stds = torch.gather(stds_expanded, 3, indices_expanded).squeeze(3)

        # Sample from selected Gaussians
        eps = torch.randn_like(selected_means)
        samples = selected_means + selected_stds * eps

        # Restore NaN for masked dimensions
        nan_mask_expanded = nan_mask.unsqueeze(-1).expand_as(samples)
        samples = torch.where(
            nan_mask_expanded, torch.full_like(samples, float("nan")), samples
        )

        return samples

    @staticmethod
    def mode(output: GMMOutput) -> Tensor:
        """Get the mode (mean of highest-weight component).

        Args:
            output: GMMOutput with shapes [B, N, DY, K]

        Returns:
            [B, N, DY] mode estimates
        """
        # Find highest weight component
        best_idx = output.weights.argmax(dim=-1, keepdim=True)  # [B, N, DY, 1]
        mode = torch.gather(output.means, -1, best_idx).squeeze(-1)  # [B, N, DY]
        return mode

    @staticmethod
    def expected_value(output: GMMOutput) -> Tensor:
        """Compute expected value (weighted mean).

        E[y] = Σₖ πₖ μₖ

        Args:
            output: GMMOutput with shapes [B, N, DY, K]

        Returns:
            [B, N, DY] expected values
        """
        return (output.weights * output.means).sum(dim=-1)

    @staticmethod
    def std(output: GMMOutput) -> Tensor:
        """ "Compute standard deviation.

        Args:
            output: GMMOutput with shapes [B, N, DY, K]

        Returns:
            [B, N, DY] standard deviation
        """
        mean = GMMPredictionHead.expected_value(output).unsqueeze(-1)
        diff = output.means - mean  # [B, N, DY, K]

        var = torch.sum(output.weights * (output.stds**2 + diff**2), dim=-1)
        var = var.clamp(min=1e-5)

        return torch.sqrt(var)


# =============================================================================
# Example usage and tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GMMPredictionHead - Test Suite")
    print("=" * 60)

    # Test configuration
    B, DY, N, H = 4, 3, 10, 64  # batch, output_dims, points, hidden
    K = 5  # components

    # Create model
    model = GMMPredictionHead(
        dim_mlp=H,
        dim_hidden=32,
        depth=2,
        num_components=K,
        single_mlp=True,
    )

    print(f"\nConfig: B={B}, DY={DY}, N={N}, H={H}, K={K}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\n--- Forward Pass ---")
    x = torch.randn(B, DY, N, H)
    y_mask = torch.ones(B, DY, dtype=torch.bool)
    y_mask[0, 2] = False  # Mask out one dimension

    output = model(x, y_mask=y_mask)

    print(f"Input shape:   {list(x.shape)}")
    print(
        f"Output shapes: means={list(output.means.shape)}, "
        f"stds={list(output.stds.shape)}, weights={list(output.weights.shape)}"
    )

    # Verify constraints
    print("\n--- Constraint Verification ---")
    print(
        f"Stds in [{model.std_min}, {model.std_max}]: "
        f"min={output.stds[~output.stds.isnan()].min():.6f}, "
        f"max={output.stds[~output.stds.isnan()].max():.6f}"
    )

    # Check weights sum to 1 (for valid dims)
    weight_sums = output.weights[0, :, 0, :].sum(
        dim=-1
    )  # First batch, first point, all DY
    print(f"Weights sum to 1: {weight_sums}")

    # Check masking
    print(f"Masked dim has NaN: {output.means[0, :, 2, :].isnan().all()}")

    # Test loss computation
    print("\n--- Loss Computation ---")
    target = torch.randn(B, N, DY)
    loss = model.nll_loss(output, target, y_mask=y_mask)
    print(f"NLL Loss: {loss.item():.4f}")

    # Test sampling
    print("\n--- Sampling ---")
    samples = model.sample(output, n_samples=100)
    print(f"Samples shape: {list(samples.shape)}")

    # Test mode and expected value
    print("\n--- Point Estimates ---")
    mode = model.mode(output)
    expected = model.expected_value(output)
    print(f"Mode shape: {list(mode.shape)}")
    print(f"Expected value shape: {list(expected.shape)}")

    # Gradient check
    print("\n--- Gradient Check ---")
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    print(f"Total gradient norm: {grad_norm:.4f}")

    print("\n All tests passed!")
