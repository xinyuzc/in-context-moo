"""Embedder module: datapoint -> token.

1. input:
    context datapoint (x, y) or target datapoint (x, ?),
    valid dimension mask for x and y,
    observed value mask for x and y (indicating missing values).
2. output: embedded tokens with missing values handled.
"""

from typing import Optional

import torch
from torch import nn, Tensor


class DimensionWiseEmbedder(nn.Module):
    """Dimension-wise embedder with missing value handling.

    Args:
        dim_mlp: Dimension of the MLP embedding.
        max_x_dim: pass for compatibility, not used here.
        max_y_dim: pass for compatibility, not used here.
    """

    def __init__(self, dim_mlp: int, max_x_dim: int, max_y_dim: int):
        super().__init__()
        self.dim_mlp = dim_mlp

        self.mlp_x = nn.Linear(1, dim_mlp)
        self.mlp_y = nn.Linear(1, dim_mlp)

        self.missing_x_marker = nn.Parameter(torch.randn(1, dim_mlp))
        self.missing_y_marker = nn.Parameter(torch.randn(1, dim_mlp))

    @staticmethod
    def _fill_missing_values(emb: Tensor, mask: Tensor, marker: Tensor) -> Tensor:
        """Fill missing values in embeddings with marker.

        Args: B = batch_size, N = num_points, d = data dimension, H = embedding dimension
            emb: [B, N, d, H]
            mask: [B, N, d], True = observed, False = missing
            marker: [1, H]

        Returns:
            emb: [B, N, d, H] with missing values filled
        """
        B, N, d, H = emb.shape

        # [B, N, d] -> [B, N, d, H], [1, H] -> [B, N, d, H]
        mask = mask.unsqueeze(-1).expand(B, N, d, H)
        marker = marker.expand(B, N, d, H)

        # If observed, keep original embedding; if missing, use marker
        emb = torch.where(mask, emb, marker)
        return emb

    def _embed_x(
        self, x: Tensor, observed_x_mask: Optional[Tensor], B: int, N: int, dx_max: int
    ) -> Tensor:
        """Returns x_emb [B, N, dx_max, dim_mlp]"""
        x = x.view(B * N * dx_max, 1)
        x_emb = self.mlp_x(x).view(B, N, dx_max, self.dim_mlp)

        if observed_x_mask is not None:
            x_emb = self._fill_missing_values(
                x_emb, observed_x_mask, self.missing_x_marker
            )

        return x_emb

    def _embed_y(
        self,
        y: Optional[Tensor],
        observed_y_mask: Optional[Tensor],
        B: int,
        N: int,
        dy_max: int,
    ) -> Tensor:
        """Returns y_emb [B, N, dy_max, dim_mlp]"""
        if y is None:
            # All values are missing
            return self.missing_y_marker.expand(B, N, dy_max, self.dim_mlp)
        else:
            # Embed values
            y = y.view(B * N * dy_max, 1)
            y_emb = self.mlp_y(y).view(B, N, dy_max, self.dim_mlp)

            # Fill missing values if mask is provided
            if observed_y_mask is not None:
                y_emb = self._fill_missing_values(
                    y_emb, observed_y_mask, self.missing_y_marker
                )
            return y_emb

    def forward(
        self,
        x: Tensor,
        x_mask: Tensor,
        y_mask: Tensor,
        y: Optional[Tensor] = None,
        observed_x_mask: Optional[Tensor] = None,
        observed_y_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Embed input and output values with missing value handling.

        Args: B = batch_size, N = num_points, dx_max = max input dim, dy_max = max output dim
            x:  [B, N, dx_max], x values
            x_mask: [B, dx_max], True = valid. Pass for compatibility, not used here.
            y_mask: [B, dy_max], True = valid. Pass for compatibility, not used here.
            y: [B, N, dy_max], optional y values
            observed_y_mask: [B, N, dy_max], True = observed
            observed_x_mask: [B, N, dx_max], True = observed

        Returns:
            Embedding [B, N, dx_max + dy_max, dim_mlp]
        """
        B, N, dx_max = x.shape
        dy_max = y_mask.shape[-1]

        x_emb = self._embed_x(x, observed_x_mask, B, N, dx_max)
        y_emb = self._embed_y(y, observed_y_mask, B, N, dy_max)

        return torch.cat([x_emb, y_emb], dim=2)


# simple test
if __name__ == "__main__":
    B = 2
    N = 3
    dx_max = 5
    dy_max = 4
    dim_mlp = 8

    # Create random input data
    x = torch.randn(B, N, dx_max)

    x_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    y_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

    observed_x_mask = x_mask.unsqueeze(1).expand(B, N, dx_max)
    observed_x_mask[0, 0, 2] = 0  # make one value missing
    observed_x_mask[1, 1, 0] = 0  # make one value missing

    observed_y_mask = y_mask.unsqueeze(1).expand(B, N, dy_max)
    observed_y_mask[0, 2, 1] = 0  # make value missing
    observed_y_mask[1, 0, 2] = 0

    dimwise_embedder = DimensionWiseEmbedder(dim_mlp, dx_max, dy_max)

    # test with and without y
    ys = [torch.randn(B, N, dy_max), None]
    for y in ys:
        emb = dimwise_embedder(x, x_mask, y_mask, y, observed_x_mask, observed_y_mask)
        # shape check
        assert emb.shape == (B, N, dx_max + dy_max, dim_mlp)

        # assert missing values are handled correctly
        for b in range(B):
            for n in range(N):
                for dx in range(dx_max):
                    if not observed_x_mask[b, n, dx]:
                        assert torch.all(
                            emb[b, n, dx, :]
                            == dimwise_embedder.missing_x_marker.squeeze(0)
                        )
                for dy in range(dy_max):
                    if not observed_y_mask[b, n, dy]:
                        assert torch.all(
                            emb[b, n, dx_max + dy, :]
                            == dimwise_embedder.missing_y_marker.squeeze(0)
                        )
