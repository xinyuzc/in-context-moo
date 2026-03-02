"""Datapoint encoder module.
Dimension-wise tokens -> do self-attention -> multiply with positional encodings -> encoded tokens.
"""

from typing import Optional, Tuple

import torch
from torch import nn, Tensor

FEEDFORWARD_MULTIPLIER = 4  # Feedforward dimension = 4 * attention dimension


class DimensionAgnosticEncoder(nn.Module):
    """Dimension-agnostic encoder.

    Args:
        dim_mlp: Dimension of the MLP embedding.
        dim_attn: Dimension of the attention mechanism.
        nhead: Number of attention heads.
        dropout: Dropout rate.
        num_layers: Number of transformer layers.
        max_x_dim: Maximum input dimension.
        max_y_dim: Maximum output dimension.
    """

    def __init__(
        self,
        dim_mlp: int,
        dim_attn: int,
        nhead: int,
        dropout: float,
        num_layers: int,
        max_x_dim: int,
        max_y_dim: int,
        **kwargs,
    ):
        super().__init__()

        self.dim_mlp = dim_mlp
        self.dim_attn = dim_attn

        # Position embeddings
        self.id_x = nn.Parameter(torch.randn(max_x_dim, dim_mlp))
        self.id_y = nn.Parameter(torch.randn(max_y_dim, dim_mlp))

        # Projections
        self.in_proj = self.build_projection(dim_mlp, dim_attn)
        self.out_proj = self.build_projection(dim_attn, dim_mlp)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_attn,
            nhead=nhead,
            dim_feedforward=FEEDFORWARD_MULTIPLIER * dim_attn,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @staticmethod
    def build_projection(dim_in: int, dim_out: int) -> nn.Module:
        return nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(
        self,
        tokens: Tensor,
        x_mask: Tensor,
        y_mask: Tensor,
        pad_mask: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        use_top_k_ids: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """tokens [B, N, dx_max + dy_max, H] -> encoded tokens [B, N, dx_max + dy_max, H]

        Args:
            tokens: [B, N, dx_max + dy_max, H]
            x_mask: [B, dx_max], valid dimension mask for x
            y_mask: [B, dy_max], valid dimension mask for y
            pad_mask: [B*N, dx_max + dy_max], padding mask, True = "ignore/pad"
            mask: [dx_max + dy_max, dx_max + dy_max], attention mask, True = "ignore"
        Returns:
            tokens: [B, N, dx_max + dy_max, H]
            x_ids: [dx_max, H]
            y_ids: [dy_max, H]
        """
        B, N, D, H = tokens.shape
        dx_max = x_mask.shape[-1]
        dy_max = y_mask.shape[-1]

        assert D == dx_max + dy_max, f"Dimension mismatch: {D} != {dx_max + dy_max}"

        # Apply transformer
        seq_in = self.in_proj(tokens).view(B * N, D, -1)
        seq_out = self.transformer(seq_in, mask=mask, src_key_padding_mask=pad_mask)
        seq_out = self.out_proj(seq_out)

        # Get position embeddings
        x_id, y_id = self._get_position_embeddings(
            dx_max, dy_max, use_top_k=use_top_k_ids
        )

        # Aggregate value and ID embeddings
        seq_out = self._multiply_position_embeddings(seq_out, x_id, y_id, B, N)
        seq_out = seq_out.view(B, N, D, H)

        return seq_out, x_id, y_id

    def _get_position_embeddings(
        self,
        dx_max: int,
        dy_max: int,
        use_top_k: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Get position embeddings for x and y dimensions.
        Returns:
        Positional embeddings of shape (dx_max, H), (dy_max, H) for x and y.
        If use_top_k, the first dx_max and dy_max position embeddings are used.
        Otherwise, randomly sample dx_max / dy_max positional embeddings.
        """
        if use_top_k:
            return self.id_x[:dx_max], self.id_y[:dy_max]
        else:
            # sample dx_max, dy_max ids from id_x and id_y respectively
            max_x_dim = self.id_x.shape[0]
            max_y_dim = self.id_y.shape[0]

            x_indices = torch.argsort(torch.rand((max_x_dim,)))[:dx_max]
            y_indices = torch.argsort(torch.rand((max_y_dim,)))[:dy_max]

            return self.id_x[x_indices], self.id_y[y_indices]

    def _multiply_position_embeddings(
        self,
        seq_out: Tensor,
        x_id: Tensor,
        y_id: Tensor,
        B: int,
        N: int,
    ) -> Tensor:
        """Multiply value embeddings with position embeddings.

        Args:
            seq_out: [B * N, dx_max + dy_max, H]
            x_id: [dx_max, H]
            y_id: [dy_max, H]
            B (int): batch size
            N (int): number of datapoints

        Returns:
        seq_out multiplied by position embeddings [B * N, dx_max + dy_max, H]
        """
        dx_max = x_id.shape[0]
        dy_max = y_id.shape[0]

        x_id_expanded = x_id.unsqueeze(0).expand(B * N, dx_max, self.dim_mlp)
        y_id_expanded = y_id.unsqueeze(0).expand(B * N, dy_max, self.dim_mlp)

        # (B * N, dx_max + dy_max, H)
        id_embeddings = torch.cat([x_id_expanded, y_id_expanded], dim=1)

        return seq_out * id_embeddings

    @staticmethod
    def make_padding_mask(
        x_mask: Tensor,
        y_mask: Tensor,
        N: int,
        observed_x_mask: Optional[Tensor] = None,
        observed_y_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Create padding mask for encoder.
        Args:
            x_mask: (B, dx_max)
            y_mask: (B, dy_max)
            N: number of datapoints
        Returns: mask of shape (B, N, dx_max + dy_max), True = ignore
        """
        B, dx_max = x_mask.shape
        _, dy_max = y_mask.shape

        # Expand masks to [B, N, dim]
        y_expanded = y_mask.unsqueeze(1).expand(B, N, dy_max)
        x_expanded = x_mask.unsqueeze(1).expand(B, N, dx_max)

        if observed_x_mask is not None and observed_y_mask is not None:
            # Set unobserved positions to False
            assert observed_x_mask.shape[1] == N
            assert observed_y_mask.shape[1] == N

            y_expanded = y_expanded & observed_y_mask
            x_expanded = x_expanded & observed_x_mask

        # Concatenate and invert (True = ignore)
        return ~torch.cat([x_expanded, y_expanded], dim=-1)


# simple test
if __name__ == "__main__":
    dim_mlp = 8
    dim_attn = 16
    nhead = 4
    dropout = 0.0
    num_layers = 2
    max_x_dim = 16
    max_y_dim = 16
    dx_max = 5
    dy_max = 4

    B = 2
    N = 3

    encoder = DimensionAgnosticEncoder(
        dim_mlp, dim_attn, nhead, dropout, num_layers, max_x_dim, max_y_dim
    )

    # random
    tokens = torch.randn(B, N, dx_max + dy_max, dim_mlp)
    x_mask = torch.randint(0, 2, (B, dx_max)).bool()
    y_mask = torch.randint(0, 2, (B, dy_max)).bool()

    seq_out, x_ids, y_ids = encoder(
        tokens=tokens, x_mask=x_mask, y_mask=y_mask, use_top_k_ids=True
    )

    assert seq_out.shape == (B, N, dx_max + dy_max, dim_mlp)
    assert x_ids.shape == (dx_max, dim_mlp)
    assert y_ids.shape == (dy_max, dim_mlp)

    assert torch.allclose(x_ids, encoder.id_x[:dx_max])
    assert torch.allclose(y_ids, encoder.id_y[:dy_max])

    seq_out, x_ids, y_ids = encoder(
        tokens=tokens, x_mask=x_mask, y_mask=y_mask, use_top_k_ids=False
    )
