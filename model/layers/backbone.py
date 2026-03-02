"""Transformer backbone: sequence of tokens -> processed sequence of tokens."""

import torch
from torch import Tensor
import torch.nn as nn

from .attention import ContextPrefixEncoderLayer


FEEDFORWARD_MULTIPLIER = 4


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim_mlp: int,
        dim_attn: int,
        nhead: int,
        dropout: float,
        num_layers: int,
        ctx_prefix_encoder_layer: bool = True,
    ):
        super().__init__()
        self.dim_mlp = dim_mlp
        self.dim_attn = dim_attn

        # use identity if dimensions match
        self.in_proj = (
            nn.Linear(dim_mlp, dim_attn) if dim_mlp != dim_attn else nn.Identity()
        )
        self.out_proj = (
            nn.Linear(dim_attn, dim_mlp) if dim_mlp != dim_attn else nn.Identity()
        )

        # create transformer encoder
        if ctx_prefix_encoder_layer:
            encoder_layer = ContextPrefixEncoderLayer(
                d_model=dim_attn,
                nhead=nhead,
                dim_feedforward=FEEDFORWARD_MULTIPLIER * dim_attn,
                dropout=dropout,
                batch_first=True,
                activation="relu",
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim_attn,
                nhead=nhead,
                dim_feedforward=FEEDFORWARD_MULTIPLIER * dim_attn,
                dropout=dropout,
                batch_first=True,
                activation="relu",
            )

        # stack layers
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

    def forward(self, tokens, mask=None, pad_mask=None):
        """tokens [B, N, H] -> seq_out [B, N, H]"""
        seq_in = self.in_proj(tokens)
        seq_out = self.transformer(seq_in, mask=mask, src_key_padding_mask=pad_mask)
        seq_out = self.out_proj(seq_out)
        return seq_out

    @staticmethod
    def make_attention_mask(x_mask: Tensor, N: int, nc: int) -> Tensor:
        """Make attention mask for transformer block.
        Args:
            x_mask: [B, dx_max]
            N: total number of points (context + target)
            nc: number of context points

        Returns:
        attention mask of shape (N, N), True = "ignore".
        Context points can be attended by all points.
        Target points can only attend to context points and itself.
        """
        assert 0 < nc <= N, f"nc={nc} must be in (0, N]"

        mask = torch.ones((N, N), dtype=torch.bool, device=x_mask.device)

        mask[:, :nc] = False
        mask[nc:N, nc:N].fill_diagonal_(False)

        # True means "ignore"
        return mask


# simple test
if __name__ == "__main__":
    import torch

    B, N, H = 2, 5, 16
    x = torch.randn(B, N, H)

    model = TransformerBlock(
        dim_mlp=16,
        dim_attn=16,
        nhead=4,
        dropout=0.1,
        num_layers=2,
        ctx_prefix_encoder_layer=True,
    )
    y = model(x)
    print(y.shape)  # should be [B, N, H]
    assert y.shape == (B, N, H)
