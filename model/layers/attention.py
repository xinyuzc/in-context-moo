from typing import Optional
import torch
from torch.nn import TransformerEncoderLayer


class ContextPrefixEncoderLayer(TransformerEncoderLayer):
    """Customized SA block for full-sequence to context-prefix attention.

    Efficiency optimization for large query sets:
    - All tokens (context + queries) attend to context only (not each other)
    - Reduces memory from O(L^2) to O(L * num_ctx) where L >> num_ctx
    - Enables processing large query sets without quadratic memory cost

    Args:
        x: [B, L, H]
        attn_mask: [L, L]
        key_padding_mask: [B, L]
        is_causal: If True, uses causal attention.

    Returns: [B, L, H]
    """

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        # Determine context length from attn_mask
        if attn_mask is None:
            # Border case: all attend to all
            num_ctx = x.shape[1]
        else:
            slice_ = attn_mask[0, :]
            zero_mask = slice_ == 0
            zero_mask = zero_mask.to(torch.float32)
            num_ctx = int(torch.sum(zero_mask).item())

        # Adjust key_padding_mask to match context length
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, :num_ctx]

        # Context prefix as Key/Value
        x = self.self_attn(
            x,
            x[:, :num_ctx, :],
            x[:, :num_ctx, :],
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)
