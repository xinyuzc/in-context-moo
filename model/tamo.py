from dataclasses import dataclass
from typing import Optional, Tuple
import gc

from einops import repeat
import torch
import torch.nn as nn
from torch import Tensor

from model.layers import (
    DimensionAgnosticEncoder,
    DimensionWiseEmbedder,
    TransformerBlock,
    Decoder,
    DecoderConfig,
    GMMOutput,
    OptimizationOutput,
)


TASK_TOKEN_PRED = "prediction"
TASK_TOKEN_OPT = "optimization"


@dataclass
class TAMOConfig:
    # Model dimensions
    max_x_dim: int = 4
    max_y_dim: int = 3
    dim_mlp: int = 64

    # Encoder
    use_missing_marker: bool = False

    # Transformer architecture
    dim_attn: int = 64
    nhead: int = 4
    dropout: float = 0.0
    num_layers_backbone: int = 4
    num_layers_encoder: int = 4
    num_layers_decoder: int = 4
    ctx_prefix_encoder_layer: bool = True

    # Head parameters
    dim_hidden: int = 128
    depth: int = 3
    num_components: int = 20

    # Policy head
    use_ar: bool = False

    # Prediction head
    std_min: float = 1e-4
    std_max: float = 1.0
    single_mlp: bool = False


class TAMO(nn.Module):
    """Task-Agnostic Multi-objective Optimization (TAMO).

    Supports prediction and policy on joint dimension space.
    For efficient sequential optimization, history (context) embeddings are cached by default.
    """

    def __init__(self, config: TAMOConfig):
        super().__init__()
        self.config = config
        self.max_y_dim = config.max_y_dim
        self.max_x_dim = config.max_x_dim
        self.dim_mlp = config.dim_mlp
        self.use_missing_marker = config.use_missing_marker

        self.embedder = DimensionWiseEmbedder(
            config.dim_mlp, config.max_x_dim, config.max_y_dim
        )

        self.encoder = DimensionAgnosticEncoder(
            dim_mlp=config.dim_mlp,
            dim_attn=config.dim_attn,
            nhead=config.nhead,
            num_layers=config.num_layers_encoder,
            dropout=config.dropout,
            max_x_dim=config.max_x_dim,
            max_y_dim=config.max_y_dim,
        )

        self.transformer_block = TransformerBlock(
            dim_mlp=config.dim_mlp,
            dim_attn=config.dim_attn,
            nhead=config.nhead,
            dropout=config.dropout,
            num_layers=config.num_layers_backbone,
        )

        self.decoder = Decoder(
            config=DecoderConfig(
                dim_mlp=config.dim_mlp,
                dim_attn=config.dim_attn,
                nhead=config.nhead,
                dropout=config.dropout,
                num_layers=config.num_layers_decoder,
                dim_hidden=config.dim_hidden,
                depth=config.depth,
                num_components=config.num_components,
                ctx_prefix_encoder_layer=config.ctx_prefix_encoder_layer,
                use_ar=config.use_ar,
                std_min=config.std_min,
                std_max=config.std_max,
                single_mlp=config.single_mlp,
            )
        )

    @staticmethod
    def mean_pooling(input: Tensor, mask: Tensor) -> Tensor:
        """Mean pooling over the second last dimension with mask.

        Args:
            input: [B, ..., L, H]
            mask: [B, ..., L, H]

        Returns: aggregate input of shape [B, ..., H]
        """
        # Zero out masked positions: (B, ..., L, H) -> (B, ..., H)
        mask_input = input.float() * mask.float()
        mask_input_sum = mask_input.sum(dim=-2)

        # Count valid positions: (B, ..., H)
        mask_input_count = mask.float().sum(dim=-2).clamp(min=1.0)

        # Take means
        input_aggregated = mask_input_sum / mask_input_count

        return input_aggregated

    def _make_tokens(
        self,
        x: Tensor,
        x_mask: Tensor,
        y_mask: Tensor,
        y: Optional[Tensor] = None,
        observed_x_mask: Optional[Tensor] = None,
        observed_y_mask: Optional[Tensor] = None,
    ):
        """Create input tokens for encoder.

        Args:
            x: [B, n, dx_max]
            x_mask: [B, dx_max], indicating valid dimensions
            y_mask: [B, dy_max], indicating valid dimensions
            y: [B, n, dy_max] or None
            observed_x_mask: [B, n, dx_max] or None, True = observed, indicating observed dimensions
            observed_y_mask: [B, n, dy_max] or None, True = observed, indicating observed dimensions

        Returns: input sequence [B, n, H], ids for x [dx_max, H], ids for y [dy_max, H].
        """
        B, dx_max = x_mask.shape
        _, dy_max = y_mask.shape
        n = x.shape[1]

        # Dimension embedding: [B, n, dx_max + dy_max, H]
        dim_token = self.embedder(
            x=x,
            y=y,
            x_mask=x_mask,
            y_mask=y_mask,
            observed_x_mask=observed_x_mask,
            observed_y_mask=observed_y_mask,
        )

        # Make padding mask: [B*n, dx_max + dy_max]
        if self.use_missing_marker:
            # If using misssing markers: unobserved dimensions are filled with missing markers
            pad_mask = self.encoder.make_padding_mask(x_mask, y_mask, n)
        else:
            # Otherwise: unobserved dimensions will also be masked
            pad_mask = self.encoder.make_padding_mask(
                x_mask, y_mask, n, observed_x_mask, observed_y_mask
            )
        pad_mask = pad_mask.view(B * n, -1)

        # Apply encoder: [B, n, dx_max + dy_max, H], [dx_max, H], [dy_max, H]
        dim_token, x_ids, y_ids = self.encoder(
            tokens=dim_token, x_mask=x_mask, y_mask=y_mask, pad_mask=pad_mask
        )

        dim_token_x = dim_token[:, :, :dx_max]  # [B, n, dx_max, H]
        dim_token_y = dim_token[:, :, dx_max:]  # [B, n, dy_max, H]

        # Aggregate into x and y tokens by mean pooling: [B, n, H]
        H = dim_token_x.shape[-1]
        x_mask_expanded = x_mask[:, None, :, None].expand(B, n, dx_max, H)
        y_mask_expanded = y_mask[:, None, :, None].expand(B, n, dy_max, H)

        token_x = self.mean_pooling(dim_token_x, x_mask_expanded)
        token_y = self.mean_pooling(dim_token_y, y_mask_expanded)

        seq_in = token_x + token_y  # [B, n, H]

        return seq_in, x_ids, y_ids

    def _combine_new_cached_sequence(
        self, new_sequence: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """Combine cached sequence and new sequence (if provided).

        Args:
            new_sequence: [B, n_new, H] or None

        Returns:
        If new_sequence is None, return cached sequence [B, n_cached, H] or None
        If new_sequence is provided, return combined sequence [B, n_cached + n_new, H] or new_sequence
        """
        cache_sequence = None
        if hasattr(self, "_embed_context"):
            cache_sequence = self._embed_context

        # Just return cache if no new sequence provided
        if new_sequence is None:
            return cache_sequence

        # Combine cache with new sequence if provided
        if cache_sequence is None:
            return new_sequence
        else:
            return torch.cat([cache_sequence, new_sequence], dim=1)

    def _update_cache(self, new_sequence) -> None:
        """Update cached sequence input. Cache: [B, n_cached, H] -> [B, n_cached + n_new, H]

        Args:
            new_sequence: [B, n_new, H]
        """
        if self._combine_new_cached_sequence() is None:
            self._embed_context = new_sequence
        else:
            self._embed_context = torch.cat([self._embed_context, new_sequence], dim=1)

    def _make_input_sequence(
        self,
        x_ctx: Tensor,
        y_ctx: Tensor,
        x_tar: Tensor,
        x_mask: Tensor,
        y_mask: Tensor,
        observed_context_x_mask: Optional[Tensor] = None,
        observed_context_y_mask: Optional[Tensor] = None,
        observed_target_x_mask: Optional[Tensor] = None,
        observed_target_y_mask: Optional[Tensor] = None,
        read_cache: bool = False,
        write_cache: bool = False,
    ):
        """Get input sequence for transformer block.

        Args:
            x_ctx: [B, nc, dx_max]
            y_ctx: [B, nc, dy_max]
            x_tar: [B, nt, dx_max]
            x_mask: [B, dx_max]
            y_mask: [B, dy_max]
            observed_context_x_mask: [B, nc, dx_max] or None, True = observed
            observed_context_y_mask: [B, nc, dy_max] or None, True = observed
            observed_target_x_mask: [B, nt, dx_max] or None, True = observed
            observed_target_y_mask: [B, nt, dy_max] or None, True = observed
            read_cache: whether to read previous context sequence from cache
            write_cache: whether to write new, full context sequence to cache

        Returns: input sequence [B, N, H],
            ids for x [dx_max, H],
            ids for y [dy_max, H],
            nc: number of context points
        """
        # Create input sequences from context: (B, nc, H)
        seq_in_ctx, x_ids, y_ids = self._make_tokens(
            x=x_ctx,
            y=y_ctx,
            x_mask=x_mask,
            y_mask=y_mask,
            observed_x_mask=observed_context_x_mask,
            observed_y_mask=observed_context_y_mask,
        )

        # Create input sequences from target: (B, nt, H)
        seq_in_tar, x_ids, y_ids = self._make_tokens(
            x=x_tar,
            y=None,
            x_mask=x_mask,
            y_mask=y_mask,
            observed_x_mask=observed_target_x_mask,
            observed_y_mask=observed_target_y_mask,
        )

        # Context cache operations -
        if read_cache:
            # Combine cache and new context sequence: [B, n_cached + n_new, H]
            seq_in_ctx_extended = self._combine_new_cached_sequence(seq_in_ctx)
            nc = seq_in_ctx_extended.shape[1]

            seq_in = torch.cat([seq_in_ctx_extended, seq_in_tar], dim=1)
        else:
            # No reading; use current context sequence
            nc = seq_in_ctx.shape[1]
            seq_in = torch.cat([seq_in_ctx, seq_in_tar], dim=1)

        # Update new context sequence to cache if required
        if write_cache:
            self._update_cache(seq_in_ctx)

        return seq_in, x_ids, y_ids, nc

    def _clear_cache(self):
        if hasattr(self, "_embed_context"):
            del self._embed_context

        gc.collect()
        torch.cuda.empty_cache()

    def _is_cache_empty(self):
        if hasattr(self, "_embed_context"):
            return False
        return True

    def _slice_context(self, x_ctx, y_ctx, read_cache, num_new: int = 1):
        """Slice context to only new points if reading from cache.
        Args:
            x_ctx : [B, nc, H]
            y_ctx: [B, nc, H]
            read_cache: whether to read previous context encoding from cache
            num_new (int): how many new context is to be encoded.
        """
        if read_cache and not self._is_cache_empty():
            assert x_ctx.shape[1] >= num_new

            # If reading from cache and cache exists, slice to only new points
            x_ctx_slice = x_ctx[:, -num_new:, :]
            y_ctx_slice = y_ctx[:, -num_new:, :]
            return x_ctx_slice, y_ctx_slice
        else:
            return x_ctx, y_ctx

    def forward(
        self,
        x_ctx: Tensor,
        y_ctx: Tensor,
        x_tar: Tensor,
        x_mask: Tensor,
        y_mask: Tensor,
        observed_target_x_mask: Optional[Tensor] = None,
        observed_target_y_mask: Optional[Tensor] = None,
        read_cache: bool = False,
        write_cache: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Encode context and target points.

        Args:
            x_ctx: Context inputs [B, nc, dx_max]
            y_ctx: Context outputs [B, nc, dy_max]
            x_tar: Target inputs [B, nt, dx_max]
            x_mask: Valid x dimensions [B, dx_max]
            y_mask: Valid y dimensions [B, dy_max]
            observed_target_x_mask: [B, dx_max]
            observed_target_y_mask: [B, dy_max]

        Returns:
            Target representations [B, nt, H], x_ids [dx_max, H], y_ids [dy_max, H]
        """
        # Expand shapes
        nt, dx_max = x_tar.shape[1:]
        dy_max = y_ctx.shape[-1]

        if dx_max > self.max_x_dim:
            raise ValueError(f"x dimension exceeds limits.")
        if dy_max > self.max_y_dim:
            raise ValueError(f"y dimension exceeds limits.")

        if observed_target_x_mask is None:
            observed_target_x_mask_exp = None
        else:
            observed_target_x_mask_exp = repeat(
                observed_target_x_mask, "b dx -> b nt dx", nt=nt
            )

        if observed_target_y_mask is None:
            observed_target_y_mask_exp = None
        else:
            observed_target_y_mask_exp = repeat(
                observed_target_y_mask, "b dy -> b nt dy", nt=nt
            )

        # Make input sequence: [B, N, H]
        seq_in, x_ids, y_ids, nc = self._make_input_sequence(
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_tar=x_tar,
            x_mask=x_mask,
            y_mask=y_mask,
            observed_target_x_mask=observed_target_x_mask_exp,
            observed_target_y_mask=observed_target_y_mask_exp,
            read_cache=read_cache,
            write_cache=write_cache,
        )
        N = seq_in.shape[1]

        # Apply transformer block: [B, N, H]
        mask = self.transformer_block.make_attention_mask(x_mask, N, nc)
        seq_out = self.transformer_block(seq_in, mask=mask)

        # Only return target sequence
        return seq_out[:, nc:], x_ids, y_ids

    def _create_dim_mask(
        self, x_mask: Tensor, x_mask_tar: Tensor, nc: int, nt: int, dx_max: int, H: int
    ) -> Tensor:
        """Create expanded dimension mask for aggregation."""
        B = x_mask.shape[0]

        if x_mask_tar is x_mask:
            return x_mask[:, None, :, None].expand(B, nc + nt, dx_max, H)
        else:
            x_mask_ctx = x_mask[:, None, :, None].expand(B, nc, dx_max, H)
            x_mask_tar = x_mask_tar[:, None, :, None].expand(B, nt, dx_max, H)
            return torch.cat([x_mask_ctx, x_mask_tar], dim=1)

    def predict(
        self,
        x_ctx: Tensor,
        y_ctx: Tensor,
        x_tar: Tensor,
        x_mask: Tensor,
        y_mask: Tensor,
        observed_target_y_mask: Optional[Tensor] = None,
        read_cache: bool = False,
    ) -> GMMOutput:
        """Predict p(y_tar | x_tar, {x_ctx, y_ctx}).
        Args:
            x_ctx: [B, nc, dx_max]
            y_ctx: [B, nc, dy_max]
            x_tar: [B, nt, dx_max]
            x_mask: [B, dx_max]
            y_mask: [B, dy_max]
            observed_target_y_mask: [B, dy_max] or None, True = observed
            read_cache: whether to read previous context sequence from cache

        Returns: Gaussian mixture model output (means, stds, weights) each of shape [B, nt, dy_max, num_components]
        """
        B, dy_max = y_mask.shape
        x_ctx, y_ctx = self._slice_context(x_ctx, y_ctx, read_cache)

        # Default observed target mask to y_mask if not provided
        if observed_target_y_mask is None:
            observed_target_y_mask = y_mask

        # Get target sequence representations
        tokens, _, y_ids = self.forward(
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_tar=x_tar,
            x_mask=x_mask,
            y_mask=y_mask,
            observed_target_y_mask=observed_target_y_mask,
            read_cache=read_cache,
            write_cache=False,  # NOTE: always False; only optimization writes cache
        )

        # Expand: [B, dy_max, nt, H], [B, dy_max, H]
        tokens_expanded = tokens.unsqueeze(1).expand(-1, dy_max, -1, -1)
        y_ids_expanded = y_ids.unsqueeze(0).expand(B, -1, -1)

        # Decode
        return self.decoder(
            task_type=TASK_TOKEN_PRED,
            tokens=tokens_expanded,
            ids=y_ids_expanded,
            x_mask=x_mask,
            y_mask=observed_target_y_mask,
        )

    def _aggregate_space_x_ids(
        self, x_ids: Tensor, query_x_mask: Tensor, B: int, n_spaces: int
    ) -> Tensor:
        """Aggregate x_ids over candidate dimensions for each query space.
        Args:
            x_ids: [dx_max, H]
            query_x_mask: [B, n_spaces, dx_max]
        Returns:
            aggregated_ids: [B, n_spaces, H]
        """
        x_ids_expanded = x_ids.unsqueeze(0).unsqueeze(0).expand(B, n_spaces, -1, -1)
        query_x_mask_expanded = query_x_mask.unsqueeze(-1).expand_as(x_ids_expanded)
        aggregated_ids = self.mean_pooling(x_ids_expanded, query_x_mask_expanded)
        return aggregated_ids

    def _expand_for_query_spaces(self, tensor: Tensor, n_spaces: int) -> Tensor:
        """Expand tensor for multiple query spaces.
        [B, ...] -> [B * n_spaces, ...]"""
        B = tensor.shape[0]
        return (
            tensor.unsqueeze(1)
            .expand(B, n_spaces, *tensor.shape[1:])
            .reshape(B * n_spaces, *tensor.shape[1:])
        )

    def action(
        self,
        x_ctx: Tensor,
        y_ctx: Tensor,
        x_mask: Tensor,
        y_mask: Tensor,
        query_chunks: Tensor,
        query_x_mask: Tensor,
        t: int,
        T: int,
        observed_target_y_mask: Optional[Tensor] = None,
        use_budget: bool = True,
        epsilon: float = 1.0,
        return_logits: bool = False,
        read_cache: bool = False,
        write_cache: bool = False,
        auto_clear_cache: bool = True,
        logit_mask: Optional[Tensor] = None,
    ) -> OptimizationOutput:
        """
        Select next action for optimization.

        Args:
            x_ctx: [B, nc, dx_max]
            y_ctx: [B, nc, dy_max]
            x_mask: [B, dx_max]
            y_mask: [B, dy_max]
            query_chunks: [B, n_query_spaces, n_cand, dx_max]
            query_x_mask: [B, n_query_spaces, dx_max]
            t (int): current timestep
            T (int): optimization horizon
            observed_target_y_mask:
            return_logits: whether to return logits
            ...
            read_cache: whether to read previous context sequence from cache
            write_cache: whether to write new, full context sequence to cache
            auto_clear_cache: whether to automatically clear cache at the end of optimization
            logit_mask: optional valid mask for logits [B, n_query_spaces, n_cand], True = "ignore"

        Returns: OptimizationOutput (next_x, indices, logp, entropy, logits)
        """
        # When read_cache is True, default one observation at a time and always write to cache;
        # TODO multiple new context points, fixed old context...
        if read_cache:
            assert write_cache is True, "If reading cache, must also write cache."
        x_ctx, y_ctx = self._slice_context(x_ctx, y_ctx, read_cache)

        B, n_spaces, n_cand, dx_max = query_chunks.shape

        # Replicate context and masks for query space
        # [B, ...] -> [B * n_spaces, ...]
        x_ctx_expanded = self._expand_for_query_spaces(x_ctx, n_spaces)
        y_ctx_expanded = self._expand_for_query_spaces(y_ctx, n_spaces)
        x_mask_expanded = self._expand_for_query_spaces(x_mask, n_spaces)
        y_mask_expanded = self._expand_for_query_spaces(y_mask, n_spaces)
        observed_target_y_mask_expanded = (
            self._expand_for_query_spaces(observed_target_y_mask, n_spaces)
            if observed_target_y_mask is not None
            else None
        )

        # Expand query masks
        observed_target_x_mask = query_x_mask.reshape(B * n_spaces, dx_max)

        # Encode
        tokens, x_ids, _ = self.forward(
            x_ctx=x_ctx_expanded,
            y_ctx=y_ctx_expanded,
            x_tar=query_chunks.reshape(B * n_spaces, n_cand, dx_max),
            x_mask=x_mask_expanded,
            y_mask=y_mask_expanded,
            observed_target_x_mask=observed_target_x_mask,
            observed_target_y_mask=observed_target_y_mask_expanded,
            read_cache=read_cache,
            write_cache=write_cache,
        )

        # Tokens and ids for decoder: [B, n_spaces, n_cand, H], [B, n_spaces, H]
        tokens = tokens.reshape(B, n_spaces, n_cand, -1)
        ids_aggregated = self._aggregate_space_x_ids(
            x_ids, query_x_mask, B=B, n_spaces=n_spaces
        )

        # Decode
        results = self.decoder(
            task_type=TASK_TOKEN_OPT,
            tokens=tokens,
            ids=ids_aggregated,
            query_chunks=query_chunks,
            query_x_mask=query_x_mask,
            use_budget=use_budget,
            t=t,
            T=T,
            logit_mask=logit_mask,
            epsilon=epsilon,
            return_logits=return_logits,
        )

        # Clear cache if at end of optimization
        if auto_clear_cache and t >= T:
            self._clear_cache()

        # Return results
        # Detach gradients from next_x and indices
        return (
            results.next_x.detach(),
            results.indices.detach(),
            results.logp,
            results.entropy,
            results.logits,
        )


if __name__ == "__main__":
    pass
