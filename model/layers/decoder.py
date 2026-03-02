"""Decoder module for prediction and optimization tasks."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical
from torchrl.modules import MaskedCategorical

from .gmm import GMMPredictionHead, GMMOutput
from .attention import ContextPrefixEncoderLayer


FEEDFORWARD_MULTIPLIER = 4  # feedforward dimension = multiplier * dim_attn
TASK_TOKEN_PRED = 0
TASK_TOKEN_OPT = 1

TASK_TOKEN_PRED_STR = "prediction"
TASK_TOKEN_OPT_STR = "optimization"


@dataclass
class DecoderConfig:
    dim_mlp: int
    dim_attn: int
    nhead: int
    dropout: float
    num_layers: int
    dim_hidden: int
    depth: int
    num_components: int = 20
    ctx_prefix_encoder_layer: bool = True
    use_ar: bool = False
    std_min: float = 1e-4  # minimum standard deviation
    std_max: float = 1.0  # maximum standard deviation
    single_mlp: bool = False


class OptimizationOutput(NamedTuple):
    next_x: Tensor  # (B, 1, d)
    indices: Tensor  # (B, n_query_spaces)
    logp: Tensor  # (B, n_query_spaces)
    entropy: Tensor  # (B, n_query_spaces)
    logits: Optional[Tensor]  # (B, n_query_spaces, n_cand) or None


class Decoder(nn.Module):
    _dim_out_policy: int = 1  # single value for policy head

    def __init__(self, config: DecoderConfig, **kwargs):
        super().__init__()

        self.dim_mlp = config.dim_mlp
        self.dim_attn = config.dim_attn
        self.use_ar = config.use_ar
        self._std_min = config.std_min
        self._std_max = config.std_max

        # Projections
        self.in_proj = self.build_projection(self.dim_mlp, self.dim_attn)
        self.out_proj = self.build_projection(self.dim_attn, self.dim_mlp)
        self.budget_proj = self.build_projection(1, self.dim_mlp)

        # Task tokens: [PRED] and [OPT]
        self.task_tokens = nn.Parameter(torch.randn(2, self.dim_mlp))

        # Autoregressive bias token
        self.ar_bias_token = nn.Parameter(torch.randn(1, self.dim_mlp))

        # Transformer
        self.transformer = self._create_transformer(
            self.dim_attn,
            config.nhead,
            config.dropout,
            config.num_layers,
            config.ctx_prefix_encoder_layer,
        )

        # Heads
        self.prediction_head = GMMPredictionHead(
            self.dim_mlp,
            config.dim_hidden,
            config.depth,
            config.num_components,
            self._std_min,
            self._std_max,
            config.single_mlp,
        )
        self.policy_head = self.build_mlp(
            self.dim_mlp, config.dim_hidden, self._dim_out_policy, config.depth
        )

    @staticmethod
    def build_mlp(dim_in: int, dim_hid: int, dim_out: int, depth: int):
        modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
        for _ in range(depth - 2):
            modules.append(nn.Linear(dim_hid, dim_hid))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dim_hid, dim_out))
        return nn.Sequential(*modules)

    @staticmethod
    def build_projection(dim_in: int, dim_out: int) -> nn.Module:
        return nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def _create_transformer(
        self,
        dim_attn: int,
        nhead: int,
        dropout: float,
        num_layers: int,
        ctx_prefix_encoder_layer: bool,
    ) -> nn.Module:
        """Create transformer encoder."""
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
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        task_type: str,
        tokens: Tensor,
        ids: Tensor,
        x_mask: Optional[Tensor] = None,
        y_mask: Optional[Tensor] = None,
        query_x_mask: Optional[Tensor] = None,
        query_chunks: Optional[Tensor] = None,
        use_budget: bool = True,
        t: Optional[int] = None,
        T: Optional[int] = None,
        epsilon: float = 1.0,
        logit_mask: Optional[Tensor] = None,  # [B, d]
        return_logits: bool = False,
    ) -> Union[GMMOutput, OptimizationOutput]:
        """
        Forward pass for prediction or optimization.

        Args:
            tokens: For prediction: (B, DY, n_cand, H)
                    For optimization: (B, n_query_spaces, n_cand, H)
            ids:    For prediction: (B, DY, H)
                    For optimization: (B, n_query_spaces, H)
            x_mask: (B, dx_max)
            y_mask: (B, dy_max)
            query_x_mask: (B, n_query_spaces, dx_max)
            query_chunks: (B, n_query_spaces, n_cand, dx_max)
            use_budget: whether to use budget token
            t: current time step
            T: total time steps
            epsilon: exploration probability
            logit_mask: (B, n_query_spaces, n_cand) boolean mask for valid logits
            return_logits: whether to return logits

        Returns:
            For prediction: GMMOutput (means, stds, weights) each of shape (B, N, DY, K)
            For optimization: OptimizationOutput (next_x, indices, logp, entropy, logits)
        """
        assert task_type in {TASK_TOKEN_PRED_STR, TASK_TOKEN_OPT_STR}

        if task_type == TASK_TOKEN_PRED_STR:
            return self._forward_prediction(tokens, ids, x_mask, y_mask)
        else:
            return self._forward_optimization(
                tokens=tokens,
                ids=ids,
                query_x_mask=query_x_mask,
                query_chunks=query_chunks,
                use_budget=use_budget,
                t=t,
                T=T,
                epsilon=epsilon,
                logit_mask=logit_mask,
                return_logits=return_logits,
            )

    def _forward_prediction(
        self,
        tokens: Tensor,
        ids: Tensor,
        x_mask: Optional[Tensor],
        y_mask: Optional[Tensor],
    ) -> GMMOutput:
        """Decoder forward pass for prediction task.

        Args:
            tokens: (B, DY, n_cand, H)
            ids: (B, DY, H)
            x_mask: (B, dx_max) or None
            y_mask: (B, dy_max) or None

        Returns:
            GMMOutput: (means, stds, weights) each of shape (B, N, DY, K)
        """
        B, DY, n_cand, H = tokens.shape

        # Prepare tokens
        tokens, ids, task_tokens, _, _ = self._make_tokens(
            tokens=tokens,
            ids=ids,
            task_type=TASK_TOKEN_PRED_STR,
            use_ar=False,
            use_budget=False,
        )

        # Create input sequence: (B, DY, seq_len, H)
        seq_in, n_cand = self._make_input_sequence(task_tokens, ids, tokens)

        # Create mask: (seq_len, seq_len)
        mask = self._make_attention_mask(n_cand, seq_in)

        # Apply transformer: (B * DY, seq_len, H)
        seq_in = seq_in.view(B * DY, -1, H)
        seq_in = self.in_proj(seq_in)
        seq_out = self.transformer(seq_in, mask=mask)
        seq_out_proj = self.out_proj(seq_out)

        # Extract candidates and predict: (B, DY, n_cand, H)
        out_candidate = self._slice_candidate(n_cand, seq_out_proj)
        out_candidate_reshaped = out_candidate.view(B, DY, n_cand, -1)

        assert out_candidate_reshaped.shape == (B, DY, n_cand, self.dim_mlp)
        return self.prediction_head(out_candidate_reshaped, x_mask, y_mask)

    def _forward_optimization(
        self,
        tokens: Tensor,
        ids: Tensor,
        query_x_mask: Tensor,
        query_chunks: Tensor,
        use_budget: bool,
        t: Optional[int],
        T: Optional[int],
        epsilon: float,
        return_logits: bool,
        logit_mask: Optional[Tensor] = None,
    ) -> OptimizationOutput:
        """Forward pass for optimization task.

        Args:
            tokens: (B, n_query_spaces, n_cand, H)
            ids: (B, n_query_spaces, H)
            query_x_mask: (B, n_query_spaces, dx_max)
            query_chunks: (B, n_query_spaces, n_cand, dx_max)
            use_budget: whether to use budget token
            t: current time step
            T: total time steps
            epsilon: exploration probability
            return_logits: whether to return logits
            logit_mask: (B, n_query_spaces, n_cand) boolean mask for valid logits

        Returns:
            OptimizationOutput (next_x, indices, logp, entropy, logits)
        """
        B, n_query_spaces, n_cand, H = tokens.shape
        use_ar = self.use_ar and n_query_spaces > 1

        # Prepare tokens
        tokens, ids, token_task, token_global, token_selected = self._make_tokens(
            tokens, ids, TASK_TOKEN_OPT_STR, use_ar, use_budget, t, T
        )

        # Permute tokens for autoregressive training
        if use_ar:
            tokens, perm_idx = self._permute(tokens)

        # Create input token sequence
        seq_in, n_cand = self._make_input_sequence(
            token_task, ids, tokens, token_global, token_selected
        )

        # Create attention mask
        mask = self._make_attention_mask(n_cand, seq_in)

        # Forward pass
        if use_ar:
            raise NotImplementedError("Autoregressive policy is under development.")
        else:
            # Fully factorized policy
            raw_results = self._forward_factorized(
                n_cand=n_cand,
                seq_in=seq_in,
                mask=mask,
                epsilon=epsilon,
                return_logits=return_logits,
                logit_mask=logit_mask,
            )

        # Unpermute results if needed
        if use_ar:
            raw_results = self._unpermute_results(raw_results, perm_idx)

        # Gather x: (B, 1, dx_max)
        next_x = self._get_x(
            query_chunks=query_chunks, indices=raw_results[0], query_x_mask=query_x_mask
        )

        return OptimizationOutput(
            next_x=next_x,
            indices=raw_results[0],
            logp=raw_results[1],
            entropy=raw_results[2],
            logits=raw_results[3],
        )

    def _get_x(
        self,
        query_chunks: Tensor,
        indices: Tensor,
        query_x_mask: Tensor,
    ) -> Tensor:
        """Get selected x from query_chunks based on indices and mask.

        Args:
            query_chunks: (B, n_query_spaces, n_cand, dx_max)
            indices: (B, n_query_spaces)
            query_x_mask: (B, n_query_spaces, dx_max)

        Returns: x of shape (B, 1, dx_max)
        """
        B, n_query_spaces, n_cand, dx_max = query_chunks.shape

        # Gather selected x: (B * n_query_spaces, 1, dx_max) -> (B, n_query_spaces, dx_max)

        x = self._gather_seq_ele(
            query_chunks.view(B * n_query_spaces, n_cand, -1),
            indices.reshape(B * n_query_spaces, -1),
        )
        x = x.squeeze(1).view(B, n_query_spaces, -1)

        # Zero out invalid dimensions and sum: (B, 1, dx_max)
        x *= query_x_mask.float()

        return torch.sum(x, dim=1, keepdim=True)

    def _forward_autoregressive(
        self,
        n_cand: int,
        seq_in: Tensor,
        mask: Tensor,
        epsilon: float,
        return_logits: bool,
    ) -> Tuple[Tensor, ...]:
        """Autoregressive policy forward pass.
        Under development!

        Args:
            n_cand: number of candidate tokens
            seq_in: (B, n_query_spaces, seq_len, H)
            mask: (seq_len, seq_len) attention mask
            epsilon: exploration probability
            return_logits: whether to return logits
        """
        B, n_query_spaces, seq_len, H = seq_in.shape
        selected_token_start = seq_len - n_cand - 1

        # Project: (B, n_query_spaces, seq_len, dim_attn)
        seq_in = self.in_proj(seq_in).view(B, n_query_spaces, seq_len, self.dim_attn)

        # Collect results
        all_indices = []
        all_logp = []
        all_entropy = []
        all_logits = [] if return_logits else None

        # Start with first query space: (B, seq_len, dim_attn)
        chunk_seq_in = seq_in[:, 0].clone()  # (B, seq_len, dim_attn)

        for i in range(n_query_spaces):
            # Apply transformer
            seq_out = self.transformer(chunk_seq_in, mask=mask)

            # Extract candidates and compute logits: (B, n_cand)
            seq_out_candidate = self._slice_candidate(n_cand, seq_out)
            out_candidate = self.out_proj(seq_out_candidate)
            logits = self.policy_head(out_candidate).squeeze(-1)

            # Sample action: (B,)
            indices, logp, entropy = self._sample(logits, epsilon)

            # Store results
            all_indices.append(indices)
            all_logp.append(logp)
            all_entropy.append(entropy)
            if return_logits:
                all_logits.append(logits)

            # Prepare input for next chunk
            if i + 1 < n_query_spaces:
                # Update selected token: (B, 1, dim_attn)
                selected_token = self._gather_seq_ele(
                    seq_out_candidate, indices.unsqueeze(-1)
                ).detach()

                # Initialize next chunk input and insert selected token
                chunk_seq_in = seq_in[:, i + 1].clone()
                chunk_seq_in[
                    :, selected_token_start : selected_token_start + 1
                ] += selected_token

        return (
            torch.stack(all_indices, dim=-1),
            torch.stack(all_logp, dim=-1),
            torch.stack(all_entropy, dim=-1),
            torch.stack(all_logits, dim=1) if return_logits else None,
        )

    def _forward_factorized(
        self,
        n_cand: int,
        seq_in: Tensor,
        mask: Tensor,
        epsilon: float,
        return_logits: bool = False,
        logit_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, ...]:
        """Fully factorized policy forward pass:
            pi = prod_i pi_i

        Args:
            n_cand: number of candidate tokens
            seq_in: (B, n_query_spaces, seq_len, H)
            mask: (seq_len, seq_len) attention mask
            epsilon: exploration probability
            return_logits: whether to return logits
            logit_mask: (B, n_query_spaces, n_cand) boolean mask for valid logits

        Returns:
        selected_indices (B, n_query_spaces),
        selected_logp (B, n_query_spaces),
        selected_entropy (B, n_query_spaces),
        chunk_logits (B, n_query_spaces, n_cand) or None.
        """
        B, n_query_spaces, seq_len, H = seq_in.shape

        # Flatten batch: (B x n_query_spaces, seq_len, H)
        seq_in = seq_in.view(B * n_query_spaces, seq_len, H)
        if logit_mask is not None:
            logit_mask = logit_mask.view(B * n_query_spaces, -1)

        # Apply transformer
        seq_in = self.in_proj(seq_in)
        seq_out = self.transformer(seq_in, mask=mask)
        seq_out_proj = self.out_proj(seq_out)

        # Extract candidates and compute logits: (B * n_query_spaces, n_cand)
        out_candidate = self._slice_candidate(n_cand, seq_out_proj)
        logits = self.policy_head(out_candidate).squeeze(-1)

        # Sample action
        indices, logp, entropy = self._sample(logits, epsilon, logit_mask)

        return (
            indices.view(B, n_query_spaces),
            logp.view(B, n_query_spaces),
            entropy.view(B, n_query_spaces),
            logits.view(B, n_query_spaces, n_cand) if return_logits else None,
        )

    def _unpermute_results(
        self, results: Tuple[Tensor, ...], perm_idx: Tensor
    ) -> Tuple[Tensor, ...]:
        """Undo permutation of results.

        Args:
            results: (indices, logp, entropy, logits) where logits may be None
                - indices, logp, entropy: (B, n_query_spaces)
                - logits: (B, n_query_spaces, n_cand) or None
            perm_idx: (B, n_query_spaces) permutation indices
        """
        inverse_perm = perm_idx.argsort(dim=1)
        indices, logp, entropy, logits = results

        # Unpermute 2D tensors (B, n_query_spaces)
        indices = torch.gather(indices, 1, inverse_perm)
        logp = torch.gather(logp, 1, inverse_perm)
        entropy = torch.gather(entropy, 1, inverse_perm)

        # Unpermute 3D tensor if present (B, n_query_spaces, n_cand)
        if logits is not None:
            expanded_perm = inverse_perm.unsqueeze(-1).expand_as(logits)
            logits = torch.gather(logits, 1, expanded_perm)

        return indices, logp, entropy, logits

    def _permute(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Randomly permute chunks for autoregressive training.

        Args:
            x: (B, n_chunks, n_cand, H)

        Returns:
            permuted x (B, n_chunks, n_cand, H), perm_idx (B, n_chunks)
        """
        B, n_chunks, n_cand, H = x.shape

        # Generate random permutation indices
        perm_idx = torch.argsort(torch.rand(B, n_chunks, device=x.device), dim=1)
        perm_idx_expanded = perm_idx.view(B, n_chunks, 1, 1).expand(
            B, n_chunks, n_cand, H
        )

        # Apply permutation
        output = torch.gather(x, 1, perm_idx_expanded)
        return output, perm_idx

    def _sample(
        self,
        logits: Tensor,
        epsilon: float = 1.0,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample from categorical distribution.

        Args:
            logits: (B, n_logits)
            epsilon: exploration probability
            mask: (B, n_logits) boolean mask for valid logits

        Returns:
        samples: (B,),
        logp: (B,),
        entropy: (B,).
        """
        # NOTE Set NaN logits to -inf: no probability
        logits = torch.nan_to_num(logits, nan=float("-inf"))

        # If a row is all -inf, set to 0 (uniform probability)
        all_neg_inf = torch.isneginf(logits).all(dim=-1)  # (B,)
        if all_neg_inf.any():
            logits[all_neg_inf] = 0.0

        B, n_logits = logits.shape

        # Create categorical distribution (using logits for numerical stability)
        if mask is None:
            dist = Categorical(logits=logits)
        else:
            dist = MaskedCategorical(logits=logits, mask=mask)

        # Sample or take argmax: (B,)
        if torch.rand(1, device=logits.device).item() < epsilon:
            samples = dist.sample().clamp(min=0, max=n_logits - 1)
        else:
            if mask is None:
                samples = logits.argmax(dim=-1)
            else:
                # Apply mask before argmax: -inf (no probability) for invalid logits
                masked_logits = logits.masked_fill(~mask.bool(), float("-inf"))
                samples = masked_logits.argmax(dim=-1)

        logp = dist.log_prob(samples)
        entropy = dist.entropy()

        return samples, logp, entropy

    def _gather_seq_ele(self, sequence: Tensor, indices: Tensor) -> Tensor:
        """Gather elements along sequence dimension.
        Args:
            sequence: (B, seq_len, H)
            indices: (B, n_indices)
        Returns: gathered elements of shape (B, n_indices, H)
        """
        B, seq_len, H = sequence.shape
        indices_expanded = indices.unsqueeze(-1).expand(B, -1, H)
        return torch.gather(sequence, dim=1, index=indices_expanded)

    @staticmethod
    def _token_or_empty(token, B, n_seq, H, device, dtype):
        """Return token or empty tensor if None."""
        if token is None:
            return torch.empty((B, n_seq, 0, H), device=device, dtype=dtype)
        else:
            return token

    def _make_input_sequence(
        self,
        task_token: Tensor,
        ids: Tensor,
        tokens: Tensor,
        optional_tokens: Optional[Tensor] = None,
        selected_token: Optional[Tensor] = None,
    ) -> Tuple[Tensor, int]:
        """Create input tokens for transformer.

        Args:
            task_token: (B, n_seq, 1, H)
            ids: (B, n_seq, 1, H)
            tokens: (B, n_seq, seq_len, H)
            optional_tokens: (B, n_seq, n_optional, H)
            selected_token: (B, n_seq, 1, H)

        Returns:
        tokens (B, n_seq, S, H), number of candidate tokens
        """

        B, n_seq, n_cand, H = tokens.shape

        optional_tokens = self._token_or_empty(
            optional_tokens, B, n_seq, H, device=tokens.device, dtype=tokens.dtype
        )
        selected_token = self._token_or_empty(
            selected_token, B, n_seq, H, device=tokens.device, dtype=tokens.dtype
        )

        seq_in = torch.cat(
            [task_token, ids, optional_tokens, selected_token, tokens], dim=-2
        )

        return seq_in, n_cand

    def _make_attention_mask(self, n_cand: int, tokens: Tensor) -> Tensor:
        """Create decoder attention mask, True = ignored.
        Non-candidate tokens can be attended by all.
        Candidate tokens can only attend to non-candidate tokens and itself.

        Args:
            n_cand: number of candidate tokens
            tokens: (B, n_seq, seq_len, H)

        Returns: attention mask of shape (seq_len, seq_len)
        """
        B, n_seq, seq_len, H = tokens.shape
        candidate_token_start = seq_len - n_cand
        assert candidate_token_start >= 0, "n_cand exceeds seq_len"

        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=tokens.device)

        # Non-candidate tokens can be attended by all
        mask[:, :candidate_token_start] = True

        # Set diagonal for candidate tokens in-place (candidate attend to itself)
        mask.diagonal()[candidate_token_start:] = True
        return ~mask

    def _slice_candidate(self, n_cand: int, sequence: Tensor) -> Tensor:
        """Get candidate tokens from sequence.
        Args:
            n_cand: number of candidate tokens
            sequence: (batch_size, seq_len, H)

        Returns: token_cand of shape (batch_size, n_cand, H)
        """
        batch_size, seq_len, H = sequence.shape
        candidate_token_start = seq_len - n_cand
        token_cand = sequence[:, candidate_token_start:]
        assert token_cand.shape == (batch_size, n_cand, H)
        return token_cand

    @staticmethod
    def _expand_to_batch(token: Tensor, B: int, n_seq: int, H: int) -> Tensor:
        """(?, H) -> (B, n_seq, ?, H)"""
        return token[None, None, :, :].expand(B, n_seq, -1, H)

    def _validate_budget_params(
        self, use_budget: bool, t: Optional[int], T: Optional[int]
    ):
        """Validate budget token parameters."""
        if use_budget:
            if t is None or T is None:
                raise ValueError("t and T must be provided when use_budget is True.")
            assert 0 <= t <= T, "t must be in the range [0, T]."

    def _make_budget_token(self, t: int, T: int, device):
        """Returns budget token of shape (1, H)."""
        ratio = (T - t) / T
        state_in = torch.full((1, 1), ratio, device=device)
        return self.budget_proj(state_in)  # (1, H)

    def _make_tokens(
        self,
        tokens: Tensor,
        ids: Tensor,
        task_type: str,
        use_ar: bool,
        use_budget: bool,
        t: Optional[int] = None,
        T: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """Prepare tokens for transformer.

        Args:
            tokens: [B, n_seq, seq_len, H]
            ids: [B, n_seq, H]
            task_type: "prediction" or "optimization"
            use_ar: whether to create AR token
            use_budget: whether to create budget token
            t: current time step
            T: total time step

        Returns:
            tokens (B, n_seq, seq_len, H),
            ids (B, n_seq, 1, H),
            task_token (B, n_seq, 1, H),
            optional_tokens (B, n_seq, n_optional, H) or None,
            selected_token (B, n_seq, 1, H) or None
        """
        B, n_seq, seq_len, H = tokens.shape
        is_optimization = task_type == TASK_TOKEN_OPT_STR

        self._validate_budget_params(use_budget, t, T)

        # task tokens
        task_idx = TASK_TOKEN_OPT if is_optimization else TASK_TOKEN_PRED
        task_token = self._expand_to_batch(
            self.task_tokens[task_idx : task_idx + 1], B, n_seq, H
        )

        # ids
        ids = ids.unsqueeze(-2).expand(B, n_seq, -1, H)

        if not is_optimization:
            return tokens, ids, task_token, None, None

        # optional tokens (only for optimization)
        optional_tokens = None
        selected_token = None
        if use_ar:
            selected_token = self._expand_to_batch(
                self.ar_bias_token.to(tokens), B, n_seq, H
            )

        if use_budget:
            budget = self._make_budget_token(t, T, tokens.device)
            optional_tokens = self._expand_to_batch(budget, B, n_seq, H)

        return tokens, ids, task_token, optional_tokens, selected_token


if __name__ == "__main__":
    B, n_seq, seq_len, H = 2, 3, 10, 8
    n_cand = 3
    device = "cpu"

    decoder = Decoder(
        DecoderConfig(
            dim_mlp=H,
            dim_attn=H,
            nhead=2,
            dropout=0.1,
            num_layers=2,
            dim_hidden=16,
            depth=2,
            num_components=5,
        )
    )

    # Test build_projection
    proj_diff = Decoder.build_projection(4, 8)
    proj_same = Decoder.build_projection(8, 8)
    assert isinstance(proj_diff, nn.Linear)
    assert isinstance(proj_same, nn.Identity)
    print("[PASS] build_projection")

    # Test _expand_to_batch
    token = torch.randn(1, H)
    expanded = decoder._expand_to_batch(token, B, n_seq, H)
    assert expanded.shape == (B, n_seq, 1, H)
    print("[PASS] _expand_to_batch")

    # Test _token_or_empty
    empty = decoder._token_or_empty(None, B, n_seq, H, device, torch.float32)
    assert empty.shape == (B, n_seq, 0, H)
    non_empty = torch.randn(B, n_seq, 2, H)
    result = decoder._token_or_empty(non_empty, B, n_seq, H, device, torch.float32)
    assert torch.equal(result, non_empty)
    print("[PASS] _token_or_empty")

    # Test _make_attention_mask
    tokens = torch.randn(B, n_seq, seq_len, H)
    mask = decoder._make_attention_mask(n_cand=n_cand, tokens=tokens)
    assert mask.shape == (seq_len, seq_len)
    # Check mask properties: non-candidate tokens should be attendable by all
    candidate_start = seq_len - n_cand
    assert (~mask[:, :candidate_start]).all()  # non-candidate cols are False (attended)
    # Candidate tokens attend only to themselves (diagonal is False)
    for i in range(candidate_start, seq_len):
        assert not mask[i, i]  # diagonal is False (can attend to self)
    print("[PASS] _make_attention_mask")

    # Test _slice_candidate
    sequence = torch.randn(B, seq_len, H)
    candidates = decoder._slice_candidate(n_cand, sequence)
    assert candidates.shape == (B, n_cand, H)
    assert torch.equal(candidates, sequence[:, -n_cand:, :])
    print("[PASS] _slice_candidate")

    # Test _gather_seq_ele
    sequence = torch.randn(B, seq_len, H)
    indices = torch.tensor([[0, 2], [1, 3]])  # (B, 2)
    gathered = decoder._gather_seq_ele(sequence, indices)
    assert gathered.shape == (B, 2, H)
    assert torch.equal(gathered[0, 0], sequence[0, 0])
    assert torch.equal(gathered[0, 1], sequence[0, 2])
    print("[PASS] _gather_seq_ele")

    # Test _validate_budget_params
    decoder._validate_budget_params(
        use_budget=False, t=None, T=None
    )  # Should not raise
    decoder._validate_budget_params(use_budget=True, t=2, T=5)  # Should not raise
    try:
        decoder._validate_budget_params(use_budget=True, t=None, T=5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] _validate_budget_params")

    # Test _make_budget_token
    budget_token = decoder._make_budget_token(t=2, T=5, device=device)
    assert budget_token.shape == (1, H)
    print("[PASS] _make_budget_token")

    # Test _make_input_sequence
    task_token = torch.randn(B, n_seq, 1, H)
    ids = torch.randn(B, n_seq, 1, H)
    tokens = torch.randn(B, n_seq, n_cand, H)
    seq_in, returned_n_cand = decoder._make_input_sequence(task_token, ids, tokens)
    assert seq_in.shape == (B, n_seq, 1 + 1 + n_cand, H)  # task + ids + tokens
    assert returned_n_cand == n_cand
    # With optional tokens
    optional = torch.randn(B, n_seq, 2, H)
    seq_in, _ = decoder._make_input_sequence(
        task_token, ids, tokens, optional_tokens=optional
    )
    assert seq_in.shape == (B, n_seq, 1 + 1 + 2 + n_cand, H)
    print("[PASS] _make_input_sequence")

    # Test _permute
    x = torch.randn(B, 4, n_cand, H)
    permuted, perm_idx = decoder._permute(x)
    assert permuted.shape == x.shape
    assert perm_idx.shape == (B, 4)
    # Verify permutation is valid (all indices present)
    for b in range(B):
        assert set(perm_idx[b].tolist()) == set(range(4))
    print("[PASS] _permute")

    # Test _unpermute_results
    n_query_spaces = 4
    indices = torch.randint(0, n_cand, (B, n_query_spaces))
    logp = torch.randn(B, n_query_spaces)
    entropy = torch.randn(B, n_query_spaces)
    logits = torch.randn(B, n_query_spaces, n_cand)
    perm_idx = torch.stack([torch.randperm(n_query_spaces) for _ in range(B)])
    results = (indices, logp, entropy, logits)
    unpermuted = decoder._unpermute_results(results, perm_idx)
    assert len(unpermuted) == 4
    assert unpermuted[0].shape == indices.shape
    assert unpermuted[3].shape == logits.shape
    # Test with None logits
    results_no_logits = (indices, logp, entropy, None)
    unpermuted_no_logits = decoder._unpermute_results(results_no_logits, perm_idx)
    assert unpermuted_no_logits[3] is None
    print("[PASS] _unpermute_results")

    # Test _sample
    logits = torch.randn(B, 5)
    samples, log_prob, ent = decoder._sample(logits, epsilon=1.0)
    assert samples.shape == (B,)
    assert log_prob.shape == (B,)
    assert ent.shape == (B,)
    assert (samples >= 0).all() and (samples < 5).all()
    # Test with mask
    mask = torch.tensor(
        [[True, True, False, False, False], [True, True, True, False, False]]
    )
    samples_masked, _, _ = decoder._sample(logits, epsilon=1.0, mask=mask)
    assert samples_masked.shape == (B,)
    # Test with NaN logits
    nan_logits = torch.tensor([[float("nan"), 1.0, 2.0], [1.0, float("nan"), 0.0]])
    samples_nan, _, _ = decoder._sample(nan_logits, epsilon=1.0)
    assert samples_nan.shape == (B,)
    print("[PASS] _sample")

    # Test _get_x
    n_query_spaces = 2
    dx_max = 4
    query_chunk = torch.randn(B, n_query_spaces, n_cand, dx_max)
    indices = torch.randint(0, n_cand, (B, n_query_spaces))
    query_x_mask = torch.ones(B, n_query_spaces, dx_max)
    query_x_mask[:, :, -1] = 0  # Mask out last dimension
    x = decoder._get_x(query_chunk, indices, query_x_mask)
    assert x.shape == (B, 1, dx_max)
    print("[PASS] _get_x")

    # Test _make_tokens for prediction
    tokens = torch.randn(B, n_seq, n_cand, H)
    ids = torch.randn(B, n_seq, H)
    out_tokens, out_ids, task_token, opt_tokens, sel_token = decoder._make_tokens(
        tokens, ids, task_type=TASK_TOKEN_PRED_STR, use_ar=False, use_budget=False
    )
    assert out_tokens.shape == tokens.shape
    assert out_ids.shape == (B, n_seq, 1, H)
    assert task_token.shape == (B, n_seq, 1, H)
    assert opt_tokens is None
    assert sel_token is None
    print("[PASS] _make_tokens (prediction)")

    # Test _make_tokens for optimization with budget
    out_tokens, out_ids, task_token, opt_tokens, sel_token = decoder._make_tokens(
        tokens,
        ids,
        task_type=TASK_TOKEN_OPT_STR,
        use_ar=False,
        use_budget=True,
        t=2,
        T=5,
    )
    assert opt_tokens.shape == (B, n_seq, 1, H)  # budget token
    assert sel_token is None  # no AR
    print("[PASS] _make_tokens (optimization with budget)")

    print("\n" + "=" * 50)
    print("All utility function tests passed!")
    print("=" * 50)
