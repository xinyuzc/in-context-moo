"""Extracting and restoring valid dimensions based on masks."""

import torch
from torch import Tensor
from typing import Optional, Tuple


def compact_by_mask(
    data: Tensor, mask: Optional[Tensor] = None, dim: int = -1
) -> Tensor:
    """Compact data by removing elements where the mask is False along dim.

    Handles two masking modes:
    - 1D mask: Shared mask across all batches, returns same shape batch
    - 2D mask: Per-batch mask, requires consistent valid_dim count to stack results

    Args:
        data: shape[dim] equals mask.shape[-1]
        mask: Valid dim mask.
            If 1D: [max_dim], broadcast for all batches
            If 2D: [b, max_dim], different for each batch
        dim (int): Dimension along which to select

    Returns:
        selected_data: shape[dim] equals to number of True in mask
    """
    # Handle trivial cases
    if data is None or mask is None:
        return data

    def _select(data, mask, dim):
        indices = mask.nonzero(as_tuple=True)[0]  # [valid_dim]
        return torch.index_select(data, dim, indices)

    # Type and shape checks
    mask = mask.to(device=data.device, dtype=torch.bool)
    max_dim = mask.shape[-1]
    if dim < 0:  # example: dim=-1, ndim=3, then dim=2
        dim = data.ndim + dim
    assert max_dim == data.shape[dim], f"Dim mismatch: {max_dim} != {data.shape[dim]}"

    if mask.ndim == 1:
        return _select(data, mask, dim)
    elif mask.ndim == 2:
        b = mask.shape[0]
        assert data.shape[0] == b, f"Batch mismatch: {data.shape[0]} != {b}"
        assert dim != 0, "Batch dim cannot be compacted."

        # Get batched selected data: [batch_size, ..., valid_dim]
        selected_data_list = []
        for i in range(b):
            selected_data = _select(data[[i]], mask[i], dim)
            selected_data_list.append(selected_data)
        try:
            return torch.stack(selected_data_list, dim=0)
        except RuntimeError as e:
            raise RuntimeError(
                f"Inconsistent number of True in mask across batch. Cannot stack results."
            ) from e
    else:
        raise ValueError(f"Mask ndim {mask.ndim} not supported.")


def restore_by_mask(data: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Restore data to original shape by filling zeros for invalid dims.

    Examples:
        data: [[[1.0, 2.0]]]
        mask: [True, False, True]
        data_full: [[[1.0, 0.0, 2.0]]]

    Args:
        data: shape[dim] equals to valid_dim
        mask: Valid dim mask, [max_dim]
        dim: Dimension along which to restore

    Returns:
        data_full: shape[dim] equals to max_dim
    """
    # Handle trivial cases
    if data is None or mask is None:
        return data

    # Type and shape checks
    assert mask.ndim == 1, f"Mask ndim {mask.ndim} not supported."
    mask = mask.to(device=data.device, dtype=torch.bool)
    max_dim = mask.shape[-1]
    valid_dim = mask.int().sum().item()
    assert data.shape[dim] == valid_dim

    # Handle trivial case
    if valid_dim == max_dim:
        return data

    # Restore data
    desired_shape = list(data.shape)
    desired_shape[dim] = max_dim
    data_full = torch.zeros(desired_shape, device=data.device, dtype=data.dtype)
    valid_indices = mask.nonzero(as_tuple=True)[0]
    data_full.index_copy_(dim, valid_indices, data)

    return data_full


def _generate_top_k_mask(k_batch: Tensor, max_dim: int, device: str) -> Tensor:
    """Generate a mask of shape [B, max_dim] with top-k dims as True."""
    B = k_batch.shape[0]
    dim_indices = torch.arange(max_dim, dtype=torch.long, device=device)
    dim_indices = dim_indices.unsqueeze(0).expand(B, -1)
    mask = dim_indices < k_batch.unsqueeze(1)
    return mask


def _get_random_permutation(B, dim, device) -> Tensor:
    random_indices = torch.argsort(torch.rand((B, dim), device=device), dim=-1)
    return random_indices


def gather_by_indices(data, indices) -> Tensor:
    """Gather data of shape [B, N, D] by indices of shape [B, D] along last dimension."""
    if indices is None:
        return data

    indices_exp = indices.unsqueeze(1).expand_as(data)
    return torch.gather(data, dim=-1, index=indices_exp)


def generate_dim_mask(
    max_dim: int,
    device: str,
    k: Optional[Tensor | int] = None,
    dim_scatter_mode: str = "random_k",
) -> Tuple[Tensor, Tensor]:
    """Generate mask given maximum dimension size and specified number of valid dimensions.

    Args:
        max_dim: Maximum dimension size
        device: Computational device
        k: Number of valid dimensions, None | int | [B]
        dim_scatter_mode: ["random_k", "top_k"]. Default as "random_k"

    Returns:
        mask: Generated mask, [max_dim] | [max_dim] | [B, max_dim]
        valid_dim_indices: Optional valid dimension indices of shape [B, max_dim] when dim_scatter_mode is "random_k".
    """
    if dim_scatter_mode not in ["random_k", "top_k"]:
        raise ValueError(f"Invalid dim_scatter_mode: {dim_scatter_mode}.")

    valid_dim_indices = None
    if k is None:
        mask = torch.ones((max_dim,), dtype=torch.bool, device=device)
    else:
        if isinstance(k, int):
            make_single_mask = True
            k_batch = torch.tensor([k], dtype=torch.long, device=device)  # [1]
        else:
            make_single_mask = False
            if not (isinstance(k, Tensor) and k.ndim == 1):
                raise ValueError(f"Invalid k: {k}. Expected k to be a 1-dim Tensor.")

            k_batch = k.to(device=device)  # [B]

        B = k_batch.shape[0]
        mask = _generate_top_k_mask(k_batch=k_batch, max_dim=max_dim, device=device)

        if dim_scatter_mode == "random_k":
            valid_dim_indices = _get_random_permutation(B=B, dim=max_dim, device=device)
            mask = torch.gather(mask, dim=-1, index=valid_dim_indices)

        if make_single_mask:
            mask = mask.squeeze(0)  # [1, max_dim] -> [max_dim]

    mask.requires_grad_(False)
    return mask, valid_dim_indices
