"""Sampling from (factorized) input spaces.
NOTE assumption: the valid dimensions of subspaces do not overlap.

Factorized representation:
    - chunks: [d, dx_max] - d samples per subspace, zero-padded to dx_max
    - chunk_mask: [n, dx_max] - n subspaces (chunks), True indicates active dims

For full policy: n=1, all dims in one space
For factorized policy: n>1, dims split into independent + joint subspaces
"""

import torch
from torch import Tensor
from einops import rearrange, repeat
import sobol_seq
from typing import Tuple, Optional, Union
from utils.types import FloatListOrNestedOrTensor
from data.base.preprocessing import make_range_tensor

GRID_SIZE = 1000


def generate_sobol_samples(
    x_range: Tensor, num_datapoints: int, grid: bool, seed: int = 0
) -> Tensor:
    """Generate sobol samples: [num_datapoints, d]

    Args:
        x_range: [d, 2]
        num_datapoints (int): Number of datapoints to generate
        grid (bool): Whether to sample from a grid or randomly
            grid is True: num_datapoints sobol samples on input space
            otherwise: max(GRID_SIZE, num_datapoints) sobol samples, and select `num_datapoints` randomly

    Returns: x of shape [num_datapoints, d]
    """
    assert x_range.ndim == 2 and x_range.shape[1] == 2
    dim_num = x_range.shape[0]
    tkwargs = {"device": x_range.device, "dtype": torch.float32}

    def _get_num_sobol_samples(grid, num_datapoints):
        """Decide number of sobol samples to generate."""
        if grid:
            return num_datapoints
        else:
            return max(GRID_SIZE, num_datapoints)

    n = _get_num_sobol_samples(grid, num_datapoints)

    # Generate Sobol samples within [0,1]^dim_num
    sobol_samples_np = sobol_seq.i4_sobol_generate(dim_num=dim_num, n=n, skip=seed)
    sobol_samples = torch.from_numpy(sobol_samples_np).to(**tkwargs)

    # Permute Sobol samples and select the first `num_datapoints`
    perm_indices = torch.randperm(n, device=x_range.device)
    samples_perm = sobol_samples[perm_indices]
    samples_perm = samples_perm[:num_datapoints]

    # Rescale to required range
    x_min, x_max = x_range[:, 0], x_range[:, 1]
    x = x_min + (x_max - x_min) * samples_perm
    x = torch.clamp(x, min=x_min, max=x_max)

    return x.detach()


def _sample_joint_subspace(
    x_range: Tensor, d: int, grid: bool, seed: int = 0
) -> Tensor:
    """Sample over joint space -> x [d, x_dim]"""
    x = generate_sobol_samples(x_range=x_range, num_datapoints=d, grid=grid, seed=seed)
    return x


def _sample_ind_subspaces(x_range: Tensor, d: int, grid: bool, seed: int = 0) -> Tensor:
    """Sample each dim independently -> [d, x_dim]"""
    D = x_range.shape[0]

    # Sample each dimension: D x [d, 1] -> [D, d, 1]
    x_list = [
        generate_sobol_samples(
            x_range=x_range[[index], :], num_datapoints=d, grid=grid, seed=seed
        )
        for index in range(D)
    ]
    x = torch.stack(x_list, dim=0)
    x = x.squeeze(2)
    x = rearrange(x, "D d -> d D")
    return x


def _sample_subspace_n_scatter_(
    sample_joint: bool,
    chunks: Tensor,
    chunk_mask: Tensor,
    x_range: Tensor,
    x_dim_indices: Tensor,
    chunk_index_slice: Union[Tensor, slice],
    d: int,
    grid: bool,
    seed: int = 0,
):
    """Sample subspaces and in-place scatter into chunks.

    Args:
        sample_joint (bool): Whether to sample jointly or independently
        chunks: [d, max_x_dim], d is number of samples
        chunk_mask: [n, max_x_dim], n is number of chunks
        x_range: [count_subspace, 2], count_subspace <= max_x_dim
        x_dim_indices: [count_subspace], indices of valid dims in subspaces
        chunk_index_slice: [count_chunk], count_chunk <= n, indices of chunks (subspaces) to fill
        d (int): Number of samples
        grid (bool): Sample from a grid or randomly
    """
    # Sample `count_subspace` subspaces: [d, count_subspace]
    if sample_joint:
        x = _sample_joint_subspace(x_range=x_range, d=d, grid=grid, seed=seed)
    else:
        x = _sample_ind_subspaces(x_range=x_range, d=d, grid=grid, seed=seed)

    # Scatter samples into chunks: [d, count_subspace] -> [d, max_x_dim]
    x_dim_indices_exp = repeat(x_dim_indices, "c -> d c", d=d)
    chunks.scatter_(dim=-1, index=x_dim_indices_exp, src=x)

    # Set mask for the subspace as True
    chunk_mask[chunk_index_slice, x_dim_indices] = True


def _sample_factorized_domain(
    x_range: FloatListOrNestedOrTensor,
    x_dim_mask: Tensor,
    q_dim_mask: Tensor,
    num_subspace_points: int,
    use_grid_sampling: bool,
    seed: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Sample from a factorized space (efficient version).

    Factorization strategy:
    - q_dim_mask=True dims: sampled independently (reduces action space size)
    - q_dim_mask=False dims: sampled jointly (preserves correlations)

    Requiring no overlapping valid dims in each subspace,
    we can use chunks of shape [num_subspace_points, dx_max],
    mask of shape [count_chunk, dx_max] to represent subspaces.

    Args:
        x_range: Input range
        x_dim_mask: Mask for valid dims, shape [x_dim]
        q_dim_mask: Mask for dims that will be independently sampled, shape [x_dim]
        num_subspace_points: number of samples in each subspace
        use_grid_sampling: whether to sample from a grid or random locations

    Returns:
        chunks: [num_subspace_points, dx_max]
        chunk_mask: [count_chunk, dx_max]
            chunk_mask[:count_ind]: independent subspaces (one per dim)
            chunk_mask[-1]: joint subspace (if exists)
    """
    device = x_dim_mask.device
    dx_max = x_dim_mask.shape[-1]

    dims = torch.arange(dx_max, device=device)
    x_range_t = make_range_tensor(x_range, num_dim=dx_max).to(device)  # [dx_max, 2]

    # Find valid dims masks for independent and joint subspaces
    ind_mask = q_dim_mask & x_dim_mask  # [dx_max]
    joint_mask = (~q_dim_mask) & x_dim_mask  # [dx_max]

    # Count valid dims for independent and joint subspaces
    count_ind = ind_mask.int().sum().item()
    count_joint = joint_mask.int().sum().item()
    count_chunk = count_ind + int(count_joint > 0)

    # Preallocate chunks and mask
    chunks = torch.zeros(
        [num_subspace_points, dx_max], device=device, dtype=torch.float32
    )
    chunk_mask = torch.zeros([count_chunk, dx_max], device=device, dtype=torch.bool)

    # Sample joint space and scatter into chunks[-1:]
    if count_joint > 0:
        index_slice_joint = slice(-1, None)
        x_range_joint = x_range_t[joint_mask, :]  # [count_joint, 2]
        dims_joint = dims[joint_mask]  # [count_joint]

        _sample_subspace_n_scatter_(
            sample_joint=True,
            x_range=x_range_joint,
            x_dim_indices=dims_joint,
            chunk_index_slice=index_slice_joint,
            chunks=chunks,
            chunk_mask=chunk_mask,
            d=num_subspace_points,
            grid=use_grid_sampling,
            seed=seed,
        )

    # Sample independent spaces and scatter into chunks[:count_ind]
    if count_ind > 0:
        index_slice_ind = torch.arange(count_ind, device=device)
        x_range_ind = x_range_t[ind_mask, :]  # [count_ind, 2]
        dims_ind = dims[ind_mask]  # [count_ind]

        _sample_subspace_n_scatter_(
            sample_joint=False,
            x_range=x_range_ind,
            x_dim_indices=dims_ind,
            chunk_index_slice=index_slice_ind,
            chunks=chunks,
            chunk_mask=chunk_mask,
            d=num_subspace_points,
            grid=use_grid_sampling,
            seed=seed,
        )

    # Final checks: no NaNs or infs in chunks
    assert not torch.isnan(chunks).any(), f"chunks has NaNs: {chunks}"
    assert not torch.isinf(chunks).any(), f"chunks has infs: {chunks}"

    return chunks, chunk_mask


def _combine_factorized_domain(
    chunks: Tensor, chunk_mask: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Combine factorized domains in the full space (efficient version).
    NOTE: chunks are combined in order, i.e., pattern exists, be careful with slices!

    The function creates a full-space grid of indices and then gathers and sums
    data from 'chunks' based on these indices, using 'chunk_mask' for dimension selection.

    Args:
        chunks: Data chunks, shape [d, dx_max].
        chunk_mask: Mask for valid dims in each chunk, shape [n_chunks, dx_max].
            Must have NO overlapping valid dims between chunks.

    Returns:
        samples: Combined samples, shape [M, dx_max], where M = d^n_chunks.
        mask: Combined mask over the full space, shape [dx_max].
        chunk_coord_grid: The grid of indices used for combination, shape [M, n_chunks].
    """
    # ==== Checks ====
    assert chunk_mask.shape[-1] == chunks.shape[-1]
    assert chunk_mask.ndim == 2 and chunks.ndim == 2

    # Check for no overlapping valid dims between chunks: [n_chunks, dx_max] -> [dx_max]
    valid_dim_counts = chunk_mask.int().sum(dim=0)
    if not torch.all((valid_dim_counts <= 1)):
        raise ValueError("chunk_mask has overlapping valid dims between chunks.")

    d, dx_max = chunks.shape
    n_chunks, _ = chunk_mask.shape

    # ==== Create index grid ====
    # Range [0, 1, ..., d-1]
    coord_range = torch.arange(d, device=chunks.device, dtype=torch.long)
    # Create list of ranges for each chunk
    chunk_index_ranges = [coord_range] * n_chunks

    # Create grid via product of chunk index (M x [chunk_1_indice, ..., chunk_n_indice])
    # NOTE For n_chunks=1, cartesian_prod returns a 1D tensor [M], not [M, 1].
    # The subsequent view/reshape handles this.
    chunk_coord_grid = torch.cartesian_prod(*chunk_index_ranges)

    # Ensure shape is [M, n_chunks]
    chunk_coord_grid = chunk_coord_grid.view(-1, n_chunks)
    M = d**n_chunks
    assert chunk_coord_grid.shape == (M, n_chunks)

    # ==== Combine samples ====
    # Pre-calculate masked chunks: [n_chunks, d, dx_max]
    masked_chunks = chunks.unsqueeze(0) * chunk_mask.unsqueeze(1).to(chunks.dtype)

    # Preallocate samples
    samples = torch.zeros(M, dx_max, device=chunks.device, dtype=chunks.dtype)

    for i in range(n_chunks):
        # Indices in the i-th chunk: [M]
        coord_grid = chunk_coord_grid[:, i]

        ## Add samples from the i-th chunk
        chunk_data = masked_chunks[i]  # [d, dx_max]
        samples.add_(chunk_data[coord_grid])  # [M, dx_max]

    # ==== Combine mask ====
    # Combine mask over full space via OR operation: [dx_max]
    mask = chunk_mask.any(dim=0)

    return samples, mask, chunk_coord_grid


def sample_factorized_domain(
    d: int,
    max_x_dim: int,
    input_bounds: FloatListOrNestedOrTensor,
    device: str,
    x_mask: Optional[Tensor] = None,
    use_grid_sampling: bool = False,
    use_factorized_policy: bool = False,
    seed: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Sample input locations from factorized subspaces.

    Args:
        d (int): Number of samples
        x_mask: Mask for valid x dims, shape [max_x_dim]
        input_bounds (FloatListOrNestedOrTensor): Input range
        use_grid_sampling (bool): Whether to sample from a grid or random locations
        use_factorized_policy (bool): Whether to use factorized policy
            True: sample from factorized subspaces and combine
            False: sample from the full space directly

    Returns:
        chunks: [d, dx_max]
        chunk_mask: [n, dx_max], n is number of chunks
    """
    # Prepare masks
    if x_mask is None:
        x_mask = torch.ones(max_x_dim, device=device, dtype=torch.bool)

    if use_factorized_policy:
        q_mask = torch.ones(max_x_dim, device=device, dtype=torch.bool)
    else:
        q_mask = torch.zeros(max_x_dim, device=device, dtype=torch.bool)

    # Sample chunks: [d, max_x_dim] and [n, max_x_dim]
    chunks, chunk_mask = _sample_factorized_domain(
        x_range=input_bounds,
        x_dim_mask=x_mask,
        q_dim_mask=q_mask,
        num_subspace_points=d,
        use_grid_sampling=use_grid_sampling,
        seed=seed,
    )

    return chunks, chunk_mask


def sample_domain(
    d: int,
    max_x_dim: int,
    device: str,
    input_bounds: FloatListOrNestedOrTensor,
    x_mask: Optional[Tensor] = None,
    use_grid_sampling: bool = False,
    use_factorized_policy: bool = False,
    seed: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Sample factorized domains and combine in the full space.

    Args:
        d (int): Number of samples
        x_mask: [max_x_dim], mask for valid dims
        input_bounds (FloatListOrNestedOrTensor): Input range
        use_grid_sampling (bool): Whether to sample from a grid or random locations
        use_factorized_policy (bool): Whether to use factorized policy
            True: sample from factorized subspaces and combine
            False: sample from the full space directly

    Returns:
        x: [m, max_x_dim], m is number of combined samples.
        chunks: [d, max_x_dim]
        chunk_mask: [n, max_x_dim], n is number of chunks
    """
    chunks, chunk_mask = sample_factorized_domain(
        d=d,
        max_x_dim=max_x_dim,
        device=device,
        input_bounds=input_bounds,
        x_mask=x_mask,
        use_grid_sampling=use_grid_sampling,
        use_factorized_policy=use_factorized_policy,
        seed=seed,
    )

    # Combine chunks on full space: [d, max_x_dim], [n, max_x_dim] -> [m, max_x_dim]
    x = _combine_factorized_domain(chunks, chunk_mask)[0]

    return x, chunks, chunk_mask


def factorized_to_flat_index(chunk_indices: Tensor, n: int, d: int) -> Tensor:
    """Turn index in factorized spaces to index in full space.

    Args:
        chunk_indices: [..., n]
        n: number of chunks
        d: number of points in each chunk

    Returns:
        full_indices: [..., 1]
    """
    assert chunk_indices.shape[-1] == n
    indices = chunk_indices.long()

    # Compute powers of Base d: [n-1, n-2, ..., 0]
    exponents = torch.arange(n - 1, -1, -1, device=indices.device, dtype=torch.long)

    # Compute powers tensor (weights): [n]
    weights = (d**exponents).long()

    # Base conversion and summation: [..., n] -> [..., 1]
    full_indices = (indices * weights).sum(dim=-1, keepdim=True)

    return full_indices


def get_num_subspace_points(
    x_dim: int, use_factorized_policy: bool = True, d: Optional[int] = None
) -> int:
    """Get number of points in each subspace."""
    if d is not None:
        return d
    assert 0 < x_dim <= 4, "x_dim must be in [1, 2, 3, 4]"

    if use_factorized_policy:
        num_categories_dict = {
            1: 128,
            2: 32,
            3: 32,
            4: 32,
        }
    else:
        num_categories_dict = {
            1: 100,
            2: 200,
            3: 300,
            4: 400,
        }

    return num_categories_dict[x_dim]
