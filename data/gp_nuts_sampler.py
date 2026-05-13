"""Fully-Bayesian GP posterior sampler.

Hyperparameters (lengthscale, outputscale, noise) are drawn from the posterior
p(theta | x_ctx, y_ctx) via pyro NUTS, using priors that mirror the rest of
this repo (see `data/sampler.py::_sample_lengthscale` for reference).

Pipeline per task:
    1. Sample a kernel type (rbf / matern12/32/52) discretely outside NUTS.
    2. Run NUTS on (log_lengthscale, std, noise_var) conditioned on (x_ctx, y_ctx_t).
    3. For each posterior theta sample, build a gpytorch ExactGP with that fixed
       theta (no MLE optimization!), condition on (x_ctx, y_ctx_t), and draw one
       function value at x_test.

"""

import math
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

from data.sampler import (
    DATA_KERNEL_TYPE_LIST,
    STD_RANGE,
    JITTER,
    MAX_TRIES,
)


# Match _sample_lengthscale in data/sampler.py: log_lengthscale ~ N(mu, sigma)
# with mu = log(2/3), sigma = 0.5, post-multiplied by sqrt(x_dim).
_MU_LOG_LS = math.log(2.0 / 3.0)
_SIGMA_LOG_LS = 0.5

# Tri-state cache for NUTS JIT compilation:
#   None  -> untried; try jit_compile=True on next run.
#   True  -> JIT works in this process; keep using it.
#   False -> JIT failed once (e.g. NVRTC arch mismatch); skip it from now on.
# Why: nvrtc errors surface only when mcmc.run() actually compiles, so we
# probe once and remember the answer for the rest of the process.
_NUTS_JIT_OK: Optional[bool] = None


def _is_jit_compile_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "nvrtc" in msg
        or "torchscript" in msg
        or "gpu-architecture" in msg
        or "jit" in msg
    )


def _kernel_correlation_matrix(
    x1: Tensor, x2: Tensor, kernel_type: str, lengthscale: Tensor
) -> Tensor:
    """Manual kernel computation (no outputscale).

    Args:
        x1: [N1, d]
        x2: [N2, d]
        lengthscale: [d] (broadcasts against the last dim of x1, x2)
        kernel_type: one of {rbf, matern12, matern32, matern52}

    Returns: [N1, N2]
    """
    diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)  # [N1, N2, d]
    scaled = diff / lengthscale
    sqdist = scaled.pow(2).sum(dim=-1)
    sqdist = sqdist.clamp_min(1e-30)  # avoid sqrt(0) -> NaN gradient

    if kernel_type == "rbf":
        return torch.exp(-0.5 * sqdist)

    r = sqdist.sqrt()
    if kernel_type == "matern12":
        return torch.exp(-r)
    if kernel_type == "matern32":
        s = math.sqrt(3.0) * r
        return (1.0 + s) * torch.exp(-s)
    if kernel_type == "matern52":
        s = math.sqrt(5.0) * r
        return (1.0 + s + (5.0 / 3.0) * sqdist) * torch.exp(-s)
    raise ValueError(f"Unsupported kernel_type for NUTS sampler: {kernel_type}")


def _gp_marginal_likelihood_model_batched(
    x: Tensor,                  # [G, N, d]
    y: Tensor,                  # [G, N]
    kernel_type: str,
    std_range: Tuple[float, float],
    jitter: float,
    mu_log_ls: float = _MU_LOG_LS,
    sigma_log_ls: float = _SIGMA_LOG_LS,
):
    """Pyro model: log Π_g p(y_g | x_g, theta_g) p(theta_g).

    All G chains share `kernel_type` (the caller groups by kernel type before
    running this). Each chain has its own (log_lengthscale [d], std, noise_var)
    drawn under `pyro.plate("bt", G)`, so a single MCMC produces G independent
    posteriors at the cost of one batched [G, N, N] Cholesky per leapfrog step.
    """
    G, N, d = x.shape
    dtype = x.dtype
    device = x.device

    with pyro.plate("bt", G):
        log_ls = pyro.sample(
            "log_lengthscale",
            dist.Normal(
                torch.full((d,), mu_log_ls, dtype=dtype, device=device),
                torch.full((d,), sigma_log_ls, dtype=dtype, device=device),
            ).to_event(1),
        )                                                # [G, d]
        std = pyro.sample(
            "std",
            dist.Uniform(
                torch.tensor(std_range[0], dtype=dtype, device=device),
                torch.tensor(std_range[1], dtype=dtype, device=device),
            ),
        )                                                # [G]
        # Weak prior on noise variance; default GaussianLikelihood initializes
        # noise around 1e-2..1e-1, this prior covers that.
        noise_var = pyro.sample(
            "noise_var",
            dist.LogNormal(
                torch.tensor(-6.0, dtype=dtype, device=device),
                torch.tensor(1.0, dtype=dtype, device=device),
            ),
        )                                                # [G]

        # Match data/sampler.py::_sample_lengthscale's sqrt(d) scaling.
        lengthscale = log_ls.exp() * math.sqrt(d)        # [G, d]
        ls_b = lengthscale.view(G, 1, 1, d)
        K = _kernel_correlation_matrix(x, x, kernel_type, ls_b)  # [G, N, N]
        K = K * std.pow(2).view(G, 1, 1)
        K = K + (noise_var.view(G, 1, 1) + jitter) * torch.eye(
            N, dtype=dtype, device=device
        )

        pyro.sample(
            "y_obs",
            dist.MultivariateNormal(
                torch.zeros(N, dtype=dtype, device=device).expand(G, N),
                covariance_matrix=K,
            ),
            obs=y,
        )


def _run_nuts_batched(
    x: Tensor,                      # [G, N, d]
    y: Tensor,                      # [G, N]
    kernel_type: str,
    std_range: Tuple[float, float],
    num_samples: int,
    warmup_steps: int,
    jitter: float,
    mu_log_ls: float = _MU_LOG_LS,
    sigma_log_ls: float = _SIGMA_LOG_LS,
) -> Dict[str, Tensor]:
    """One MCMC over G chains. Returns thetas with leading [num_samples, G, ...].

    `kernel_type` is captured by the JIT trace, so this must be called once per
    kernel type group (the caller arranges that).
    """
    global _NUTS_JIT_OK

    def _build_and_run(jit_compile: bool):
        pyro.clear_param_store()
        nuts_kernel = NUTS(
            _gp_marginal_likelihood_model_batched,
            jit_compile=jit_compile,
            full_mass=False,
        )
        mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            num_chains=1,
            disable_progbar=True,
        )
        mcmc.run(x, y, kernel_type, std_range, jitter, mu_log_ls, sigma_log_ls)
        return mcmc, nuts_kernel

    if _NUTS_JIT_OK is False:
        mcmc, nuts_kernel = _build_and_run(jit_compile=False)
    else:
        try:
            mcmc, nuts_kernel = _build_and_run(jit_compile=True)
            _NUTS_JIT_OK = True
        except RuntimeError as e:
            if not _is_jit_compile_error(e):
                raise
            print(
                f"[gp_nuts_sampler] jit_compile=True failed ({e.__class__.__name__}); "
                "falling back to jit_compile=False for the rest of this process."
            )
            _NUTS_JIT_OK = False
            if x.is_cuda:
                torch.cuda.empty_cache()
            mcmc, nuts_kernel = _build_and_run(jit_compile=False)

    # Clone samples so they survive the MCMC teardown below (sample tensors
    # may otherwise alias MCMC internal buffers).
    samples = {k: v.clone() for k, v in mcmc.get_samples().items()}

    # Free MCMC state and JIT trace cache before the next group's run.
    del mcmc, nuts_kernel
    pyro.clear_param_store()
    if x.is_cuda:
        torch.cuda.empty_cache()

    return samples


def _batched_cholesky_with_retry(
    A: Tensor, jitter: float, max_tries: int
) -> Tensor:
    """Cholesky on [..., N, N] with geometrically-growing jitter on retries.

    Mirrors gpytorch's `cholesky_max_tries`: if Cholesky fails (any sub-batch
    indefinite), add `jitter * 10**i * I` and retry up to `max_tries` times.
    """
    N = A.shape[-1]
    eye = torch.eye(N, dtype=A.dtype, device=A.device)
    last_err = None
    for i in range(max(max_tries, 1)):
        try:
            return torch.linalg.cholesky(A + (jitter * 10.0**i) * eye)
        except (torch.linalg.LinAlgError, RuntimeError) as e:
            last_err = e
    raise last_err


def _draw_functions_batched(
    x_ctx: Tensor,                  # [B, N, d]
    y_ctx_t: Tensor,                # [B, N]
    x_test: Tensor,                 # [T, d]
    kernel_type: str,
    log_lengthscale: Tensor,        # [B, d]
    std: Tensor,                    # [B]
    noise_var: Tensor,              # [B]
    jitter: float,
    max_tries: int,
) -> Tensor:
    """Posterior function draws at x_test, batched over B (context, theta) pairs.

    Caller is expected to flatten `(group_size × num_draws)` into B; both the
    context (x_ctx, y_ctx_t) and the hyperparams (log_lengthscale, std,
    noise_var) vary along this leading dim. K_NN, K_TN, K_TT are computed in
    one shot of shape `[B, *, *]`.

    Returns: [B, T] latent function samples (no observation noise).
    """
    B, N, d = x_ctx.shape
    T = x_test.shape[0]
    dtype = x_ctx.dtype
    device = x_ctx.device

    lengthscale = log_lengthscale.exp() * math.sqrt(d)        # [B, d]
    ls_b = lengthscale.view(B, 1, 1, d)                       # broadcast against [N1, N2, d]
    outputscale = std.pow(2).view(B, 1, 1)
    nv = noise_var.clamp_min(1e-8).view(B, 1, 1)

    K_NN = _kernel_correlation_matrix(x_ctx, x_ctx, kernel_type, ls_b) * outputscale
    K_TN = _kernel_correlation_matrix(x_test, x_ctx, kernel_type, ls_b) * outputscale
    K_TT = _kernel_correlation_matrix(x_test, x_test, kernel_type, ls_b) * outputscale

    eye_N = torch.eye(N, dtype=dtype, device=device)
    K_NN = K_NN + nv * eye_N                                  # noise inside; jitter via retry

    L = _batched_cholesky_with_retry(K_NN, jitter, max_tries)  # [B, N, N]

    # Posterior mean: μ = K_TN @ K_NN^{-1} @ y
    y = y_ctx_t.unsqueeze(-1)                                  # [B, N, 1]
    alpha = torch.cholesky_solve(y, L)                         # [B, N, 1]
    mu = torch.bmm(K_TN, alpha).squeeze(-1)                    # [B, T]

    # Posterior cov: Σ = K_TT - K_TN K_NN^{-1} K_NT
    K_NT = K_TN.transpose(-1, -2)                              # [B, N, T]
    v = torch.linalg.solve_triangular(L, K_NT, upper=False)    # [B, N, T]
    Sigma = K_TT - torch.bmm(v.transpose(-1, -2), v)           # [B, T, T]
    # Symmetrize against round-off before Cholesky.
    Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2))

    L_T = _batched_cholesky_with_retry(Sigma, jitter, max_tries)  # [B, T, T]
    eps = torch.randn(B, T, 1, dtype=dtype, device=device)
    f = mu + torch.bmm(L_T, eps).squeeze(-1)                   # [B, T]
    return f


def multi_output_gp_posterior_sampler_nuts(
    x_ctx: Tensor,                  # [B, N, d]
    y_ctx: Tensor,                  # [B, N, num_tasks]
    x_test: Tensor,                 # [T, d]
    num_draws: int,
    kernel_types: List[str] = DATA_KERNEL_TYPE_LIST,
    sample_kernel_weights: Optional[List[float]] = None,
    std_range: Tuple[float, float] = STD_RANGE,
    nuts_warmup: int = 128,
    nuts_thinning: int = 1,
    jitter: float = JITTER,
    max_tries: int = MAX_TRIES,
    device: str = "cuda",
) -> Tensor:
    """Vectorized fully-Bayesian GP posterior sampler over (B, num_tasks).

    Pipeline:
        1. Sample one kernel_type per (b, task), index `g = b * num_tasks + t`.
        2. Group the G = B*num_tasks chains by kernel_type. Each group becomes
           one batched MCMC run (`_run_nuts_batched`) with `pyro.plate("bt", Gk)`,
           replacing what used to be Gk separate NUTS calls.
        3. For each group, draw `num_draws` posterior functions at x_test using
           one batched compute over (Gk × num_draws) flattened.

    Args:
        x_ctx: [B, N, d] context inputs (one set per batch element).
        y_ctx: [B, N, num_tasks] context outputs.
        x_test: [T, d] test inputs (shared across all (b, task)).
        num_draws: posterior draws per (b, task).
        kernel_types, sample_kernel_weights: kernel type prior.
        std_range: prior bounds on outputscale-std.
        nuts_warmup: NUTS warmup steps.
        nuts_thinning: keep every Nth sample (use >1 if mixing is poor).

    Returns:
        y of shape [B, num_draws, T, num_tasks].
    """
    if sample_kernel_weights is None:
        sample_kernel_weights = [1.0] * len(kernel_types)

    # Use float64 for NUTS numerical stability; callers cast back if needed.
    work_dtype = torch.float64
    x_ctx = x_ctx.to(device=device, dtype=work_dtype)
    y_ctx = y_ctx.to(device=device, dtype=work_dtype)
    x_test = x_test.to(device=device, dtype=work_dtype)

    B, N, d = x_ctx.shape
    M = y_ctx.shape[-1]
    T = x_test.shape[0]
    G = B * M

    # Sample kernel_type per (b, t). g = b * M + t.
    kt_per_g = random.choices(kernel_types, weights=sample_kernel_weights, k=G)

    # Flatten contexts to (b, t) order matching the indexing above.
    # x_ctx shared across t -> repeat for each task.
    x_ctx_all = (
        x_ctx.unsqueeze(1).expand(B, M, N, d).reshape(G, N, d).contiguous()
    )
    # y_ctx[b, :, t] -> y_ctx_all[b * M + t]
    y_ctx_all = y_ctx.permute(0, 2, 1).reshape(G, N).contiguous()

    nuts_total = num_draws * nuts_thinning
    out = torch.empty(G, num_draws, T, dtype=work_dtype, device=device)

    # One batched MCMC per kernel type. Worst case: |kernel_types| runs.
    for kt in dict.fromkeys(kt_per_g):  # dedupe, preserve first-seen order
        idx_list = [g for g, k in enumerate(kt_per_g) if k == kt]
        idx = torch.tensor(idx_list, dtype=torch.long, device=device)
        Gk = idx.numel()

        x_g = x_ctx_all.index_select(0, idx)  # [Gk, N, d]
        y_g = y_ctx_all.index_select(0, idx)  # [Gk, N]

        thetas = _run_nuts_batched(
            x=x_g,
            y=y_g,
            kernel_type=kt,
            std_range=tuple(std_range),
            num_samples=nuts_total,
            warmup_steps=nuts_warmup,
            jitter=jitter,
        )
        log_ls = thetas["log_lengthscale"]   # [nuts_total, Gk, d]
        std_s = thetas["std"]                 # [nuts_total, Gk]
        nv_s = thetas["noise_var"]            # [nuts_total, Gk]

        if nuts_thinning > 1:
            log_ls = log_ls[::nuts_thinning][:num_draws]
            std_s = std_s[::nuts_thinning][:num_draws]
            nv_s = nv_s[::nuts_thinning][:num_draws]
        # Now [num_draws, Gk, ...].

        # Batched draw: flatten (num_draws, Gk) -> S = num_draws * Gk along leading dim.
        # Each S-slot pairs one (x_g[k], y_g[k]) context with one theta.
        S = num_draws * Gk
        x_flat = (
            x_g.unsqueeze(0).expand(num_draws, Gk, N, d).reshape(S, N, d).contiguous()
        )
        y_flat = (
            y_g.unsqueeze(0).expand(num_draws, Gk, N).reshape(S, N).contiguous()
        )
        log_ls_flat = log_ls.reshape(S, d)
        std_flat = std_s.reshape(S)
        nv_flat = nv_s.reshape(S)

        f = _draw_functions_batched(
            x_ctx=x_flat,
            y_ctx_t=y_flat,
            x_test=x_test,
            kernel_type=kt,
            log_lengthscale=log_ls_flat,
            std=std_flat,
            noise_var=nv_flat,
            jitter=jitter,
            max_tries=max_tries,
        )  # [S, T]
        # Reshape back to [Gk, num_draws, T] (note the flatten order).
        f = f.reshape(num_draws, Gk, T).transpose(0, 1).contiguous()

        out.index_copy_(0, idx, f)

    # [G, num_draws, T] = [B*M, num_draws, T] -> [B, num_draws, T, M]
    return out.reshape(B, M, num_draws, T).permute(0, 2, 3, 1).contiguous()


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(0)
    random.seed(0)

    device = "cpu"
    B, N, d, T, M = 2, 16, 2, 32, 2
    x_ctx = torch.rand(B, N, d, device=device)
    # Make y_ctx data-like: sample from a simple lengthscale-1 RBF prior, per batch.
    diff = x_ctx.unsqueeze(2) - x_ctx.unsqueeze(1)         # [B, N, N, d]
    K = torch.exp(-0.5 * diff.pow(2).sum(-1)) + 1e-6 * torch.eye(N)  # [B, N, N]
    L = torch.linalg.cholesky(K)
    y_ctx = torch.bmm(L, torch.randn(B, N, M)).to(torch.float64)  # [B, N, M]
    x_test = torch.rand(T, d, device=device)

    print("Running NUTS sampler smoke test...")
    y = multi_output_gp_posterior_sampler_nuts(
        x_ctx=x_ctx,
        y_ctx=y_ctx,
        x_test=x_test,
        num_draws=4,
        kernel_types=["rbf", "matern32", "matern52"],
        sample_kernel_weights=[1.0, 1.0, 1.0],
        nuts_warmup=64,
        device=device,
    )
    print(f"  y shape: {tuple(y.shape)} (expected ({B}, {4}, {T}, {M}))")
    assert y.shape == (B, 4, T, M)
    assert torch.isfinite(y).all()
    # Diversity across draws within a batch element.
    diff_draws = (y[0, 0] - y[0, 1]).abs().mean().item()
    print(f"  mean abs diff between draws 0 and 1 (batch 0): {diff_draws:.4f}")
    assert diff_draws > 1e-3, "Draws are too similar — diversity check failed"
    # Diversity across batch elements (different x_ctx, y_ctx).
    diff_batch = (y[0, 0] - y[1, 0]).abs().mean().item()
    print(f"  mean abs diff between batch 0 and batch 1 (draw 0): {diff_batch:.4f}")
    assert diff_batch > 1e-3, "Batches are too similar — batched-NUTS check failed"
    print("  PASSED")
