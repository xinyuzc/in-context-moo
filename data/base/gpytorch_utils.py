import torch
from torch import Tensor
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.means import ZeroMean
from gpytorch.models import ExactGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal

from typing import Optional


class MultitaskGPModel(ExactGP):
    """Multi-task GP model with output correlations."""

    def __init__(
        self,
        likelihood: MultitaskGaussianLikelihood,
        kernel: Kernel,
        num_tasks: int,
        rank: int,
        train_x: Optional[Tensor] = None,  # [num_datapoints, x_dim]
        train_y: Optional[Tensor] = None,  # [num_datapoints, y_dim]
    ):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(ZeroMean(), num_tasks=num_tasks)

        # `rank=1`: highly correlated
        # `rank=num_tasks`: highly independent
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            data_covar_module=kernel, num_tasks=num_tasks, rank=rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class ExactGPModel(ExactGP):
    """GP model with a single output."""

    def __init__(
        self,
        kernel: Kernel,
        likelihood: GaussianLikelihood,
        train_x: Optional[Tensor] = None,  # [num_datapoints, x_dim]
        train_y: Optional[Tensor] = None,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class RotatedARDKernel(gpytorch.kernels.Kernel):
    def __init__(self, num_dims, batch_shape=torch.Size([]), **kwargs):
        super().__init__(batch_shape=batch_shape, **kwargs)
        initial_R = torch.eye(num_dims)
        self.register_parameter(name="R", parameter=torch.nn.Parameter(initial_R))
        self.register_parameter(
            name="raw_lengthscales",
            parameter=torch.nn.Parameter(
                torch.zeros(batch_shape + torch.Size([num_dims]))
            ),
        )

    def forward(self, x1, x2, diag=False, **params):
        U, S, V = torch.svd(self.R)
        R_ortho = U @ V.t()
        lengthscales = self.raw_lengthscales.exp()
        x1_rotated = torch.matmul(x1, R_ortho)
        x2_rotated = torch.matmul(x2, R_ortho)

        if diag:
            dist_sq = ((x1_rotated - x2_rotated) / lengthscales).pow(2).sum(dim=-1)
        else:
            dist_sq = torch.sum(
                (x1_rotated.unsqueeze(1) - x2_rotated.unsqueeze(0)).pow(2)
                / lengthscales.pow(2),
                dim=-1,
            )

        return torch.exp(-0.5 * dist_sq)


def sample_orthonormal_matrix(d):
    A = torch.randn(d, d)
    Q, _ = torch.linalg.qr(A)
    return Q
