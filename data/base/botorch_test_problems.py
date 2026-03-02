"""BoTorch-based test problems."""

from typing import Optional, List, Dict
from collections import OrderedDict

from botorch.test_functions import SyntheticTestFunction
from botorch.test_functions.synthetic import Ackley, Rosenbrock, Rastrigin
from botorch.test_functions.base import BaseTestProblem, ABC, MultiObjectiveTestProblem
from botorch.utils.torch import BufferDict
from botorch.exceptions.errors import InputDataError
from torch import Tensor
import torch

from data.HPO_3DGS.functions import (
    set_NERF_scene,
    NERF_synthetic,
    NERF_synthetic_fnum_3,
)


class Forrester(SyntheticTestFunction):
    dim = 1
    _bounds = [(0.0, 1.0)]
    _optimal_value = -6.020740
    _optimizers = [(0.75724876,)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        return ((6 * X - 2) ** 2) * torch.sin(12 * X - 4)


class AckleyRastrigin(BaseTestProblem, ABC):
    dim = 2
    num_objectives = 2
    _bounds = [(0.0, 1.0), (0.0, 1.0)]
    _ref_point = None
    _max_hv = None

    def __init__(
        self,
        noise_std=None,
        negate=False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        if isinstance(noise_std, list) and len(noise_std) != len(self._ref_point):
            raise InputDataError(
                f"If specified as a list, length of noise_std ({len(noise_std)}) "
                f"must match the number of objectives ({len(self._ref_point)})"
            )
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)
        if self._ref_point is not None:
            ref_point = torch.tensor(self._ref_point, dtype=dtype)
            if negate:
                ref_point *= -1
            self.register_buffer("ref_point", ref_point)

        self._ackley = Ackley(
            dim=self.dim, negate=negate, noise_std=noise_std, dtype=dtype
        )
        self._rastrigin = Rastrigin(
            dim=self.dim, negate=negate, noise_std=noise_std, dtype=dtype
        )

    def _rescaled_ackley(self, X: Tensor) -> Tensor:
        # return to Ackley bounds
        X = X * 65.536 - 32.768
        return self._ackley(X)

    def _rescaled_rastrigin(self, X: Tensor) -> Tensor:
        # return to Rastrigin bounds
        X = X * 10.24 - 5.12
        return self._rastrigin(X)

    def evaluate_true(self, X: Tensor) -> Tensor:
        ackley = self._rescaled_ackley(X=X)
        rastrigin = self._rescaled_rastrigin(X=X)
        return torch.stack([ackley, rastrigin], dim=-1)

    @property
    def max_hv(self) -> float:
        if self._max_hv is not None:
            return self._max_hv
        else:
            raise NotImplementedError(
                f"Problem {self.__class__.__name__} does not specify maximal "
                "hypervolume."
            )

    def gen_pareto_front(self, n: int) -> Tensor:
        r"""Generate `n` pareto optimal points."""
        raise NotImplementedError


class AckleyRosenbrock(BaseTestProblem, ABC):
    dim = 2
    num_objectives = 2
    _bounds = [(0.0, 1.0), (0.0, 1.0)]
    _ref_point = None
    _max_hv = None

    def __init__(
        self,
        noise_std=None,
        negate=False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        if isinstance(noise_std, list) and len(noise_std) != len(self._ref_point):
            raise InputDataError(
                f"If specified as a list, length of noise_std ({len(noise_std)}) "
                f"must match the number of objectives ({len(self._ref_point)})"
            )
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)
        if self._ref_point is not None:
            ref_point = torch.tensor(self._ref_point, dtype=dtype)
            if negate:
                ref_point *= -1
            self.register_buffer("ref_point", ref_point)

        self._ackley = Ackley(
            dim=self.dim, negate=negate, noise_std=noise_std, dtype=dtype
        )
        self._rosenbrock = Rosenbrock(
            dim=self.dim, negate=negate, noise_std=noise_std, dtype=dtype
        )

    def _rescaled_ackley(self, X: Tensor) -> Tensor:
        # return to Ackley bounds
        X = X * 65.536 - 32.768
        return self._ackley(X)

    def _rescaled_rosenbrock(self, X: Tensor) -> Tensor:
        # return to Rosenbrock bounds
        # X = X * 15.0 - 5.0
        X = X * 4.096 - 2.048
        return self._rosenbrock(X)

    def evaluate_true(self, X: Tensor) -> Tensor:
        ackley = self._rescaled_ackley(X=X)
        rosenbrock = self._rescaled_rosenbrock(X=X)
        return torch.stack([ackley, rosenbrock], dim=-1)

    @property
    def max_hv(self) -> float:
        if self._max_hv is not None:
            return self._max_hv
        else:
            raise NotImplementedError(
                f"Problem {self.__class__.__name__} does not specify maximal "
                "hypervolume."
            )

    def gen_pareto_front(self, n: int) -> Tensor:
        r"""Generate `n` pareto optimal points."""
        raise NotImplementedError


class DiscreteTestProblem(BaseTestProblem):
    """
    References: https://github.com/facebookresearch/bo_pr/blob/main/discrete_mixed_bo/problems/base.py
    """

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        integer_indices: Optional[List[int]] = None,
        categorical_indices: Optional[List[int]] = None,
    ) -> None:
        super().__init__(negate=negate, noise_std=noise_std)
        self._setup(
            integer_indices=integer_indices, categorical_indices=categorical_indices
        )

    def _setup(
        self,
        integer_indices: Optional[List[int]] = None,
        categorical_indices: Optional[List[int]] = None,
    ) -> None:
        dim = self.bounds.shape[-1]
        discrete_indices = []
        if integer_indices is None:
            integer_indices = []
        if categorical_indices is None:
            categorical_indices = []
        self.register_buffer(
            "_orig_integer_indices", torch.tensor(integer_indices, dtype=torch.long)
        )
        discrete_indices.extend(integer_indices)
        self.register_buffer(
            "_orig_categorical_indices",
            torch.tensor(sorted(categorical_indices), dtype=torch.long),
        )
        discrete_indices.extend(categorical_indices)
        if len(discrete_indices) == 0:
            raise ValueError("Expected at least one discrete feature.")
        cont_indices = sorted(list(set(range(dim)) - set(discrete_indices)))
        self.register_buffer(
            "_orig_cont_indices",
            torch.tensor(
                cont_indices,
                dtype=torch.long,
                device=self.bounds.device,
            ),
        )
        self.register_buffer("_orig_bounds", self.bounds.clone())
        # remap inputs so that categorical features come after all of
        # the ordinal features
        remapper = torch.zeros(
            self.bounds.shape[-1], dtype=torch.long, device=self.bounds.device
        )
        reverse_mapper = remapper.clone()
        for i, orig_idx in enumerate(
            cont_indices + integer_indices + categorical_indices
        ):
            remapper[i] = orig_idx
            reverse_mapper[orig_idx] = i
        self.register_buffer("_remapper", remapper)
        self.register_buffer("_reverse_mapper", reverse_mapper)
        self.bounds = self.bounds[:, remapper]
        self.register_buffer("cont_indices", reverse_mapper[cont_indices])
        self.register_buffer("integer_indices", reverse_mapper[integer_indices])
        self.register_buffer("categorical_indices", reverse_mapper[categorical_indices])

        self.effective_dim = (
            self.cont_indices.shape[0]
            + self.integer_indices.shape[0]
            + int(sum(self.categorical_features.values()))
        )

        one_hot_bounds = torch.zeros(
            2, self.effective_dim, dtype=self.bounds.dtype, device=self.bounds.device
        )
        one_hot_bounds[1] = 1
        one_hot_bounds[:, self.integer_indices] = self.integer_bounds
        one_hot_bounds[:, self.cont_indices] = self.cont_bounds
        self.register_buffer("one_hot_bounds", one_hot_bounds)

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate the function on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                function.
            noise: If `True`, add observation noise as specified by `noise_std`.

        Returns:
            A `batch_shape`-dim tensor of function evaluations.
        """
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        # remap to original space
        X = X[..., self._reverse_mapper]
        f = self.evaluate_true(X=X)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        if self.negate:
            f = -f
        return f if batch else f.squeeze(0)

    def evaluate_slack(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate the constraint function on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                function.
            noise: If `True`, add observation noise as specified by `noise_std`.

        Returns:
            A `batch_shape x n_constraints`-dim tensor of function evaluations.
        """
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        # remap to original space
        X = X[..., self._reverse_mapper]
        f = self.evaluate_slack_true(X=X)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        return f if batch else f.squeeze(0)

    @property
    def integer_bounds(self) -> Optional[Tensor]:
        if self.integer_indices is not None:
            return self.bounds[:, self.integer_indices]
        return None

    @property
    def cont_bounds(self) -> Optional[Tensor]:
        if self.cont_indices is not None:
            return self.bounds[:, self.cont_indices]
        return None

    @property
    def categorical_bounds(self) -> Optional[Tensor]:
        if self.categorical_indices is not None:
            return self.bounds[:, self.categorical_indices]
        return None

    @property
    def categorical_features(self) -> Optional[Dict[int, int]]:
        # Return dictionary mapping indices to cardinalities
        if self.categorical_indices is not None:
            categ_bounds = self.categorical_bounds
            return OrderedDict(
                zip(
                    self.categorical_indices.tolist(),
                    (categ_bounds[1] - categ_bounds[0] + 1).long().tolist(),
                )
            )
        return None

    @property
    def objective_weights(self) -> Optional[Tensor]:
        return None

    @property
    def is_moo(self) -> bool:
        return isinstance(self, MultiObjectiveTestProblem) and (
            self.objective_weights is None
        )


class OilSorbentContinuousMid(BaseTestProblem, ABC):
    """2-dimensional continuous test problem based on the Oil Sorbent HPO benchmark."""

    _bounds = [
        (0.0, 1.0),  # V1: continuous
        (0.0, 1.0),  # V5: continuous
    ]
    dim = 2
    num_objectives = 3
    _ref_point = None

    _discrete_values = {
        "V2": [0.7, 1, 1.4, 1.7, 2],
        "V3": [12, 15, 18, 21, 24],
        "V4": [0.12, 0.135, 0.15, 0.165, 0.18],
        "V6": [16, 20, 26, 28],
        "V7": [0.41, 0.6, 0.84, 1.32],
    }

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        if isinstance(noise_std, list) and len(noise_std) != len(self._ref_point):
            raise InputDataError(
                f"If specified as a list, length of noise_std ({len(noise_std)}) "
                f"must match the number of objectives ({len(self._ref_point)})"
            )
        super().__init__(negate=negate, noise_std=noise_std, dtype=dtype)
        if self._ref_point is not None:
            ref_point = torch.tensor(self._ref_point, dtype=dtype)
            if negate:
                ref_point *= -1
            self.register_buffer("ref_point", ref_point)

        self.discrete_values = BufferDict()
        for k, v in self._discrete_values.items():
            # 1. turn into tensor, 2. normalize to [0, 1]
            self.discrete_values[k] = torch.tensor(v, dtype=torch.float)
            self.discrete_values[k] /= self.discrete_values[k].max()

    def evaluate_true(self, X: Tensor) -> Tensor:
        # The input X now has a shape of (..., 2)
        V1, V5 = torch.split(X, 1, -1)

        drop_features = {"V2": None, "V3": None, "V4": None, "V6": None, "V7": None}
        discrete_values = self.discrete_values.to(X.device)
        for drop_name in drop_features.keys():
            if drop_name in self._discrete_values:
                # Always select the first value
                fixed_idx = torch.zeros_like(V1).view(-1).long()
                fixed_val = discrete_values[drop_name][fixed_idx].view(V1.shape)
                drop_features[drop_name] = fixed_val

        wca = (
            -197.0928
            - 78.3309 * V1
            + 98.6355 * drop_features["V2"]
            + 300.0701 * drop_features["V3"]
            + 89.8360 * drop_features["V4"]
            + 208.2343 * V5
            + 332.9341 * drop_features["V6"]
            + 135.6621 * drop_features["V7"]
            - 11.0715 * V1 * drop_features["V2"]
            + 201.8934 * V1 * drop_features["V3"]
            + 17.1270 * V1 * drop_features["V4"]
            + 2.5198 * V1 * V5
            - 109.3922 * V1 * drop_features["V6"]
            + 30.1607 * V1 * drop_features["V7"]
            - 46.1790 * drop_features["V2"] * drop_features["V3"]
            + 19.2888 * drop_features["V2"] * drop_features["V4"]
            - 102.9493 * drop_features["V2"] * V5
            - 19.1245 * drop_features["V2"] * drop_features["V6"]
            + 53.6297 * drop_features["V2"] * drop_features["V7"]
            - 73.0649 * drop_features["V3"] * drop_features["V4"]
            - 37.7181 * drop_features["V3"] * V5
            - 219.1268 * drop_features["V3"] * drop_features["V6"]
            - 55.3704 * drop_features["V3"] * drop_features["V7"]
            + 3.8778 * drop_features["V4"] * V5
            - 6.9252 * drop_features["V4"] * drop_features["V6"]
            - 105.1650 * drop_features["V4"] * drop_features["V7"]
            - 34.3181 * V5 * drop_features["V6"]
            - 36.3892 * V5 * drop_features["V7"]
            - 82.3222 * drop_features["V6"] * drop_features["V7"]
            - 16.7536 * V1.pow(2)
            - 45.6507 * drop_features["V2"].pow(2)
            - 91.4134 * drop_features["V3"].pow(2)
            - 76.8701 * V5.pow(2)
        )
        q = (
            -212.8531
            + 245.7998 * V1
            - 127.3395 * drop_features["V2"]
            + 305.8461 * drop_features["V3"]
            + 638.1605 * drop_features["V4"]
            + 301.2118 * V5
            - 451.3796 * drop_features["V6"]
            - 115.5485 * drop_features["V7"]
            + 42.8351 * V1 * drop_features["V2"]
            + 262.3775 * V1 * drop_features["V3"]
            - 103.5274 * V1 * drop_features["V4"]
            - 196.1568 * V1 * V5
            - 394.7975 * V1 * drop_features["V6"]
            - 176.3341 * V1 * drop_features["V7"]
            + 74.8291 * drop_features["V2"] * drop_features["V3"]
            + 4.1557 * drop_features["V2"] * drop_features["V4"]
            - 133.8683 * drop_features["V2"] * V5
            + 65.8711 * drop_features["V2"] * drop_features["V6"]
            - 42.6911 * drop_features["V2"] * drop_features["V7"]
            - 323.9363 * drop_features["V3"] * drop_features["V4"]
            - 107.3983 * drop_features["V3"] * V5
            - 323.2353 * drop_features["V3"] * drop_features["V6"]
            + 46.9172 * drop_features["V3"] * drop_features["V7"]
            - 144.4199 * drop_features["V4"] * V5
            + 272.3729 * drop_features["V4"] * drop_features["V6"]
            + 49.0799 * drop_features["V4"] * drop_features["V7"]
            + 318.4706 * V5 * drop_features["V6"]
            - 236.2498 * V5 * drop_features["V7"]
            + 252.4848 * drop_features["V6"] * drop_features["V7"]
            - 286.0182 * drop_features["V4"].pow(2)
            + 393.5992 * drop_features["V6"].pow(2)
        )
        sigma = (
            7.7696
            + 15.4344 * V1
            - 10.6190 * drop_features["V2"]
            - 17.9367 * drop_features["V3"]
            + 17.1385 * drop_features["V4"]
            + 2.5026 * V5
            - 24.3010 * drop_features["V6"]
            + 10.6058 * drop_features["V7"]
            - 1.2041 * V1 * drop_features["V2"]
            - 37.2207 * V1 * drop_features["V3"]
            - 3.2265 * V1 * drop_features["V4"]
            + 7.3121 * V1 * V5
            + 52.3994 * V1 * drop_features["V6"]
            + 9.7485 * V1 * drop_features["V7"]
            - 15.9371 * drop_features["V2"] * drop_features["V3"]
            - 1.1706 * drop_features["V2"] * drop_features["V4"]
            - 2.6297 * drop_features["V2"] * V5
            + 7.0225 * drop_features["V2"] * drop_features["V6"]
            - 1.4938 * drop_features["V2"] * drop_features["V7"]
            + 30.2786 * drop_features["V3"] * drop_features["V4"]
            + 14.5061 * drop_features["V3"] * V5
            + 48.5021 * drop_features["V3"] * drop_features["V6"]
            - 11.4857 * drop_features["V3"] * drop_features["V7"]
            - 3.1381 * drop_features["V4"] * V5
            - 14.9747 * drop_features["V4"] * drop_features["V6"]
            + 4.5204 * drop_features["V4"] * drop_features["V7"]
            - 17.6907 * V5 * drop_features["V6"]
            - 19.2489 * V5 * drop_features["V7"]
            - 9.8219 * drop_features["V6"] * drop_features["V7"]
            - 18.7356 * V1.pow(2)
            + 12.1928 * drop_features["V2"].pow(2)
            - 17.5460 * drop_features["V4"].pow(2)
            + 5.4997 * V5.pow(2)
            - 26.2718 * drop_features["V6"].pow(2)
        )
        return -torch.cat([wca, q, sigma], dim=-1)


class HPO_3DGS(BaseTestProblem, ABC):
    """HPO_3DGS benchmark problem for 2 or 3 objectives."""

    dim = 5
    _bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    num_objectives = 2
    _ref_point = None
    _max_hv = None

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = True,  # default minimization
        dtype: torch.dtype = torch.double,
        scene: str = "ship",
        num_objectives: int = 2,
    ):
        if isinstance(noise_std, list) and len(noise_std) != len(self._ref_point):
            raise InputDataError(
                f"If specified as a list, length of noise_std ({len(noise_std)}) "
                f"must match the number of objectives ({len(self._ref_point)})"
            )
        super().__init__(negate=negate, noise_std=noise_std, dtype=dtype)
        if self._ref_point is not None:
            ref_point = torch.tensor(self._ref_point, dtype=dtype)
            if negate:
                ref_point *= -1
            self.register_buffer("ref_point", ref_point)

        assert scene in ["lego", "materials", "mic", "ship"]
        set_NERF_scene(scene)

        if num_objectives == 2:
            self.nerf_synthetic = NERF_synthetic
        elif num_objectives == 3:
            self.nerf_synthetic = NERF_synthetic_fnum_3
        else:
            raise ValueError("num_objectives must be 2 or 3")

        self.scene = scene
        self.num_objectives = num_objectives

    def evaluate_true(self, X: Tensor) -> Tensor:
        """Evaluate the function on a set of points.

        Args:
            X: Tensor of shape (*batch_shape, d) in [0, 1]^d

        Returns:
            y: Tensor of shape (*batch_shape, 2) or (*batch_shape, 3) corresponding to objectives
        """
        X_np = X.cpu().numpy()
        y_np = self.nerf_synthetic(X_np)
        y = torch.from_numpy(y_np).to(X.device).to(X.dtype)
        return y


if __name__ == "__main__":
    hpo_3dgs = HPO_3DGS(scene="ship", num_objectives=3)
    X = torch.rand(4, 5)
    y = hpo_3dgs.evaluate_true(X)
    print(X)
    print(y)
