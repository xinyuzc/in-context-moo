"""Create environment for multi-objective minimization based on:
- Botorch function, or
- Linear interpolation of dataset

Supports:
    - Evaluate at input points
    - Sampling from functions
    - Hypervolume computation
    - Regret computation

Classes:
    - TestFunction: Base class
    - BenchmarkFunction: built from botorch synthetic function
    - IntepolatorFunction: built by interpolating at datapoints
"""

from typing import List, Dict, Tuple, Optional

from einops import repeat
import torch
from torch import Tensor
import numpy as np
from botorch.test_functions import (
    Branin,
    BraninCurrin,
    EggHolder,
    Ackley,
    Rastrigin,
)
from data.base.botorch_test_problems import (
    Forrester,
    AckleyRosenbrock,
    AckleyRastrigin,
    OilSorbentContinuousMid,
    HPO_3DGS,
)
from data.function_sampling import sample_domain
from data.base.preprocessing import (
    tuple_list_to_nested_list,
    make_range_nested_list,
    make_range_tensor,
    transform,
    estimate_objective_bounds,
    NPInterpolatorExtTensor,
)
from data.base.masking import compact_by_mask, restore_by_mask
from data.moo import MOO
from utils.types import FloatListOrNestedOrTensor, NestedFloatList, FloatListOrNested

# Supported benchmark with y bounds (loose)
MO_BENCHMARK = {
    "BraninCurrin": BraninCurrin,
    "AckleyRosenbrock": AckleyRosenbrock,
    "AckleyRastrigin": AckleyRastrigin,
}
MO_REALWORLD = {
    "OilSorbentContinuousMid": OilSorbentContinuousMid,
    "NERF_synthetic": HPO_3DGS,
    "NERF_synthetic_fnum_3": HPO_3DGS,
}
SO_BENCHMARK = {
    "Branin": Branin,
    "Forrester": Forrester,
    "EggHolder": EggHolder,
    "Ackley": Ackley,
    "Rastrigin": Rastrigin,
}
SYN_FUNCTIONS = {**MO_BENCHMARK, **SO_BENCHMARK, **MO_REALWORLD}

SO_Y_BOUNDS = {
    "Forrester": [[-6.020740, 16.0]],
    "Branin": [[0.397887, 309.0]],
    "Currin": [[1.18, 14.0]],
    "Ackley": [[0.0, 23.0]],
    "Rosenbrock": [[0.0, 3907.0]],
    "Rastrigin": [[0.0, 81.0]],
    "EggHolder": [[-959.6407, 1050.0]],
}

MO_Y_BOUNDS = {
    "AckleyRastrigin": [SO_Y_BOUNDS["Ackley"][0], SO_Y_BOUNDS["Rastrigin"][0]],
    "AckleyRosenbrock": [SO_Y_BOUNDS["Ackley"][0], SO_Y_BOUNDS["Rosenbrock"][0]],
    "BraninCurrin": [SO_Y_BOUNDS["Branin"][0], SO_Y_BOUNDS["Currin"][0]],
}
RW_Y_BOUNDS = {
    "NERF_synthetic": [[-1.0, 0.0], [-1.0, 0.0]],
    "NERF_synthetic_fnum_3": [[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
}
BENCHMARK_Y_BOUNDS = {**SO_Y_BOUNDS, **MO_Y_BOUNDS, **RW_Y_BOUNDS}


class TestFunction:
    """Test function environment base class;
    `get_metadata()` must be implemented in subclasses.

    Attrs:
        function_name (str): Function name
        func (callable): Callable function instance
        x_dim (int): Dimension of the input space
        y_dim (int): Dimension of the output space
        x_bounds (Tensor): Input bounds in the truth domain, shape [dx, 2]
        y_bounds (Tensor): Output bounds in the truth domain, shape [dy, 2]
        ref_point (Tensor): Reference point in the truth domain, shape [dy]
        max_hv (float): Maximum hypervolume value from `ref_point` in the truth domain
    """

    func: callable = None
    x_dim: int = None
    y_dim: int = None
    x_bounds: Tensor = None
    y_bounds: Tensor = None
    ref_point: Tensor = None
    max_hv: float = None
    function_name: str = "undefined_function"

    def __init__(self, **kwargs):
        metadata = self.get_metadata(**kwargs)
        self.init_from_metadata(metadata)

    def get_metadata(self, **kwargs) -> Dict:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement `get_metadata()` method to return dictionary of metadata."
        )

    def init_from_metadata(self, metadata: Dict) -> List:
        """Initialize function environment from metadata."""

        def _get_required(key: str, default=None):
            val = metadata.get(key)
            if val is None:
                if default is None:
                    raise ValueError(f"Function metadata must contain '{key}'.")
                else:
                    val = default
            return val

        self.x_bounds = torch.tensor(_get_required("x_bounds"))
        self.y_bounds = torch.tensor(_get_required("y_bounds"))
        self.ref_point = torch.tensor(_get_required("ref_point"))
        self.func = _get_required("func")
        self.max_hv = _get_required("max_hv")
        self.function_name = _get_required("function_name", self.function_name)

        self.x_dim = len(self.x_bounds)
        self.y_dim = len(self.y_bounds)

    def transform_inputs(
        self,
        inputs: Tensor,
        input_bounds: FloatListOrNestedOrTensor,
        transform_method: str = "min_max",
    ) -> Tensor:
        """Scale inputs from its original domain (`input_bounds`) to function input domain (`x_bounds`).

        Args:
            inputs: [..., dx]
            input_bounds: [dx, 2] | [2]

        Returns:
            scaled_inputs: [..., dx]
        """
        num_dim = inputs.shape[-1]
        tkwargs = {"device": inputs.device, "dtype": inputs.dtype}

        inp_bounds = make_range_tensor(input_bounds, num_dim=num_dim).to(**tkwargs)
        out_bounds = self.x_bounds.to(**tkwargs)

        return transform(
            data=inputs,
            inp_bounds=inp_bounds,
            out_bounds=out_bounds,
            transform_method=transform_method,
        )

    def transform_outputs(
        self,
        outputs: Tensor,
        output_bounds: FloatListOrNestedOrTensor,
        transform_method: str = "min_max",
    ) -> Tensor:
        """Scale outputs from function output domain (`y_bounds`) to target domain (`output_bounds`).

        Args:
            outputs: [..., dy]
            output_bounds: [dy, 2] | [2]

        Returns:
            scaled_outputs [..., dy]
        """
        num_dim = outputs.shape[-1]
        tkwargs = {"device": outputs.device, "dtype": outputs.dtype}

        inp_bounds = self.y_bounds.to(**tkwargs)
        out_bounds = make_range_tensor(output_bounds, num_dim=num_dim).to(**tkwargs)

        return transform(
            data=outputs,
            inp_bounds=inp_bounds,
            out_bounds=out_bounds,
            transform_method=transform_method,
        )

    @staticmethod
    def get_ref_point(
        bounds: Tensor | NestedFloatList, candidates: Tensor = None
    ) -> List:
        """Get reference point [dy] given bounds and optional candidates."""
        if candidates is None:
            # Use upper bounds
            return [bounds[i][1] for i in range(len(bounds))]
        else:
            # Use maximum objective values in candidates
            assert candidates.shape[-1] == len(bounds)
            candidate_max = candidates.max(dim=0).values  # [dy]
            return candidate_max.tolist()

    @staticmethod
    def get_max_hv(
        ref_point: List, bounds: NestedFloatList, candidates: Tensor = None
    ) -> float:
        if candidates is None:
            # Use lower bounds
            return np.prod([ref_point[i] - bounds[i][0] for i in range(len(bounds))])
        else:
            # Use minimum objective values in candidates
            assert candidates.shape[-1] == len(bounds)
            candidates_min = candidates.min(dim=0).values  # [dy]
            return np.prod(
                (np.array(ref_point) - candidates_min.cpu().numpy()).clip(min=0.0)
            )

    @staticmethod
    def _update_context(new: Tensor, old: Optional[Tensor]) -> Tensor:
        """Update context with new data points.
        [B, num_old | None, DY] -> [B, num_old + num_new | num_new, DY]
        """
        if old is None:
            return new.clone()
        else:
            B, _, DY = old.shape
            B_new, _, DY_new = new.shape
            assert B_new == B and DY_new == DY, f"{new.shape} != {old.shape}"
            return torch.cat((old, new), dim=1)

    def _sample(
        self,
        num_subspace_points: int,
        input_bounds: FloatListOrNestedOrTensor,
        use_grid_sampling: bool,
        use_factorized_policy: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        x_mask: Optional[Tensor] = None,
        y_mask: Optional[Tensor] = None,
        seed: int = 0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample a batch (x, y, chunks, chunk_mask) from the function."""
        # Sample domain
        if x_mask is None:
            max_x_dim = self.x_dim
        else:
            max_x_dim = x_mask.shape[-1]

        x, chunks, chunk_mask = sample_domain(
            d=num_subspace_points,
            max_x_dim=max_x_dim,
            device=device,
            x_mask=x_mask,
            input_bounds=input_bounds,
            use_grid_sampling=use_grid_sampling,
            use_factorized_policy=use_factorized_policy,
            seed=seed,
        )

        # Evaluate function at x: [m, max_y_dim]
        y = self.evaluate(x=x, input_bounds=input_bounds, x_mask=x_mask, y_mask=y_mask)

        return (x, y, chunks, chunk_mask)

    def sample(
        self,
        input_bounds: FloatListOrNestedOrTensor,
        num_subspace_points: int,
        use_grid_sampling: bool,
        batch_size: int,
        use_factorized_policy: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        x_mask: Optional[Tensor] = None,
        y_mask: Optional[Tensor] = None,
        seed: int = 0,
    ):
        """Sample batches (batch_x, batch_y, batch_chunks, batch_chunk_mask) from the function."""
        assert batch_size > 0, "`batch_size` must be a positive integer."

        x_list, y_list, chunks_list = [], [], []

        for _ in range(batch_size):
            x, y, chunks, chunk_mask = self._sample(
                num_subspace_points=num_subspace_points,
                input_bounds=input_bounds,
                use_grid_sampling=use_grid_sampling,
                use_factorized_policy=use_factorized_policy,
                device=device,
                x_mask=x_mask,
                y_mask=y_mask,
                seed=seed,
            )

            x_list.append(x)
            y_list.append(y)
            chunks_list.append(chunks)

        batch_x = torch.stack(x_list, dim=0)
        batch_y = torch.stack(y_list, dim=0)
        batch_chunks = torch.stack(chunks_list, dim=0)
        # shared across batches
        batch_chunk_mask = repeat(chunk_mask, "n d -> b n d", b=batch_size)

        return (batch_x, batch_y, batch_chunks, batch_chunk_mask)

    def evaluate(
        self,
        x: Tensor,
        input_bounds: FloatListOrNestedOrTensor,
        x_mask: Optional[Tensor] = None,
        y_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluate function at x.

        Args:
            x: [m, dx_max]
            input_bounds: [dx_max, 2]
            x_mask: Optional valid x dim mask, [dx_max]
            y_mask: Optional valid y dim mask, [dy_max]

        Returns:
            y: [m, dy_max]
        """
        if x_mask is None and y_mask is None:
            return self.__call__(x=x, input_bounds=input_bounds)
        else:
            # Compact x and input_bounds by x_mask
            x_valid = compact_by_mask(data=x, mask=x_mask, dim=-1)

            input_bounds = make_range_tensor(input_bounds, num_dim=x_valid.shape[-1])
            input_bounds = input_bounds.to(x.device)
            input_bounds_valid = compact_by_mask(data=input_bounds, mask=x_mask, dim=0)

            # Evaluate at compacted x: [m, dy]
            y_valid = self.__call__(x=x_valid, input_bounds=input_bounds_valid)

            # Restore full dimensions: [m, dy_max]
            y = restore_by_mask(data=y_valid, mask=y_mask, dim=-1)
            return y

    def compute_hv(self, solutions: Tensor, y_mask: Optional[Tensor] = None) -> Tensor:
        bounds = self.y_bounds.to(solutions.device)
        ref_point = self.ref_point.to(solutions.device)

        # Compact solutions by y_mask
        solutions = compact_by_mask(solutions, y_mask, dim=-1)

        hv = MOO.compute_hv(
            solutions=solutions,
            ref_point=ref_point,
            minimum=bounds[:, 0],
            maximum=bounds[:, 1],
            normalize=False,
            # y_mask=y_mask,
        )[0]

        hv = torch.from_numpy(hv)
        hv = hv.to(device=solutions.device, dtype=solutions.dtype)
        return hv

    def compute_regret(
        self,
        solutions: Tensor,
        candidates: Optional[Tensor] = None,
        regret_type: str = "ratio",
        y_mask: Optional[Tensor] = None,
    ) -> Tensor:
        bounds = self.y_bounds.to(solutions.device)
        ref_point = self.ref_point.to(solutions.device)

        # Compact solutions and candidates by y_mask
        solutions = compact_by_mask(solutions, y_mask, dim=-1)
        candidates = compact_by_mask(candidates, y_mask, dim=-1)

        regret = MOO.compute_regret(
            solutions=solutions,
            candidates=candidates,
            ref_point=ref_point,
            minimum=bounds[:, 0],
            maximum=bounds[:, 1],
            regret_type=regret_type,
            optimal_value=getattr(self.func, "_optimal_value", None),
            max_hv=self.max_hv,
        )

        regret = torch.from_numpy(regret)
        regret = regret.to(device=solutions.device, dtype=solutions.dtype)
        return regret

    def step(
        self,
        x_new: Tensor,
        input_bounds: FloatListOrNestedOrTensor,
        x_ctx: Optional[Tensor] = None,
        y_ctx: Optional[Tensor] = None,
        x_mask: Optional[Tensor] = None,
        y_mask: Optional[Tensor] = None,
        compute_hv: bool = True,
        compute_regret: bool = True,
        regret_type: str = "ratio",
        solution_candidate_set: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[np.ndarray], Optional[np.ndarray]]:
        """Evaluate at `x_new`, update context, optionally compute reward / regret.

        Args:
            x_new: Input points, shape [B, num_new, dx_max]
            input_bounds: Input bounds for scaling
            x_ctx: Optional context input points, shape [B, num_ctx, dx_max]
            y_ctx: Optional context output points, shape [B, num_ctx, dy_max]
            x_mask: Optional mask for valid x dims, [dx_max]
            y_mask: Optional mask for valid y dims, [dy_max]
            compute_reward: compute reward from choosing `x_new` if True
            compute_regret: compute regret from choosing `x_new` if True
            regret_type: Type of regret to compute, defaults to "ratio"
            solution_candidate_set: Optional set of candidate solutions for regret computation

        Returns:
            x_ctx: Updated context input points, shape [B, num_ctx + num_new, dx_max]
            y_ctx: Updated context output points, shape [B, num_ctx + num_new, dy_max]
            reward (np.ndarray): Reward from choosing `x_new`, shape [B] or None if not computed
            regret (np.ndarray): Regret from choosing `x_new`, shape [B] or None if not computed
        """
        # Evaluate at x_new: [B, num_new, dy_max]
        y_new = self.evaluate(
            x=x_new, input_bounds=input_bounds, x_mask=x_mask, y_mask=y_mask
        )

        # Update context
        x_ctx = TestFunction._update_context(new=x_new, old=x_ctx)
        y_ctx = TestFunction._update_context(new=y_new, old=y_ctx)

        # Compute hypervolume / regret
        if compute_hv:
            reward = self.compute_hv(solutions=y_ctx, y_mask=y_mask)
        else:
            reward = None

        if compute_regret:
            regret = self.compute_regret(
                solutions=y_ctx,
                candidates=solution_candidate_set,
                regret_type=regret_type,
                y_mask=y_mask,
            )
        else:
            regret = None

        return x_ctx, y_ctx, reward, regret

    def init(
        self,
        input_bounds: FloatListOrNestedOrTensor,
        batch_size: int,
        num_initial_points: int,
        regret_type: str,
        compute_hv: bool = True,
        compute_regret: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        x_mask: Optional[Tensor] = None,
        y_mask: Optional[Tensor] = None,
        solution_candidate_set: Optional[Tensor] = None,
        seed: int = 0,
    ):
        """Sample initial samples.

        Returns:
            x_ctx: [B, num_initial_points, dx_max]
            y_ctx: [B, num_initial_points, dy_max]
            reward (np.ndarray): [B] or None
            regret (np.ndarray): [B] or None
        """
        if x_mask is None:
            max_x_dim = self.x_dim
        else:
            max_x_dim = x_mask.shape[-1]

        x_init = sample_domain(
            d=num_initial_points,
            max_x_dim=max_x_dim,
            device=device,
            x_mask=x_mask,
            input_bounds=input_bounds,
            use_grid_sampling=True,
            use_factorized_policy=False,
            seed=seed,
        )[0]

        # Repeat across batch
        x_init = repeat(x_init, "n dx -> b n dx", b=batch_size)

        # Evaluate and compute reward / regret
        x_ctx, y_ctx, reward, regret = self.step(
            x_new=x_init,
            input_bounds=input_bounds,
            x_ctx=None,
            y_ctx=None,
            x_mask=x_mask,
            y_mask=y_mask,
            compute_hv=compute_hv,
            compute_regret=compute_regret,
            regret_type=regret_type,
            solution_candidate_set=solution_candidate_set,
        )

        return x_ctx, y_ctx, reward, regret

    def __call__(self, x: Tensor, input_bounds: FloatListOrNestedOrTensor) -> Tensor:
        """Function forward pass: scale x, evaluate function, and return y.

        Args: x [..., dx], input_bounds: [dx, 2] | [2]

        Returns: y [..., dy]
        """
        dx = x.shape[-1]
        assert dx == self.x_dim, f"Input dimension mismatch: {dx} != {self.x_dim}"

        # Transform inputs
        x_transformed = self.transform_inputs(inputs=x, input_bounds=input_bounds)

        # Evaluate function (ensure func is on the same device as input)
        if hasattr(self.func, 'to'):
            self.func = self.func.to(x_transformed.device)
        y = self.func(x_transformed)
        y = y.reshape(*x.shape[:-1], self.y_dim)

        return y


class SyntheticFunction(TestFunction):
    """Synthetic function environment.

    Args:
        function_name (str): Synthetic function name
    """

    def __init__(self, function_name: str, **kwargs):
        super().__init__(function_name=function_name, **kwargs)

    @staticmethod
    def get_function_constructor(function_name: str) -> Optional[callable]:
        
        func_constructor = SYN_FUNCTIONS.get(function_name)
        return func_constructor

    @staticmethod
    def _init_function(function_name, func_constructor, scene):
        """Initialize function instance based on function name.
        - NERF_synthetic: 2-objective minimization (negate=True)
        - NERF_synthetic_fnum_3: 3-objective minimization (negate=True)
        - Others: minimization (negate=False)
        """
        if function_name == "NERF_synthetic":
            func = func_constructor(negate=True, num_objectives=2, scene=scene)
        elif function_name == "NERF_synthetic_fnum_3":
            func = func_constructor(negate=True, num_objectives=3, scene=scene)
        else:
            func = func_constructor(negate=False)
        return func

    def get_metadata(
        self,
        function_name: str,
        x_range_list: Optional[FloatListOrNested] = None,
        **kwargs,
    ) -> Dict:
        """Get metadata for synthetic function.

        Args:
            function_name: Name of the synthetic function
            x_range_list: Optional input bounds to override default function bounds

        Returns: None if function not found, otherwise a dictionary with the following keys:
            function_name (str): Function name
            func: BoTorch function instance
            x_bounds (List): Input bounds for the function, DX x [[x_min, x_max]]
            y_bounds (List): Output bounds for the function, DY x [[y_min, y_max]]
            ref_point (List): botorch defined if `ref_point` attribute is found, otherwise upper bounds
            max_hv (float): botorch defined if `max_hv` attribute is found, otherwise computed from reference point and lower bounds
        """
        # Get function
        func_constructor = self.get_function_constructor(function_name)
        if func_constructor is None:
            raise ValueError(
                "Function not implmented; "
                f"Only {list(SYN_FUNCTIONS.keys())} are available."
            )
        func = self._init_function(
            function_name=function_name,
            func_constructor=func_constructor,
            scene=kwargs.get("scene", "ship"),
        )

        # Prepare input bounds: DX x [[x_min, x_max]]
        x_dim = len(func._bounds)
        if x_range_list is None:
            x_bounds = tuple_list_to_nested_list(func._bounds)
        else:
            # Override with provided x_range_list
            x_bounds = make_range_nested_list(range_list=x_range_list, num_dim=x_dim)

        # Prepare output bounds: DY x [[y_min, y_max]]
        y_bounds = BENCHMARK_Y_BOUNDS.get(function_name, None)
        if y_bounds is None:
            # Estimate output bounds if not predefined
            num_objectives = getattr(func, "num_objectives", 1)
            y_bounds = estimate_objective_bounds(
                func=func,
                num_objectives=num_objectives,
                x_bounds=x_bounds,
            )

        # Prepare reference point and max hv
        ref_point = getattr(func, "_ref_point", None)
        max_hv = getattr(func, "_max_hv", None)

        if ref_point is None:
            ref_point = self.get_ref_point(y_bounds)  # [dy]
        if max_hv is None:
            max_hv = self.get_max_hv(ref_point, y_bounds)

        return {
            "function_name": function_name,
            "func": func,
            "x_bounds": x_bounds,
            "y_bounds": y_bounds,
            "ref_point": ref_point,
            "max_hv": max_hv,
        }


class IntepolatorFunction(TestFunction):
    """Interpolator function environment.

    Args:
        function_name: Name of the function - only for identification purposes
        train_x: Training input points, shape [n, dx]
        train_y: Training output points, shape [n, dy]
        train_x_bounds: Input bounds for training data, shape [dx, 2]
        train_y_bounds: Output bounds for training data, shape [dy, 2]
    """

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        train_x_bounds: FloatListOrNested,
        train_y_bounds: FloatListOrNested,
        function_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            train_x=train_x,
            train_y=train_y,
            train_x_bounds=train_x_bounds,
            train_y_bounds=train_y_bounds,
            function_name=function_name,
            **kwargs,
        )

    @staticmethod
    def is_valid_input(train_x, train_y, train_x_bounds, train_y_bounds) -> bool:
        if train_x is None or train_y is None:
            raise ValueError("Training data must be provided.")

        if train_x_bounds is None or train_y_bounds is None:
            raise ValueError("Training bounds must be provided.")

        if train_x.ndim != 2 or train_y.ndim != 2:
            raise ValueError(f"Training data must be 2D Tensors. ")

        if train_x.shape[:-1] != train_y.shape[:-1]:
            raise ValueError(
                f"Data shapes mismatch: {train_x.shape[:-1]} != {train_y.shape[:-1]}"
            )

    def get_metadata(
        self,
        train_x: Tensor,
        train_y: Tensor,
        train_x_bounds: FloatListOrNestedOrTensor,
        train_y_bounds: FloatListOrNestedOrTensor,
        function_name: Optional[str] = "interpolator_function",
        **kwargs,
    ) -> Dict:
        """Get metadata for interpolator function.

        Args:
            train_x: Input training points, shape [n, dx]
            train_y: Output training values, shape [n, dy]
            train_x_bounds: Input bounds for training data, DX x [[x_min, x_max]]
            train_y_bounds: Output bounds for training data, DY x [[y_min, y_max]]

        Returns: A dictionary with the following keys:
            function_name (str): Function name
            func (BatchNDInterpolatorExtTensor): Interpolator function instance
            x_bounds (List): train_x_bounds
            y_bounds (List): train_y_bounds
            ref_point (List): Upper bounds
            max_hv (float): Maximum hypervolume computed from reference point and lower bounds
        """
        # Validate inputs
        self.is_valid_input(train_x, train_y, train_x_bounds, train_y_bounds)

        x_dim = train_x.shape[-1]
        y_dim = train_y.shape[-1]
        train_x_bounds = make_range_nested_list(train_x_bounds, num_dim=x_dim)
        train_y_bounds = make_range_nested_list(train_y_bounds, num_dim=y_dim)

        # Get function: [N, DX] -> [N, DY]
        func = NPInterpolatorExtTensor(points=train_x, values=train_y)

        ref_point = self.get_ref_point(bounds=train_y_bounds, candidates=train_y)
        max_hv = self.get_max_hv(
            ref_point=ref_point, bounds=train_y_bounds, candidates=train_y
        )

        return {
            "function_name": function_name,
            "func": func,
            "x_bounds": train_x_bounds,
            "y_bounds": train_y_bounds,
            "ref_point": ref_point,
            "max_hv": max_hv,
        }
