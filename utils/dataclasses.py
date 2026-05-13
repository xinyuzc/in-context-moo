from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, ClassVar
import random
import time

import torch

from utils.config import get_train_x_range, get_train_y_range


@dataclass
class DataConfig:
    """Configurations for data.

    Args:
        function_name: Function name
        scene: only used for HPO3DGS problems
        x_dim_list: x dimensions to sample from
        y_dim_list: y dimensions to sample from
        max_x_dim (int): maximal number of x dimensions
        max_y_dim (int): maximal number of y dimensions
        data_id (str): optional data category - for dataset retrieval
        x_range: get_train_x_range()
        y_range: get_train_y_range()
        ...
    """

    function_name: str = "gp"
    data_id: Optional[str] = None
    scene: Optional[str] = "ship"

    dim_scatter_mode: str = "random_k"

    x_range: List[float] = field(default_factory=lambda: get_train_x_range())
    y_range: List[float] = field(default_factory=lambda: get_train_y_range())
    max_x_dim: int = 4
    max_y_dim: int = 3
    x_dim_list: List[int] = field(default_factory=lambda: [1, 2, 3])
    y_dim_list: List[int] = field(default_factory=lambda: [1, 2])
    sampler_list: List[str] = field(
        default_factory=lambda: [
            "multi_task_gp_prior_sampler",
            "multi_output_gp_prior_sampler",
        ]
    )
    sampler_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])
    data_kernel_type_list: List[str] = field(
        default_factory=lambda: ["rbf", "matern32", "matern52"]
    )
    sample_kernel_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    lengthscale_range: List[float] = field(default_factory=lambda: [0.1, 2.0])
    std_range: List[float] = field(default_factory=lambda: [0.1, 1.0])
    min_rank: int = 1
    max_rank: Optional[int] = None
    p_iso: float = 0.5
    standardize: bool = True
    jitter: float = 1e-3
    max_tries: int = 10

    sigma: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def assert_dims_within_limits(self, max_x_dim, max_y_dim):
        for dx in self.x_dim_list:
            assert dx <= max_x_dim, f"x_dim {dx} exceeds max_x_dim {max_x_dim}"
        for dy in self.y_dim_list:
            assert dy <= max_y_dim, f"y_dim {dy} exceeds max_y_dim {max_y_dim}"


@dataclass
class LossConfig:
    use_cumulative_rewards: bool = True
    batch_standardize: bool = True
    clip_rewards: bool = True
    loss_weight: float = 1.0
    discount_factor: float = 0.99
    max_norm: float = 1.0


@dataclass
class PredictionConfig:
    batch_size: int = 32
    min_nc: int = 2
    max_nc: int = 50
    nc: Optional[int] = None
    read_cache: bool = False

    def __post_init__(self):
        assert self.min_nc >= 0, f"min_nc {self.min_nc} < 0"
        assert self.max_nc >= self.min_nc, f"max_nc {self.max_nc} < {self.min_nc}"

        if self.nc is not None:
            assert self.nc >= self.min_nc, f"nc {self.nc} < min_nc {self.min_nc}"
            assert self.nc <= self.max_nc, f"nc {self.nc} > max_nc {self.max_nc}"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class OptimizationConfig:
    """Configuration for optimization tasks.

    Args:
        ...
        T (int): optimization horizon. If not specified, T will be sampled from [min_T, max_T] each time being called.
    """

    use_grid_sampling: bool
    use_fixed_query_set: bool
    use_factorized_policy: bool
    use_time_budget: bool
    batch_size: int
    T: Optional[int] = None
    min_T: int = 10
    max_T: int = 100
    regret_type: str = "norm_ratio"
    num_initial_points: int = 1
    random_num_initial: bool = False
    num_samples: int = 1
    dim_mask_gen_mode: str = "full"
    num_query_points: int = 256
    single_obs_x_dim: Optional[int] = None
    single_obs_y_dim: Optional[int] = None
    read_cache: bool = False
    write_cache: bool = False
    epsilon: float = 1.0
    use_curriculum: bool = False
    use_logit_mask: bool = False

    # Batch query with fantasized outcomes
    q: int = 1
    fantasy: bool = False

    # Cost associates with each objective or instance
    cost_mode: bool = False
    cost: float = 1.0

    # How to sample query points
    sampling_mode: str = "full"
    num_reweighted_samples: int = 2048

    # Two-pass adaptive query set (Phase 1: deployment-loop only)
    use_adaptive_query_set: bool = False
    adaptive_top_k: int = 8
    adaptive_samples_per_top: int = 32
    # Trust-region radius as a multiple of Q0's average nearest-neighbour
    # spacing per dim (i.e. range_width * (1/d)^(1/dx)). Values >= 1 let
    # cubes bridge gaps between adjacent Q0 grid points; smaller values keep
    # cubes purely local. For "gaussian" this is the std multiplier; for
    # "cube_sobol" this is the cube-half-width multiplier.
    adaptive_sigma_frac: float = 3.0
    # Stochastic top-K: sample K without replacement from softmax(logits / T).
    # 0.0 = deterministic topk (legacy); higher = more diverse seeds.
    adaptive_topk_temperature: float = 0.0
    # Trust-region geometry around each top-K seed: "cube_sobol" draws Sobol
    # samples uniformly from the (clipped) cube, "gaussian" draws from N(seed, std).
    adaptive_trust_region: str = "cube_sobol"
    use_adaptive_global_region: bool = True
    # Epsilon (sample-vs-argmax) for pass-2 over Q1. 1.0 = always sample
    # (preserves prior hardcoded behavior; useful at train time to keep
    # exploration on the refined candidate set). At deployment, set 0.0 to take
    # the argmax of Q1's policy scores. Independent from outer `epsilon`.
    adaptive_pass2_epsilon: float = 1.0

    def __post_init__(self):
        if self.T is not None:
            assert isinstance(self.T, int) and self.T > 0, "T must be a positive integer."

        if self.sampling_mode not in ["full", "top_d", "reweighted"]: 
            raise ValueError(f"sampling mode `{self.sampling_mode} not supported.")

        # use_logit_mask can be True only if considering a full, fixed query set during optimization...
        if self.sampling_mode != "full":
            self.use_logit_mask = False
        if not self.use_fixed_query_set: 
            self.use_logit_mask = False

    def sample_T(self) -> int:
        """Return T if fixed, otherwise sample from [min_T, max_T]."""
        if self.T is not None:
            return self.T
        return random.randint(self.min_T, self.max_T)

    params_map: ClassVar[Dict[str, str]] = {
        "use_grid_sampling": "Grid",
        "use_fixed_query_set": "Fixq",
        "use_factorized_policy": "Fact",
        "use_time_budget": "Tbud",
        "batch_size": "B",
        "T": "T",
        "min_T": "MinT",
        "max_T": "MaxT",
        "regret_type": "Regr",
        "num_initial_points": "Nspt",
        "num_samples": "Nsmp",
    }

    # Allowlist of fields that contribute to the run-identity filename.
    # Adding new fields here is opt-in: forgetting to list one keeps it OUT of
    # the path so old data still loads under the same folder name.
    filename_fields: ClassVar[List[str]] = [
        "use_grid_sampling",
        "use_fixed_query_set",
        "use_factorized_policy",
        "use_time_budget",
        "batch_size",
        "T",
        "min_T",
        "max_T",
        "regret_type",
        "num_initial_points",
        "random_num_initial",
        "num_samples",
        "dim_mask_gen_mode",
        "num_query_points",
        "single_obs_x_dim",
        "single_obs_y_dim",
        "read_cache",
        "write_cache",
        "epsilon",
        "use_curriculum",
        "use_logit_mask",
        "q",
        "fantasy",
        "cost_mode",
        "cost",
        "sampling_mode",
        "num_reweighted_samples",
    ]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class TrainConfig:
    num_total_epochs: int = 400000
    num_burnin_epochs: int = 395000
    nc_burnin_ratio: float = 0.8
    lr1: float = 1e-3
    lr2: float = 4e-5
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine_with_warmup"
    weight_decay: float = 1e-2
    num_warmup_steps: Optional[int] = None
    num_workers: int = 2
    prefetch_factor: int = 2

    def __post_init__(self):
        assert self.num_burnin_epochs <= self.num_total_epochs
        assert self.nc_burnin_ratio >= 0.0
        assert self.nc_burnin_ratio <= 1.0
        self.num_nc_burnin_epochs = int(self.nc_burnin_ratio * self.num_burnin_epochs)
        if self.num_workers == 0:
            self.prefetch_factor = None


@dataclass
class ExConfig:
    seed: Optional[int] = None
    mode: str = "train"
    task: str = "optimization"
    model_name: str = "TAMO"
    device: str = "cuda"
    expid: Optional[str] = None
    resume: bool = False
    override: bool = False
    log_to_wandb: bool = True

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(1, 10000)

        assert isinstance(self.seed, int), f"Seed {self.seed} should be integer"
        assert self.mode in ["train", "validation", "test"]
        if self.device == "cuda":
            assert torch.cuda.is_available(), "CUDA is not available."

        if self.expid is None:
            timestamp = time.strftime("%Y%m%d%H%M%S")
            self.expid = f"{self.model_name}_seed{self.seed}_{timestamp}"


@dataclass
class LogConfig:
    # Training time
    freq_log: int = 200
    freq_save: int = 500
    freq_save_extra_burnin: int = 25000
    freq_save_extra: int = 5000
    freq_log_grad: int = 1000

    # Test time
    plot_enabled: bool = True
    plot_nc_list: Optional[List] = None
    plot_per_n_steps: int = 25
