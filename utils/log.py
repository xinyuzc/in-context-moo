from typing import Optional, Dict, List, Any, Union
import os
import os.path as osp
from collections import OrderedDict
import time
import logging
from logging import Logger
from dataclasses import dataclass, field


import numpy as np
from torch import Tensor
import torch

ROOT = ""


def get_log_filename(
    model_name: str,
    expid: str,
    logid: Optional[str] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    log_dir: str = "logs",
) -> str:
    """Create logging filename.
    Returns: logs/{model_name}/{expid}/{prefix}_{log_id}_{suffix}.log
    """
    log_dir = osp.join(ROOT, log_dir, model_name, expid)
    os.makedirs(log_dir, exist_ok=True)

    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(logid if logid else time.strftime("%Y%m%d%H%M%S"))
    if suffix:
        parts.append(suffix)

    filename = "_".join(parts) + ".log"
    return osp.join(log_dir, filename)


def log_fn(logger: None):
    return logger.info if logger is not None else print


def get_logger(filename: str, mode="a") -> Logger:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()

    # remove old handlers to avoid duplicated outputs.
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    logger.addHandler(logging.FileHandler(filename, mode=mode))
    logger.addHandler(logging.StreamHandler())

    logger.info(f"Logging to file:\t{filename}")
    return logger


def get_log_fn(filename: str, mode="a") -> callable:
    logger = get_logger(filename=filename, mode=mode)
    return log_fn(logger)


class Averager(object):
    """Organize averaged metrics in dictionary-style."""

    def __init__(self, *keys):
        self.sum = OrderedDict()
        self.cnt = OrderedDict()
        self.clock = time.time()
        for key in keys:
            self.sum[key] = 0
            self.cnt[key] = 0

    def batch_update(self, batch_dict):
        for key, val in batch_dict.items():
            self.update(key, val)

    def update(self, key, val):
        if isinstance(val, torch.Tensor):
            val = val.item()

        if self.sum.get(key, None) is None:
            self.sum[key] = val
            self.cnt[key] = 1
        else:
            self.sum[key] = self.sum[key] + val
            self.cnt[key] += 1

    def reset(self):
        for key in self.sum.keys():
            self.sum[key] = 0
            self.cnt[key] = 0

        self.clock = time.time()

    def get(self, key) -> float:
        if key not in self.sum:
            return None
        return self.sum[key] / self.cnt[key] if self.cnt[key] > 0 else 0

    def info(self) -> str:
        line = ""
        for key in self.sum.keys():
            val = self.sum[key] / self.cnt[key]  # average
            line += f"{key}: {val:.4f} "

        line += f"({time.time()-self.clock:.3f} secs)"
        return line


@dataclass
class MetricTracker:
    """Evaluation metric tracker.

    Attributes:
        Optimization metrics:
            hv: Hypervolume over time [B, num_steps]
            hv_queries: Hypervolume of individual queries [B, num_steps]
            regret: Regret over time [B, num_steps]
            entropy: Policy entropy over time [B, num_steps]
            time: Inference time per step [num_steps]

        Prediction metrics (optional):
            nll_c: Context negative log-likelihood [num_steps]
            nll_t: Target negative log-likelihood [num_steps]
            mse_c: Context MSE per dimension [dy, num_steps]
            mse_t: Target MSE per dimension [dy, num_steps]

        Query tracking:
            x_queries: Queried input points [B, num_steps*q, dx]
            y_queries: Queried output values [B, num_steps*q, dy]
            y_mask_history: Observation mask history [num_steps, dy]

        Auxiliary information:
            acq_values: Acquisition function values (when available)
            refinement_info: Gradient refinement details (when used)
    """

    # Optimization metrics
    hv_list: List[Tensor] = field(default_factory=list)
    hv_queries_list: List[Tensor] = field(default_factory=list)
    regret_list: List[Tensor] = field(default_factory=list)
    entropy_list: List[Tensor] = field(default_factory=list)
    time_list: List[float] = field(default_factory=list)

    # Prediction metrics
    nll_c_list: List[Tensor] = field(default_factory=list)
    nll_t_list: List[Tensor] = field(default_factory=list)
    mse_c_list: List[Tensor] = field(default_factory=list)
    mse_t_list: List[Tensor] = field(default_factory=list)

    # Query tracking
    x_queries_list: List[Tensor] = field(default_factory=list)
    y_queries_list: List[Tensor] = field(default_factory=list)
    y_mask_history: List[Tensor] = field(default_factory=list)

    # Auxiliary information
    acq_values_list: List[Optional[Tensor]] = field(default_factory=list)
    refinement_info_list: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _match_query_dim(metric: Union[Tensor, List], n_queries: int):
        """[B, 1] or [B,] -> [B, n_queries]"""
        if isinstance(metric, Tensor):
            B = len(metric)
            metric = metric.view(B, -1)
            n_metrics = metric.shape[1]
            expand = lambda m: m.expand(-1, n_queries)
        else:
            n_metrics = len(metric)
            expand = lambda m: m * n_queries

        if n_metrics == 1:
            return expand(metric)
        elif n_metrics == n_queries:
            return metric
        else:
            raise ValueError(
                f"metric has {n_metrics} entries but expected 1 or {n_queries}"
            )

    def add_optimization_step(
        self,
        hv: Tensor,
        hv_query: Tensor,
        regret: Tensor,
        entropy: Tensor,
        time: float | List[float],
        x_query: Tensor,
        y_query: Tensor,
        acq_values: Optional[Tensor] = None,
        match_query_dim: bool = True,
    ):
        """Add metrics from one optimization step."""
        if match_query_dim:
            # Replicate metric values along query dim to match queries shape
            # (For batch settings - a batch of queries versus a single metric)
            n_queries = x_query.shape[1]
            hv = self._match_query_dim(hv, n_queries=n_queries)
            hv_query = self._match_query_dim(hv_query, n_queries=n_queries)
            regret = self._match_query_dim(regret, n_queries=n_queries)
            entropy = self._match_query_dim(entropy, n_queries=n_queries)
            time = self._match_query_dim(time, n_queries=n_queries)

        self.hv_list.append(hv.detach().cpu())
        self.hv_queries_list.append(hv_query.detach().cpu())
        self.regret_list.append(regret.detach().cpu())
        self.entropy_list.append(entropy.detach().cpu())
        self.time_list += time if isinstance(time, list) else [time]
        self.x_queries_list.append(x_query.detach().cpu())
        self.y_queries_list.append(y_query.detach().cpu())

        if acq_values is not None:
            self.acq_values_list.append(acq_values.detach().cpu())
        else:
            self.acq_values_list.append(None)

    def add_prediction_step(
        self, nll_c: Tensor, nll_t: Tensor, mse_c: Tensor, mse_t: Tensor
    ):
        """Add prediction metrics from one step."""
        self.nll_c_list.append(nll_c)
        self.nll_t_list.append(nll_t)
        self.mse_c_list.append(mse_c)
        self.mse_t_list.append(mse_t)

    def add_refinement_info(self, info: Dict[str, Any]):
        """Add gradient refinement information."""
        self.refinement_info_list.append(info)

    def get_stacked_metrics(self, device: torch.device) -> Dict[str, Tensor]:
        """Convert lists to stacked tensors for saving/plotting."""
        metrics = {
            "hv": torch.cat(self.hv_list, dim=-1).cpu(),  # [B, T+1]
            "instant_hv": torch.cat(self.hv_queries_list, dim=-1).cpu(),  # [B, T+1]
            "regret": torch.cat(self.regret_list, dim=-1).cpu(),  # [B, T+1]
            "entropy": torch.cat(self.entropy_list, dim=-1).cpu(),  # [B, T+1]
            "time": torch.tensor(self.time_list, device=device).cpu(),  # [T+1]
        }

        # Add prediction metrics if available
        if self.nll_c_list:
            metrics.update(
                {
                    "nll_c": torch.stack(self.nll_c_list, dim=-1).cpu(),  # [T+1]
                    "nll_t": torch.stack(self.nll_t_list, dim=-1).cpu(),  # [T+1]
                    "mse_c": torch.stack(self.mse_c_list, dim=-1).cpu(),  # [dy, T+1]
                    "mse_t": torch.stack(self.mse_t_list, dim=-1).cpu(),  # [dy, T+1]
                }
            )

        return metrics

    def get_latest_values(self) -> Dict[str, Any]:
        """Get the most recent metric values for logging."""
        return {
            "hv": self.hv_list[-1] if self.hv_list else None,
            "regret": self.regret_list[-1] if self.regret_list else None,
            "entropy": self.entropy_list[-1] if self.entropy_list else None,
            "x_query": self.x_queries_list[-1] if self.x_queries_list else None,
            "y_query": self.y_queries_list[-1] if self.y_queries_list else None,
        }

    def get_statistics(self) -> Dict[str, float]:
        """Compute summary statistics across all steps."""
        stats = {}

        if self.hv_list:
            hv_tensor = torch.cat(self.hv_list, dim=-1)  # [B, T]
            stats["hv_final_mean"] = hv_tensor[:, -1].mean().item()
            stats["hv_final_std"] = hv_tensor[:, -1].std().item()
            stats["hv_max"] = hv_tensor.max().item()

        if self.regret_list:
            regret_tensor = torch.cat(self.regret_list, dim=-1)  # [B, T]
            stats["regret_final_mean"] = regret_tensor[:, -1].mean().item()
            stats["regret_final_std"] = regret_tensor[:, -1].std().item()
            stats["regret_min"] = regret_tensor.min().item()

        if self.time_list:
            stats["total_time"] = sum(self.time_list)
            stats["avg_time_per_step"] = np.mean(self.time_list)
            stats["std_time_per_step"] = np.std(self.time_list)

        if self.nll_t_list:
            nll_t_tensor = torch.stack(self.nll_t_list)  # [T]
            stats["nll_t_final"] = nll_t_tensor[-1].item()
            stats["nll_t_mean"] = nll_t_tensor.mean().item()
            stats["nll_t_std"] = nll_t_tensor.std().item()

        return stats
