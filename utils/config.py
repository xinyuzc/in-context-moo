"""Global configuration.

Global variables:
X_RANGE, Y_RANGE, MODEL_DICT, SPLITS, TASKS, SIGMA

Functions:
...
- save_checkpoint: Saves the model checkpoint.
- load_checkpoint: Loads a model checkpoint from disk.
- build_scheduler: Constructs a learning rate scheduler.
- build_optimizer: Constructs an optimizer for training.
- build_model: Constructs a model based on the specified configuration.
- build_dataloader: Constructs a data loader for the dataset.
"""

from typing import Optional, Tuple, List
import math
import os
import os.path as osp

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torchdata.stateful_dataloader import StatefulDataLoader
from pymoo.config import Config

from model import TAMO, TAMOConfig


Config.warnings["not_compiled"] = False

SIGMA = 0.0

X_RANGE = [-5.0, 5.0]
Y_RANGE = [-1.0, 1.0]
SPLITS = ["train", "validation", "test"]

# Remap keys that were renamed during the clean-up...
_ckpt_key_map = {
    "decoder.id_task": "decoder.task_tokens",
    "decoder.token_selected": "decoder.ar_bias_token",
}

def remap_checkpoint_keys(state_dict: dict) -> dict:
    return {_ckpt_key_map.get(k, k): v for k, v in state_dict.items()}

def load_checkpoint(
    exp_path: str,
    device: str,
    resume: bool = False,
    ckpt_name: str = "ckpt.tar",
    weights_only: bool = False,
):
    """try loading checkpoint from `exp_path`/`ckpt_name`

    Returns: If resume, try reading checkpoint otherwise raise not found error.
    otherwise, create new experiment directory and return an empty dict.
    """
    ckpt_path = osp.join(exp_path, ckpt_name)

    if resume:
        if not osp.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=weights_only)
    else:
        if osp.exists(ckpt_path):
            raise FileExistsError(f"Checkpoint {ckpt_path} already exists.")

        os.makedirs(exp_path, exist_ok=True)
        ckpt = {}

    return ckpt


def build_tamo(model_cfg: dict = {}) -> TAMO:
    """Construct a TAMO model from flat model config dict."""
    valid_fields = set(TAMOConfig.__dataclass_fields__.keys())
    tamo_kwargs = {k: v for k, v in model_cfg.items() if k in valid_fields}
    config = TAMOConfig(**tamo_kwargs)
    return TAMO(config)


def get_train_x_range(function_name: Optional[str] = None) -> list:
    """Returns: [-5.0, 5.0]"""
    return X_RANGE


def get_train_y_range(function_name: Optional[str] = None) -> list:
    """Returns: [-1.0, 1.0]"""
    return Y_RANGE


def save_checkpoint(
    exp_path: str,
    model: Module,
    epoch: int,
    seed: int,
    dataloader=None,
    optimizer=None,
    scheduler=None,
    ckpt_name: str = "ckpt.tar",
    stats=None,
) -> Tuple[dict, str]:
    """Save checkpoint."""
    with torch.no_grad():
        model_state_dict = model.state_dict()
        ckpt = {"model": model_state_dict, "epoch": epoch, "seed": seed}

        def _save_optional(ckpt, key, value=None) -> None:
            if value is None:
                return
            ckpt[key] = value.state_dict() if hasattr(value, "state_dict") else value

        _save_optional(ckpt, "dataloader", dataloader)
        _save_optional(ckpt, "optimizer", optimizer)
        _save_optional(ckpt, "scheduler", scheduler)

        # TODO save stats

    ckpt_filepath = osp.join(exp_path, ckpt_name)
    torch.save(ckpt, ckpt_filepath)

    return ckpt, ckpt_filepath


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    num_warmup_steps: Optional[int] = None,
    ratio_warmup: float = 0.05,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build scheduler with type in ["cosine", "cosine_with_warmup"]."""
    if num_warmup_steps is None:
        # Rule of thumb: 5% of total training steps
        num_warmup_steps = int(ratio_warmup * num_training_steps)

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
    elif scheduler_type == "cosine_with_warmup":
        assert num_warmup_steps < num_training_steps and num_warmup_steps >= 0

        def get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps: int,
            num_training_steps: int,
            num_cycles: float = 0.5,
            last_epoch: int = -1,
        ):
            """Linear increase from 0 to lr during warmup, cosine decay to 0 after warmup.
            Reference: https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104
            """

            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))

                progress = float(current_step - num_warmup_steps) / float(
                    max(1, num_training_steps - num_warmup_steps)
                )
                return max(
                    0.0,
                    0.5
                    * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
                )

            return LambdaLR(optimizer, lr_lambda, last_epoch)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    else:
        raise NotImplementedError(f"Scheduler {scheduler_type} is not implemented.")

    return scheduler


def build_optimizer(
    model: Module,
    optimizer_type: str,
    lr: float,
    weight_decay: float = 1e-2,
    **kwargs: dict,
) -> torch.optim.Optimizer:
    """Build optimizer: ["adam", "adamw"]."""
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {optimizer_type} is not implemented.")
    return optimizer


def build_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    split: str,
    device: str,
    num_workers: int = 0,
    prefetch_factor: Optional[int] = None,
) -> StatefulDataLoader:
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split=="train",
        generator=torch.Generator(device="cpu"),
        pin_memory=(device != "cpu"),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
    )

    return dataloader
