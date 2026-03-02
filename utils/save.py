from dataclasses import asdict
import os.path as osp
import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure

from torch import Tensor
import torch
import wandb


def _convert_value_to_str(val):
    """Convert a value to a string representation."""
    if isinstance(val, str):
        val_str = val
    elif val is None:
        val_str = "none"
    elif isinstance(val, bool):
        val_str = "1" if val else "0"
    elif isinstance(val, int):
        # NOTE must be after bool
        val_str = str(val)
    elif isinstance(val, float):
        # 0.04 -> "4e02"
        # val_str = f"{val:.0e}"
        # val_str.replace("-", "")
        # val_str = val_str.replace(".", "")
        val_str = f"{val:.0e}"
        val_str.replace("-", "")
        val_str = val_str.replace(".", "")

        ## 0.045 -> 4.5e-2 -> 4p5e-2 -> 4p5em2
        # val_str = f"{val:g}"
        # val_str = val_str.replace(".", "p")
        # val_str = val_str.replace("+", "")
        # val_str = val_str.replace("-", "m")
    elif isinstance(val, list):
        # ["a", "b", "c"] -> "abc"
        val_str = "".join(_convert_value_to_str(v) for v in val)
    else:
        raise TypeError(
            f"Unsupported type {type(val)} for value '{val}'. "
            "Supported types are None, bool, float, and str."
        )

    # Remove spaces and slashes from the string
    val_str = val_str.replace(" ", "")
    val_str = val_str.replace("/", "")

    return val_str


def params_to_string(
    params: dict,
    preffix: dict = {},
    suffix: dict = {},
    params_map: dict = {},
    exclude: list[str] = [],
) -> str:
    """Generate a string summarizing config attributes.

    Examples:
        `Grid1_T64_gs1_fq1_fp1_tb0_rt_norm_ratio_n0_1_sg0_0e+00_t32`
    """

    def _kv2str(key, val):
        """Convert key and value to a string representation, if key is not empty."""
        if key == "":
            return None
        return f"{key}{val}"

    def _append(cur_list: list, new_dict: dict, params_map: dict = {}) -> list:
        """Create string items from `new_dict` and append to `cur_list`."""
        if new_dict is None or not new_dict:
            return cur_list
        if params_map is None:
            params_map = {}

        for key, val in sorted(new_dict.items()):
            if key in exclude:
                continue

            key_str = params_map.get(key, key[:4])
            val_str = _convert_value_to_str(val)
            item_str = _kv2str(key_str, val_str)

            if item_str is not None:
                cur_list.append(item_str)

        return cur_list

    exclude = set(exclude or [])

    item_list = []
    item_list = _append(item_list, preffix)
    item_list = _append(item_list, params, params_map)
    item_list = _append(item_list, suffix)

    return "_".join(item_list)


def adapt_save_fig(fig, filename="test.pdf"):
    """Remove right and top spines, set bbox_inches and dpi."""

    for ax in fig.get_axes():
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    fig.savefig(filename, bbox_inches="tight", dpi=300)


def save_fig(
    fig: Figure,
    path: str,
    config,
    filename: str,
    override: bool = False,
    log: callable = print,
    log_to_wandb: bool = False,
):
    log(f"--- Saving figure {filename} ---")
    if fig is None:
        log(f"    Figure is None. Skipping save.")
        return None
    folder_name = params_to_string(asdict(config))
    fig_path = osp.join(path, folder_name, f"{filename}.pdf")
    log(f"    Full path:\t{fig_path}")

    if osp.exists(fig_path) and not override:
        log(f"\n    Figure exists. Skipping save.")
        return fig_path
    elif osp.exists(fig_path) and override:
        log(f"\n    Figure exists but overrides.")
    else:
        os.makedirs(osp.join(path, folder_name), exist_ok=True)
        log(f"\n    Figure saves.")

    adapt_save_fig(fig, fig_path)
    if log_to_wandb:
        wandb.log({filename: wandb.Image(fig)})
    fig.clf()
    plt.close(fig)
    del fig
    return fig_path


def save_data(
    data: Tensor,
    path: str,
    filename: str,
    config,
    override: bool = False,
    log: callable = print,
    use_pickle: bool = False,
):
    log(f"--- Saving data {filename} ---")
    if data is None:
        log(f"    Data is None. Skipping save.")
        return None
    folder_name = params_to_string(asdict(config))
    ext = "pkl" if use_pickle else "pt"
    data_path = osp.join(path, folder_name, f"{filename}.{ext}")

    log(f"    Full path:\t{data_path}")
    if osp.exists(data_path) and not override:
        log(f"    File exists. Skipping save.")
        return data_path
    elif osp.exists(data_path) and override:
        log(f"    File exists but overrides.")
    else:
        os.makedirs(osp.join(path, folder_name), exist_ok=True)
        log(f"    File saves.")

    if use_pickle:
        with open(data_path, "wb") as f:
            pickle.dump(data, f)
    else:
        torch.save(data, data_path)
    return data_path
