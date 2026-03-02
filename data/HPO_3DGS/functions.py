"""Benchmark functions constructed from 3DGS HPO benchmark datasets.
Reference:
https://github.com/NYCU-RL-Bandits-Lab/BOFormer/blob/main/Environment/benchmark_functions.py
"For HPO-3DGS, there are about 68,000 data points related to 4 different objects: Lego, Materials,
Mic, Ship, and Chairs. Additionally, there are 64 different scenes involving chairs. While testing on
chairs, each episode is conducted with an individual scene"

Functions: 
- set_noise_level(level): define the perturbation level 
- set_NERF_scene(scene): define the scene
- NERF_synthetic_fnum_3(x): (psnr, size, npt)
- NERF_synthetic(x): (psnr, size)

Details: (11518, 9)
1. 11518 instances
2. 5 inputs (sh_degree, lambda_dssim, percent_dense, scale_pos_scale_lr, scale_feat_opacity_lr)
3. 3 outputs (psnr, size, npt)
4. 4 scenes: "lego", "materials", "mic", "ship"
5. Optimization:
    - maximize psnr, minimize size, minimize npt
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd


noise_level = 0.0


def set_noise_level(level: float) -> None:
    """Set the global noise level for output perturbation.

    The perturbation noise will be sampled from a uniform distribution U[-level, level].
    """
    global noise_level
    noise_level = level


dir = Path(__file__).resolve().parent


def set_NERF_scene(x, ins="6838708"):
    """Set glboal scene: 'lego', 'materials', 'mic', 'ship', 'chairs'"""
    global scene, instance, df_synthetic

    if x == "chairs":
        raise NotImplementedError("Chairs scene data missing!")
    else:
        print(f"Setting NERF synthetic scene to {x}...")
        scene = x


def get_NERF_scene():
    """Get global scene: 'lego', 'materials', 'mic', 'ship', 'chairs'"""
    global scene, instance
    return scene, instance


df_synthetic = pd.read_csv(f"{dir}/record_synthetic_v2.csv")

# clean data
df_synthetic = df_synthetic.drop("dataset", axis=1)
df_synthetic = df_synthetic.drop("ssim", axis=1)
df_synthetic = df_synthetic.drop("lpips", axis=1)


def NERF_synthetic_fnum_3(x):
    """Wrapper function for NERF synthetic dataset.

    Args:
        x: np.array of shape (d,) or (..., d)

    Returns:
        y: np.array of shape (3,) or (..., 3) corresponding to (psnr, size, npt)
    """
    f_list = [_NERF_synthetic_1, _NERF_synthetic_2, _NERF_synthetic_3]
    if x.ndim > 1:
        dim = x.shape[-1]
        x_flat = x.reshape(-1, dim)

        y = [[f(xi) for f in f_list] for xi in x_flat]  # n x [[(2,), (2, )]]
        y = np.array(y).reshape(*x.shape[:-1], -1)  # restore shape
    else:
        y = [f(x) for f in f_list]
        y = np.array(y)

    return y


def NERF_synthetic(x):
    """Wrapper function for NERF synthetic dataset.

    Args:
        x: np.array of shape (d,) or (..., d)

    Returns:
        y: np.array of shape (2,) or (..., 2) corresponding to (psnr, size)
    """
    f_list = [_NERF_synthetic_1, _NERF_synthetic_2]
    if x.ndim > 1:
        dim = x.shape[-1]
        x_flat = x.reshape(-1, dim)

        y = [[f(xi) for f in f_list] for xi in x_flat]  # n x [[(2,), (2, )]]
        y = np.array(y).reshape(*x.shape[:-1], -1)  # restore shape
    else:
        y = [f(x) for f in f_list]
        y = np.array(y)

    return y


def _NERF_synthetic_1(x):
    """Maximize psnr"""
    global scene
    df = df_synthetic
    if scene != "chairs":
        df = df[df["scene"] == scene]
    df = df.drop("scene", axis=1)
    df = df.drop("size", axis=1)
    df = df.drop("npt", axis=1)

    # normalize data
    df = (df - df.min()) / (df.max() - df.min())
    if scene == "chairs":
        df = df.fillna(0)

    # get all y
    y = df["psnr"]
    df = df.drop("psnr", axis=1)

    # find row
    df = df - x
    df["norm"] = df.apply(np.linalg.norm, axis=1)
    min_norm_index = df["norm"].idxmin()
    y = y.loc[min_norm_index]
    y = norm_add_noise(y, 1, 0)
    return y


def _NERF_synthetic_2(x):
    """Minimize size"""
    global scene
    df = df_synthetic
    if scene != "chairs":
        df = df[df["scene"] == scene]
    df = df.drop("scene", axis=1)
    df = df.drop("psnr", axis=1)
    df = df.drop("npt", axis=1)

    # normalize data
    df = (df - df.min()) / (df.max() - df.min())
    if scene == "chairs":
        df = df.fillna(0)
    # get all y
    y = -df["size"] + 1
    df = df.drop("size", axis=1)

    # find row
    df = df - x
    df["norm"] = df.apply(np.linalg.norm, axis=1)
    min_norm_index = df["norm"].idxmin()
    y = y.loc[min_norm_index]
    y = norm_add_noise(y, 1, 0)
    return y


def _NERF_synthetic_3(x):
    """Minimize npt"""
    global scene
    df = df_synthetic
    if scene != "chairs":
        df = df[df["scene"] == scene]
    df = df.drop("scene", axis=1)
    df = df.drop("size", axis=1)
    df = df.drop("psnr", axis=1)

    # normalize data
    df = (df - df.min()) / (df.max() - df.min())
    if scene == "chairs":
        df = df.fillna(0)
    # get all y
    y = -df["npt"] + 1
    df = df.drop("npt", axis=1)

    # find row
    df = df - x
    df["norm"] = df.apply(np.linalg.norm, axis=1)
    min_norm_index = df["norm"].idxmin()
    y = y.loc[min_norm_index]
    y = norm_add_noise(y, 1, 0)
    return y


def norm_add_noise(y, max_f, min_f, disable=False):
    if disable:
        return y
    global noise_level
    if y == np.inf or y > max_f:
        return (max_f * (1 + random.uniform(-noise_level, noise_level)) - min_f) / (
            max_f - min_f
        )
    elif np.isnan(y) or y < min_f:
        return (min_f * (1 + random.uniform(-noise_level, noise_level)) - min_f) / (
            max_f - min_f
        )
    else:
        return (y * (1 + random.uniform(-noise_level, noise_level)) - min_f) / (
            max_f - min_f
        )


if __name__ == "__main__":
    print(f"df_synthetic shape: {df_synthetic.shape}")
    print(f"columns: {df_synthetic.columns}")

    set_NERF_scene("ship")
    d = 5
    x_list = []
    for i in range(5):
        x = np.random.random((d,))
        x_list.append(x)
        f = NERF_synthetic_fnum_3(x)
        print(f"x:\n{x}\n f:\n{f}")
    x_batch = np.stack(x_list, axis=0)
    f_batch = NERF_synthetic_fnum_3(x_batch)
    print(f"x_batch:\n{x_batch}\nf_batch:\n{f_batch}")
