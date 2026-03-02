"""Plot functions for optimization, prediction, and metric visualization."""

from typing import Optional

import torch
from torch import Tensor
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib as mpl

from model import TAMO
from model.layers import GMMPredictionHead
from utils.types import NestedFloatList

from data.function import TestFunction


CONTOURF_LEVELS = 20
# Context points
CONTEXT_COLOR = "black"
CONTEXT_SIZE = 30
CONTEXT_MARKER = "o"
CONTEXT_ZORDER = 2

# Target points
TARGET_COLOR = "fuchsia"
TARGET_SIZE = 30
TARGET_MARKER = "o"
TARGET_ZORDER = 2

# Confidence interval
CF_ZORDER = 0
CF_ALPHA = 0.3

# Predicted function
PREDICTION_LINESTYLE = "-"
PREDICTION_COLOR = "blue"
PREDICTION_LINE_WIDTH_25 = 2.5
PREDICTION_LINE_WIDTH_05 = 0.5
PREDICTION_ZORDER = 1
PREDICTION_ALPHA_10 = 1
PREDICTION_ALPHA_005 = 0.05

# Query points
QUERY_COLOR = "red"
QUERY_SIZE = 50
QUERY_MARKER = "o"
QUERY_ZORDER = 2
QUERY_ALPHA = 0.7

# Query point histogram
HIST_LINEWIDTH = 1.5
HIST_BINS = 10
HIST_ZORDER = QUERY_ZORDER - 1

# Query point sizes
MIN_SIZE = 1
MAX_SIZE = 100

# Acquisition function
ACQ_FUNC_COLOR = "red"
ACQ_FUNC_LINESTYLE = "-"
ACQ_FUNC_ALPHA = 0.5
ACQ_FUNC_ZORDER = 2

# Optimum
OPT_COLOR = "black"
OPT_MARKER = "*"
OPT_SIZE = 50
OPT_ZORDER = 2

# Function ground truth
GROUND_TRUTH_COLOR = "black"
GROUND_TRUTH_LINESTYLE = "dashed"
GROUND_TRUTH_ALPHA = 0.8
GROUND_TRUTH_ZORDER = 0

CMAP = "viridis"
OBS_CMAP = "plasma"

GRID_RES = 100
FIGSIZE_W_COEF = 8
FIGSIZE_H_COEF = 6
POINT_PLOT_STEP = 5

# Synthetic pareto front
NUM_PF_POINTS = 1000


def tnp(x: Tensor | np.ndarray) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, Tensor) else x


def set_matplotlib_params():
    """Set matplotlib params."""

    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rc("font", family="serif")
    mpl.rcParams.update(
        {
            "font.size": 18,
            "lines.linewidth": 2,
            "lines.markersize": 14,
            "axes.labelsize": 20,  # fontsize for x and y labels
            "axes.titlesize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "axes.linewidth": 2,
            "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
            "text.usetex": False,  # use LaTeX to write all text
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "legend.shadow": False,
            "legend.fancybox": False,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )


set_matplotlib_params()

def _gather_valid(
    b: int, x: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    if mask is None:
        return x[b]
    else:
        return x[b][:, mask[b]]


def _softmax(logits: np.ndarray):
    """logits [N, 1] -> probs [N, 1] via softmax."""
    if logits.ndim == 1:  # [N] -> [N, 1]
        logits = logits[:, None]

    logits_max = np.max(logits, axis=0, keepdims=True)
    logits = logits - logits_max  # For numerical stability
    logits_exp = np.exp(logits)

    probs = logits_exp / np.sum(logits_exp)
    return probs


def _plot_1d_optimization(
    ax: Axes,
    x: np.ndarray,  # [N, 1]
    y: np.ndarray,  # [N, 1]
    x_query: np.ndarray,  # [m, 1]
    y_query: np.ndarray,  # [m, 1]
    logits: Optional[np.ndarray] = None,  # [N, 1]
    x_bounds: Optional[np.ndarray] = None,
    grid_res: int = GRID_RES,
):
    """Plot query points on 1-D x and 1-D y.

    Args:
        ax: Axes to plot on
        x: Input locations, [N, 1]
        y: Function values, [N, 1]
        x_query: Query input locations, [m, 1]
        y_query: Query function values, [m, 1]
        logits: Optional acquisition function values at x, [N, 1]
        x_bounds: Optional bounds of input, [1, 2]
    """
    # Larger markers for later queries
    m = x_query.shape[0]
    N = x.shape[0]

    sizes = np.linspace(MIN_SIZE, MAX_SIZE, m)

    x_query = x_query.squeeze(-1)  # [m]
    y_query = y_query.squeeze(-1)  # [m]

    # Sort x and y: [N]
    indices_sorted = np.argsort(x, axis=0)  # [N, 1]
    x_sorted = np.take_along_axis(x, indices_sorted, axis=0).squeeze(-1)
    y_sorted = np.take_along_axis(y, indices_sorted, axis=0).squeeze(-1)

    # Find global minimum
    indices_min = np.argmin(y_sorted)  # [1]
    x_min = x_sorted[indices_min]  # [1]
    y_min = y_sorted[indices_min]  # [1]

    if x_bounds is not None:
        # Use the true bounds if provided - better ground truth visualization
        x_bounds_min = x_bounds[0, 0].item()
        x_bounds_max = x_bounds[0, 1].item()
    else:
        x_bounds_min = x_sorted.min().item()
        x_bounds_max = x_sorted.max().item()

    # Interpolate y values for better visualization
    x_interp = np.linspace(x_bounds_min, x_bounds_max, grid_res)  # [grid_res]
    y_interp = np.interp(x_interp, x_sorted, y_sorted)  # [grid_res]

    # Plot ground truth
    ax.plot(
        x_interp,
        y_interp,
        color=GROUND_TRUTH_COLOR,
        alpha=GROUND_TRUTH_ALPHA,
        linestyle=GROUND_TRUTH_LINESTYLE,
        zorder=GROUND_TRUTH_ZORDER,
    )

    # Plot minium
    ax.scatter(
        x_min,
        y_min,
        color=OPT_COLOR,
        marker=OPT_MARKER,
        s=OPT_SIZE,
        zorder=OPT_ZORDER,
        facecolors="none",
    )

    # Plot queries
    ax.scatter(
        x_query,
        y_query,
        color=QUERY_COLOR,
        s=sizes,
        marker=QUERY_MARKER,
        alpha=QUERY_ALPHA,
        zorder=QUERY_ZORDER,
    )

    # Add histogram for query points
    divider = make_axes_locatable(ax)
    ax_hist = divider.append_axes("top", size="15%", pad=0.1, sharex=ax)
    plt.setp(ax_hist.get_xticklabels(), visible=False)

    ax_hist.hist(
        x_query,
        histtype="step",
        linewidth=HIST_LINEWIDTH,
        bins=HIST_BINS,
        color="black",
        orientation="vertical",
        zorder=HIST_ZORDER,
    )

    ax_hist.set_ylabel("Freq(x)")

    # Add softmax acquisition function if logits are provided
    if logits is not None:
        # Sort logits: [N, 1]
        assert logits.shape == (N, 1)
        logits_sorted = np.take_along_axis(logits, indices_sorted, axis=0)
        probs = _softmax(logits_sorted)  # [N, 1]

        # Scale probabilities to fit in the y-axis
        ax_hist2 = ax_hist.twinx()
        plt.setp(ax_hist2.get_xticklabels(), visible=False)

        ax_hist2.plot(
            x_sorted,
            probs,  # no scaling here
            color=ACQ_FUNC_COLOR,
            alpha=ACQ_FUNC_ALPHA,
            linestyle=ACQ_FUNC_LINESTYLE,
            zorder=ACQ_FUNC_ZORDER,
        )
        ax_hist2.set_ylabel("Prob")

    ax.grid(True, alpha=0.5, linestyle="dotted")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _plot_2d_optimization(
    ax: Axes,
    fig: Figure,
    x: np.ndarray,  # [N, 2]
    y: np.ndarray,  # [N, 1]
    x_query: np.ndarray,  # [m, 2]
    y_query: np.ndarray,  # [m, 1]
    grid_res: int = GRID_RES,
    logits: Optional[np.ndarray] = None,  # [N, 2]
    x_bounds: Optional[np.ndarray] = None,
):
    """Optimization on 2-D x and 1-D y.

    Args:
        ax: Axes to plot on
        x: Input data of shape [N, 2]
        y: Output data of shape [N, 1]
        x_query: Query input data of shape [m, 2]
        y_query: Query output data of shape [m, 1]
        grid_res: Resolution for grid interpolation in 2D plots
        logits: Optional acquisition function values at x, shape [N, 2]
        x_bounds: Optional bounds for x
    """
    m = x_query.shape[0]
    N = x.shape[0]

    if x_bounds is not None:
        # Use the true bounds if provided - better ground truth visualization
        x_bounds_min = x_bounds[:, 0]
        x_bounds_max = x_bounds[:, 1]
    else:
        x_bounds_min = x.min(axis=0)
        x_bounds_max = x.max(axis=0)

    # Larger markers for later queries
    sizes = np.linspace(MIN_SIZE, MAX_SIZE, m)

    y = y.squeeze(-1)  # [N]
    y_query = y_query.squeeze(-1)  # [m]

    # Create grid for x and intepolate y values: [grid_res, grid_res]
    xi = np.linspace(x_bounds_min[0], x_bounds_max[0], grid_res)
    xj = np.linspace(x_bounds_min[1], x_bounds_max[1], grid_res)
    xi_grid, xj_grid = np.meshgrid(xi, xj)

    y_grid = griddata(x, y, (xi_grid, xj_grid), method="cubic")

    # Plot interpolated data
    mappable = ax.contourf(xi_grid, xj_grid, y_grid, levels=20, cmap=CMAP)
    fig.colorbar(mappable, ax=ax)

    # Plot query points
    ax.scatter(
        x_query[:, 0],
        x_query[:, 1],
        c=y_query,
        cmap=CMAP,
        s=sizes,
        marker=QUERY_MARKER,
        zorder=QUERY_ZORDER,
        alpha=QUERY_ALPHA,
        edgecolor="white",
    )

    # Add histograms for query points along x and y axes
    # Add new axes to the top and right
    divider = make_axes_locatable(ax)

    ax_hist_x = divider.append_axes("top", size="15%", pad=0.1, sharex=ax)
    ax_hist_y = divider.append_axes("right", size="15%", pad=0.1, sharey=ax)

    # Hide tick labels on histogram axes
    plt.setp(ax_hist_x.get_xticklabels(), visible=False)
    plt.setp(ax_hist_y.get_yticklabels(), visible=False)

    ax_hist_x.tick_params(axis="x", which="both", bottom=False, top=False)
    ax_hist_y.tick_params(axis="y", which="both", left=False, right=False)

    # Histogram for x1 (along x-axis)
    ax_hist_x.hist(
        x_query[:, 0],
        histtype="step",
        linewidth=HIST_LINEWIDTH,
        bins=HIST_BINS,
        color="blue",
        orientation="vertical",
        zorder=HIST_ZORDER,
    )

    ax_hist_x.set_ylabel("Freq(x1)")

    # Histogram for x2 (along y-axis)
    ax_hist_y.hist(
        x_query[:, 1],
        histtype="step",
        linewidth=HIST_LINEWIDTH,
        bins=HIST_BINS,
        color="red",
        orientation="horizontal",
        zorder=HIST_ZORDER,
    )

    ax_hist_y.set_xlabel("Freq(x2)")

    if logits is not None:
        assert logits.shape == (N, 2)

        # Sort x0, x1, and logits accordingly
        x0_sort_indices = np.argsort(x[:, 0])
        x1_sort_indices = np.argsort(x[:, 1])

        x0_sorted = np.take_along_axis(x[:, 0], x0_sort_indices, axis=0)
        x1_sorted = np.take_along_axis(x[:, 1], x1_sort_indices, axis=0)

        logit0_sorted = np.take_along_axis(
            logits[:, 0], x0_sort_indices, axis=0
        )  # [N, 1]
        logit1_sorted = np.take_along_axis(
            logits[:, 1], x1_sort_indices, axis=0
        )  # [N, 1]

        probs0 = _softmax(logit0_sorted).squeeze()
        probs1 = _softmax(logit1_sorted).squeeze()

        # Interpolate logit
        probs0_interp = np.interp(xi, x0_sorted, probs0)
        probs1_interp = np.interp(xj, x1_sorted, probs1)

        # Plot acquisition function
        ax_hist_x_2 = ax_hist_x.twinx()
        plt.setp(ax_hist_x_2.get_xticklabels(), visible=False)

        ax_hist_x_2.plot(
            xi,
            probs0_interp,
            color="blue",
            alpha=ACQ_FUNC_ALPHA,
            linestyle="-.",
            zorder=ACQ_FUNC_ZORDER,
        )
        ax_hist_x_2.set_ylabel("Prob")

        ax_hist_y_2 = ax_hist_y.twiny()
        plt.setp(ax_hist_y_2.get_xticklabels(), visible=False)

        ax_hist_y_2.plot(
            probs1_interp,
            xj,
            color="red",
            alpha=ACQ_FUNC_ALPHA,
            linestyle="-.",
            zorder=ACQ_FUNC_ZORDER,
        )
        ax_hist_y_2.set_xlabel("Prob")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def plot_optimization(
    x: Tensor,  # [B, N, dx]
    y: Tensor,  # [B, N, dy]
    x_query: Tensor,  # [B, m, dx]
    y_query: Tensor,  # [B, m, dy]
    x_mask: Optional[Tensor] = None,  # [B, dx]
    y_mask: Optional[Tensor] = None,  # [B, dy]
    x_bounds: Optional[NestedFloatList] = None,
    grid_res: int = GRID_RES,
    logits: Optional[Tensor] = None,  # [B, d, dx]
    **kwargs,
) -> Figure:
    """Plot query points for optimization on 1- or 2-D x.

    Args:
        x: Input data of shape [B, N, dx_max]
        y: Output data of shape [B, N, dy_max]
        x_query: Query input data of shape [B, m, dx_max]
        y_query: Query output data of shape [B, m, dy_max]
        x_mask: Mask for x data of shape [B, dx_max]
        y_mask: Mask for y data of shape [B, dy_max]
        x_bounds: Optional bounds for x, dx_max x [2]
        grid_res: Resolution for grid interpolation in 2D plots
        logits: Optional acquisition function values at x, shape [B, d, dx_max]
    """
    B, _, dx_max = x.shape
    _, _, dy_max = y.shape

    m = x_query.shape[1]

    assert x_query.shape == (B, m, dx_max)
    assert y_query.shape == (B, m, dy_max)

    # Set default masks if not provided
    if x_mask is None:
        x_mask = torch.ones((B, dx_max), dtype=torch.bool)
    if y_mask is None:
        y_mask = torch.ones((B, dy_max), dtype=torch.bool)

    assert x_mask.shape == (B, dx_max)
    assert y_mask.shape == (B, dy_max)

    # Move data to CPU and convert to numpy
    x, y, x_query, y_query, x_mask, y_mask = (
        tnp(t) for t in (x, y, x_query, y_query, x_mask, y_mask)
    )

    # Convert x_bounds to numpy array of shape [dx_max, 2]
    if x_bounds is not None:
        if isinstance(x_bounds, list):
            x_bounds_np = np.array(x_bounds, dtype=np.float32)
        elif isinstance(x_bounds, torch.Tensor):
            x_bounds_np = x_bounds.cpu().numpy()

        assert x_bounds_np.shape == (dx_max, 2)
    else:
        x_bounds_np = None

    if logits is not None:
        assert logits.shape[0] == B
        logits = tnp(logits)

    # Create a figure for [B x dy_max] subplots
    fig, axes = plt.subplots(
        B,
        dy_max,
        figsize=(FIGSIZE_W_COEF * dy_max, FIGSIZE_H_COEF * B),
        squeeze=False,
    )

    logits_valid = None
    for b in range(B):
        x_valid = _gather_valid(b, x, x_mask)  # [N, dx_valid_b]
        xq_valid = _gather_valid(b, x_query, x_mask)  # [m, dx_valid_b]

        if x_bounds_np is not None:
            x_bounds_np_valid = x_bounds_np[x_mask[b], :]  # [dx_valid_b, 2]
            assert x_bounds_np_valid.ndim == 2, f"{x_bounds_np_valid}"
        else:
            x_bounds_np_valid = None

        if logits is not None:
            logits_valid = _gather_valid(b, logits, x_mask)

        yb = y[b]
        yb_query = y_query[b]  # [m, dy_max]
        # Plot for each valid y
        for i in range(dy_max):
            ax = axes[b, i]

            # Skip invalid y
            if i >= y[b].shape[1] or not y_mask[b, i]:
                ax.axis("off")
                continue

            y_valid = yb[:, [i]]
            yq_valid = yb_query[:, [i]]

            if x_valid.shape[1] == 1:
                # 1D inputs
                _plot_1d_optimization(
                    ax=ax,
                    x=x_valid,  # [N, 1]
                    y=y_valid,  # [N, 1]
                    x_query=xq_valid,  # [m, 1]
                    y_query=yq_valid,  # [m, 1]
                    logits=logits_valid,
                    x_bounds=x_bounds_np_valid,
                )
            elif x_valid.shape[1] == 2:
                # 2D inputs
                _plot_2d_optimization(
                    ax=ax,
                    fig=fig,
                    x=x_valid,  # [N, 2]
                    y=y_valid,  # [N, 1]
                    x_query=xq_valid,  # [m, 2]
                    y_query=yq_valid,  # [m, 1]
                    grid_res=grid_res,
                    logits=logits_valid,
                    x_bounds=x_bounds_np_valid,
                )
            else:
                print(f"Unsupported input dimension to plot: {x_valid.shape[1]}")
                fig = None

    plt.tight_layout()
    return fig


def _plot_1d_prediction(
    mean: np.ndarray,  # [N, 1]
    std: np.ndarray,  # [N, 1]
    x: np.ndarray,  # [N, 1]
    y: np.ndarray,  # [N, 1]
    xctx: np.ndarray,  # [N, 1]
    yctx: np.ndarray,  # [N, 1]
    ax_true: Axes,  # Ground truth axis
    ax_pred: Axes,  # Prediction axis
    x_bounds: Optional[np.ndarray] = None,  # [2, 1]
    grid_res: int = GRID_RES,
    is_y_observed: Optional[np.ndarray] = None,  # [nc, 1]
    num_samples: int= 10
):
    """Plot prediction for 1-D x and 1-D y."""
    if is_y_observed is not None:
        is_y_observed = is_y_observed.squeeze(-1)

    # Sort data: [N]
    indices_sorted = np.argsort(x, axis=0)  # [N, 1]

    x_sorted = np.take_along_axis(x, indices_sorted, axis=0).squeeze(-1)
    y_sorted = np.take_along_axis(y, indices_sorted, axis=0).squeeze(-1)

    mean_sorted = np.take_along_axis(mean, indices_sorted, axis=0).squeeze(-1)
    std_sorted = np.take_along_axis(std, indices_sorted, axis=0).squeeze(-1)

    xctx = xctx.squeeze(-1)  # [N]
    yctx = yctx.squeeze(-1)  # [N]

    if x_bounds is not None:
        # Use the true bounds if provided - better ground truth visualization
        x_bounds_min = x_bounds[0, 0].item()
        x_bounds_max = x_bounds[0, 1].item()
    else:
        x_bounds_min = x_sorted.min().item()
        x_bounds_max = x_sorted.max().item()

    # Interpolate y values for better visualization
    x_interp = np.linspace(x_bounds_min, x_bounds_max, grid_res)  # [grid_res]
    y_interp = np.interp(x_interp, x_sorted, y_sorted)  # [grid_res]

    # ax_true: Plot ground truth with context points
    ax_true.plot(
        x_interp,
        y_interp,
        color=GROUND_TRUTH_COLOR,
        alpha=GROUND_TRUTH_ALPHA,
        linestyle=GROUND_TRUTH_LINESTYLE,
        zorder=GROUND_TRUTH_ZORDER,
    )

    if is_y_observed is None:
        ax_true.scatter(
            xctx,
            yctx,
            color=CONTEXT_COLOR,
            s=CONTEXT_SIZE,
            marker=CONTEXT_MARKER,
            zorder=CONTEXT_ZORDER,
        )
    else:
        xctx_plot = xctx[is_y_observed]
        yctx_plot = yctx[is_y_observed]

        ax_true.scatter(
            xctx_plot,
            yctx_plot,
            color=CONTEXT_COLOR,
            s=CONTEXT_SIZE,
            marker=CONTEXT_MARKER,
            zorder=CONTEXT_ZORDER,
        )

    # ax_pred: Plot prediction with context points
    ax_pred.plot(
        x_sorted,
        mean_sorted,
        color=PREDICTION_COLOR,
        alpha=PREDICTION_ALPHA_10,
        linestyle=PREDICTION_LINESTYLE,
        linewidth=PREDICTION_LINE_WIDTH_25,
        zorder=PREDICTION_ZORDER,
    )

    cf = std_sorted * 1.96  # 95% confidence interval

    # Interpolate onto a fine grid for a smoother confidence band
    from scipy.interpolate import make_interp_spline

    _n_smooth = 300
    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), _n_smooth)
    mean_smooth = make_interp_spline(x_sorted, mean_sorted, k=3)(x_smooth)
    cf_smooth = make_interp_spline(x_sorted, cf, k=3)(x_smooth)

    ax_pred.fill_between(
        x_smooth,
        mean_smooth - cf_smooth,
        mean_smooth + cf_smooth,
        color=PREDICTION_COLOR,
        alpha=CF_ALPHA,
        zorder=CF_ZORDER,
    )

    if is_y_observed is None:
        ax_pred.scatter(
            xctx,
            yctx,
            color=CONTEXT_COLOR,
            s=CONTEXT_SIZE,
            marker=CONTEXT_MARKER,
            zorder=CONTEXT_ZORDER,
        )
    else:
        ax_pred.scatter(
            xctx_plot,
            yctx_plot,
            color=CONTEXT_COLOR,
            s=CONTEXT_SIZE,
            marker=CONTEXT_MARKER,
            zorder=CONTEXT_ZORDER,
        )

    # Also plot ground truth
    ax_pred.plot(
        x_sorted,
        y_sorted,
        color=GROUND_TRUTH_COLOR,
        alpha=GROUND_TRUTH_ALPHA,
        linestyle=GROUND_TRUTH_LINESTYLE,
        zorder=GROUND_TRUTH_ZORDER,
    )

    ax_true.grid(True, alpha=0.5, linestyle="dotted")
    ax_pred.grid(True, alpha=0.5, linestyle="dotted")

    ax_true.set_title(f"Ground Truth")
    ax_pred.set_title(f"Prediction")

    ax_true.set_xlabel("x")
    ax_true.set_ylabel("y")

    ax_pred.set_xlabel("x")
    ax_pred.set_ylabel("y")


def _plot_2d_prediction(
    mean: np.ndarray,  # [N, 1]
    std: np.ndarray,  # [N, 1]
    x: np.ndarray,  # [N, 2]
    y: np.ndarray,  # [N, 1]
    xctx: np.ndarray,  # [nc, 2]
    yctx: np.ndarray,  # [nc, 1]
    ax_true: Axes,
    ax_pred: Axes,
    fig: Figure,
    is_y_observed=None,  # [nc, 1]
    x_bounds: Optional[np.ndarray] = None,  # [2, 2]
    grid_res: int = GRID_RES,
    plot_mean: bool = True,
    plot_order: bool = False,
):
    def _reduce_y_dim(tensor):
        if tensor is None:
            return None
        if tensor.ndim == 1:
            return tensor
        return tensor.squeeze(-1)

    y = _reduce_y_dim(y)
    yctx = _reduce_y_dim(yctx)
    mean = _reduce_y_dim(mean)
    std = _reduce_y_dim(std)
    is_y_observed = _reduce_y_dim(is_y_observed)

    y_pred_plot = mean if plot_mean else std

    x_bounds_min = x.min(axis=0) if x_bounds is None else x_bounds[:, 0]
    x_bounds_max = x.max(axis=0) if x_bounds is None else x_bounds[:, 1]

    # Create grid: [grid_res, grid_res]
    xi = np.linspace(x_bounds_min[0] - 0.5, x_bounds_max[0] + 0.5, grid_res)
    xj = np.linspace(x_bounds_min[1] - 0.5, x_bounds_max[1] + 0.5, grid_res)
    xi, xj = np.meshgrid(xi, xj)

    # Interpolate y values on grid
    y_true_grid = griddata(x, y, (xi, xj), method="cubic")
    y_pred_grid = griddata(x, y_pred_plot, (xi, xj), method="cubic")

    # Plot interpolated data
    mappable_true = ax_true.contourf(
        xi, xj, y_true_grid, levels=CONTOURF_LEVELS, cmap=CMAP
    )
    mappable_pred = ax_pred.contourf(
        xi, xj, y_pred_grid, levels=CONTOURF_LEVELS, cmap=CMAP
    )
    fig.colorbar(mappable_pred, ax=ax_pred)
    fig.colorbar(mappable_true, ax=ax_true)

    def _get_markersizes(points, plot_order):
        if plot_order:
            s = np.linspace(MIN_SIZE, MAX_SIZE, len(points))
        else:
            s = CONTEXT_SIZE
        return s

    # Plot context points
    x_ctx_plot = xctx if is_y_observed is None else xctx[is_y_observed, :]
    y_ctx_plot = yctx if is_y_observed is None else yctx[is_y_observed]
    ax_true.scatter(
        x_ctx_plot[:, 0],
        x_ctx_plot[:, 1],
        c=y_ctx_plot,
        s=_get_markersizes(x_ctx_plot, plot_order),
        marker=CONTEXT_MARKER,
        zorder=CONTEXT_ZORDER,
        edgecolor="white",
    )
    ax_pred.scatter(
        x_ctx_plot[:, 0],
        x_ctx_plot[:, 1],
        c=y_ctx_plot,
        s=_get_markersizes(x_ctx_plot, plot_order),
        marker=CONTEXT_MARKER,
        zorder=CONTEXT_ZORDER,
        edgecolor="white",
    )


def plot_prediction(
    mean: Tensor,
    std: Tensor,
    x: Tensor,
    y: Tensor,
    xctx: Tensor,
    yctx: Tensor,
    x_mask: Tensor,
    y_mask: Tensor,
    y_mask_history: Optional[Tensor] = None,  # [nc, dy_max]
    x_bounds: Optional[NestedFloatList] = None,
    grid_res: int = GRID_RES,
    plot_mean: bool = True,
    plot_order: bool = False,
    **kwargs,
) -> Figure:
    """Plot predictions for 1 dimensional or 2 dimensional x.

    Args:
        mean: [B, N, dy_max]
        std: [B, N, dy_max]
        x: [B, N, dx_max]
        y: [B, N, dy_max]
        xctx: [B, nc, dx_max]
        yctx: [B, nc, dy_max]
        x_mask: [B, dx_max] or [dx_max]
        y_mask: [B, dy_max] or [dy_max]
        y_mask_history (optional): [nc, dy_max], mask for observed y in context points
        x_bounds: Optional input bounds, dx_max x [2]. The mins and maxs will be used if not provided
        grid_res: Resolution for grid interpolation in 2D plots
    """
    B, N, dx_max = x.shape
    _, _, dy_max = y.shape
    nc = xctx.shape[1]

    assert mean.shape == (B, N, dy_max), f"{mean.shape} != ({B}, {N}, {dy_max})"
    assert std.shape == (B, N, dy_max), f"{std.shape} != ({B}, {N}, {dy_max})"
    assert xctx.shape == (B, nc, dx_max) and yctx.shape == (B, nc, dy_max)
    assert x_mask.shape == (B, dx_max) or x_mask.shape == (dx_max,)
    assert y_mask.shape == (B, dy_max) or y_mask.shape == (dy_max,)
    
    x_mask = x_mask.view(B, -1)
    y_mask = y_mask.view(B, -1)

    mean, std, x, y, xctx, yctx, x_mask, y_mask = (
        tnp(t) for t in (mean, std, x, y, xctx, yctx, x_mask, y_mask)
    )

    if y_mask_history is not None:
        assert y_mask_history.shape == (nc, dy_max)
        y_mask_history = y_mask_history.expand(B, nc, dy_max)
        y_mask_history = tnp(y_mask_history)

    x_bounds_np = None
    if x_bounds is not None:
        x_bounds_np = np.array(x_bounds, dtype=np.float32)
        assert x_bounds_np.shape == (dx_max, 2), f"{x_bounds_np.shape} != ({dx_max}, 2)"

    # Figure: [(2 * B), dy_max]
    nrows = 2 * B
    ncols = dy_max
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(FIGSIZE_W_COEF * ncols, FIGSIZE_H_COEF * nrows),
        squeeze=False,
    )

    for b in range(B):
        # Inputs: [N, dx_valid_b]
        x_valid = _gather_valid(b, x, x_mask)
        xc_valid = _gather_valid(b, xctx, x_mask)

        # Outputs: [N, dy_max]
        yb = _gather_valid(b, y)
        yctxb = _gather_valid(b, yctx)
        gmm_mean_b = _gather_valid(b, mean)
        gmm_std_b = _gather_valid(b, std)

        x_bounds_valid = None
        if x_bounds_np is not None:
            x_bounds_valid = x_bounds_np[x_mask[b], :]
            assert x_bounds_valid.ndim == 2, f"{x_bounds_valid}"

        has_plot_ylabel = False
        for i in range(dy_max):
            if i >= y.shape[-1] or not y_mask[b, i]:
                axes[2 * b, i].axis("off")
                axes[2 * b + 1, i].axis("off")
                continue
            else:
                ax_true = axes[2 * b, i]
                ax_pred = axes[2 * b + 1, i]

            # Take along valid dim
            y_valid = yb[:, [i]]  # [N, 1]
            yc_valid = yctxb[:, [i]]  # [N, 1]
            gmm_mean_valid = gmm_mean_b[:, [i]]  # [N, 1]
            gmm_std_valid = gmm_std_b[:, [i]]  # [N, 1]

            is_yi_observed = None
            if y_mask_history is not None:
                # NOTE [N, 1]
                is_yi_observed = y_mask_history[b]
                is_yi_observed = is_yi_observed[..., [i]]

            if x_valid.shape[1] == 1:
                _plot_1d_prediction(
                    mean=gmm_mean_valid,
                    std=gmm_std_valid,
                    x=x_valid,
                    y=y_valid,
                    xctx=xc_valid,
                    yctx=yc_valid,
                    ax_true=ax_true,
                    ax_pred=ax_pred,
                    x_bounds=x_bounds_valid,
                    is_y_observed=is_yi_observed,
                )
            elif x_valid.shape[1] == 2:
                _plot_2d_prediction(
                    mean=gmm_mean_valid,
                    std=gmm_std_valid,
                    x=x_valid,
                    y=y_valid,
                    xctx=xc_valid,
                    yctx=yc_valid,
                    ax_true=ax_true,
                    ax_pred=ax_pred,
                    fig=fig,
                    grid_res=grid_res,
                    x_bounds=x_bounds_valid,
                    is_y_observed=is_yi_observed,
                    plot_mean=plot_mean,
                    plot_order=plot_order,
                )

                ax_true.set_title(f"Objective {i+1}")
                if not has_plot_ylabel:
                    ax_true.set_ylabel(f"Ground Truth")
                    ax_pred.set_ylabel(f"Prediction")
            else:
                print(f"Unsupported input dimension to plot: {x_valid.shape[1]}")
                fig = None

    plt.tight_layout()
    return fig


def plot_1d(
    title: str,
    y_vals: Tensor | np.ndarray,
    x_vals: Optional[Tensor | np.ndarray] = None,
    ylabel: str = "Value",
    xlabel: str = "Iteration",
    point_plot_step: int = POINT_PLOT_STEP,
    color: str = "blue",
    marker: str = "o",
    label: str = "",
    alpha: float = 0.3,
    fontsize: int = 8,
    logscale: bool = False,
    show_final_mean: bool = True,
    fig=None,
):
    """Plot 1-D data.

    Args:
        y_vals: 1-D values to plot, shape [B, N]
        x_vals: 1-D x values, shape [N] or None for default range
    """
    if y_vals.ndim == 1:
        y_vals = y_vals.unsqueeze(0)  # [1, N]

    N = y_vals.shape[1]
    x_vals = np.array(range(N)) if x_vals is None else x_vals
    assert x_vals.shape == (N,)

    x_vals = tnp(x_vals)
    y_vals = tnp(y_vals)
    y_means = y_vals.mean(axis=0)  # [N]
    y_stds = y_vals.std(axis=0)  # [N]
    y_cf = y_stds * 1.96 / np.sqrt(N)

    fig = plt.figure(figsize=(10, 5)) if fig is None else fig
    plt.plot(x_vals, y_means, "-", color=color)
    plt.fill_between(
        x_vals,
        y_means - y_cf,
        y_means + y_cf,
        color=color,
        alpha=alpha,
    )
    plt.plot(
        x_vals[::point_plot_step],
        y_means[::point_plot_step],
        marker,
        color=color,
        markersize=5,
        label=label,
    )
    if show_final_mean:
        plt.text(
            plt.xlim()[1] + 0.15 * (plt.xlim()[1] - plt.xlim()[0]),
            (y_means - y_cf)[-1],
            f"{y_means[-1]:.4f} ± {y_cf[-1]:.4f}",
            fontsize=fontsize,
            verticalalignment="bottom",
            horizontalalignment="right",
        )

    plt.grid(True, alpha=0.5, linestyle="dotted")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logscale:
        plt.yscale("log")
    plt.title(title)
    plt.legend()

    return fig


def plot_acq_values(q_chunk: Tensor, acq_values: Tensor, grid_res: int = GRID_RES):
    """q_chunk: [d, dx] or [B, n, d, dx], acq_values: [B, n, d]"""
    if acq_values is None:
        return None
    b, n, d = acq_values.shape
    assert n == 1
    if q_chunk.ndim == 2:
        x = q_chunk.unsqueeze(0).unsqueeze(0).expand(b, 1, -1, -1)  # [B, 1, d, dx]
    x = x.squeeze(1)
    acq_values = acq_values.squeeze(1)  # [B, d]

    dx = x.shape[2]
    x, acq_values = tnp(x), tnp(acq_values)

    if dx == 1:
        ncols = min(4, b)
        nrows = int(b // ncols) + 1 if b % ncols != 0 else int(b // ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(FIGSIZE_W_COEF * ncols, FIGSIZE_H_COEF * nrows),
            squeeze=False,
        )
        for i in range(b):
            ax = axes[i // ncols, i % ncols]
            x_b = x[i].squeeze(-1)
            xb_sorted = np.sort(x_b, axis=0)
            acq_sorted = acq_values[i][np.argsort(x_b, axis=0)]
            acq_sorted = _softmax(acq_sorted).squeeze(-1)
            ax.plot(xb_sorted, acq_sorted, "-", color="blue")
            ax.set_title(f"Acq values (batch {i})")
    elif dx == 2:
        ncols = min(4, b)
        nrows = int(b // ncols) + 1 if b % ncols != 0 else int(b // ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(FIGSIZE_W_COEF * ncols, FIGSIZE_H_COEF * nrows),
            squeeze=False,
        )
        # Intepolate
        for i in range(b):
            ax = axes[i // ncols, i % ncols]
            x_b = x[i]
            acq_b = _softmax(acq_values[i]).squeeze(-1)
            xb_min = x_b.min(axis=0)  # [2]
            xb_max = x_b.max(axis=0)

            xi = np.linspace(xb_min[0], xb_max[0], grid_res)
            xj = np.linspace(xb_min[1], xb_max[1], grid_res)
            xi_grid, xj_grid = np.meshgrid(xi, xj)
            y_grid = griddata(x_b, acq_b, (xi_grid, xj_grid), method="cubic")

            # Plot interpolated data
            mappable = ax.contourf(xi_grid, xj_grid, y_grid, levels=20, cmap=CMAP)
            fig.colorbar(mappable, ax=ax)
            ax.set_title(f"Acq values (batch {i})")
    else:
        return None
    plt.tight_layout()
    return fig


def plot_prediction_batch(
    model: TAMO,
    nc: int,
    x: Tensor,
    y: Tensor,
    x_mask: Tensor,
    y_mask: Tensor,
    y_mask_tar: Optional[Tensor] = None,
    xc: Optional[Tensor] = None,
    yc: Optional[Tensor] = None,
    read_cache: bool = False,
    y_mask_history: Optional[Tensor] = None,  # [nc, dy_max]
    plot_mean: bool = True,
    plot_order: bool = False,
) -> Figure:
    """Plot predictions of batches.

    Args:
        ...
        y_mask: [B, dy] boolean tensor for context points
        y_mask_tar: [B, dy] boolean tensor for target points
        xc: [B, nc, dx] context inputs, if None, use first nc points from x
        yc: [B, nc, dy] context outputs, if None, use first nc points from y
        ...
        y_mask_history (optional): [nc, dy_max] boolean tensor for history context points

    Returns:
        fig: Figure object
    """
    x_plot, y_plot = x.clone(), y.clone()

    if xc is None or yc is None:
        xc = x_plot[:, :nc]
        yc = y_plot[:, :nc]

    if y_mask_tar is None:
        y_mask_tar = y_mask  # Use y_mask for target if not provided

    out = model.predict(
        x_ctx=xc,
        y_ctx=yc,
        x_tar=x_plot,
        x_mask=x_mask,
        y_mask=y_mask,
        observed_target_y_mask=y_mask_tar,
        read_cache=read_cache,
    )

    mean = GMMPredictionHead.expected_value(out)
    std = GMMPredictionHead.std(out)

    fig = plot_prediction(
        mean=mean,
        std=std,
        x=x_plot,
        y=y_plot,
        xctx=xc,
        yctx=yc,
        x_mask=x_mask,
        y_mask=y_mask,
        y_mask_history=y_mask_history,
        plot_mean=plot_mean,
        plot_order=plot_order,
    )

    del x_plot, y_plot, out, mean, std
    return fig


def plot_optimization_batch(
    test_function: TestFunction,
    x_query: Tensor,
    y_query: Tensor,
    input_range_list: Optional[NestedFloatList] = None,
    grid_res: int = 2000,
    seed: int = 0,
) -> Figure:
    """Plot optimization results for a batch."""
    b, _, x_dim = x_query.shape
    _, _, y_dim = y_query.shape

    # Sanity check
    assert x_dim == test_function.x_dim
    assert y_dim == test_function.y_dim

    # Sample a grid from function
    input_bounds = input_range_list or test_function.x_bounds
    x, y, _, _ = test_function.sample(
        input_bounds=input_bounds,
        num_subspace_points=grid_res,
        batch_size=b,
        use_grid_sampling=True,
        use_factorized_policy=False,
        device=x_query.device,
        seed=seed,
    )

    fig = plot_optimization(x=x, y=y, x_query=x_query, y_query=y_query)
    return fig
