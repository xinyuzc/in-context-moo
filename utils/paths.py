from typing import Optional

import os
import os.path as osp

ROOT = ""
DATASETS_PATH = osp.join(ROOT, "datasets")
RESULT_PATH = osp.join(ROOT, "results")

SPLITS = ["train", "validation", "test"]


def get_split_dataset_path(split: str):
    """Get datasets path based on dataset split type: `DATASETS_PATH/split`"""
    if split in SPLITS:
        return os.path.join(DATASETS_PATH, split)
    else:
        raise ValueError(
            f"Unsupported split: {split}. "
            "Supported types are 'train', 'validation', and 'test'."
        )


def get_exp_path(model_name: str, expid: str) -> str:
    """Create experiment directory.
    Returns: {RESULT_PATH}/{model_name}/{expid}
    """
    ex_dir = osp.join(RESULT_PATH, model_name, expid)
    os.makedirs(ex_dir, exist_ok=True)
    return ex_dir


def _get_result_subpath(
    model_name, expid, task_type, filename_base: Optional[str] = None
):
    path = osp.join(model_name, expid, task_type)
    if filename_base:
        path = osp.join(path, filename_base)
    return path


def get_result_plot_path(
    model_name: str, expid: str, task_type: str, filename_base: Optional[str] = None
) -> str:
    """Get path for saving result plots:
    `RESULT_PATH/plots/model_name/expid/task_type/suffix`
    """
    subpath = _get_result_subpath(
        model_name=model_name,
        expid=expid,
        task_type=task_type,
        filename_base=filename_base,
    )

    return os.path.join(RESULT_PATH, "plots", subpath)


def get_result_data_path(
    model_name: str, expid: str, task_type: str, filename_base: Optional[str] = None
) -> str:
    """Get path for result data:
    `RESULT_PATH/data/model_name/exp_id/task_type/filename_base`
    """
    subpath = _get_result_subpath(
        model_name=model_name,
        expid=expid,
        task_type=task_type,
        filename_base=filename_base,
    )

    return os.path.join(RESULT_PATH, "data", subpath)


def get_filename_base(
    function_name: str,
    ckpt_name: str,
    suffix_segment: Optional[str] = None,
) -> str:
    """Construct a filename base for saving results:
    function_name/ckpt_name_base/[suffix_segment]
    """
    parts = [function_name]

    ckpt_name_base = ckpt_name.split(".")[0]
    parts.append(ckpt_name_base)

    if suffix_segment is not None:
        parts.append(suffix_segment)
    return "/".join(parts)
