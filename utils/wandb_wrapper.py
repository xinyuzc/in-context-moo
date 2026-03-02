import os
import os.path as osp

from omegaconf import DictConfig
import wandb
from wandb.sdk.wandb_run import Run
from dotenv import load_dotenv
import flatten_dict

ROOT = ""
WANDB_PATH = osp.join(ROOT, "wandb")


def init(config: DictConfig, dir: str = WANDB_PATH, **kwargs) -> Run:
    """Init wandb run.

    Args:
        config: saving inputs to wandb.
        dir: metadata saving directory.
        **kwargs: additional arguments for `wandb.init()`.

    Returns: wandb run.
    """
    # Load environment vars
    load_dotenv(dotenv_path=os.path.join(ROOT, ".env"))

    if "WANDB_API_KEY" not in os.environ:
        raise ValueError("Please add `WANDB_API_KEY` to the file `.env`")

    if "project" not in kwargs:
        raise ValueError("Please specify `project`in the file `.env`")

    # Make wandb metadata saving dir
    os.makedirs(dir, exist_ok=True)

    # Start a new wandb run
    config = flatten_dict.flatten(config, reducer="path")
    run = wandb.init(config=config, dir=dir, **kwargs)

    return run


def save_artifact(
    run: Run, local_path: str, name: str, type: str, log: callable = print
):
    """Saving an artifact to wandb."""
    try:
        artifact = wandb.Artifact(name=name, type=type)
        artifact.add_file(local_path)
        run.log_artifact(artifact)

        log(f"Checkpoint logged to wandb: {local_path}")
    except Exception as e:
        log(f"Failed to log checkpoint to wandb: {e}")
