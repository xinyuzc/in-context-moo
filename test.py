"""Test scripts: evaluate tamo on prediction or optimization problems."""

import torch
from omegaconf import DictConfig
import hydra

from utils.wandb_wrapper import init as wandb_init
from utils.log import get_log_filename, get_log_fn
from utils.config import load_checkpoint, build_tamo
from utils.dataclasses import (
    ExConfig,
    DataConfig,
    PredictionConfig,
    OptimizationConfig,
    LogConfig,
)
from utils.paths import (
    get_exp_path,
    get_result_plot_path,
    get_result_data_path,
    get_filename_base,
)
from data.dataset import map_function_to_gp_datapath, get_function_environment
from evaluate import evaluate_prediction, evaluate_optimization
from utils.seed import set_all_seeds

# ------------------------------------------------------------------
# Hack - checkpoint key remapping from clean-up
# ------------------------------------------------------------------
_CKPT_KEY_REMAP = {
    "decoder.id_task": "decoder.task_tokens",
    "decoder.token_selected": "decoder.ar_bias_token",
}


def remap_checkpoint_keys(state_dict: dict) -> dict:
    return {_CKPT_KEY_REMAP.get(k, k): v for k, v in state_dict.items()}


@hydra.main(version_base=None, config_path="configs", config_name="test.yaml")
def main(config: DictConfig):
    assert config.experiment.mode == "test", f"Set mode to 'test'!"

    torch.set_printoptions(threshold=torch.inf)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    set_all_seeds(config.experiment.seed)

    # ------------------------------------------------------------------
    # Config dataclasses, enabling dot access
    # ------------------------------------------------------------------
    cfg_map = {
        "experiment": ExConfig,
        "prediction": PredictionConfig,
        "optimization": OptimizationConfig,
        "data": DataConfig,
        "log": LogConfig,
    }
    cfgs = {k: v(**config[k]) for k, v in cfg_map.items()}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_filename = get_log_filename(
        model_name=cfgs["experiment"].model_name,
        expid=cfgs["experiment"].expid,
        prefix=cfgs["experiment"].mode,
    )
    log = get_log_fn(filename=log_filename)

    exp_path = get_exp_path(
        model_name=cfgs["experiment"].model_name,
        expid=cfgs["experiment"].expid,
    )

    log(
        f"""--- Setup logging and experiment path ---
        Logging information is saving to:\t{log_filename}
        Experiment checkpoint will be read from:\t{exp_path}"""
    )

    if cfgs["experiment"].log_to_wandb:
        log(f"--- Setup W&B ---\n{config.wandb}")
        wandb_init(config=config, **config.wandb)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt = load_checkpoint(
        exp_path=exp_path,
        device=cfgs["experiment"].device,
        resume=cfgs["experiment"].resume,
        ckpt_name=config.extra.ckpt_name,
    )
    model_state_dict = ckpt.get("model", {})
    if not model_state_dict:
        raise RuntimeError(
            f"Invalid checkpoint loaded from {exp_path}. "
            "Checkpoint is either empty or missing the 'model' key."
        )

    # Remap keys that were renamed during the clean-up
    model_state_dict = remap_checkpoint_keys(model_state_dict)

    # ------------------------------------------------------------------
    # Build model and load weights
    # ------------------------------------------------------------------
    model = build_tamo(dict(config.model))
    model = model.to(cfgs["experiment"].device)
    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    if missing:
        log(f"[WARNING] Missing keys after checkpoint load:\n  " + "\n  ".join(missing))
    if unexpected:
        log(f"[WARNING] Unexpected keys in checkpoint:\n  " + "\n  ".join(unexpected))

    log(
        f"--- Model built: TAMO ---\n"
        f"  Config: {model.config}\n"
        f"  Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    # ------------------------------------------------------------------
    # Data / Function env
    # ------------------------------------------------------------------
    if cfgs["experiment"].task == "prediction":
        datapaths, function_name = map_function_to_gp_datapath(
            function_name=cfgs["data"].function_name,
            mode=cfgs["experiment"].mode,
            data_id=cfgs["data"].data_id,
        )
        assert datapaths, f"Unsupported function name: {cfgs['data'].function_name}"
    elif cfgs["experiment"].task == "optimization":
        test_function = get_function_environment(
            function_name=cfgs["data"].function_name,
            seed=cfgs["experiment"].seed,
            device=cfgs["experiment"].device,
            data_id=cfgs["data"].data_id,
            scene=cfgs["data"].scene,
        )
        function_name = cfgs["data"].function_name
    else:
        raise ValueError(
            f"Unsupported task: {cfgs['experiment'].task}. "
            "Supported tasks are 'prediction' and 'optimization'."
        )

    # ------------------------------------------------------------------
    # Save paths
    # ------------------------------------------------------------------
    _filename_base = get_filename_base(
        function_name=function_name,
        ckpt_name=config.extra.ckpt_name,
        suffix_segment=config.extra.suffix_segment,
    )
    plot_save_path = get_result_plot_path(
        model_name=cfgs["experiment"].model_name,
        expid=cfgs["experiment"].expid,
        task_type=cfgs["experiment"].task,
        filename_base=_filename_base,
    )
    data_save_path = get_result_data_path(
        model_name=cfgs["experiment"].model_name,
        expid=cfgs["experiment"].expid,
        task_type=cfgs["experiment"].task,
        filename_base=_filename_base,
    )

    log(
        f"""--- Setup saving paths ---
        plot_save_path:\t{plot_save_path}
        data_save_path:\t{data_save_path}"""
    )

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    if cfgs["experiment"].task == "prediction":
        evaluate_prediction(
            model=model,
            datapaths=datapaths,
            data_save_path=data_save_path,
            plot_save_path=plot_save_path,
            exp_cfg=cfgs["experiment"],
            pred_cfg=cfgs["prediction"],
            data_cfg=cfgs["data"],
            log_cfg=cfgs["log"],
            log=log,
        )
    else:
        evaluate_optimization(
            model=model,
            test_function=test_function,
            plot_save_path=plot_save_path,
            data_save_path=data_save_path,
            exp_cfg=cfgs["experiment"],
            opt_cfg=cfgs["optimization"],
            data_cfg=cfgs["data"],
            pred_cfg=cfgs["prediction"],
            log_cfg=cfgs["log"],
            log=log,
        )


if __name__ == "__main__":
    main()
