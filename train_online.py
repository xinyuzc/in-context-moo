"""Training script."""

import random
import time
import os.path as osp
from typing import Dict

import torch
import torch.nn.functional as F

from omegaconf import DictConfig
import hydra
import wandb

from data.gp_sample_function import prepare_prediction_batches, GPSampleFunction
from data.base.preprocessing import has_nan_or_inf
from utils.paths import get_exp_path
from utils.log import Averager, get_log_filename, get_log_fn
from utils.dataclasses import (
    ExConfig,
    PredictionConfig,
    OptimizationConfig,
    DataConfig,
    LossConfig,
    TrainConfig,
    LogConfig,
)
from utils.config import (
    build_tamo,
    build_optimizer,
    build_scheduler,
    load_checkpoint,
    save_checkpoint,
)
from utils.seed import set_all_seeds
from forwards import optimization_forward, prediction_forward
from utils.wandb_wrapper import init as wandb_init, save_artifact

N = 300


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(config: DictConfig):
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")

    # Setup configurations
    exp_cfg = ExConfig(**config.experiment)
    train_cfg = TrainConfig(**config.train)
    pred_cfg = PredictionConfig(**config.prediction)
    opt_cfg = OptimizationConfig(**config.optimization)
    loss_cfg = LossConfig(**config.loss)
    data_cfg = DataConfig(**config.data)
    log_cfg = LogConfig(**config.log)

    # Setup logging
    log_filename = get_log_filename(
        model_name=exp_cfg.model_name, expid=exp_cfg.expid, prefix=exp_cfg.mode
    )
    log = get_log_fn(filename=log_filename)
    log(f"Logs will be saved to:\t{log_filename}")

    if exp_cfg.log_to_wandb:
        log(f"wandb configuration:{config.wandb}\n")
        wandb_init(config=config, **config.wandb)

    # Setup experiment path
    exp_path = get_exp_path(model_name=exp_cfg.model_name, expid=exp_cfg.expid)
    log(f"exp_path:\t{exp_path}")

    train(
        exp_path=exp_path,
        model_kwargs=dict(config.model),
        exp_cfg=exp_cfg,
        opt_cfg=opt_cfg,
        pred_cfg=pred_cfg,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        loss_cfg=loss_cfg,
        log_cfg=log_cfg,
        log=log,
    )


def train(
    exp_path: str,
    model_kwargs: Dict,
    exp_cfg: ExConfig,
    opt_cfg: OptimizationConfig,
    pred_cfg: PredictionConfig,
    train_cfg: TrainConfig,
    loss_cfg: LossConfig,
    data_cfg: DataConfig,
    log_cfg: LogConfig = LogConfig(),
    log: callable = print,
):
    # Set random seed
    set_all_seeds(exp_cfg.seed)
    log(f"seed:\t{exp_cfg.seed}")

    # ===============================================
    # Load checkpoint
    # ===============================================
    ckpt = load_checkpoint(
        exp_path=exp_path, device=exp_cfg.device, resume=exp_cfg.resume
    )

    epoch = ckpt.get("epoch", -1)
    model_state_dict = ckpt.get("model", None)
    optimizer_state_dict = ckpt.get("optimizer", None)
    scheduler_state_dict = ckpt.get("scheduler", None)

    # ===============================================
    # Setup epochs
    # ===============================================
    num_total_epochs = train_cfg.num_total_epochs
    num_burnin_epochs = train_cfg.num_burnin_epochs
    num_after_burnin_epochs = num_total_epochs - num_burnin_epochs
    num_context_size_burnin_epochs = train_cfg.num_nc_burnin_epochs

    log(
        f"==== epochs ===="
        f"  last epoch:\t{epoch}"
        f"  num_total_epochs:\t{num_total_epochs}"
        f"  num_burnin_epochs:\t{num_burnin_epochs}"
        f"  num_nc_burnin_epochs:\t{num_context_size_burnin_epochs}"
    )

    # ===============================================
    # Setup model
    # ===============================================
    model = build_tamo(model_kwargs).to(exp_cfg.device)
    if model_state_dict:
        missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
        if missing:
            log(
                f"[WARNING] Missing keys after checkpoint load:\n  "
                + "\n  ".join(missing)
            )
        if unexpected:
            log(
                f"[WARNING] Unexpected keys in checkpoint:\n  "
                + "\n  ".join(unexpected)
            )

    log(
        f"==== Model built: TAMO ====\n"
        f"  Config: {model.config}\n"
        f"  Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    if exp_cfg.log_to_wandb:
        wandb.watch(model, log="gradients", log_freq=log_cfg.freq_log_grad)

    # ===============================================
    # Setup optimizer and scheduler
    # ===============================================
    log(f"Initializing optimizer...")
    optimizer = build_optimizer(
        model=model,
        optimizer_type=train_cfg.optimizer_type,
        lr=train_cfg.lr1,
        weight_decay=train_cfg.weight_decay,
    )

    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)

    log(f"Initializing scheduler...")
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type=train_cfg.scheduler_type,
        num_training_steps=(num_burnin_epochs, num_after_burnin_epochs)[
            epoch >= num_burnin_epochs
        ],
        last_epoch=(
            epoch if epoch < num_burnin_epochs else max(epoch - num_burnin_epochs, -1)
        ),
        num_warmup_steps=train_cfg.num_warmup_steps,
    )
    if scheduler_state_dict:
        scheduler.load_state_dict(scheduler_state_dict)

    ravg = Averager()

    while epoch < num_total_epochs:
        x_list = []
        y_list = []
        valid_x_counts_list = []
        valid_y_counts_list = []
        while len(x_list) < pred_cfg.batch_size:
            x_dim = random.choice(data_cfg.x_dim_list)
            y_dim = random.choice(data_cfg.y_dim_list)

            x_i, y_i, _, _ = GPSampleFunction.sample_from_syn(
                x_dim=x_dim,
                y_dim=y_dim,
                batch_size=1,
                d=N,
                use_grid_sampling=False,
                use_factorized_policy=False,
                x_range=data_cfg.x_range,
                sampler_list=data_cfg.sampler_list,
                sampler_weights=data_cfg.sampler_weights,
                data_kernel_type_list=data_cfg.data_kernel_type_list,
                sample_kernel_weights=data_cfg.sample_kernel_weights,
                lengthscale_range=data_cfg.lengthscale_range,
                std_range=data_cfg.std_range,
                min_rank=data_cfg.min_rank,
                max_rank=data_cfg.max_rank,
                p_iso=data_cfg.p_iso,
                jitter=data_cfg.jitter,
                max_tries=data_cfg.max_tries,
                standardize=data_cfg.standardize,
                device=exp_cfg.device,
                sampler_type=data_cfg.function_name,
            )

            # Pad to max dimensions: [1, N, x_dim] -> [1, N, max_x_dim]
            x_i = F.pad(x_i, (0, data_cfg.max_x_dim - x_dim))
            y_i = F.pad(y_i, (0, data_cfg.max_y_dim - y_dim))

            if has_nan_or_inf(x_i, "x", log) or has_nan_or_inf(y_i, "y", log):
                continue

            x_list.append(x_i)
            y_list.append(y_i)
            valid_x_counts_list.append(x_dim)
            valid_y_counts_list.append(y_dim)

        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        valid_x_counts = torch.tensor(
            valid_x_counts_list, dtype=x.dtype, device=x.device
        )
        valid_y_counts = torch.tensor(
            valid_y_counts_list, dtype=y.dtype, device=y.device
        )

        del x_list, y_list, valid_x_counts_list, valid_y_counts_list

        epoch += 1

        # ===============================================
        # Reinit optimizer and scheduler when starting policy learning
        # ===============================================
        if epoch == num_burnin_epochs:
            log(
                f"Start policy learning at epoch {epoch}; "
                f"Re-build optimizer and scheduler with lr2: {train_cfg.lr2}"
            )
            optimizer = build_optimizer(
                model=model,
                optimizer_type=train_cfg.optimizer_type,
                lr=train_cfg.lr2,
                weight_decay=train_cfg.weight_decay,
            )
            scheduler = build_scheduler(
                optimizer=optimizer,
                scheduler_type=train_cfg.scheduler_type,
                num_training_steps=num_after_burnin_epochs,
                num_warmup_steps=train_cfg.num_warmup_steps,
            )

            # Loss curve would change - to avoid confusion!
        if epoch == num_context_size_burnin_epochs:
            log(f"Start training on prediction batches of random context size.")

        t1 = time.time()

        model.train()
        optimizer.zero_grad()

        # ===============================================
        # Prediction batch
        # ===============================================
        xc, yc, xt, yt, x_mask, y_mask = prepare_prediction_batches(
            x=x,
            y=y,
            valid_x_counts=valid_x_counts,
            valid_y_counts=valid_y_counts,
            dim_scatter_mode=data_cfg.dim_scatter_mode,
            min_nc=pred_cfg.min_nc,
            max_nc=pred_cfg.max_nc,
            warmup=epoch <= num_context_size_burnin_epochs,
        )

        # ===============================================
        # Forwards
        # ===============================================
        # Prediction forward (model + loss)
        loss_pre, mse_mean, _ = prediction_forward(
            model=model,
            x_ctx=xc,
            y_ctx=yc,
            x_tar=xt,
            y_tar=yt,
            x_mask=x_mask,
            y_mask=y_mask,
            read_cache=pred_cfg.read_cache,
        )

        loss_pre_val = loss_pre.detach().item()
        mse_mean = mse_mean.detach()

        # Prediction loss backward and free up graph
        if epoch >= num_burnin_epochs:
            loss_weight = loss_cfg.loss_weight
        else:
            loss_weight = 1.0

        (loss_weight * loss_pre).backward()

        del loss_pre
        del xc, yc, xt, yt
        del x_mask, y_mask, valid_x_counts, valid_y_counts

        # Optimization forward (model + loss)
        loss_acq_val = 0.0
        step_reward_mean = 0.0
        final_step_reward_mean = 0.0
        final_step_entropy_mean = 0.0
        T = 0

        if epoch >= num_burnin_epochs:
            T = opt_cfg.sample_T()
            (
                loss_acq,
                step_reward_mean,
                final_step_reward_mean,
                final_step_entropy_mean,
            ) = optimization_forward(
                model=model,
                data_cfg=data_cfg,
                opt_config=opt_cfg,
                loss_config=loss_cfg,
                T=T,
                device=exp_cfg.device,
            )
            loss_acq_val = loss_acq.detach().item()

            # optimization loss backward and free up graph
            loss_acq.backward()
            del loss_acq

            # gradient clipping (must unscale before clipping)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=loss_cfg.max_norm)
        optimizer.step()
        scheduler.step()

        # ===============================================
        # Tracking and Logging
        # ===============================================
        epoch_time = time.time() - t1
        mse_dict = {
            f"train/mse_{j}": mse_mean[j].detach().item()
            for j in range(mse_mean.shape[0])
        }
        log_dict = {
            "train/epoch": epoch,
            "train/loss_pre": loss_pre_val,
            "train/loss_acq": loss_acq_val,
            "train/loss": loss_pre_val + loss_acq_val,
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "train/step_reward": step_reward_mean,
            "train/step_reward_final": final_step_reward_mean,
            "train/step_entropy_final": final_step_entropy_mean,
            "train/epoch_time": epoch_time,
            "train/num_query_points": (
                opt_cfg.num_query_points if epoch >= num_burnin_epochs else 0
            ),
            "train/opt_batch_size": (
                opt_cfg.batch_size if epoch >= num_burnin_epochs else 0
            ),
            "train/opt_num_samples": (
                opt_cfg.num_samples if epoch >= num_burnin_epochs else 0
            ),
            "train/T": T,
            **mse_dict,
        }

        # Tracking
        ravg.batch_update(log_dict)

        if exp_cfg.log_to_wandb:
            #    wandb.log(ravg.get_averages())
            wandb.log(log_dict)

        # Logging
        if epoch > 0 and epoch % log_cfg.freq_log == 0:
            line = (
                f"[epoch {epoch} / {num_total_epochs}] "
                f"lr: {optimizer.param_groups[0]['lr']:.3e} "
                f"[train] "
                f"{ravg.info()}"
            )
            log(line)
            ravg.reset()

            # Saving
        if (epoch > 0 and epoch % log_cfg.freq_save == 0) or (
            epoch == num_total_epochs - 1
        ):
            log(f"Saving checkpoint at epoch {epoch} to {exp_path}")
            ckpt, ckpt_filepath = save_checkpoint(
                exp_path=exp_path,
                model=model,
                epoch=epoch,
                seed=exp_cfg.seed,
                optimizer=optimizer,
                scheduler=scheduler,
                ckpt_name="ckpt.tar",
            )
            torch.save(ckpt, ckpt_filepath)

            # Backup checkpoints
            freq_backup = (
                log_cfg.freq_save_extra_burnin,
                log_cfg.freq_save_extra,
            )[epoch >= num_burnin_epochs]
            if epoch % freq_backup == 0:
                epoch_ckpt_filepath = osp.join(exp_path, f"ckpt_epoch_{epoch}.tar")
                torch.save(ckpt, epoch_ckpt_filepath)

                # Save to WandB artifact
                if exp_cfg.log_to_wandb:
                    save_artifact(
                        run=wandb.run,
                        local_path=ckpt_filepath,
                        name=f"checkpoint_epoch_{epoch}.tar",
                        type="model",
                        log=log,
                    )


if __name__ == "__main__":
    main()
