import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from trainings.config.config import (
    derive_shapes,
    initialise_instance,
    read_training_params_from_yaml,
)
from trainings.dataloader.dataloader import H5Dataset
from trainings.models.mlp import parse_hidden_dims
from trainings.models.utils import import_model
from trainings.utils.common import setup_seed
from trainings.utils.learning_rate_scheduler import LRScheduler
from trainings.utils.logger import Logger
from trainings.utils.run_manifest import write_run_manifest


def training(
    output_folder: str | Path,
    path_yaml: str | Path = None,
    overwrite_output: bool = False,
    num_layers: int | None = None,
    hidden_dims=None,
    pca_rotation: bool | None = None,
    head_type: str | None = None,
) -> None:
    tp, path_yaml = read_training_params_from_yaml(path_yaml)

    if num_layers is not None:
        tp["num_layers"] = num_layers
    if hidden_dims is not None:
        tp["hidden_dims"] = hidden_dims
    if pca_rotation is not None:
        tp["pca_rotation"] = pca_rotation
    if head_type is not None:
        tp["head_type"] = head_type

    tp["num_layers"] = tp.get("num_layers", 1) or 1
    tp["hidden_dims"] = parse_hidden_dims(
        tp.get("hidden_dims"), tp["num_layers"]
    )

    setup_seed(tp["seed"])

    logger = Logger(
        output_folder,
        path_yaml,
        overwrite=overwrite_output,
        plot_frequency=tp["plot_frequency"],
    )

    log_print = print if logger is None else logger.print

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log_print("device : ", device)

    tp["logger"] = logger
    tp["device"] = device

    dataset = initialise_instance(H5Dataset, tp)

    derive_shapes(tp, dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=tp["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=tp["num_workers"],
        pin_memory=True,
    )

    Decoder = import_model(tp["model_name"])

    decoder = initialise_instance(Decoder, tp)
    decoder = decoder.to(device)

    log_print(decoder)

    logger.log_model(decoder)

    decoder.apply_mask(dataset.mask_reduced.to(device), tp["batch_size"])

    log_print(f"The rank of Y is {decoder.get_rank_Y()}")

    mse_loss = nn.MSELoss()

    betas = (tp["beta1"], tp["beta2"])
    optimizer = optim.Adam(decoder.parameters(), lr=tp["lr"], betas=betas)

    if tp["lr_scheduler_type"] is not None:
        tp["lr_scheduler_lr_init_linear"] = tp["lr"]
        tp["optimizer"] = optimizer
        scheduler = initialise_instance(LRScheduler, tp)
        del tp["optimizer"]

    # save all the parameters used
    path_config_all = Path(output_folder, "all_params.yaml")
    with open(path_config_all, "w") as f:
        yaml.dump(tp, f, default_flow_style=False)

    # E0 / AUG-02: machine-readable provenance next to the human-readable YAML.
    # Written *before* the training loop so a crashed or killed run still leaves a
    # traceable record of what was attempted.
    manifest_path = write_run_manifest(
        output_folder,
        run_kind="training",
        config=tp,
        seed=tp["seed"],
        dataset=dataset,
        model_fingerprint={
            "model_name": tp["model_name"],
            "num_layers": tp["num_layers"],
            "hidden_dims": tp["hidden_dims"],
            "head_type": tp.get("head_type"),
            "n_repeat": tp["n_repeat"],
            "n_features": tp["n_features"],
            "dim_output": tp["dim_output"],
            "n_trainable_params": sum(
                p.numel() for p in decoder.parameters() if p.requires_grad
            ),
        },
        extra={"path_yaml": str(path_yaml), "device": str(device), "status": "started"},
    )
    log_print(f"run manifest : {manifest_path}")

    start = time.time()
    log_print("Start training...")

    # inv variant refits W in closed form each epoch against the full X
    X_full_for_refit = (
        dataset.get_X().to(device).to(torch.float32) if hasattr(decoder, "refit_W") else None
    )

    for epoch in range(tp["n_epochs"]):
        if X_full_for_refit is not None:
            decoder.refit_W(X_full_for_refit)

        for batch_idx, (y, mask) in enumerate(dataloader):
            y = y.to(device).to(torch.float32)
            mask = mask.to(device)

            predicted_x = decoder(batch_size=y.shape[0], batch_idx=batch_idx)

            loss = mse_loss(predicted_x, y)

            optimizer.zero_grad()
            loss.backward()
            # E11 wants gradient norms alongside the loss curve to tell "the two-layer
            # advantage appeared here" from "the optimiser was still moving here".
            # Cheap: one extra reduction per step over parameters already in memory.
            grad_sq = 0.0
            for p in decoder.parameters():
                if p.grad is not None:
                    grad_sq += float(p.grad.detach().pow(2).sum().item())
            grad_norm = grad_sq**0.5
            optimizer.step()

        if epoch % tp["print_frequency"] == 0:
            lr = optimizer.param_groups[0]["lr"]
            log_print(
                f"epoch: {epoch}, training loss: {loss.item():.6f}, time: {(time.time() - start):.2f}s, lr: {lr:.7f}"
            )

        # to debug
        if epoch < 3 and tp["model_name"] == "model_trainable_inputs":
            logger.log_matrix(
                f"Y_with_mask_epoch_{epoch}.txt",
                decoder.Y_with_same_and_mask.detach().cpu().numpy(),
            )

        if tp["lr_scheduler_type"] is not None:
            scheduler.step(epoch)

        logger.log("rec_loss", iteration=epoch, value=loss.detach().item())
        logger.log("grad_norm", iteration=epoch, value=grad_norm)

        if (
            tp["save_model_frequency"] is not None
            and epoch % tp["save_model_frequency"] == 0
        ):
            # Was `logger.log("Saving model.")`, which raised TypeError (Logger.log needs
            # key/iteration/value) and so made save_model_frequency unusable; and it wrote
            # to a single model.pt, overwriting rather than accumulating. E11 needs the
            # *trajectory*, so keep one file per checkpoint epoch.
            ckpt_dir = Path(logger.log_folder, "checkpoints")
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(decoder.state_dict(), ckpt_dir / f"model_epoch_{epoch:05d}.pt")
            log_print(f"Saved checkpoint at epoch {epoch}.")

    # Capture before the denormalisation block below rebinds `loss`.
    final_train_loss_normalized = float(loss.item())
    final_train_loss_denormalized = None

    # The tied-code decoder relies on nn.Embedding(padding_idx=0) keeping row 0 at zero,
    # so that an inactive property contributes nothing and the one-layer model is exactly
    # additive in the property indicators (the premise of E1/C6). The MPS backend does not
    # honour padding_idx gradient masking, so on Apple silicon that row trains and the
    # model picks up an extra -W_k sigma(y_0) term per inactive block. The model is still
    # additive, but v_k = W_k sigma(y_k) is then the wrong extraction. Record the fact
    # rather than silently changing optimisation behaviour.
    padding_row_is_zero = None
    padding_row_absmax = None
    Y_param = getattr(decoder, "Y", None)
    if isinstance(Y_param, nn.Embedding) and Y_param.padding_idx is not None:
        with torch.no_grad():
            row = Y_param.weight[Y_param.padding_idx]
            padding_row_absmax = float(row.abs().max().item())
            padding_row_is_zero = padding_row_absmax == 0.0
        if not padding_row_is_zero:
            log_print(
                f"WARNING: Y padding row is non-zero (abs max {padding_row_absmax:.3e}). "
                f"device={device}. nn.Embedding padding_idx gradient masking is not "
                f"honoured on this backend; downstream concept-vector extraction must use "
                f"v_k = W_k(sigma(y_k) - sigma(y_0)). See analysis/equivalence_ridge.py."
            )

    # log the denormalized loss
    if tp["normalize"] is not None:
        with torch.no_grad():
            y = y.cpu()
            predicted_x = predicted_x.cpu()
            y = dataset.denormalize(y)
            predicted_x = dataset.denormalize(predicted_x)
            loss = mse_loss(predicted_x, y)
            final_train_loss_denormalized = float(loss.item())
            log_print(f"Final loss denormalised : {loss.item()}")

    torch.save(decoder.state_dict(), logger.log_folder + "/model.pt")

    # E11 consumes the loss / gradient-norm series as data, not as PNGs.
    logs_path = logger.dump_logs()
    log_print(f"scalar logs : {logs_path}")

    # Re-emit with the completion status and final loss so a finished run is
    # distinguishable from an interrupted one without reading training.log.
    write_run_manifest(
        output_folder,
        run_kind="training",
        config=tp,
        seed=tp["seed"],
        dataset=dataset,
        model_fingerprint={
            "model_name": tp["model_name"],
            "num_layers": tp["num_layers"],
            "hidden_dims": tp["hidden_dims"],
            "head_type": tp.get("head_type"),
            "n_repeat": tp["n_repeat"],
            "n_features": tp["n_features"],
            "dim_output": tp["dim_output"],
            "n_trainable_params": sum(
                p.numel() for p in decoder.parameters() if p.requires_grad
            ),
        },
        extra={
            "path_yaml": str(path_yaml),
            "device": str(device),
            "status": "completed",
            "n_epochs_run": tp["n_epochs"],
            "final_train_loss_normalized": final_train_loss_normalized,
            "final_train_loss_denormalized": final_train_loss_denormalized,
            "elapsed_sec": float(time.time() - start),
            "padding_row_is_zero": padding_row_is_zero,
            "padding_row_absmax": padding_row_absmax,
        },
    )


if __name__ == "__main__":
    training(output_folder="results/training1", overwrite_output=True)
