import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from trainings.config.config import initialise_instance, read_training_params_from_yaml
from trainings.dataloader.dataloader import H5Dataset
from trainings.models.utils import import_model
from trainings.utils.common import setup_seed
from trainings.utils.learning_rate_scheduler import LRScheduler
from trainings.utils.logger import Logger


def training(
    output_folder: str | Path,
    path_yaml: str | Path = None,
    overwrite_output: bool = False,
) -> None:
    tp, path_yaml = read_training_params_from_yaml(path_yaml)

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

    tp["n_properties"] = dataset.properties.n_properties
    tp["n_categories"] = dataset.properties.n_categories
    tp["n_properties_situation"] = dataset.same_id.n_pid_never_same
    tp["n_features"] = dataset.properties.n_properties * tp["n_repeat"]
    tp["n_features_situation"] = (
        dataset.properties.n_properties * dataset.same_id.n_pid_never_same
    )
    tp["tid_same"] = dataset.same_id.tid_same
    tp["dim_output"] = dataset.dim_x
    tp["n_prompts"] = len(dataset)
    tp["backbone"] = dataset.backbone_name

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

    start = time.time()
    log_print("Start training...")

    for epoch in range(tp["n_epochs"]):
        for batch_idx, (y, mask) in enumerate(dataloader):
            y = y.to(device).to(torch.float32)
            mask = mask.to(device)

            predicted_x = decoder(batch_size=y.shape[0], batch_idx=batch_idx)

            loss = mse_loss(predicted_x, y)

            optimizer.zero_grad()
            loss.backward()
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

        if (
            tp["save_model_frequency"] is not None
            and epoch % tp["save_model_frequency"] == 0
        ):
            logger.log("Saving model.")
            torch.save(decoder.state_dict(), logger.log_folder + "/model.pt")

    # log the denormalized loss
    if tp["normalize"] is not None:
        with torch.no_grad():
            y = y.cpu()
            predicted_x = predicted_x.cpu()
            y = dataset.denormalize(y)
            predicted_x = dataset.denormalize(predicted_x)
            loss = mse_loss(predicted_x, y)
            log_print(f"Final loss denormalised : {loss.item()}")

    torch.save(decoder.state_dict(), logger.log_folder + "/model.pt")


if __name__ == "__main__":
    training(output_folder="results/training1", overwrite_output=True)
