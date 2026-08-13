"""Linear probe for T5 embedding space."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence


class CategoricalLinearProbe(nn.Module):
    """Linear probe that predicts one property per category from centered embeddings."""

    def __init__(
        self, embedding_dim: int, category_pid_groups: Sequence[Sequence[int]]
    ) -> None:
        super().__init__()
        # total number of properties across all categories
        n_properties = sum(len(group) for group in category_pid_groups)
        self.linear = nn.Linear(embedding_dim, n_properties, bias=True)
        # store as list of lists for JSON serialization
        self.category_pid_groups = [list(group) for group in category_pid_groups]

    def forward(self, centered_embeddings: torch.Tensor) -> torch.Tensor:
        """(n, d) -> (n, 26) logits."""
        return self.linear(centered_embeddings)

    def loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Average of category-wise cross-entropy losses.
        mask: (n, 26) binary, one-hot per category.
        For category pids [p0,p1,p2]: logits_cat = logits[:,[p0,p1,p2]],
        target = argmax(mask[:,[p0,p1,p2]], dim=1)
        Average the n_categories losses.
        """
        n_categories = len(self.category_pid_groups)
        losses = []
        for pids in self.category_pid_groups:
            logits_cat = logits[:, pids]  # (n, n_pids_in_category)
            mask_cat = mask[:, pids]  # (n, n_pids_in_category)
            # target is the index within this category's pid group
            target = mask_cat.argmax(dim=1)  # (n,)
            cat_loss = F.cross_entropy(logits_cat, target, reduction="mean")
            losses.append(cat_loss)
        return sum(losses) / n_categories

    def predict_pids(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Returns (n, n_categories) tensor of predicted pids (actual pid values, not local indices).
        For each category, argmax of the category's logit slice maps to the actual pid.
        """
        n = logits.shape[0]
        n_categories = len(self.category_pid_groups)
        predicted_pids = torch.zeros((n, n_categories), dtype=torch.long)
        for cid, pids in enumerate(self.category_pid_groups):
            logits_cat = logits[:, pids]  # (n, n_pids_in_category)
            local_idx = logits_cat.argmax(dim=1)  # (n,)
            # map local index to actual pid
            pids_tensor = torch.tensor(pids, dtype=torch.long, device=logits.device)
            predicted_pids[:, cid] = pids_tensor[local_idx]
        return predicted_pids

    def category_accuracy(
        self, logits: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """(n_categories,) tensor of per-category accuracy."""
        n_categories = len(self.category_pid_groups)
        accuracies = torch.zeros(n_categories)
        for cid, pids in enumerate(self.category_pid_groups):
            logits_cat = logits[:, pids]
            mask_cat = mask[:, pids]
            pred_local = logits_cat.argmax(dim=1)
            target_local = mask_cat.argmax(dim=1)
            accuracies[cid] = (pred_local == target_local).float().mean()
        return accuracies


def fit_probe(
    train_X: torch.Tensor,
    train_M: torch.Tensor,
    cal_X: torch.Tensor,
    cal_M: torch.Tensor,
    category_pid_groups: Sequence[Sequence[int]],
    *,
    embedding_dim: int,
    max_epochs: int = 300,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 25,
    weight_decay_grid: Sequence[float] = (0.0, 1e-6, 1e-5, 1e-4, 1e-3),
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[CategoricalLinearProbe, dict]:
    """
    Fit probe via AdamW. Returns (best_probe, fit_info).
    fit_info keys: selected_weight_decay, best_cal_accuracy, cal_accuracy_per_wd,
                   n_epochs_trained, early_stopped, train_loss_final, fit_mean.

    - Compute fit_mean = train_X.mean(dim=0), subtract from both train_X and cal_X before use.
    - For each wd in weight_decay_grid: train from scratch, track best cal mean categorical accuracy.
    - Early stopping: stop if cal accuracy hasn't improved for patience epochs.
    - Select wd with highest best cal accuracy; ties go to LARGER wd.
    - Store fit_mean in fit_info.
    - Keep embedding matrices on CPU, transfer mini-batches to device.
    """
    torch.manual_seed(seed)
    device = torch.device(device)

    # compute fit_mean on CPU
    fit_mean = train_X.mean(dim=0)
    train_X_centered = train_X - fit_mean
    cal_X_centered = cal_X - fit_mean

    n_train = train_X_centered.shape[0]

    cal_accuracy_per_wd = {}
    best_cal_acc = -1.0
    best_wd = None
    best_probe_state = None
    best_n_epochs = 0
    best_early_stopped = False
    best_train_loss = None

    for wd in weight_decay_grid:
        probe = CategoricalLinearProbe(embedding_dim, category_pid_groups).to(device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

        best_cal_acc_this_wd = -1.0
        best_state_this_wd = None
        epochs_since_improvement = 0
        early_stopped = False
        final_train_loss = None

        for epoch in range(max_epochs):
            probe.train()
            # shuffle training indices
            perm = torch.randperm(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for start_idx in range(0, n_train, batch_size):
                end_idx = min(start_idx + batch_size, n_train)
                batch_indices = perm[start_idx:end_idx]

                # transfer batch to device
                batch_X = train_X_centered[batch_indices].to(device)
                batch_M = train_M[batch_indices].to(device)

                optimizer.zero_grad()
                logits = probe(batch_X)
                loss = probe.loss(logits, batch_M)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            final_train_loss = epoch_loss / n_batches

            # evaluate on calibration set
            probe.eval()
            with torch.no_grad():
                # evaluate in batches to avoid memory issues
                all_logits = []
                for start_idx in range(0, cal_X_centered.shape[0], batch_size):
                    end_idx = min(start_idx + batch_size, cal_X_centered.shape[0])
                    batch_X = cal_X_centered[start_idx:end_idx].to(device)
                    batch_logits = probe(batch_X)
                    all_logits.append(batch_logits.cpu())
                cal_logits = torch.cat(all_logits, dim=0)
                cal_acc_per_cat = probe.category_accuracy(cal_logits, cal_M)
                cal_acc = cal_acc_per_cat.mean().item()

            # track best for this weight decay
            if cal_acc > best_cal_acc_this_wd:
                best_cal_acc_this_wd = cal_acc
                best_state_this_wd = {
                    k: v.cpu().clone() for k, v in probe.state_dict().items()
                }
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # early stopping
            if epochs_since_improvement >= patience:
                early_stopped = True
                break

        cal_accuracy_per_wd[wd] = best_cal_acc_this_wd

        # select this wd if it's better, or if tied and wd is larger
        if (
            best_cal_acc_this_wd > best_cal_acc
            or (best_cal_acc_this_wd == best_cal_acc and wd > best_wd)
        ):
            best_cal_acc = best_cal_acc_this_wd
            best_wd = wd
            best_probe_state = best_state_this_wd
            best_n_epochs = epoch + 1
            best_early_stopped = early_stopped
            best_train_loss = final_train_loss

    # load best probe
    best_probe = CategoricalLinearProbe(embedding_dim, category_pid_groups)
    best_probe.load_state_dict(best_probe_state)

    fit_info = {
        "selected_weight_decay": best_wd,
        "best_cal_accuracy": best_cal_acc,
        "cal_accuracy_per_wd": cal_accuracy_per_wd,
        "n_epochs_trained": best_n_epochs,
        "early_stopped": best_early_stopped,
        "train_loss_final": best_train_loss,
        "fit_mean": fit_mean.cpu(),
    }

    return best_probe, fit_info
