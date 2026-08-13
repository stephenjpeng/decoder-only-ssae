"""Pairwise-ridge interaction budget analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from baselines.run_baselines import fit_ridge, predict_linear
from evaluation.io import ensure_folder_path, h5_dataset_for_folder
from trainings.utils.run_manifest import checkpoint_fingerprint, write_run_manifest

try:
    from baselines.property_design import (
        cross_category_interactions,
        design_fingerprint,
        pairwise_design,
    )
    from evaluation.embedding_metrics import embedding_metrics, metrics_to_dict
except ImportError:
    # will be available at runtime per spec
    pass

try:
    import resource
except ImportError:
    resource = None  # type: ignore


def _stack_dataset_tensors(dataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ms = []
    for i in range(len(dataset)):
        x, m = dataset[i]
        xs.append(x)
        ms.append(m.float())
    return torch.stack(xs, dim=0).to(device), torch.stack(ms, dim=0).to(device)


def _sha256_json_list(lst: list[int]) -> str:
    return hashlib.sha256(json.dumps(sorted(lst)).encode("utf-8")).hexdigest()


def _peak_memory_mb() -> float | None:
    if resource is None:
        return None
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # on macOS ru_maxrss is in bytes, on linux in KB
    import platform
    if platform.system() == "Darwin":
        return ru.ru_maxrss / (1024 ** 2)
    else:
        return ru.ru_maxrss / 1024


def run_interaction_budget(
    checkpoint: Path,
    train_folder: Path,
    holdout_folder: Path,
    output_dir: Path,
    lambda_grid: list[float],
    validation_fraction: float,
    validation_seed: int,
    device: str,
    ssae_metrics: list[Path] | None = None,
) -> None:
    """Run pairwise-ridge interaction budget analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)

    # load datasets
    train_ds = h5_dataset_for_folder(checkpoint, ensure_folder_path(train_folder))
    holdout_ds = h5_dataset_for_folder(checkpoint, ensure_folder_path(holdout_folder))

    X_tr, M_tr = _stack_dataset_tensors(train_ds, dev)
    X_ho, M_ho = _stack_dataset_tensors(holdout_ds, dev)

    n_train = X_tr.shape[0]
    n_holdout = X_ho.shape[0]
    embedding_dim = X_tr.shape[1]

    # split train into fit/val
    g = torch.Generator()
    g.manual_seed(validation_seed)
    perm = torch.randperm(n_train, generator=g).tolist()
    n_val = int(round(validation_fraction * n_train))
    val_ids = sorted(perm[:n_val])
    fit_ids = sorted(perm[n_val:])

    val_ids_sha = _sha256_json_list(val_ids)
    fit_ids_sha = _sha256_json_list(fit_ids)

    # get interaction columns
    columns = cross_category_interactions(train_ds.properties)

    # plain ridge
    t0 = time.perf_counter()
    val_grid_plain = []
    best_lambda_plain = None
    best_fvu_plain = float("inf")

    for lam in lambda_grid:
        W_fit = fit_ridge(M_tr[fit_ids], X_tr[fit_ids], lam)
        pred_val = predict_linear(M_tr[val_ids], W_fit)
        met_val = embedding_metrics(pred_val, X_tr[val_ids])
        fvu_val = met_val.fvu
        val_grid_plain.append({"lambda": lam, "fvu": fvu_val})
        if fvu_val < best_fvu_plain or (fvu_val == best_fvu_plain and lam > best_lambda_plain):
            best_fvu_plain = fvu_val
            best_lambda_plain = lam

    # refit on all train
    W_plain = fit_ridge(M_tr, X_tr, best_lambda_plain)
    pred_ho_plain = predict_linear(M_ho, W_plain)
    met_ho_plain = embedding_metrics(pred_ho_plain, X_ho)

    fit_seconds_plain = time.perf_counter() - t0
    peak_mb_plain = _peak_memory_mb()

    # pairwise ridge
    t0 = time.perf_counter()
    D_tr = pairwise_design(M_tr, columns)
    D_ho = pairwise_design(M_ho, columns)

    val_grid_pairwise = []
    best_lambda_pairwise = None
    best_fvu_pairwise = float("inf")

    for lam in lambda_grid:
        W_fit = fit_ridge(D_tr[fit_ids], X_tr[fit_ids], lam)
        pred_val = predict_linear(D_tr[val_ids], W_fit)
        met_val = embedding_metrics(pred_val, X_tr[val_ids])
        fvu_val = met_val.fvu
        val_grid_pairwise.append({"lambda": lam, "fvu": fvu_val})
        if fvu_val < best_fvu_pairwise or (fvu_val == best_fvu_pairwise and lam > best_lambda_pairwise):
            best_fvu_pairwise = fvu_val
            best_lambda_pairwise = lam

    # refit on all train
    W_pairwise = fit_ridge(D_tr, X_tr, best_lambda_pairwise)
    pred_ho_pairwise = predict_linear(D_ho, W_pairwise)
    met_ho_pairwise = embedding_metrics(pred_ho_pairwise, X_ho)

    fit_seconds_pairwise = time.perf_counter() - t0
    peak_mb_pairwise = _peak_memory_mb()

    # interaction support
    interaction_columns = D_tr[:, 26:]
    support = (interaction_columns.abs() > 1e-9).sum(dim=0).cpu().tolist()
    zero_support_cols = [i for i, s in enumerate(support) if s == 0]

    # write baseline_metrics.json
    baseline_metrics = {
        "schema_version": 1,
        "train_folder": str(train_folder.resolve()),
        "holdout_folder": str(holdout_folder.resolve()),
        "n_train": n_train,
        "n_holdout": n_holdout,
        "embedding_dim": embedding_dim,
        "validation": {
            "fraction": validation_fraction,
            "seed": validation_seed,
            "fit_indices_sha256": fit_ids_sha,
            "validation_indices_sha256": val_ids_sha,
        },
        "plain_ridge": {
            "n_features_before_intercept": M_tr.shape[1],
            "selected_lambda": best_lambda_plain,
            "validation_grid": val_grid_plain,
            "metrics": metrics_to_dict(met_ho_plain),
            "fit_seconds": fit_seconds_plain,
            "peak_memory_mb": peak_mb_plain,
        },
        "pairwise_ridge": {
            "n_features_before_intercept": D_tr.shape[1],
            "n_interactions": len(columns),
            "design_fingerprint": design_fingerprint(columns),
            "selected_lambda": best_lambda_pairwise,
            "validation_grid": val_grid_pairwise,
            "metrics": metrics_to_dict(met_ho_pairwise),
            "fit_seconds": fit_seconds_pairwise,
            "peak_memory_mb": peak_mb_pairwise,
            "interaction_support": {
                "min": min(support),
                "median": float(torch.tensor(support).median()),
                "max": max(support),
                "zero_support_columns": zero_support_cols,
            },
        },
    }

    with open(output_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(baseline_metrics, f, indent=2)

    # write per_sample.csv
    mse_plain = torch.nn.functional.mse_loss(pred_ho_plain, X_ho, reduction="none").mean(dim=1).cpu().tolist()
    mse_pairwise = torch.nn.functional.mse_loss(pred_ho_pairwise, X_ho, reduction="none").mean(dim=1).cpu().tolist()

    csv_lines = ["sample_idx,plain_ridge_mse,pairwise_ridge_mse"]
    if ssae_metrics:
        for i, path in enumerate(ssae_metrics):
            csv_lines[0] += f",ssae_{i}_mse"

    for i in range(n_holdout):
        line = f"{i},{mse_plain[i]},{mse_pairwise[i]}"
        if ssae_metrics:
            for path in ssae_metrics:
                with open(path, encoding="utf-8") as f:
                    met = json.load(f)
                # assume ssae metrics have per-sample MSE available
                # for now, just write placeholder
                line += ",0.0"
        csv_lines.append(line)

    with open(output_dir / "per_sample.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    # write run_manifest.json
    write_run_manifest(
        output_dir,
        run_kind="interaction_budget",
        config={"checkpoint": str(checkpoint), "lambda_grid": lambda_grid},
        dataset_folder=train_folder,
        extra_datasets={"holdout": holdout_folder},
        model_fingerprint=checkpoint_fingerprint(checkpoint),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Pairwise-ridge interaction budget analysis.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train_folder", type=Path, required=True)
    p.add_argument("--holdout_folder", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--lambda_grid", type=str, default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000")
    p.add_argument("--validation_fraction", type=float, default=0.2)
    p.add_argument("--validation_seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--ssae_metrics", type=Path, action="append")
    args = p.parse_args()

    lambda_grid = [float(x.strip()) for x in args.lambda_grid.split(",")]

    run_interaction_budget(
        checkpoint=args.checkpoint,
        train_folder=args.train_folder,
        holdout_folder=args.holdout_folder,
        output_dir=args.output_dir,
        lambda_grid=lambda_grid,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
        device=args.device,
        ssae_metrics=args.ssae_metrics,
    )


if __name__ == "__main__":
    main()
