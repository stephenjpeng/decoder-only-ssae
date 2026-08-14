"""Validation suite for the PCA-truncation training option.

Covers:
- exact round-trip when K equals the intrinsic rank / d,
- residual and replace inference semantics,
- explained-variance vs empirical holdout R^2 sanity,
- simulated end-to-end training smoke test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from backbones.base import StreamSpec
from trainings.dataloader.dataloader import (
    H5Dataset,
    MANIFEST_FILENAME,
    TRUNCATE_PCA,
    TRUNCATE_RANGE,
)


# --- fixture helpers ---------------------------------------------------------


def _write_simulated_fixture(folder: Path, n_prompts: int = 24) -> Path:
    """Create a minimal folder that H5Dataset(simulated=True) accepts.

    We still need properties.json / prompts.json / properties_same.json /
    embds/manifest.json even in simulated mode because the dataloader reads
    them at init.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    embds_dir = folder / "embds"
    embds_dir.mkdir(exist_ok=True)

    categories = {
        "hair": ["blond", "brunette"],
        "eyes": ["blue eyes", "brown eyes"],
        "action": ["walking", "running"],
    }
    (folder / "properties.json").write_text(json.dumps(categories))

    props_same = {k: False for k in categories}
    (folder / "properties_same.json").write_text(json.dumps(props_same))

    property_list = [p for props in categories.values() for p in props]
    prompts = []
    for i in range(n_prompts):
        picks = [property_list[(i + j) % len(property_list)] for j in range(3)]
        prompts.append({"prompt": ", ".join(picks)})
    (folder / "prompts.json").write_text(json.dumps(prompts))

    manifest = {
        "backbone": "sim",
        "backbone_kwargs": {},
        "streams": [
            {"name": "t5", "shape": [8], "dtype": "float32", "h5_file": "embds.h5"},
        ],
    }
    (embds_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest))

    # per-prompt subfolders + empty h5 (unused in simulated mode but the
    # dataset still enumerates them)
    for i in range(n_prompts):
        sub = embds_dir / f"embds_{i}"
        sub.mkdir(exist_ok=True)
        with h5py.File(sub / "embds.h5", "w") as f:
            f.create_dataset("vector", data=np.zeros(8, dtype=np.float32))
        (sub / "prompts.txt").write_text(prompts[i]["prompt"])

    return folder


def _make_simulated_dataset(
    tmp_path: Path,
    K: int,
    method: str = TRUNCATE_PCA,
    semantics: str = "residual",
    dim: int = 200,
    n_prompts: int = 24,
    seed: int = 0,
) -> H5Dataset:
    fixture = _write_simulated_fixture(tmp_path, n_prompts=n_prompts)
    ds = H5Dataset(
        folder_path=str(fixture) + "/",
        simulated=True,
        dim_clip_simulated=dim,
        truncate_embds_topk=K,
        truncate_embds_method=method,
        pca_semantics=semantics,
        normalize=None,
    )
    torch.manual_seed(seed)
    return ds


# --- 1. round trip -----------------------------------------------------------


def test_pca_roundtrip_exact_at_full_rank(tmp_path):
    """K = d recovers the input exactly (up to float32 noise)."""
    n, d = 32, 24
    torch.manual_seed(0)
    X = torch.randn(n, d, dtype=torch.float32)

    ds = _make_simulated_dataset(tmp_path / "roundtrip_full", K=d, dim=d, n_prompts=n)
    # override X_simulated so we control the exact matrix
    ds.pca_mean = None
    ds.pca_components = None
    ds.X_simulated = X
    ds.X = X
    ds._compute_pca_projection()

    mean = ds.pca_mean
    P = ds.pca_components
    X_proj = (X - mean) @ P.T
    X_recon = X_proj @ P + mean
    mse = torch.mean((X - X_recon) ** 2).item()
    assert mse < 1e-4, f"full-rank PCA should round-trip; got mse={mse:.3e}"


def test_pca_roundtrip_close_at_n_minus_one(tmp_path):
    """K = n-1 keeps almost all variance for a rank-<=n-1 centered matrix."""
    n, d = 64, 200
    torch.manual_seed(1)
    X = torch.randn(n, d, dtype=torch.float32)
    K = n - 1

    ds = _make_simulated_dataset(
        tmp_path / "roundtrip_nm1", K=K, dim=d, n_prompts=n
    )
    ds.pca_mean = None
    ds.pca_components = None
    ds.X_simulated = X
    ds.X = X
    ds._compute_pca_projection()

    mean = ds.pca_mean
    P = ds.pca_components
    X_recon = (X - mean) @ P.T @ P + mean
    mse = torch.mean((X - X_recon) ** 2).item()
    assert mse < 1e-4, f"K=n-1 should reconstruct centered data; got mse={mse:.3e}"


# --- 2 & 3. residual / replace semantics -------------------------------------


class _FakeStreamSpec:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.flat_dim = int(np.prod(shape))


class _FakeDataset:
    """Minimal stand-in for H5Dataset for testing overwrite_full_embedding."""

    def __init__(self, mean, P, semantics, source, d):
        self.pca_mean = mean
        self.pca_components = P
        self.pca_semantics = semantics
        self.indices_truncate_embds_topk = None
        self.normalize = None
        self.stream_specs = [_FakeStreamSpec("t5", (d,))]
        self._source = source

    def denormalize(self, v):
        return v

    def __getitem__(self, idx):
        return self._source, None


class _FakeInference:
    """Bypass SFDInference.__init__ so we can exercise the PCA branch alone."""

    def __init__(self, dataset):
        self.dataset = dataset

    # borrow the two methods from the real class
    from inference.abstract import SFDInference

    get_full_embedding = SFDInference.get_full_embedding
    overwrite_full_embedding = SFDInference.overwrite_full_embedding


def _fit_pca(X, K):
    mean = X.mean(dim=0)
    Xc = X - mean
    U, S, V = torch.pca_lowrank(Xc, q=min(K, min(X.shape) - 1), niter=6)
    P = V[:, :K].T.contiguous()
    return mean, P


def test_residual_semantics_perfect_ssae_returns_source():
    """If SSAE reconstructs P(x - mean) perfectly, residual mode reproduces x."""
    n, d, K = 32, 40, 5
    torch.manual_seed(2)
    X = torch.randn(n, d, dtype=torch.float32)
    mean, P = _fit_pca(X, K)

    source = X[0]
    y = (source - mean) @ P.T  # perfect SSAE output in projected space

    ds = _FakeDataset(mean, P, semantics="residual", source=source.clone(), d=d)
    inf = _FakeInference(ds)
    streams = inf.overwrite_full_embedding(y, idx=0)
    out = streams["t5"].reshape(-1)

    max_err = torch.max(torch.abs(out - source)).item()
    assert max_err < 1e-4, f"residual mode should recover source; max_err={max_err:.3e}"


def test_replace_semantics_ignores_residual():
    """Replace mode outputs P.T y + mean regardless of the source residual."""
    n, d, K = 32, 40, 5
    torch.manual_seed(3)
    X = torch.randn(n, d, dtype=torch.float32)
    mean, P = _fit_pca(X, K)

    source = X[0]
    y = (source - mean) @ P.T
    expected = y @ P + mean

    ds = _FakeDataset(mean, P, semantics="replace", source=source.clone(), d=d)
    inf = _FakeInference(ds)
    streams = inf.overwrite_full_embedding(y, idx=0)
    out = streams["t5"].reshape(-1)

    max_err = torch.max(torch.abs(out - expected)).item()
    assert max_err < 1e-4, f"replace mode mismatch; max_err={max_err:.3e}"

    # and it should NOT equal the source (unless K = d, which it isn't)
    diff_to_source = torch.max(torch.abs(out - source)).item()
    assert diff_to_source > 1e-3, "replace mode should discard the residual"


# --- 4. explained variance vs holdout R^2 -----------------------------------


def test_explained_variance_bounds_holdout_r2(tmp_path):
    n_train, n_hold, d, K = 64, 32, 50, 8
    torch.manual_seed(4)
    # Low-rank + noise so the first PCs carry most of the variance.
    latent = torch.randn(n_train + n_hold, K, dtype=torch.float32)
    W = torch.randn(K, d, dtype=torch.float32)
    data = latent @ W + 0.05 * torch.randn(n_train + n_hold, d, dtype=torch.float32)
    X_train, X_hold = data[:n_train], data[n_train:]

    ds = _make_simulated_dataset(
        tmp_path / "ev", K=K, dim=d, n_prompts=n_train
    )
    ds.pca_mean = None
    ds.pca_components = None
    ds.X_simulated = X_train
    ds.X = X_train
    ds._compute_pca_projection()

    ev_ratio = ds.pca_explained_variance_ratio[:K].sum().item()

    mean = ds.pca_mean
    P = ds.pca_components
    X_hold_recon = (X_hold - mean) @ P.T @ P + mean

    hold_ss_res = torch.mean((X_hold - X_hold_recon) ** 2).item()
    hold_ss_tot = torch.mean((X_hold - X_hold.mean(dim=0)) ** 2).item()
    r2_hold = 1.0 - hold_ss_res / (hold_ss_tot + 1e-12)

    # EV on train should exceed R^2 on holdout (train fit >= holdout fit).
    # Small slack for finite-sample fluctuation.
    assert ev_ratio + 0.05 >= r2_hold, (
        f"train EV ({ev_ratio:.3f}) should upper-bound holdout R^2 ({r2_hold:.3f})"
    )
    assert r2_hold > 0.5, f"holdout R^2 too low ({r2_hold:.3f})"


# --- 5. simulated end-to-end smoke ------------------------------------------


def test_simulated_training_smoke(tmp_path):
    """1-epoch training run through the CLI entry point in PCA mode."""
    import yaml

    from trainable_inputs_all_clips import training

    fixture = _write_simulated_fixture(tmp_path / "smoke_data", n_prompts=24)

    yaml_params = {
        "training": {
            "model": {"model_name": "model_avg_feature", "using_blocs": False},
            "dataloader": {
                "folder_path": str(fixture) + "/",
                "truncate_n_prompts": None,
                "truncate_embds_topk": 8,
                "truncate_embds_method": "pca",
                "pca_semantics": "residual",
                "add_property_is_the_same": True,
                "normalize": None,
                "num_workers": 0,
                "simulated": {"simulated": True, "dim_clip_simulated": 32},
            },
            "training": {
                "n_epochs": 1,
                "print_frequency": 1,
                "save_model_frequency": None,
                "plot_frequency": 1,
                "seed": 0,
                "batch_size": 4,
                "lr": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "lr_scheduler": {
                    "lr_scheduler_type": None,
                    "lr_scheduler_linear": {"lr_scheduler_lr_final_linear": 0.0001},
                },
            },
            "sparse_feature_design": {"n_repeat": 2},
        }
    }
    yaml_path = tmp_path / "params.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_params, f)

    out = tmp_path / "run"
    training(output_folder=str(out), path_yaml=str(yaml_path), overwrite_output=True)

    assert (out / "model.pt").exists(), "training did not produce a checkpoint"
    assert (fixture / "pca_top_8.npz").exists(), "PCA cache was not written"
