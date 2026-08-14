"""Content-addressed cache of non-SSAE baselines for ``run_image_benchmark``.

Baselines (gt_embed, mean_arithmetic, ridge_embed, native_prompt, prompt_only, and
prompt_modified_packed) are independent of the SSAE checkpoint. Caching them by the
holdout data, training data, render contract, and packing policy lets subsequent runs
reuse the same images and metrics.

The cache identity has an explicit version. Render-contract changes therefore create a
new directory instead of silently reusing images produced under older semantics.
Simulated benchmark runs bypass this shared cache entirely.

Cache layout::

    <root>/<dataset_id>/
        manifest.json          # key components + methods populated
        per_sample.csv         # rows for cached baseline methods (one per (method, idx))
        summary.json           # per-method aggregates for cached methods
        images/<baseline>/00000.png ...
        images_pre_edit/<baseline>/...   # present iff drop mode populated
        images_swapped/<baseline>/...    # present iff swap mode populated
        fits/baselines_fits.npz          # cached MA + ridge params

Caveats:

* Locality flags (``locality_drop_one_attr`` / ``locality_swap_one_attr``) are not part of
  the dataset id — the cache grows as new variants are requested.
* Optional per-row metrics (DINO, LPIPS) are captured at population time. A downstream run
  that enables a metric absent from the cache gets an empty value; pre-warm with the
  desired metrics enabled if you need them everywhere.
* No cross-process lock. Concurrent populates of the same ``dataset_id`` may duplicate work
  but do not corrupt state; final CSV is rewritten on ``write``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


# Re-exported from the shared registry so method identity has exactly one definition.
from evaluation.method_labels import BASELINE_METHODS  # noqa: E402,F401

BASELINE_CACHE_IDENTITY_VERSION = 2
VARIANTS: tuple[str, ...] = ("post", "pre_edit", "swapped")


def variant_dir_name(variant: str) -> str:
    return "images" if variant == "post" else f"images_{variant}"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ordered_indices_fingerprint(
    indices: Iterable[int] | np.ndarray | torch.Tensor | None,
) -> dict:
    """Fingerprint the ordered top-k coordinates without storing the full list"""
    if indices is None:
        return {"algorithm": "sha256", "count": None, "sha256": None}

    ordered = torch.as_tensor(indices, dtype=torch.int64).detach().cpu().reshape(-1)
    canonical = json.dumps(ordered.tolist(), separators=(",", ":")).encode("utf-8")
    return {
        "algorithm": "sha256",
        "count": int(ordered.numel()),
        "sha256": _sha256_bytes(canonical),
    }


def compute_dataset_id(
    *,
    holdout_folder: Path,
    train_x: torch.Tensor,
    train_mask: torch.Tensor,
    base_seed: int,
    ridge_lambda: float,
    sd3_fingerprint: dict,
    truncate_embds_topk_indices: Iterable[int] | np.ndarray | torch.Tensor | None,
    packer_fingerprint: dict | None = None,
) -> tuple[str, dict]:
    holdout_hash = _sha256_file(Path(holdout_folder) / "prompts.json")
    train_x_np = train_x.detach().to(torch.float32).cpu().contiguous().numpy()
    train_mask_np = train_mask.detach().to(torch.float32).cpu().contiguous().numpy()
    train_hash = _sha256_bytes(train_x_np.tobytes() + train_mask_np.tobytes())
    key = {
        "baseline_cache_identity_version": BASELINE_CACHE_IDENTITY_VERSION,
        "holdout_hash": holdout_hash,
        "train_hash": train_hash,
        "train_shape": list(train_x_np.shape),
        "mask_shape": list(train_mask_np.shape),
        "base_seed": int(base_seed),
        "ridge_lambda": float(ridge_lambda),
        "sd3_fingerprint": sd3_fingerprint,
        "truncate_embds_topk_indices_fingerprint": ordered_indices_fingerprint(
            truncate_embds_topk_indices
        ),
        "packer_fingerprint": packer_fingerprint or {},
    }
    canonical = json.dumps(key, sort_keys=True).encode("utf-8")
    return _sha256_bytes(canonical)[:16], key


@dataclass
class BaselineCache:
    root: Path
    dataset_id: str
    key: dict
    rows: dict[tuple[str, int], dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dir = Path(self.root) / self.dataset_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "images").mkdir(exist_ok=True)
        (self.dir / "fits").mkdir(exist_ok=True)
        self._manifest_path = self.dir / "manifest.json"
        self._csv_path = self.dir / "per_sample.csv"
        self._load_rows()

    def _load_rows(self) -> None:
        if not self._csv_path.exists():
            return
        with open(self._csv_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    key = (r["method"], int(r["sample_idx"]))
                except (KeyError, ValueError):
                    continue
                self.rows[key] = dict(r)

    # image paths -----------------------------------------------------------
    def image_path(self, method: str, sample_idx: int, variant: str = "post") -> Path:
        return self.dir / variant_dir_name(variant) / method / f"{sample_idx:05d}.png"

    def has_image(self, method: str, sample_idx: int, variant: str = "post") -> bool:
        return self.image_path(method, sample_idx, variant).exists()

    def ensure_variant_dir(self, method: str, variant: str) -> Path:
        d = self.dir / variant_dir_name(variant) / method
        d.mkdir(parents=True, exist_ok=True)
        return d

    # rows ------------------------------------------------------------------
    def has_row(self, method: str, sample_idx: int) -> bool:
        return (method, sample_idx) in self.rows

    def get_row(self, method: str, sample_idx: int) -> dict | None:
        return self.rows.get((method, sample_idx))

    def upsert_row(self, row: dict) -> None:
        self.rows[(row["method"], int(row["sample_idx"]))] = dict(row)

    # fits ------------------------------------------------------------------
    def fits_path(self) -> Path:
        return self.dir / "fits" / "baselines_fits.npz"

    def save_fits(
        self,
        *,
        mu_ma: torch.Tensor,
        deltas_ma: torch.Tensor,
        W_ridge: torch.Tensor,
    ) -> None:
        np.savez(
            self.fits_path(),
            mu_ma=mu_ma.detach().cpu().numpy(),
            deltas_ma=deltas_ma.detach().cpu().numpy(),
            W_ridge=W_ridge.detach().cpu().numpy(),
        )

    def load_fits(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        path = self.fits_path()
        if not path.exists():
            return None
        z = np.load(path)
        return (
            torch.from_numpy(z["mu_ma"]),
            torch.from_numpy(z["deltas_ma"]),
            torch.from_numpy(z["W_ridge"]),
        )

    # finalize --------------------------------------------------------------
    def write(
        self,
        *,
        methods: Iterable[str],
        summary: dict | None = None,
        extra_manifest: dict | None = None,
    ) -> None:
        # Union with methods already present in the cache: a run that requests a subset
        # must not make the manifest claim the others were never populated.
        existing = {m for (m, _idx) in self.rows}
        methods = list(dict.fromkeys([*methods, *sorted(existing)]))
        if self.rows:
            all_keys: set[str] = set()
            for r in self.rows.values():
                all_keys.update(r.keys())
            preferred = [
                "task",
                "sample_idx",
                "method",
                "prompt",
                "edit_pid",
                "edit_attribute",
                "swap_target_pid",
                "swap_target_attribute",
                "swapped_prompt",
            ]
            ordered = [k for k in preferred if k in all_keys] + sorted(
                all_keys - set(preferred)
            )

            def _sk(r: dict) -> tuple[int, int]:
                m = r["method"]
                mi = methods.index(m) if m in methods else len(methods)
                return (int(r["sample_idx"]), mi)

            rows_sorted = sorted(self.rows.values(), key=_sk)
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=ordered)
                w.writeheader()
                for r in rows_sorted:
                    w.writerow({k: r.get(k, "") for k in ordered})

        manifest = {
            "dataset_id": self.dataset_id,
            "key": self.key,
            "methods": methods,
        }
        if extra_manifest:
            manifest.update(extra_manifest)
        self._manifest_path.write_text(json.dumps(manifest, indent=2))
        if summary is not None:
            (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))


def load_or_create(
    root: Path,
    *,
    holdout_folder: Path,
    train_x: torch.Tensor,
    train_mask: torch.Tensor,
    base_seed: int,
    ridge_lambda: float,
    sd3_fingerprint: dict,
    truncate_embds_topk_indices: Iterable[int] | np.ndarray | torch.Tensor | None,
    packer_fingerprint: dict | None = None,
) -> BaselineCache:
    dataset_id, key = compute_dataset_id(
        holdout_folder=holdout_folder,
        train_x=train_x,
        train_mask=train_mask,
        base_seed=base_seed,
        ridge_lambda=ridge_lambda,
        sd3_fingerprint=sd3_fingerprint,
        truncate_embds_topk_indices=truncate_embds_topk_indices,
        packer_fingerprint=packer_fingerprint,
    )
    return BaselineCache(root=Path(root), dataset_id=dataset_id, key=key)
