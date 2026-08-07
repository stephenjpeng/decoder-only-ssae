"""Machine-readable run provenance (E0 / AUG-02).

Every training run and every evaluation run should be traceable to the exact code,
config, data and seed that produced it. Before this module the repo recorded provenance
in three partial and non-machine-readable ways:

* ``Logger._register_git_commit`` wrote the SHA as free text into ``training.log``;
* ``trainable_inputs_all_clips.training`` dumped ``all_params.yaml`` from a ``tp`` dict
  that still held a live ``Logger`` and ``torch.device``, so the YAML carried Python
  object tags;
* ``run_image_benchmark`` wrote a ``manifest.json`` with no code, model or data identity
  at all.

``write_run_manifest`` emits a single ``run_manifest.json`` with a stable schema, so
downstream tooling (report builders, the freeze step on Aug 25) can join a result row to
its origin without parsing logs.

Nothing here raises on missing information — a run on a git-less rsync'd copy, or with no
dataset attached, still produces a manifest with the corresponding fields set to ``None``
and a note in ``warnings``. Provenance capture must never break a run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

MANIFEST_FILENAME = "run_manifest.json"

# Files that define a split's identity. Hashed when present; silently skipped otherwise.
# Sidecars (top-k indices, min/max) are resolved dynamically because their names embed k.
_DATASET_IDENTITY_FILES = ("prompts.json", "properties.json", "properties_same.json")


# --------------------------------------------------------------------------- hashing


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str | None:
    """Streaming SHA-256 of a file, or ``None`` if it does not exist."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json_obj(obj: Any) -> str:
    """SHA-256 of a canonical JSON encoding — order-insensitive for dicts."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# ------------------------------------------------------------------------------ git


def git_provenance(repo_dir: str | Path = ".") -> dict:
    """Git SHA, branch and dirty flag. All ``None`` outside a repo."""
    out: dict = {"sha": None, "branch": None, "dirty": None, "describe": None}

    def _run(args: list[str]) -> str | None:
        try:
            r = subprocess.run(
                ["git", *args],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if r.returncode != 0:
            return None
        return r.stdout.strip()

    sha = _run(["rev-parse", "HEAD"])
    if sha is None:
        return out
    out["sha"] = sha
    out["branch"] = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    out["describe"] = _run(["describe", "--always", "--dirty", "--tags"])
    status = _run(["status", "--porcelain"])
    # Untracked-only changes still count as dirty: an unstaged new module can change
    # behaviour just as much as a modified one.
    out["dirty"] = bool(status) if status is not None else None
    return out


# ------------------------------------------------------------------------- environment


def environment_provenance() -> dict:
    env: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": None,
        "cuda_available": None,
        "cuda_device_name": None,
        "numpy": None,
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if env["cuda_available"]:
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - torch is a hard dep in practice
        pass
    try:
        import numpy

        env["numpy"] = numpy.__version__
    except Exception:  # pragma: no cover
        pass
    return env


# ----------------------------------------------------------------------------- config


def jsonable_config(tp: dict) -> dict:
    """Strip a resolved ``tp`` dict down to something JSON round-trippable.

    ``tp`` accumulates live objects during ``training()`` (``logger``, ``device``,
    ``optimizer``). Those are replaced by a ``"<repr>"`` string rather than dropped, so the
    manifest still shows that the key was set.
    """
    out: dict = {}
    for k, v in sorted(tp.items()):
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            try:
                json.dumps(list(v))
                out[k] = list(v)
            except TypeError:
                out[k] = f"<{type(v).__name__}>"
        elif isinstance(v, dict):
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = f"<{type(v).__name__}>"
        else:
            out[k] = f"<{type(v).__name__}>"
    return out


# ---------------------------------------------------------------------------- dataset


def dataset_fingerprint(dataset: Any = None, *, folder_path: str | Path | None = None) -> dict:
    """Identity of the data a run consumed.

    Accepts an ``H5Dataset`` (preferred — also captures derived shapes) and/or a bare
    folder path. Hashes the split-defining JSONs plus the truncation/normalisation
    sidecars for the *specific* top-k in use, since those determine which coordinates the
    model ever sees.
    """
    fp: dict = {
        "folder_path": None,
        "n_prompts": None,
        "dim_x": None,
        "truncate_embds_topk": None,
        "normalize": None,
        "pca_rotation": None,
        "backbone": None,
        "n_properties": None,
        "n_categories": None,
        "files": {},
    }

    if dataset is not None:
        folder_path = folder_path or getattr(dataset, "folder_path", None)
        fp["dim_x"] = _int_or_none(getattr(dataset, "dim_x", None))
        fp["truncate_embds_topk"] = _int_or_none(
            getattr(dataset, "truncate_embds_topk", None)
        )
        fp["normalize"] = getattr(dataset, "normalize", None)
        fp["pca_rotation"] = _bool_or_none(getattr(dataset, "pca_rotation", None))
        fp["backbone"] = getattr(dataset, "backbone_name", None)
        try:
            fp["n_prompts"] = len(dataset)
        except Exception:
            pass
        props = getattr(dataset, "properties", None)
        if props is not None:
            fp["n_properties"] = _int_or_none(getattr(props, "n_properties", None))
            fp["n_categories"] = _int_or_none(getattr(props, "n_categories", None))

    if folder_path is None:
        return fp

    folder = Path(str(folder_path))
    fp["folder_path"] = folder.as_posix()
    if not folder.is_dir():
        return fp

    for name in _DATASET_IDENTITY_FILES:
        digest = sha256_file(folder / name)
        if digest is not None:
            fp["files"][name] = digest

    manifest = folder / "embds" / "manifest.json"
    digest = sha256_file(manifest)
    if digest is not None:
        fp["files"]["embds/manifest.json"] = digest

    k = fp["truncate_embds_topk"]
    if k is not None:
        # Mirrors H5Dataset._cache_suffix(): PCA and non-PCA sidecars coexist in one
        # folder, so hash only the variant this run actually reads.
        suffix = "_pca" if fp["pca_rotation"] else ""
        for name in (
            f"indices_top_{k}{suffix}.json",
            f"embds_min_top_{k}{suffix}.json",
            f"embds_max_top_{k}{suffix}.json",
        ):
            digest = sha256_file(folder / name)
            if digest is not None:
                fp["files"][name] = digest

    return fp


def _int_or_none(v: Any) -> int | None:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _bool_or_none(v: Any) -> bool | None:
    return None if v is None else bool(v)


# --------------------------------------------------------------------------- assembly


def build_run_manifest(
    *,
    run_kind: str,
    output_dir: str | Path,
    config: dict | None = None,
    seed: int | None = None,
    dataset: Any = None,
    dataset_folder: str | Path | None = None,
    extra_datasets: dict[str, Any] | None = None,
    packer_fingerprint: dict | None = None,
    model_fingerprint: dict | None = None,
    extra: dict | None = None,
    repo_dir: str | Path = ".",
) -> dict:
    """Assemble the manifest dict. See :func:`write_run_manifest` for the usual entry point.

    ``run_kind`` is a free-form tag (``"training"``, ``"image_benchmark"``, ``"analysis"``)
    used by report builders to decide how to interpret the rest.
    """
    warnings: list[str] = []

    git_info = git_provenance(repo_dir)
    if git_info["sha"] is None:
        warnings.append("git provenance unavailable (not a git repo or git not on PATH)")
    elif git_info["dirty"]:
        warnings.append(
            "working tree is dirty — this run is not reproducible from the recorded SHA alone"
        )

    cfg = jsonable_config(config) if config else {}

    datasets: dict[str, dict] = {}
    if dataset is not None or dataset_folder is not None:
        datasets["primary"] = dataset_fingerprint(dataset, folder_path=dataset_folder)
    for name, ds in (extra_datasets or {}).items():
        if isinstance(ds, (str, Path)):
            datasets[name] = dataset_fingerprint(folder_path=ds)
        else:
            datasets[name] = dataset_fingerprint(ds)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_kind": run_kind,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_dir": Path(output_dir).as_posix(),
        "git": git_info,
        "command": {
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
        },
        "environment": environment_provenance(),
        "seed": _int_or_none(seed),
        "config": cfg,
        "config_hash": sha256_json_obj(cfg) if cfg else None,
        "datasets": datasets,
        "packer_fingerprint": packer_fingerprint,
        "model_fingerprint": model_fingerprint,
        "extra": extra or {},
        "warnings": warnings,
    }
    return manifest


def write_run_manifest(
    output_dir: str | Path,
    *,
    run_kind: str,
    filename: str = MANIFEST_FILENAME,
    **kwargs: Any,
) -> Path:
    """Build and write ``<output_dir>/run_manifest.json``. Returns the path written.

    Never raises: provenance capture must not be able to fail a run. On error the
    exception text is written into a minimal stub manifest so the failure is visible.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    try:
        manifest = build_run_manifest(run_kind=run_kind, output_dir=out, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_kind": run_kind,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error": f"{type(exc).__name__}: {exc}",
            "warnings": ["manifest generation failed"],
        }
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def checkpoint_fingerprint(checkpoint_dir: str | Path) -> dict:
    """Identify a trained run folder: its path, its ``model.pt`` hash and its config hash.

    Used by evaluation runs so a ``bench_out`` directory can be joined back to the exact
    weights it scored. Hashing a ~100 MB ``model.pt`` costs well under a second and is
    worth it — filenames are reused across sweeps.
    """
    d = Path(checkpoint_dir)
    fp: dict = {
        "checkpoint_dir": d.as_posix(),
        "model_pt_sha256": sha256_file(d / "model.pt"),
        "params_yaml_sha256": sha256_file(d / "params.yaml"),
        "all_params_yaml_sha256": sha256_file(d / "all_params.yaml"),
        "train_run_manifest": None,
    }
    train_manifest = d / MANIFEST_FILENAME
    if train_manifest.is_file():
        try:
            m = json.loads(train_manifest.read_text(encoding="utf-8"))
            fp["train_run_manifest"] = {
                "git_sha": (m.get("git") or {}).get("sha"),
                "seed": m.get("seed"),
                "config_hash": m.get("config_hash"),
                "created_utc": m.get("created_utc"),
            }
        except (OSError, json.JSONDecodeError):
            pass
    return fp


def iter_manifests(root: str | Path) -> Iterable[tuple[Path, dict]]:
    """Yield ``(path, manifest)`` for every ``run_manifest.json`` under ``root``."""
    for path in sorted(Path(root).rglob(MANIFEST_FILENAME)):
        try:
            yield path, json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
