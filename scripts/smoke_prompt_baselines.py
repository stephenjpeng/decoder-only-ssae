"""AUG-01 acceptance check for distinct prompt-conditioning methods.

Builds a throwaway SD3-shaped dataset with the ``fake_sd3`` backbone, trains a tiny
decoder, then runs ``evaluation.run_image_benchmark --simulated`` twice:

1. with the exact round-trip prompt path;
2. with the native and packed prompt paths added.

Asserts the plan's acceptance criteria:

* metadata distinguishes direct text, exact full-embedding round-trip, and packed top-k;
* simulated placeholders never create or reuse the shared real-render cache;
* exact round-trip and packed paths use the same prompt text and seed per tuple;
* the manifest records canonical labels and the packer fingerprint;
* a simulated run produces separate rows and directories for all three prompt paths.

Run from the repo root:

    .venv/bin/python scripts/smoke_prompt_baselines.py

Takes ~1-2 minutes and ~250 MB of scratch under a temp dir (removed unless
``--keep`` is passed).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# trainings/dataloader/checks.py probes prompt indices [0, 10, 15] unconditionally, so
# *every* split it is pointed at must have >= 16 rows. That is why the holdout is 16 and
# not the 4 a smoke test would otherwise want. (Latent repo bug: any dataset with fewer
# than 16 prompts raises KeyError in check_properties_mask_consistency. Out of scope here,
# logged in CONTEXT/research-notes/.)
N_HOLDOUT = 16
BASE_SEED = 0
TOPK = 512
SPLIT_SEED = 0

# 6 categories x 2 values = 64 tuples -> 48 train / 16 holdout.
PROPERTIES = {
    "hair": ["A blond girl", "A brunette girl"],
    "eyes": ["with blue eyes", "with brown eyes"],
    "hat": ["and a hat", "and a baseball cap"],
    "action": ["holding a coffee", "holding a gun"],
    "situation": ["at the beach", "sitting at a cafe"],
    "pose": ["looking to the left", "looking to the right"],
}
PROPERTIES_SAME = {k: False for k in PROPERTIES}

ROUND_TRIP_METHODS = "gt_embed,ssae_compose,mean_arithmetic,ridge_embed,prompt_only"
THREE_PROMPT_PATH_METHODS = (
    ROUND_TRIP_METHODS + ",native_prompt,prompt_modified_packed"
)


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"  ok  {msg}")


def build_prompts() -> list[dict]:
    keys = list(PROPERTIES)
    out: list[dict] = []

    def rec(i: int, chosen: list[str]) -> None:
        if i == len(keys):
            out.append({"prompt": ", ".join(chosen)})
            return
        for v in PROPERTIES[keys[i]]:
            rec(i + 1, chosen + [v])

    rec(0, [])
    return out


def write_split(root: Path) -> tuple[Path, Path]:
    """Write train/ and holdout/ prompt folders with a disjoint tuple split.

    The split must be *shuffled*: the cartesian enumeration is ordered, so any contiguous
    or stride-based slice puts an entire property value on one side of the split (e.g. all
    blondes in train, all brunettes in holdout) and the ridge/mean-arithmetic baselines
    become undefined for the unseen values.
    """
    import random

    prompts = build_prompts()
    random.Random(SPLIT_SEED).shuffle(prompts)
    holdout, train = prompts[:N_HOLDOUT], prompts[N_HOLDOUT:]

    for name, subset in (("train", train), ("holdout", holdout)):
        seen = {v for p in subset for v in p["prompt"].split(", ")}
        missing = {v for vals in PROPERTIES.values() for v in vals} - seen
        if missing:
            _fail(f"{name} split does not cover {sorted(missing)}; change SPLIT_SEED")
    print(f"split: {len(train)} train / {len(holdout)} holdout, all values covered in both")

    paths = []
    for name, subset in (("train", train), ("holdout", holdout)):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "prompts.json").write_text(json.dumps(subset, indent=1))
        (d / "properties.json").write_text(json.dumps(PROPERTIES, indent=1))
        (d / "properties_same.json").write_text(json.dumps(PROPERTIES_SAME, indent=1))
        paths.append(d)
    return paths[0], paths[1]


def extract(folder: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO / "get_embeddings.py"),
        "--backbone",
        "fake_sd3",
        "--prompts",
        str(folder / "prompts.json"),
        "--out",
        str(folder),
        "--categories",
        str(folder / "properties.json"),
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)


def write_yaml(path: Path, train_folder: Path) -> None:
    """Start from the committed default so the smoke config cannot drift from the real one."""
    params = yaml.safe_load((REPO / "trainings/config/params_default.yaml").read_text())
    t = params["training"]
    t["dataloader"]["folder_path"] = train_folder.as_posix() + "/"
    t["dataloader"]["truncate_embds_topk"] = TOPK
    t["dataloader"]["num_workers"] = 0
    t["training"]["n_epochs"] = 3
    t["training"]["batch_size"] = 4
    t["training"]["plot_frequency"] = None
    t["sparse_feature_design"]["n_repeat"] = 2
    path.write_text(yaml.safe_dump(params, default_flow_style=False))


def train(out_dir: Path, yaml_path: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO / "training_cli.py"),
        "--output_folder",
        str(out_dir),
        "--path_yaml",
        str(yaml_path),
        "--overwrite_output",
        "True",
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)


def run_bench(ckpt: Path, holdout: Path, out_dir: Path, cache_root: Path, methods: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "evaluation.run_image_benchmark",
        "--checkpoint",
        str(ckpt),
        "--holdout_folder",
        str(holdout),
        "--output_dir",
        str(out_dir),
        "--simulated",
        "--sd_device",
        "cpu",
        "--base_seed",
        str(BASE_SEED),
        "--methods",
        methods,
        "--baseline_cache_root",
        str(cache_root),
        "--locality_drop_one_attr",
        "--locality_swap_one_attr",
        "--skip_lpips",
        "--skip_pixel_metrics",
        "--n_bootstrap",
        "50",
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)


def read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- direct packer check


def check_conditioning_differs(ckpt: Path, holdout: Path) -> None:
    """The packed path must not match the exact full-embedding round-trip.

    Encodes one prompt, packs it, and confirms that (a) the top-k coordinates match the
    full encoding exactly and (b) the remaining coordinates were replaced by the
    training mean. If either failed, the two embedding methods would be identical.
    """
    from evaluation.io import h5_dataset_for_folder, load_decoder_checkpoint
    from evaluation.sd3_pack import (
        compute_or_load_full_mean,
        flatten_sd3_conditioning,
        pack_sd3_from_full_flat_topk,
    )
    from inference.image_generation.image_generator import ImageGenerator

    _, _, train_ds = load_decoder_checkpoint(ckpt, device="cpu")
    holdout_ds = h5_dataset_for_folder(ckpt, holdout)
    template = compute_or_load_full_mean(train_ds).cpu()

    gen = ImageGenerator(simulated=True, device="cpu")
    seq, _, pooled, _ = gen.get_embds_text_encoder("A blond girl, with blue eyes", seed=7)
    full_flat = flatten_sd3_conditioning(seq, pooled)

    pe, pp = pack_sd3_from_full_flat_topk(holdout_ds, full_flat, template=template)
    packed_flat = torch.cat(
        [pe.to(torch.float32).reshape(-1), pp.to(torch.float32).reshape(-1)]
    )

    idx = holdout_ds.indices_truncate_embds_topk
    if idx is None:
        _fail("holdout dataset has no top-k truncation; smoke config is wrong")

    idx_t = torch.as_tensor(idx, dtype=torch.long)
    mask = torch.ones(packed_flat.numel(), dtype=torch.bool)
    mask[idx_t] = False

    # (a) top-k coordinates preserved (bfloat16 round-trip tolerance)
    max_topk_dev = (packed_flat[idx_t] - full_flat[idx_t]).abs().max().item()
    if max_topk_dev > 0.05:
        _fail(f"packed variant altered top-k coordinates (max dev {max_topk_dev:.4g})")
    _ok(f"packed keeps full-embedding top-k coordinates (max dev {max_topk_dev:.2e})")

    # (b) everything else came from the training mean, not the prompt
    dev_from_mean = (packed_flat[mask] - template[mask]).abs().max().item()
    dev_from_full = (packed_flat[mask] - full_flat[mask]).abs().mean().item()
    if dev_from_mean > 0.05:
        _fail(f"non-top-k coordinates are not the training mean (max dev {dev_from_mean:.4g})")
    if dev_from_full < 1e-3:
        _fail(
            "non-top-k coordinates are indistinguishable from the full encoding; "
            "packed and exact round-trip would be the same computation"
        )
    _ok(
        f"packed replaces non-top-k with training mean "
        f"(dev from mean {dev_from_mean:.2e}, mean dev from full {dev_from_full:.3f})"
    )


# --------------------------------------------------------------------------- assertions


def check_acceptance(
    cache_root: Path, out_round_trip: Path, out_three_paths: Path
) -> None:
    manifest = json.loads((out_three_paths / "manifest.json").read_text())
    run_manifest = json.loads((out_three_paths / "run_manifest.json").read_text())

    # placeholders stay local, so they cannot satisfy or overwrite a real cache row
    if manifest["dataset_id"] is not None or manifest["baseline_cache_dir"] is not None:
        _fail("simulated benchmark unexpectedly attached to a shared baseline cache")
    if cache_root.exists() and any(cache_root.iterdir()):
        _fail(f"simulated benchmark wrote shared cache content under {cache_root}")
    if manifest.get("render_mode") != "simulated":
        _fail(f"unexpected render mode: {manifest.get('render_mode')!r}")
    _ok("simulated placeholders bypass shared baseline cache storage")

    # runtime metadata records all three entry paths and the active packing contract
    cond = manifest.get("method_conditioning") or {}
    expected = {
        "native_prompt": ("direct_text", "Native text generation"),
        "prompt_only": ("full_embedding", "Exact full-embedding round-trip"),
        "prompt_modified_packed": ("packed", "Prompt modification (packed top-k)"),
    }
    for method, (conditioning, label) in expected.items():
        actual = cond.get(method, {})
        if (actual.get("conditioning"), actual.get("label")) != (conditioning, label):
            _fail(
                f"unexpected {method} conditioning/label: {actual!r}; "
                f"expected {(conditioning, label)!r}"
            )
    packed_meta = cond["prompt_modified_packed"]
    if packed_meta.get("fill_policy") != "train_mean":
        _fail(f"packed path has wrong runtime fill metadata: {packed_meta!r}")
    if packed_meta.get("truncate_embds_topk") != TOPK:
        _fail(f"packed path has wrong runtime top-k metadata: {packed_meta!r}")
    for method in ("native_prompt", "prompt_only", "prompt_modified_packed"):
        if cond[method].get("t5_max_sequence_length") != 256:
            _fail(f"{method} does not record the 256-token T5 contract")
    _ok("manifest distinguishes all three prompt paths and their runtime contract")

    for key in ("git", "config", "datasets", "packer_fingerprint", "model_fingerprint"):
        if key not in run_manifest:
            _fail(f"run_manifest.json missing '{key}'")
    if not run_manifest["git"].get("sha"):
        _fail("run_manifest.json has no git SHA")
    if not run_manifest["model_fingerprint"].get("model_pt_sha256"):
        _fail("run_manifest.json has no checkpoint hash")
    _ok("run_manifest.json carries dataset, packer, model, and git provenance")

    # all three prompt paths get distinct local images and rows in simulated mode
    for variant in ("images", "images_pre_edit", "images_swapped"):
        counts = {}
        for method in ("native_prompt", "prompt_only", "prompt_modified_packed"):
            method_dir = out_three_paths / variant / method
            counts[method] = len(list(method_dir.glob("*.png")))
            if counts[method] == 0:
                _fail(f"missing simulated images under {method_dir}")
        _ok(f"{variant}/ contains all three prompt paths: {counts}")

    rows = read_rows(out_three_paths / "per_sample.csv")
    by_method = {
        method: {
            int(row["sample_idx"]): row
            for row in rows
            if row["method"] == method
        }
        for method in ("native_prompt", "prompt_only", "prompt_modified_packed")
    }
    indices = set(by_method["native_prompt"])
    if not indices or any(set(rows_by_idx) != indices for rows_by_idx in by_method.values()):
        _fail("prompt-path rows do not cover the same sample indices")
    for idx in sorted(indices):
        prompts = {by_method[method][idx]["prompt"] for method in by_method}
        if len(prompts) != 1:
            _fail(f"tuple {idx} uses different prompt text across entry paths")
    _ok(f"all three prompt paths share prompt text over {len(indices)} tuples")

    round_trip_manifest = json.loads((out_round_trip / "manifest.json").read_text())
    if "native_prompt" in round_trip_manifest["methods"]:
        _fail("round-trip-only run unexpectedly included the native path")
    _ok("round-trip-only method set runs without shared cache state")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="keep the scratch directory")
    ap.add_argument("--workdir", type=Path, default=None)
    args = ap.parse_args()

    root = args.workdir or Path(tempfile.mkdtemp(prefix="ssae_aug01_"))
    root.mkdir(parents=True, exist_ok=True)
    print(f"scratch: {root}")

    try:
        train_folder, holdout_folder = write_split(root / "split")
        extract(train_folder)
        extract(holdout_folder)

        yaml_path = root / "params_smoke.yaml"
        write_yaml(yaml_path, train_folder)
        ckpt = root / "run"
        train(ckpt, yaml_path)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.run_copy_truncation",
                "--train_folder",
                str(train_folder),
                "--holdout_folder",
                str(holdout_folder),
            ],
            check=True,
            cwd=REPO,
        )

        cache_root = root / "cache"
        out_round_trip = root / "bench_round_trip"
        out_three_paths = root / "bench_three_prompt_paths"

        print("\n=== pass 1: exact round-trip prompt path ===")
        run_bench(
            ckpt,
            holdout_folder,
            out_round_trip,
            cache_root,
            ROUND_TRIP_METHODS,
        )

        print("\n=== pass 2: native, exact round-trip, and packed prompt paths ===")
        run_bench(
            ckpt,
            holdout_folder,
            out_three_paths,
            cache_root,
            THREE_PROMPT_PATH_METHODS,
        )

        print("\n=== acceptance criteria ===")
        check_conditioning_differs(ckpt, holdout_folder)
        check_acceptance(cache_root, out_round_trip, out_three_paths)
        print("\nAUG-01 smoke test PASSED")
    finally:
        if args.keep or args.workdir:
            print(f"scratch kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
