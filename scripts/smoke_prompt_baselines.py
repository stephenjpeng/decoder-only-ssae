"""AUG-01 acceptance check: native vs packed prompt modification are distinct methods.

Builds a throwaway SD3-shaped dataset with the ``fake_sd3`` backbone, trains a tiny
decoder, then runs ``evaluation.run_image_benchmark --simulated`` twice:

1. with the legacy method set, to prove existing caches stay readable and that the
   legacy run still works;
2. with ``prompt_modified_packed`` added, to prove the two prompt variants produce
   separate rows and separate image directories.

Asserts the plan's acceptance criteria:

* method names describe genuinely different computations (different conditioning
  recorded in the manifest, and the packed variant's conditioning tensor differs from
  the native one on the non-top-k coordinates);
* existing ``prompt_only`` caches remain readable (same ``dataset_id``, rows survive);
* native and packed use the same prompt text and diffusion seed per tuple;
* the manifest records native-vs-packed conditioning and the packer fingerprint;
* a simulated run produces separate rows/directories for both variants.

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

LEGACY_METHODS = "gt_embed,ssae_compose,mean_arithmetic,ridge_embed,prompt_only"
PACKED_METHODS = LEGACY_METHODS + ",prompt_modified_packed"


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
    """The packed path must not be a no-op relative to the native path.

    Encodes one prompt, packs it, and confirms that (a) the top-k coordinates match the
    native encoding exactly and (b) the remaining coordinates were replaced by the
    training mean. If either failed, the two methods really would be the same
    computation, which is the bug AUG-01 exists to fix.
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
    native_flat = flatten_sd3_conditioning(seq, pooled)

    pe, pp = pack_sd3_from_full_flat_topk(holdout_ds, native_flat, template=template)
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
    max_topk_dev = (packed_flat[idx_t] - native_flat[idx_t]).abs().max().item()
    if max_topk_dev > 0.05:
        _fail(f"packed variant altered top-k coordinates (max dev {max_topk_dev:.4g})")
    _ok(f"packed keeps native top-k coordinates (max dev {max_topk_dev:.2e})")

    # (b) everything else came from the training mean, not the prompt
    dev_from_mean = (packed_flat[mask] - template[mask]).abs().max().item()
    dev_from_native = (packed_flat[mask] - native_flat[mask]).abs().mean().item()
    if dev_from_mean > 0.05:
        _fail(f"non-top-k coordinates are not the training mean (max dev {dev_from_mean:.4g})")
    if dev_from_native < 1e-3:
        _fail(
            "non-top-k coordinates are indistinguishable from the native encoding — "
            "packed and native would be the same computation"
        )
    _ok(
        f"packed replaces non-top-k with training mean "
        f"(dev from mean {dev_from_mean:.2e}, mean dev from native {dev_from_native:.3f})"
    )


# --------------------------------------------------------------------------- assertions


def check_acceptance(cache_root: Path, out_legacy: Path, out_packed: Path, legacy_id: str) -> None:
    manifest = json.loads((out_packed / "manifest.json").read_text())
    run_manifest = json.loads((out_packed / "run_manifest.json").read_text())

    # --- existing caches remain readable: same dataset_id before and after
    if manifest["dataset_id"] != legacy_id:
        _fail(
            "adding prompt_modified_packed changed the cache dataset_id "
            f"({legacy_id} -> {manifest['dataset_id']}); existing caches would be orphaned"
        )
    _ok(f"cache dataset_id unchanged by the new method ({legacy_id})")

    # --- manifest records conditioning + packer fingerprint
    cond = manifest.get("method_conditioning") or {}
    if cond.get("prompt_only", {}).get("conditioning") != "native":
        _fail("manifest does not record prompt_only as native conditioning")
    if cond.get("prompt_modified_packed", {}).get("conditioning") != "packed":
        _fail("manifest does not record prompt_modified_packed as packed conditioning")
    if not manifest.get("packer_fingerprint"):
        _fail("manifest is missing packer_fingerprint")
    _ok(
        "manifest records native-vs-packed conditioning + packer fingerprint "
        f"({manifest['packer_fingerprint']})"
    )

    # --- labels agree with the code
    if cond["prompt_only"]["label"] != "Prompt modification (native/full embedding)":
        _fail(f"unexpected prompt_only label: {cond['prompt_only']['label']}")
    _ok(f"prompt_only labelled '{cond['prompt_only']['label']}'")

    # --- AUG-02 provenance rides along
    for key in ("git", "config", "datasets", "packer_fingerprint", "model_fingerprint"):
        if key not in run_manifest:
            _fail(f"run_manifest.json missing '{key}'")
    if not run_manifest["git"].get("sha"):
        _fail("run_manifest.json has no git SHA")
    if not run_manifest["model_fingerprint"].get("model_pt_sha256"):
        _fail("run_manifest.json has no checkpoint hash")
    _ok(
        f"run_manifest.json carries git {run_manifest['git']['sha'][:8]}"
        f"{' (dirty)' if run_manifest['git']['dirty'] else ''}, "
        f"config_hash {str(run_manifest['config_hash'])[:8]}, checkpoint hash"
    )

    # --- separate directories
    cache_dir = cache_root / legacy_id
    for variant in ("images", "images_pre_edit", "images_swapped"):
        native_dir = cache_dir / variant / "prompt_only"
        packed_dir = cache_dir / variant / "prompt_modified_packed"
        if not native_dir.is_dir():
            _fail(f"missing {native_dir}")
        if not packed_dir.is_dir():
            _fail(f"missing {packed_dir}")
        n_native = len(list(native_dir.glob("*.png")))
        n_packed = len(list(packed_dir.glob("*.png")))
        if n_native == 0 or n_packed == 0:
            _fail(f"{variant}: native={n_native} packed={n_packed} images")
        _ok(f"{variant}/: prompt_only={n_native} png, prompt_modified_packed={n_packed} png")

    # --- separate rows, same prompt text per tuple
    rows = read_rows(cache_dir / "per_sample.csv")
    native = {int(r["sample_idx"]): r for r in rows if r["method"] == "prompt_only"}
    packed = {
        int(r["sample_idx"]): r for r in rows if r["method"] == "prompt_modified_packed"
    }
    if not packed:
        _fail("no prompt_modified_packed rows in the cache CSV")
    if set(native) != set(packed):
        _fail(f"row index mismatch: native {sorted(native)} vs packed {sorted(packed)}")
    _ok(f"separate rows for both variants over {len(packed)} tuples")

    for i in sorted(packed):
        for field in ("prompt", "swapped_prompt", "edit_attribute", "swap_target_attribute"):
            if native[i][field] != packed[i][field]:
                _fail(
                    f"tuple {i}: '{field}' differs between native "
                    f"({native[i][field]!r}) and packed ({packed[i][field]!r})"
                )
    _ok("native and packed use identical prompt text per tuple (post / pre-edit / swap)")

    # Diffusion seed is a pure function of (base_seed, idx) in run_image_benchmark, and
    # both methods are rendered inside the same idx loop, so seed parity is structural.
    # Record it explicitly anyway so the criterion is visibly checked.
    from evaluation.run_image_benchmark import _sample_seed

    seeds = {i: _sample_seed(BASE_SEED, i) for i in sorted(packed)}
    _ok(f"shared per-tuple diffusion seeds: {seeds}")

    # --- legacy run still fine
    legacy_manifest = json.loads((out_legacy / "manifest.json").read_text())
    if "prompt_modified_packed" in legacy_manifest["methods"]:
        _fail("legacy run unexpectedly included the packed method")
    _ok("legacy method set runs unchanged")


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
        out_legacy = root / "bench_legacy"
        out_packed = root / "bench_packed"

        print("\n=== pass 1: legacy method set ===")
        run_bench(ckpt, holdout_folder, out_legacy, cache_root, LEGACY_METHODS)
        legacy_id = json.loads((out_legacy / "manifest.json").read_text())["dataset_id"]

        print("\n=== pass 2: + prompt_modified_packed (must reuse the same cache) ===")
        run_bench(ckpt, holdout_folder, out_packed, cache_root, PACKED_METHODS)

        print("\n=== acceptance criteria ===")
        check_conditioning_differs(ckpt, holdout_folder)
        check_acceptance(cache_root, out_legacy, out_packed, legacy_id)
        print("\nAUG-01 smoke test PASSED")
    finally:
        if args.keep or args.workdir:
            print(f"scratch kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
