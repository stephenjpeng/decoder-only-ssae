"""Generate training YAMLs for small hyperparameter sweeps (ablations)."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


def deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_value(v_raw: str):
    v_raw = v_raw.strip()
    if v_raw.lower() in ("true", "false"):
        return v_raw.lower() == "true"
    if v_raw.isdigit() or (v_raw.startswith("-") and v_raw[1:].isdigit()):
        return int(v_raw)
    if _is_float(v_raw):
        return float(v_raw)
    return v_raw


def _patch_from_spec(spec: str) -> dict:
    """``spec``: comma-separated ``path/to/key=value`` (path segments with ``/``)."""
    patch_flat: dict = {}
    for kv in spec.split(","):
        kv = kv.strip()
        if not kv:
            continue
        key, val = kv.split("=", 1)
        path = key.strip().split("/")
        cur: dict = patch_flat
        for seg in path[:-1]:
            cur = cur.setdefault(seg, {})
        cur[path[-1]] = _parse_value(val)
    return patch_flat


def main() -> None:
    p = argparse.ArgumentParser(
        description="Write YAML configs under output_dir from a template + grid."
    )
    p.add_argument(
        "--template",
        type=Path,
        default=Path("trainings/config/params_default.yaml"),
    )
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--grid",
        type=str,
        default="",
        help="Semicolon-separated override specs. Each spec: comma-separated path/to/key=value "
        '(path with "/"). Example: '
        '"training/sparse_feature_design/n_repeat=5;training/sparse_feature_design/n_repeat=10"',
    )
    args = p.parse_args()

    with open(args.template, "r", encoding="utf-8") as f:
        template = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.grid.strip():
        entries = [
            {"training": {"sparse_feature_design": {"n_repeat": v}}}
            for v in (1, 5, 10, 20)
        ]
    else:
        entries = []
        for part in args.grid.split(";"):
            part = part.strip()
            if not part:
                continue
            entries.append(_patch_from_spec(part))

    written = []
    for i, flat_patch in enumerate(entries):
        merged = deep_merge(template, flat_patch)
        name = f"config_{i:03d}.yaml"
        out_path = args.output_dir / name
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(merged, f, default_flow_style=False, sort_keys=False)
        written.append(str(out_path.resolve()))

    manifest = {"template": str(args.template.resolve()), "configs": written}
    with open(args.output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
