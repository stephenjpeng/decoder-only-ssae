"""Materialize a compositional split by hard-linking embedding row directories from source pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_prompts_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_manifest_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_tuple_key_map(prompts: list[dict]) -> dict[tuple[str, ...], int]:
    """Map tuple_key (as tuple of strings) to row index (0-based position in JSON)."""
    mapping = {}
    for row_idx, entry in enumerate(prompts):
        tuple_key = tuple(entry["tuple_key"])
        if tuple_key in mapping:
            raise ValueError(f"duplicate tuple_key in source prompts: {tuple_key}")
        mapping[tuple_key] = row_idx
    return mapping


def verify_manifests_compatible(manifests: list[tuple[Path, dict]]) -> None:
    """Verify all source manifests agree on backbone, backbone_kwargs, streams, flat_dim, style_suffix."""
    if not manifests:
        return
    ref_path, ref = manifests[0]
    ref_fields = {
        "backbone": ref.get("backbone"),
        "backbone_kwargs": ref.get("backbone_kwargs"),
        "streams": ref.get("streams"),
        "flat_dim": ref.get("flat_dim"),
        "style_suffix": ref.get("style_suffix"),
    }
    for path, m in manifests[1:]:
        for field, ref_val in ref_fields.items():
            if m.get(field) != ref_val:
                raise ValueError(
                    f"incompatible {field}: {ref_path} has {ref_val}, {path} has {m.get(field)}"
                )


def materialize_embedding_split(
    source_folders: list[Path],
    destination_folder: Path,
    mode: str = "hardlink",
    resume: bool = False,
) -> None:
    """Materialize a compositional split by linking embedding row directories from sources."""
    # load source prompts and manifests
    source_maps = []
    source_manifests = []
    for src_folder in source_folders:
        prompts = load_prompts_json(src_folder / "prompts.json")
        mapping = build_tuple_key_map(prompts)
        source_maps.append((src_folder, mapping))
        manifest_path = src_folder / "embds" / "manifest.json"
        manifest = load_manifest_json(manifest_path)
        source_manifests.append((manifest_path, manifest))

    verify_manifests_compatible(source_manifests)

    # load destination prompts
    dest_prompts_path = destination_folder / "prompts.json"
    if not dest_prompts_path.exists():
        raise FileNotFoundError(f"destination prompts.json not found: {dest_prompts_path}")
    dest_prompts = load_prompts_json(dest_prompts_path)

    # build reverse map: tuple_key -> source folder and source row index
    tuple_to_source = {}
    for src_folder, mapping in source_maps:
        for tup_key, row_idx in mapping.items():
            if tup_key in tuple_to_source:
                raise ValueError(
                    f"duplicate tuple_key across sources: {tup_key} found in both "
                    f"{tuple_to_source[tup_key][0]} and {src_folder}"
                )
            tuple_to_source[tup_key] = (src_folder, row_idx)

    # verify all destination tuple_keys are covered
    for dest_idx, entry in enumerate(dest_prompts):
        tup_key = tuple(entry["tuple_key"])
        if tup_key not in tuple_to_source:
            raise ValueError(f"destination row {dest_idx} tuple_key {tup_key} not found in any source")

    # check destination embds directory
    dest_embds = destination_folder / "embds"
    if dest_embds.exists() and not resume:
        raise FileExistsError(
            f"destination embds directory already exists: {dest_embds}. use --resume to skip existing rows"
        )
    dest_embds.mkdir(parents=True, exist_ok=True)

    # materialize each destination row
    file_counts = []
    for dest_idx, entry in enumerate(dest_prompts):
        tup_key = tuple(entry["tuple_key"])
        src_folder, src_row_idx = tuple_to_source[tup_key]
        src_row_dir = src_folder / "embds" / f"embds_{src_row_idx}"
        dest_row_dir = dest_embds / f"embds_{dest_idx}"

        if dest_row_dir.exists():
            if resume:
                continue
            raise FileExistsError(f"destination row directory already exists: {dest_row_dir}")

        dest_row_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for src_file in src_row_dir.iterdir():
            if src_file.is_file():
                dest_file = dest_row_dir / src_file.name
                try:
                    if mode == "hardlink":
                        dest_file.hardlink_to(src_file)
                    elif mode == "copy":
                        shutil.copy2(src_file, dest_file)
                    else:
                        raise ValueError(f"invalid mode: {mode}")
                except OSError as e:
                    if mode == "hardlink" and e.errno == 18:  # EXDEV
                        raise OSError(
                            f"cross-device hard link not supported (source and destination on different filesystems). "
                            f"use --mode copy instead"
                        ) from e
                    raise
                count += 1
        file_counts.append(count)

    # verify row count
    if len(dest_prompts) != len(file_counts):
        raise ValueError(
            f"row count mismatch: dest prompts has {len(dest_prompts)}, materialized {len(file_counts)}"
        )

    # write embds/manifest.json
    ref_manifest = source_manifests[0][1]
    dest_manifest = {
        "backbone": ref_manifest.get("backbone"),
        "backbone_kwargs": ref_manifest.get("backbone_kwargs"),
        "streams": ref_manifest.get("streams"),
        "flat_dim": ref_manifest.get("flat_dim"),
        "style_suffix": ref_manifest.get("style_suffix"),
        "n_prompts": len(dest_prompts),
        "materialization": {
            "source_folders": [str(f.resolve()) for f in source_folders],
            "mode": mode,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    with open(dest_embds / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(dest_manifest, f, indent=2)

    # write embedding_materialization_manifest.json
    n_sources = {str(src.resolve()): len(mapping) for src, mapping in source_maps}
    materialization_manifest = {
        "source_folders": [str(f.resolve()) for f in source_folders],
        "destination_folder": str(destination_folder.resolve()),
        "mode": mode,
        "n_destination": len(dest_prompts),
        "n_sources": n_sources,
        "file_counts": file_counts,
        "prompts_json_sha256": sha256_file(dest_prompts_path),
        "embds_manifest_sha256": sha256_file(dest_embds / "manifest.json"),
    }
    with open(destination_folder / "embedding_materialization_manifest.json", "w", encoding="utf-8") as f:
        json.dump(materialization_manifest, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Materialize a compositional split by linking embedding row directories from sources."
    )
    p.add_argument("--source_folder", action="append", required=True, type=Path, help="source folder(s)")
    p.add_argument("--destination_folder", required=True, type=Path, help="destination folder")
    p.add_argument(
        "--mode", choices=["hardlink", "copy"], default="hardlink", help="link mode (default: hardlink)"
    )
    p.add_argument("--resume", action="store_true", help="skip existing destination rows")
    args = p.parse_args()

    materialize_embedding_split(
        source_folders=args.source_folder,
        destination_folder=args.destination_folder,
        mode=args.mode,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
