"""Generate native-model qualification plans, images, and scoring artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import secrets
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from backbones import get_backbone, list_backbones
from evaluation.image_qualification import (
    PromptSpec,
    RenderRow,
    build_audit_plan,
    build_bakeoff_plan,
    validate_coverage,
)

_MODEL_BASE_CONFIGS: dict[str, dict[str, Any]] = {
    "sd35_large_turbo": {
        "model_id": "stabilityai/stable-diffusion-3.5-large-turbo",
        "revision": None,
        "configuration": {
            "text_encoder_3_model_id": "diffusers/t5-nf4",
            "transformer_quantization": "nf4",
            "torch_dtype": "bfloat16",
        },
    },
    "sd35_large": {
        "model_id": "stabilityai/stable-diffusion-3.5-large",
        "revision": None,
        "configuration": {"torch_dtype": "bfloat16"},
    },
    "flux_dev": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "revision": None,
        "configuration": {"torch_dtype": "bfloat16"},
    },
}
_OBSERVABLE_CONFIG_ATTRIBUTES = (
    "dtype",
    "max_length",
    "variant",
    "torch_dtype",
)
_EXPLICIT_RENDER_PARAMETERS: dict[str, dict[str, Any]] = {
    "sd35_large_turbo": {
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "max_sequence_length": 512,
    },
    "sd35_large": {
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "max_sequence_length": 256,
    },
    "flux_dev": {
        "num_inference_steps": 50,
        "guidance_scale": 3.5,
        "height": 1024,
        "width": 1024,
        "max_sequence_length": 512,
    },
}
_CONFIG_MARKER_FIELDS = (
    "_name_or_path",
    "_commit_hash",
    "_diffusers_version",
    "revision",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate native model qualification images"
    )
    parser.add_argument(
        "--prompt_spec",
        type=str,
        default="evaluation/config/image_validity_prompts.yaml",
        help="path to prompt spec YAML",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output directory for the manifest, images, and scoring sheet",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="write the plan without loading model weights or generating images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device for model inference (ignored in dry-run mode)",
    )
    return parser.parse_args()


def load_existing_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Load existing manifest rows keyed by row ID."""
    if not manifest_path.exists():
        return {}

    rows: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                rows[row["row_id"]] = row
    return rows


def write_manifest(manifest_path: Path, rows: dict[str, dict[str, Any]]) -> None:
    """Atomically write one stable manifest record per row ID."""
    temporary_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    with temporary_path.open("w") as file:
        for row_id in sorted(rows):
            file.write(json.dumps(rows[row_id], sort_keys=True) + "\n")
    temporary_path.replace(manifest_path)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Create a mode-0600 mapping once so resume cannot replace its identities."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    path.chmod(0o600)


def _validate_blinding_mapping(mapping: dict[str, Any], rows: list[RenderRow]) -> None:
    """Validate that a persisted mapping covers the current plan exactly once."""
    if mapping.get("version") != 1 or not isinstance(mapping.get("rows"), dict):
        raise ValueError("blinded row mapping has an unsupported format")

    expected_ids = {row.row_id for row in rows}
    mapped_ids = set(mapping["rows"])
    if mapped_ids != expected_ids:
        raise ValueError("blinded row mapping does not match the current plan")

    scoring_order = mapping.get("scoring_order")
    if not isinstance(scoring_order, list) or set(scoring_order) != expected_ids:
        raise ValueError("blinded scoring order does not match the current plan")
    if len(scoring_order) != len(expected_ids):
        raise ValueError("blinded scoring order contains duplicate row IDs")

    blinded_ids = [entry["blinded_row_id"] for entry in mapping["rows"].values()]
    image_paths = [entry["image_path"] for entry in mapping["rows"].values()]
    if len(blinded_ids) != len(set(blinded_ids)):
        raise ValueError("blinded row mapping contains duplicate opaque IDs")
    if len(image_paths) != len(set(image_paths)):
        raise ValueError("blinded row mapping contains duplicate image paths")


def load_or_create_blinding_mapping(
    output_dir: Path, rows: list[RenderRow]
) -> dict[str, Any]:
    """Load stable opaque identities or create them with cryptographic randomness."""
    mapping_path = output_dir / "blinded_row_mapping.json"
    if mapping_path.exists():
        # the mapping unblinds every score, so repair permissive legacy modes first
        mapping_path.chmod(0o600)
        with mapping_path.open("r") as file:
            mapping = json.load(file)
        _validate_blinding_mapping(mapping, rows)
        return mapping

    entries: dict[str, dict[str, str]] = {}
    used_tokens: set[str] = set()
    for row in rows:
        token = secrets.token_hex(16)
        while token in used_tokens:
            token = secrets.token_hex(16)
        used_tokens.add(token)
        entries[row.row_id] = {
            "blinded_row_id": token,
            "image_path": f"blinded_images/{token}.png",
        }

    # hide the deterministic property and model grouping from the scorer
    scoring_order = [row.row_id for row in rows]
    original_order = list(scoring_order)
    secrets.SystemRandom().shuffle(scoring_order)
    if len(scoring_order) > 1 and scoring_order == original_order:
        scoring_order = scoring_order[1:] + scoring_order[:1]

    mapping = {"version": 1, "rows": entries, "scoring_order": scoring_order}
    _validate_blinding_mapping(mapping, rows)
    _write_json_exclusive(mapping_path, mapping)
    return mapping


def write_scoring_sheet(output_dir: Path, mapping: dict[str, Any]) -> None:
    """Create a blinded scoring sheet without replacing existing annotations."""
    scoring_path = output_dir / "scoring_sheet.csv"
    if scoring_path.exists():
        print(f"preserving existing scoring sheet at {scoring_path}")
        return

    with scoring_path.open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "blinded_row_id",
                "image_path",
                "target_present",
                "target_visible",
                "prompt_ambiguous",
                "notes",
            ]
        )
        for row_id in mapping["scoring_order"]:
            entry = mapping["rows"][row_id]
            writer.writerow(
                [entry["blinded_row_id"], entry["image_path"], "", "", "", ""]
            )
    print(f"wrote blinded scoring sheet to {scoring_path}")
    print(f"keep {output_dir / 'blinded_row_mapping.json'} private")


def _json_value(value: Any) -> Any:
    """Convert observable configuration values to stable JSON values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _config_marker(config: Any, field: str) -> Any:
    """Read a marker from mapping-like or attribute-based model configs."""
    if isinstance(config, Mapping):
        return config.get(field)
    return getattr(config, field, None)


def _loaded_model_markers(backbone: Any) -> dict[str, dict[str, Any]]:
    """Capture revision and source markers exposed by the loaded pipeline."""
    pipeline = getattr(backbone, "pipeline", None)
    if pipeline is None:
        return {}

    markers: dict[str, dict[str, Any]] = {}
    components = {
        "pipeline": pipeline,
        "transformer": getattr(pipeline, "transformer", None),
        "text_encoder": getattr(pipeline, "text_encoder", None),
        "text_encoder_2": getattr(pipeline, "text_encoder_2", None),
        "text_encoder_3": getattr(pipeline, "text_encoder_3", None),
        "vae": getattr(pipeline, "vae", None),
    }
    for name, component in components.items():
        config = getattr(component, "config", None)
        if config is None:
            continue
        component_markers = {
            field: _json_value(value)
            for field in _CONFIG_MARKER_FIELDS
            if (value := _config_marker(config, field)) is not None
        }
        if component_markers:
            markers[name] = component_markers
    return markers


def _resolved_revision(backbone: Any, markers: dict[str, dict[str, Any]]) -> str:
    """Resolve an explicit revision label from the instance or loaded configs."""
    revision = getattr(backbone, "revision", None)
    if revision is not None:
        return str(revision)
    for component in markers.values():
        commit_hash = component.get("_commit_hash")
        if commit_hash:
            return str(commit_hash)
    return "default"


def resolved_render_parameters(backbone_name: str, backbone: Any) -> dict[str, Any]:
    """Resolve the explicit parameters passed for qualification renders."""
    parameters = dict(_EXPLICIT_RENDER_PARAMETERS[backbone_name])
    if backbone_name == "flux_dev":
        parameters["max_sequence_length"] = int(backbone.max_length)
    return parameters


def _effective_inference_dtype(backbone_name: str, backbone: Any) -> Any:
    """Resolve the dtype used by the backbone during image generation."""
    configured_dtype = getattr(backbone, "dtype", None)
    if configured_dtype is None:
        configured_dtype = _MODEL_BASE_CONFIGS[backbone_name]["configuration"].get(
            "torch_dtype"
        )

    device = getattr(backbone, "device", "unknown")
    device_type = getattr(device, "type", str(device).split(":", maxsplit=1)[0])
    if (
        backbone_name == "flux_dev"
        and device_type != "cuda"
        and str(configured_dtype) != "float32"
    ):
        return "float32"
    return _json_value(configured_dtype)


def build_model_configuration(backbone_name: str, backbone: Any) -> dict[str, Any]:
    """Record the resolved model, runtime, render parameters, and stream contract."""
    if backbone_name not in _MODEL_BASE_CONFIGS:
        raise ValueError(f"no qualification model configuration for {backbone_name!r}")

    base = _MODEL_BASE_CONFIGS[backbone_name]
    observable_configuration = dict(base["configuration"])
    for attribute in _OBSERVABLE_CONFIG_ATTRIBUTES:
        value = getattr(backbone, attribute, None)
        if value is not None:
            observable_configuration[attribute] = _json_value(value)

    loaded_model_markers = _loaded_model_markers(backbone)
    return {
        "backbone_name": backbone_name,
        "backbone_class": type(backbone).__name__,
        "model_id": _json_value(getattr(backbone, "model_id", base["model_id"])),
        "revision": _resolved_revision(backbone, loaded_model_markers),
        "loaded_model_markers": loaded_model_markers,
        "configuration": observable_configuration,
        "runtime": {
            "device": str(getattr(backbone, "device", "unknown")),
            "effective_inference_dtype": _effective_inference_dtype(
                backbone_name, backbone
            ),
        },
        "render_parameters": resolved_render_parameters(backbone_name, backbone),
        "stream_contract": [spec.to_dict() for spec in backbone.stream_specs],
    }


def fingerprint_model_configuration(configuration: dict[str, Any]) -> str:
    """Hash a canonical model configuration for stable manifest identity."""
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def manifest_success_matches(
    manifest_row: dict[str, Any],
    row: RenderRow,
    spec_fingerprint: str,
    model_configuration: dict[str, Any],
    model_fingerprint: str,
) -> bool:
    """Return whether a saved success matches the full current render contract."""
    expected = {
        "row_id": row.row_id,
        "stage": row.stage,
        "model_backbone": row.model_backbone,
        "property_id": row.property_id,
        "context_id": row.context_id,
        "seed": row.seed,
        "prompt": row.prompt,
        "spec_fingerprint": spec_fingerprint,
        "model_configuration": model_configuration,
        "model_fingerprint": model_fingerprint,
        "status": "success",
    }
    return all(manifest_row.get(key) == value for key, value in expected.items())


def _write_dry_run_plan(
    output_dir: Path,
    spec_fingerprint: str,
    audit_rows: list[RenderRow],
    bakeoff_rows: list[RenderRow],
) -> None:
    """Write the deterministic dry-run plan."""
    all_rows = audit_rows + bakeoff_rows
    plan_path = output_dir / "plan.json"
    with plan_path.open("w") as file:
        json.dump(
            {
                "spec_fingerprint": spec_fingerprint,
                "audit_count": len(audit_rows),
                "bakeoff_count": len(bakeoff_rows),
                "total_count": len(all_rows),
                "rows": [row.to_dict() for row in all_rows],
            },
            file,
            indent=2,
        )
        file.write("\n")
    print(f"dry-run mode: wrote plan to {plan_path}")


def main() -> None:
    """Run the qualification plan or materialize it without model loading."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading prompt spec from {args.prompt_spec}")
    spec = PromptSpec.load(args.prompt_spec)
    spec_fingerprint = spec.fingerprint()
    audit_rows = build_audit_plan(spec)
    bakeoff_rows = build_bakeoff_plan(spec)
    validate_coverage(audit_rows, bakeoff_rows, spec)
    all_rows = audit_rows + bakeoff_rows
    print(
        f"validated {len(audit_rows)} audit rows and "
        f"{len(bakeoff_rows)} bakeoff rows ({spec_fingerprint})"
    )

    # scorer identities remain private and stable across retries
    blinding_mapping = load_or_create_blinding_mapping(output_dir, all_rows)
    write_scoring_sheet(output_dir, blinding_mapping)

    if args.dry_run:
        _write_dry_run_plan(output_dir, spec_fingerprint, audit_rows, bakeoff_rows)
        return

    manifest_path = output_dir / "manifest.jsonl"
    manifest = load_existing_manifest(manifest_path)
    planned_ids = {row.row_id for row in all_rows}

    # migrate legacy image locations, but verify the full contract after model load
    for row in all_rows:
        manifest_row = manifest.get(row.row_id)
        if manifest_row is None or manifest_row.get("status") != "success":
            continue
        target_path = output_dir / blinding_mapping["rows"][row.row_id]["image_path"]
        legacy_value = manifest_row.get("image_path")
        legacy_path = Path(legacy_value) if legacy_value else None
        if legacy_path is not None and not legacy_path.is_absolute():
            output_relative_path = output_dir / legacy_path
            if not legacy_path.is_file() and output_relative_path.is_file():
                legacy_path = output_relative_path
        if (
            not target_path.is_file()
            and legacy_path is not None
            and legacy_path.is_file()
        ):
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_path, target_path)
        if target_path.is_file():
            manifest_row["image_path"] = str(target_path)

    # rewriting here canonicalizes duplicate legacy IDs even if every row is valid
    write_manifest(manifest_path, manifest)

    failed_ids = {
        row_id
        for row_id, row in manifest.items()
        if row_id in planned_ids and row.get("status") != "success"
    }
    print(f"checking {len(all_rows)} rows against the current model contracts")
    print(f"retrying {len(failed_ids)} previously failed rows")

    rows_by_backbone: dict[str, list[RenderRow]] = {}
    for row in all_rows:
        rows_by_backbone.setdefault(row.model_backbone, []).append(row)

    registered = list_backbones()
    for backbone_name in rows_by_backbone:
        if backbone_name not in registered:
            print(
                f"error: backbone {backbone_name!r} not registered; "
                f"available: {registered}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    for backbone_name, rows in rows_by_backbone.items():
        print(f"loading backbone {backbone_name!r}")
        backbone = get_backbone(backbone_name, device=args.device)
        try:
            backbone.load()
            model_configuration = build_model_configuration(backbone_name, backbone)
            model_fingerprint = fingerprint_model_configuration(model_configuration)
            render_parameters = resolved_render_parameters(backbone_name, backbone)
            decode_signature = inspect.signature(type(backbone).decode)
            accepted_render_parameters = {
                name: value
                for name, value in render_parameters.items()
                if name in decode_signature.parameters
            }

            skipped_count = 0
            for index, row in enumerate(rows, start=1):
                relative_image_path = blinding_mapping["rows"][row.row_id]["image_path"]
                image_path = output_dir / relative_image_path
                existing_row = manifest.get(row.row_id, {})
                if image_path.is_file() and manifest_success_matches(
                    existing_row,
                    row,
                    spec_fingerprint,
                    model_configuration,
                    model_fingerprint,
                ):
                    skipped_count += 1
                    continue

                print(f"[{index}/{len(rows)}] {row.row_id}")
                image_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_row: dict[str, Any] = {
                    "row_id": row.row_id,
                    "stage": row.stage,
                    "model_backbone": row.model_backbone,
                    "model_fingerprint": model_fingerprint,
                    "model_configuration": model_configuration,
                    "property_id": row.property_id,
                    "context_id": row.context_id,
                    "seed": row.seed,
                    "prompt": row.prompt,
                    "spec_fingerprint": spec_fingerprint,
                    "image_path": str(image_path),
                    "status": "success",
                    "error": None,
                }
                try:
                    # a stale partial file must not make a no-op decode look successful
                    image_path.unlink(missing_ok=True)
                    streams = backbone.encode(row.prompt)
                    backbone.decode(
                        streams,
                        image_path,
                        seed=row.seed,
                        **accepted_render_parameters,
                    )
                    if not image_path.is_file():
                        raise RuntimeError(
                            f"backbone decode did not create image at {image_path}"
                        )
                except Exception as err:
                    print(f"render error for {row.row_id}: {err}")
                    manifest_row["status"] = "error"
                    manifest_row["error"] = str(err)
                    manifest_row["image_path"] = None

                # checkpoint each result so an interruption loses at most one render
                manifest[row.row_id] = manifest_row
                write_manifest(manifest_path, manifest)
            print(f"skipped {skipped_count} valid {backbone_name!r} rows")
        finally:
            backbone.unload()
            print(f"unloaded backbone {backbone_name!r}")

    print(f"rendering complete. manifest: {manifest_path}")


if __name__ == "__main__":
    main()
