"""Generate direct-text qualification plans, images, and blinded scoring artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import secrets
import shutil
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
        "configuration": {
            "text_encoder_3_model_id": "diffusers/t5-nf4",
            "transformer_quantization": "nf4",
            "torch_dtype": "bfloat16",
        },
    },
    "sd35_large": {
        "model_id": "stabilityai/stable-diffusion-3.5-large",
        "configuration": {"torch_dtype": "bfloat16"},
    },
    "flux_dev": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "configuration": {"torch_dtype": "bfloat16"},
    },
}
_OBSERVABLE_CONFIG_ATTRIBUTES = ("dtype", "max_length", "variant", "torch_dtype")
_EXPLICIT_RENDER_PARAMETERS: dict[str, dict[str, Any]] = {
    "sd35_large_turbo": {
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "max_sequence_length": 256,
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
_SCORE_FIELDS = ("target_present", "target_visible", "prompt_ambiguous", "notes")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate native model qualification images"
    )
    parser.add_argument(
        "--prompt_spec",
        default="evaluation/config/image_validity_prompts.yaml",
        help="path to prompt spec YAML",
    )
    parser.add_argument(
        "--output_dir", required=True, help="qualification output directory"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="write the plan without loading model weights or rendering images",
    )
    parser.add_argument("--device", default="cuda", help="inference device")
    return parser.parse_args()


def load_existing_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Load existing private manifest rows keyed by canonical row ID."""
    if not manifest_path.exists():
        return {}
    manifest_path.chmod(0o600)
    rows: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                rows[row["row_id"]] = row
    return rows


def write_manifest(manifest_path: Path, rows: dict[str, dict[str, Any]]) -> None:
    """Atomically write the private render manifest with mode 0600."""
    temporary_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        for row_id in sorted(rows):
            file.write(json.dumps(rows[row_id], sort_keys=True) + "\n")
    temporary_path.replace(manifest_path)
    manifest_path.chmod(0o600)


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write private JSON with mode 0600."""
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary_path.replace(path)
    path.chmod(0o600)


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


def _resolved_revision(backbone: Any, markers: dict[str, dict[str, Any]]) -> str | None:
    """Resolve a concrete revision when the instance or loaded config exposes one."""
    revision = getattr(backbone, "revision", None)
    if revision:
        return str(revision)
    for component in markers.values():
        commit_hash = component.get("_commit_hash")
        if commit_hash:
            return str(commit_hash)
    return None


def resolved_render_parameters(backbone_name: str, backbone: Any) -> dict[str, Any]:
    """Resolve parameters passed to direct-text qualification generation."""
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
    """Record model provenance, runtime, direct render parameters, and streams."""
    if backbone_name not in _MODEL_BASE_CONFIGS:
        raise ValueError(f"no qualification model configuration for {backbone_name!r}")
    base = _MODEL_BASE_CONFIGS[backbone_name]
    observable_configuration = dict(base["configuration"])
    for attribute in _OBSERVABLE_CONFIG_ATTRIBUTES:
        value = getattr(backbone, attribute, None)
        if value is not None:
            observable_configuration[attribute] = _json_value(value)
    markers = _loaded_model_markers(backbone)
    revision = _resolved_revision(backbone, markers)
    return {
        "backbone_name": backbone_name,
        "backbone_class": type(backbone).__name__,
        "model_id": _json_value(getattr(backbone, "model_id", base["model_id"])),
        "revision": revision,
        "revision_is_concrete": revision is not None,
        "loaded_model_markers": markers,
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
    """Hash a canonical model configuration for stable render identity."""
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def qualification_identity(
    row: RenderRow, spec_fingerprint: str, model_fingerprint: str
) -> str:
    """Bind one rendered pixel artifact to its spec, model, and render contract."""
    payload = {
        "row": row.to_dict(),
        "spec_fingerprint": spec_fingerprint,
        "model_fingerprint": model_fingerprint,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


def manifest_success_matches(
    manifest_row: dict[str, Any],
    row: RenderRow,
    spec_fingerprint: str,
    model_configuration: dict[str, Any],
    model_fingerprint: str,
) -> bool:
    """Return whether a saved success matches the complete direct render contract."""
    expected = {
        **row.to_dict(),
        "spec_fingerprint": spec_fingerprint,
        "model_configuration": model_configuration,
        "model_fingerprint": model_fingerprint,
        "render_contract_fingerprint": qualification_identity(
            row, spec_fingerprint, model_fingerprint
        ),
        "status": "success",
    }
    if not all(manifest_row.get(key) == value for key, value in expected.items()):
        return False
    artifact_identity = manifest_row.get("qualification_identity")
    render_attempt_id = manifest_row.get("render_attempt_id")
    return (
        isinstance(render_attempt_id, str)
        and bool(render_attempt_id)
        and artifact_identity
        == f"{expected['render_contract_fingerprint']}-{render_attempt_id}"
    )


def _load_existing_scores(path: Path) -> dict[str, dict[str, str]]:
    """Load annotations by opaque ID so unchanged pixels retain their scores."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as file:
        return {
            row["blinded_row_id"]: row
            for row in csv.DictReader(file)
            if row.get("blinded_row_id")
        }


def _load_mapping(path: Path) -> dict[str, Any]:
    """Load the private v2 mapping or return an empty mapping."""
    if not path.exists():
        return {"version": 2, "rows": {}, "scoring_order": []}
    path.chmod(0o600)
    with path.open("r", encoding="utf-8") as file:
        mapping = json.load(file)
    if mapping.get("version") != 2 or not isinstance(mapping.get("rows"), dict):
        return {"version": 2, "rows": {}, "scoring_order": []}
    return mapping


def _shuffle_public_copy_rows(
    pending: list[tuple[RenderRow, Path, str]],
) -> list[tuple[RenderRow, Path, str]]:
    """Randomize public file creation order without retaining an arm-order collision."""
    shuffled = list(pending)
    if len(shuffled) < 2:
        return shuffled

    original_ids = [row.row_id for row, _, _ in pending]
    original_models = [row.model_backbone for row, _, _ in pending]
    secrets.SystemRandom().shuffle(shuffled)
    shuffled_ids = [row.row_id for row, _, _ in shuffled]
    shuffled_models = [row.model_backbone for row, _, _ in shuffled]
    if shuffled_ids == original_ids or (
        len(set(original_models)) > 1 and shuffled_models == original_models
    ):
        # sample only rotations that cannot reproduce the deterministic arm schedule
        offsets = [
            offset
            for offset in range(1, len(shuffled))
            if len(set(original_models)) == 1
            or original_models[offset:] + original_models[:offset] != original_models
        ]
        offset = offsets[secrets.randbelow(len(offsets))]
        shuffled = shuffled[offset:] + shuffled[:offset]
    return shuffled


def build_scoring_package(
    output_dir: Path,
    rows: list[RenderRow],
    manifest: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Publish opaque copies and a score sheet for successful current renders only."""
    mapping_path = output_dir / "blinded_row_mapping.json"
    scoring_path = output_dir / "scoring_sheet.csv"
    prior_mapping = _load_mapping(mapping_path)
    prior_scores = _load_existing_scores(scoring_path)
    prior_entries = prior_mapping["rows"]
    blinded_dir = output_dir / "blinded_images"
    blinded_dir.mkdir(exist_ok=True)

    entries: dict[str, dict[str, str]] = {}
    scorer_rows = [
        row
        for row in rows
        if row.design == "bakeoff" and row.conditioning == "direct_text"
    ]
    pending_copies: list[tuple[RenderRow, Path, str]] = []
    for row in scorer_rows:
        rendered = manifest.get(row.row_id, {})
        if rendered.get("status") != "success":
            continue
        source_path = Path(rendered["image_path"])
        if not source_path.is_file():
            continue
        identity = rendered["qualification_identity"]
        previous = prior_entries.get(row.row_id, {})
        previous_path = output_dir / previous.get("image_path", "missing")
        if (
            previous.get("qualification_identity") == identity
            and previous_path.is_file()
        ):
            # retained files stay untouched so resume cannot restamp ctime in arm order
            entries[row.row_id] = previous
            continue

        pending_copies.append((row, source_path, identity))

    # file creation metadata follows an independent random order, not model order
    for row, source_path, identity in _shuffle_public_copy_rows(pending_copies):
        token = secrets.token_hex(16)
        while (blinded_dir / f"{token}.png").exists():
            token = secrets.token_hex(16)
        destination = blinded_dir / f"{token}.png"
        shutil.copyfile(source_path, destination)
        destination.chmod(0o644)
        os.utime(destination, (0, 0))
        entries[row.row_id] = {
            "blinded_row_id": token,
            "image_path": str(destination.relative_to(output_dir)),
            "qualification_identity": identity,
        }

    # remove obsolete opaque images after the replacement is safely copied
    retained_paths = {entry["image_path"] for entry in entries.values()}
    for previous in prior_entries.values():
        old_path = previous.get("image_path")
        if old_path and old_path not in retained_paths:
            (output_dir / old_path).unlink(missing_ok=True)

    row_by_id = {row.row_id: row for row in scorer_rows}
    scoring_order = [
        row_id for row_id in prior_mapping.get("scoring_order", []) if row_id in entries
    ]
    new_ids = [row_id for row_id in entries if row_id not in scoring_order]
    secrets.SystemRandom().shuffle(new_ids)
    scoring_order.extend(new_ids)
    mapping = {"version": 2, "rows": entries, "scoring_order": scoring_order}
    _write_private_json(mapping_path, mapping)

    temporary_path = scoring_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "blinded_row_id",
            "image_path",
            "target_phrase",
            "prompt",
            *_SCORE_FIELDS,
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_id in scoring_order:
            entry = entries[row_id]
            old_score = prior_scores.get(entry["blinded_row_id"], {})
            row = row_by_id[row_id]
            writer.writerow(
                {
                    "blinded_row_id": entry["blinded_row_id"],
                    "image_path": entry["image_path"],
                    "target_phrase": row.target_phrase,
                    "prompt": row.prompt,
                    **{field: old_score.get(field, "") for field in _SCORE_FIELDS},
                }
            )
    temporary_path.replace(scoring_path)
    return mapping


def _write_dry_run_plan(
    output_dir: Path,
    spec_fingerprint: str,
    audit_rows: list[RenderRow],
    bakeoff_rows: list[RenderRow],
) -> None:
    """Write the deterministic dry-run plan without creating scoring artifacts."""
    all_rows = audit_rows + bakeoff_rows
    plan_path = output_dir / "plan.json"
    with plan_path.open("w", encoding="utf-8") as file:
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
    """Render the qualification plan through native direct-text dispatch."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    if args.dry_run:
        _write_dry_run_plan(output_dir, spec_fingerprint, audit_rows, bakeoff_rows)
        return

    manifest_path = output_dir / "manifest.jsonl"
    manifest = load_existing_manifest(manifest_path)
    # canonicalize duplicate legacy rows even when every render can be resumed
    write_manifest(manifest_path, manifest)
    rows_by_backbone: dict[str, list[RenderRow]] = {}
    for row in all_rows:
        rows_by_backbone.setdefault(row.model_backbone, []).append(row)
    registered = list_backbones()
    unknown = sorted(set(rows_by_backbone) - set(registered))
    if unknown:
        raise SystemExit(
            f"unregistered qualification backbones: {unknown}; available: {registered}"
        )

    private_images = output_dir / "private_images"
    private_images.mkdir(exist_ok=True)
    private_images.chmod(0o700)
    for backbone_name, rows in rows_by_backbone.items():
        print(f"loading backbone {backbone_name!r}")
        backbone = get_backbone(backbone_name, device=args.device)
        try:
            backbone.load()
            model_configuration = build_model_configuration(backbone_name, backbone)
            model_fingerprint = fingerprint_model_configuration(model_configuration)
            render_parameters = resolved_render_parameters(backbone_name, backbone)
            generate_signature = inspect.signature(type(backbone).generate)
            accepted_parameters = {
                name: value
                for name, value in render_parameters.items()
                if name in generate_signature.parameters
            }
            skipped_count = 0
            for index, row in enumerate(rows, start=1):
                contract_identity = qualification_identity(
                    row, spec_fingerprint, model_fingerprint
                )
                image_path = private_images / f"{row.row_id}.{contract_identity}.png"
                existing = manifest.get(row.row_id, {})
                existing_path_value = existing.get("image_path")
                existing_path = (
                    Path(existing_path_value) if existing_path_value else None
                )
                if (
                    existing_path is not None
                    and existing_path.is_file()
                    and manifest_success_matches(
                        existing,
                        row,
                        spec_fingerprint,
                        model_configuration,
                        model_fingerprint,
                    )
                ):
                    skipped_count += 1
                    continue

                print(f"[{index}/{len(rows)}] {row.row_id}")
                old_path_value = existing.get("image_path")
                render_attempt_id = secrets.token_hex(8)
                manifest_row: dict[str, Any] = {
                    **row.to_dict(),
                    "model_fingerprint": model_fingerprint,
                    "model_configuration": model_configuration,
                    "spec_fingerprint": spec_fingerprint,
                    "render_contract_fingerprint": contract_identity,
                    "render_attempt_id": render_attempt_id,
                    "qualification_identity": f"{contract_identity}-{render_attempt_id}",
                    "image_path": str(image_path),
                    "status": "success",
                    "error": None,
                }
                try:
                    image_path.unlink(missing_ok=True)
                    backbone.generate(
                        row.prompt,
                        image_path,
                        seed=row.seed,
                        **accepted_parameters,
                    )
                    if not image_path.is_file():
                        raise RuntimeError(
                            f"backbone generate did not create image at {image_path}"
                        )
                    if old_path_value and Path(old_path_value) != image_path:
                        Path(old_path_value).unlink(missing_ok=True)
                except Exception as err:
                    print(f"render error for {row.row_id}: {err}")
                    manifest_row["status"] = "error"
                    manifest_row["error"] = str(err)
                    manifest_row["image_path"] = None
                manifest[row.row_id] = manifest_row
                write_manifest(manifest_path, manifest)
            print(f"skipped {skipped_count} current {backbone_name!r} rows")
        finally:
            backbone.unload()
            print(f"unloaded backbone {backbone_name!r}")

    mapping = build_scoring_package(output_dir, all_rows, manifest)
    successful = len(mapping["rows"])
    failed = sum(
        manifest.get(row.row_id, {}).get("status") == "error" for row in all_rows
    )
    print(
        f"rendering complete: {successful} bakeoff scorer rows, {failed} render failures"
    )
    print(f"private manifest: {manifest_path}")
    print(f"blinded scoring sheet: {output_dir / 'scoring_sheet.csv'}")


if __name__ == "__main__":
    main()
