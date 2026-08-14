"""
Image validity scoring analysis with Wilson confidence intervals.

Consumes human-scored target presence/visibility/ambiguity for rendered images
and computes per-model and per-property validity rates with 95% Wilson CIs.
Applies the agreed 90% native validity gate and calculates source-valid
eligibility for edit stages.
"""

from __future__ import annotations

import csv
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ScoringRow:
    """One human-scored image row"""

    row_id: str
    target_present: bool
    target_visible: bool
    prompt_ambiguous: bool
    notes: str


@dataclass
class ManifestRow:
    """One rendered image manifest row"""

    row_id: str
    model_backbone: str
    property_id: str
    context_id: str
    seed: int
    stage: str
    image_path: str | None
    source_row_id: str | None
    design: str = "bakeoff"
    conditioning: str = "direct_text"
    status: str = "success"
    source_property_id: str | None = None
    target_property_id: str | None = None
    qualification_identity: str | None = None


@dataclass
class ValidityStats:
    """Validity statistics for a group (model, property, or overall)"""

    n_total: int
    n_valid: int
    rate: float
    ci_lower: float
    ci_upper: float
    passes_gate: bool


@dataclass
class EligibilityStats:
    """Edit eligibility statistics (attempted vs source-valid eligible)"""

    n_attempted: int
    n_eligible: int
    rate: float


def parse_bool_strict(value: str | None, field_name: str, row_id: str) -> bool:
    """
    Parse boolean from CSV with strict validation.

    Args:
        value: string value from CSV (or None for missing columns)
        field_name: name of the field for error messages
        row_id: row identifier for error messages

    Returns:
        parsed boolean

    Raises:
        ValueError: if value is None, empty, or not 'true' or 'false' (case-insensitive)
    """
    if value is None:
        raise ValueError(f"missing value for {field_name} in row {row_id}")
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError(f"empty value for {field_name} in row {row_id}")
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(
        f"invalid boolean for {field_name} in row {row_id}: "
        f"got '{value}', expected 'true' or 'false'"
    )


def import_blinded_scores(
    scoring_path: Path,
    mapping_path: Path,
    manifest_path: Path,
    output_path: Path | None = None,
) -> dict[str, ScoringRow]:
    """Authorize unblinding and translate opaque score IDs to canonical row IDs."""
    if not mapping_path.exists():
        raise FileNotFoundError(f"blinding mapping not found: {mapping_path}")
    if stat.S_IMODE(mapping_path.stat().st_mode) != 0o600:
        raise ValueError("blinding mapping must have mode 0600 before unblinding")
    with mapping_path.open("r", encoding="utf-8") as file:
        mapping = json.load(file)
    if mapping.get("version") != 2 or not isinstance(mapping.get("rows"), dict):
        raise ValueError("blinding mapping has an unsupported format")

    manifest = load_manifest_jsonl(manifest_path)
    expected_rows = {
        row.row_id: row
        for row in manifest
        if row.stage == "native"
        and row.design == "bakeoff"
        and row.conditioning == "direct_text"
        and row.status == "success"
    }
    if set(mapping["rows"]) != set(expected_rows):
        raise ValueError(
            "blinding mapping does not match current successful bakeoff rows"
        )

    opaque_to_row: dict[str, str] = {}
    for row_id, entry in mapping["rows"].items():
        opaque_id = entry.get("blinded_row_id")
        if not isinstance(opaque_id, str) or not opaque_id:
            raise ValueError(f"mapping row {row_id} has no opaque ID")
        if opaque_id in opaque_to_row:
            raise ValueError(f"duplicate opaque ID in mapping: {opaque_id}")
        mapped_identity = entry.get("qualification_identity")
        current_identity = expected_rows[row_id].qualification_identity
        if not current_identity or mapped_identity != current_identity:
            raise ValueError(
                f"blinding mapping identity does not match current manifest row {row_id}"
            )
        opaque_to_row[opaque_id] = row_id

    if not scoring_path.exists():
        raise FileNotFoundError(f"scoring CSV not found: {scoring_path}")
    scores: dict[str, ScoringRow] = {}
    with scoring_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "blinded_row_id",
            "target_present",
            "target_visible",
            "prompt_ambiguous",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("blinded scoring CSV is missing required score columns")
        for line_num, row in enumerate(reader, 2):
            opaque_id = (row.get("blinded_row_id") or "").strip()
            if opaque_id not in opaque_to_row:
                raise ValueError(
                    f"blinded scoring row {line_num} has an unknown opaque ID: {opaque_id}"
                )
            row_id = opaque_to_row[opaque_id]
            if row_id in scores:
                raise ValueError(f"duplicate blinded score for canonical row {row_id}")
            scores[row_id] = ScoringRow(
                row_id=row_id,
                target_present=parse_bool_strict(
                    row.get("target_present"), "target_present", opaque_id
                ),
                target_visible=parse_bool_strict(
                    row.get("target_visible"), "target_visible", opaque_id
                ),
                prompt_ambiguous=parse_bool_strict(
                    row.get("prompt_ambiguous"), "prompt_ambiguous", opaque_id
                ),
                notes=(row.get("notes") or "").strip(),
            )
    missing = set(mapping["rows"]) - set(scores)
    if missing:
        raise ValueError(
            f"blinded scoring CSV is missing {len(missing)} successful row(s)"
        )

    if output_path is not None:
        descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "row_id",
                    "target_present",
                    "target_visible",
                    "prompt_ambiguous",
                    "notes",
                ]
            )
            for row_id in sorted(scores):
                score = scores[row_id]
                writer.writerow(
                    [
                        row_id,
                        str(score.target_present).lower(),
                        str(score.target_visible).lower(),
                        str(score.prompt_ambiguous).lower(),
                        score.notes,
                    ]
                )
        output_path.chmod(0o600)
    return scores


def load_scoring_csv(path: Path) -> dict[str, ScoringRow]:
    """
    Load human scores from CSV.

    Expected columns: row_id, target_present, target_visible, prompt_ambiguous, notes

    Args:
        path: path to scoring CSV

    Returns:
        mapping from row_id to ScoringRow

    Raises:
        FileNotFoundError: if CSV does not exist
        ValueError: for duplicate row_ids or invalid boolean values
    """
    if not path.exists():
        raise FileNotFoundError(f"scoring CSV not found: {path}")

    scores: dict[str, ScoringRow] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required = {"row_id", "target_present", "target_visible", "prompt_ambiguous"}
        if not required.issubset(fieldnames):
            raise ValueError(
                f"scoring CSV missing required columns. "
                f"Expected {sorted(required)}, got {sorted(fieldnames)}"
            )
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("scoring CSV contains duplicate column names")

        for line_num, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(
                    f"scoring CSV row {line_num} has more values than columns"
                )

            raw_row_id = row["row_id"]
            if raw_row_id is None or not raw_row_id.strip():
                raise ValueError(f"empty row_id in scoring CSV row {line_num}")
            row_id = raw_row_id.strip()

            if row_id in scores:
                raise ValueError(f"duplicate row_id in scoring CSV: {row_id}")

            scores[row_id] = ScoringRow(
                row_id=row_id,
                target_present=parse_bool_strict(
                    row["target_present"], "target_present", row_id
                ),
                target_visible=parse_bool_strict(
                    row["target_visible"], "target_visible", row_id
                ),
                prompt_ambiguous=parse_bool_strict(
                    row["prompt_ambiguous"], "prompt_ambiguous", row_id
                ),
                notes=(row.get("notes") or "").strip(),
            )

    return scores


def load_manifest_jsonl(path: Path) -> list[ManifestRow]:
    """
    Load render manifest from JSONL.

    Expected fields: row_id, model_backbone, property_id, context_id, seed, stage,
                     image_path, plus optional source_row_id for edit stages

    Args:
        path: path to manifest JSONL

    Returns:
        list of ManifestRow objects

    Raises:
        FileNotFoundError: if JSONL does not exist
        ValueError: for duplicate row_ids or missing required fields
    """
    if not path.exists():
        raise FileNotFoundError(f"manifest JSONL not found: {path}")

    rows: list[ManifestRow] = []
    seen_ids: set[str] = set()
    string_fields = {
        "row_id",
        "model_backbone",
        "property_id",
        "context_id",
        "stage",
    }
    required = string_fields | {"seed"}

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"invalid JSON at line {line_num}: {err}") from err
            if not isinstance(obj, dict):
                raise ValueError(f"manifest line {line_num} must be a JSON object")

            missing = required - obj.keys()
            if missing:
                raise ValueError(
                    f"manifest line {line_num} missing required fields: {sorted(missing)}"
                )
            for field_name in string_fields:
                value = obj[field_name]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"manifest line {line_num} field {field_name} "
                        "must be a nonblank string"
                    )
            if not isinstance(obj["seed"], int) or isinstance(obj["seed"], bool):
                raise ValueError(
                    f"manifest line {line_num} field seed must be an integer"
                )
            if obj["stage"] not in {"native", "edit"}:
                raise ValueError(
                    f"manifest line {line_num} has unsupported stage: {obj['stage']}"
                )
            status = obj.get("status", "success")
            if status not in {"success", "error"}:
                raise ValueError(
                    f"manifest line {line_num} has unsupported status: {status}"
                )
            image_path = obj.get("image_path")
            if status == "success" and (
                not isinstance(image_path, str) or not image_path.strip()
            ):
                raise ValueError(
                    f"manifest line {line_num} successful row needs a nonblank image_path"
                )
            if status == "error" and image_path not in {None, ""}:
                raise ValueError(
                    f"manifest line {line_num} failed row image_path must be blank"
                )
            design = obj.get("design", "bakeoff")
            conditioning = obj.get("conditioning", "direct_text")
            if design not in {"audit", "bakeoff"}:
                raise ValueError(
                    f"manifest line {line_num} has unsupported design: {design}"
                )
            if not isinstance(conditioning, str) or not conditioning.strip():
                raise ValueError(
                    f"manifest line {line_num} conditioning must be a nonblank string"
                )

            source_row_id = obj.get("source_row_id")
            if source_row_id is not None and (
                not isinstance(source_row_id, str) or not source_row_id.strip()
            ):
                raise ValueError(
                    f"manifest line {line_num} field source_row_id "
                    "must be a nonblank string when present"
                )
            property_join_fields = ("source_property_id", "target_property_id")
            if obj["stage"] == "edit":
                for field_name in property_join_fields:
                    value = obj.get(field_name)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(
                            f"manifest line {line_num} edit row requires nonblank "
                            f"{field_name}"
                        )

            row_id = obj["row_id"].strip()
            if row_id in seen_ids:
                raise ValueError(f"duplicate row_id in manifest JSONL: {row_id}")
            seen_ids.add(row_id)

            rows.append(
                ManifestRow(
                    row_id=row_id,
                    model_backbone=obj["model_backbone"].strip(),
                    property_id=obj["property_id"].strip(),
                    context_id=obj["context_id"].strip(),
                    seed=obj["seed"],
                    stage=obj["stage"],
                    image_path=image_path.strip()
                    if isinstance(image_path, str)
                    else None,
                    source_row_id=(
                        source_row_id.strip() if source_row_id is not None else None
                    ),
                    design=design,
                    conditioning=conditioning.strip(),
                    status=status,
                    source_property_id=(
                        obj["source_property_id"].strip()
                        if obj.get("source_property_id") is not None
                        else None
                    ),
                    target_property_id=(
                        obj["target_property_id"].strip()
                        if obj.get("target_property_id") is not None
                        else None
                    ),
                    qualification_identity=obj.get("qualification_identity"),
                )
            )

    return rows


def wilson_confidence_interval(
    n_success: int, n_total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a binomial proportion.

    Handles edge cases: n_total=0, all failures, all successes.

    Args:
        n_success: number of successes
        n_total: total number of trials
        confidence: confidence level (default 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound) tuple
    """
    if n_total == 0:
        return (0.0, 0.0)

    # z-score for given confidence level
    # for 95%, z ≈ 1.96
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence)
    if z is None:
        raise ValueError(f"unsupported confidence level: {confidence}")

    p = n_success / n_total
    z2 = z * z

    # Wilson score interval formula
    denominator = 1 + z2 / n_total
    center = (p + z2 / (2 * n_total)) / denominator
    margin = (
        z
        * math.sqrt((p * (1 - p) / n_total + z2 / (4 * n_total * n_total)))
        / denominator
    )

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)


def calculate_validity(
    manifest: list[ManifestRow],
    scores: dict[str, ScoringRow],
    gate_threshold: float = 0.90,
) -> ValidityStats:
    """
    Calculate overall validity statistics.

    Args:
        manifest: list of manifest rows
        scores: mapping from row_id to score
        gate_threshold: minimum observed rate to pass (default 0.90)

    Returns:
        ValidityStats with rate, CI, and gate decision

    Raises:
        ValueError: if any manifest row lacks a score
    """
    n_total = len(manifest)
    n_valid = 0

    for row in manifest:
        # render failures remain in the denominator and count as invalid
        if row.status == "error":
            continue
        if row.row_id not in scores:
            raise ValueError(f"manifest row {row.row_id} has no corresponding score")
        if scores[row.row_id].target_present:
            n_valid += 1

    rate = n_valid / n_total if n_total > 0 else 0.0
    ci_lower, ci_upper = wilson_confidence_interval(n_valid, n_total)

    # gate applies to observed rate, not CI lower bound
    passes_gate = rate >= gate_threshold

    return ValidityStats(
        n_total=n_total,
        n_valid=n_valid,
        rate=rate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        passes_gate=passes_gate,
    )


def calculate_per_model_validity(
    manifest: list[ManifestRow],
    scores: dict[str, ScoringRow],
    gate_threshold: float = 0.90,
) -> dict[str, ValidityStats]:
    """
    Calculate validity statistics per model.

    Args:
        manifest: list of manifest rows
        scores: mapping from row_id to score
        gate_threshold: minimum observed rate to pass

    Returns:
        mapping from model name to ValidityStats
    """
    by_model: dict[str, list[ManifestRow]] = {}
    for row in manifest:
        by_model.setdefault(row.model_backbone, []).append(row)

    results = {}
    for model, rows in by_model.items():
        results[model] = calculate_validity(rows, scores, gate_threshold)

    return results


def calculate_per_property_validity(
    manifest: list[ManifestRow],
    scores: dict[str, ScoringRow],
    gate_threshold: float = 0.90,
) -> dict[str, ValidityStats]:
    """
    Calculate validity statistics per property.

    Args:
        manifest: list of manifest rows
        scores: mapping from row_id to score
        gate_threshold: minimum observed rate to pass

    Returns:
        mapping from property_id to ValidityStats
    """
    by_property: dict[str, list[ManifestRow]] = {}
    for row in manifest:
        by_property.setdefault(row.property_id, []).append(row)

    results = {}
    for prop_id, rows in by_property.items():
        results[prop_id] = calculate_validity(rows, scores, gate_threshold)

    return results


def _validate_edit_source(
    row: ManifestRow, source_rows: dict[str, ManifestRow]
) -> ManifestRow:
    """Return the native source only when every join identity field matches"""
    if row.stage != "edit":
        raise ValueError(f"eligibility row {row.row_id} must have stage edit")
    if not row.source_row_id:
        raise ValueError(f"edit row {row.row_id} has no source_row_id")
    if row.source_row_id == row.row_id:
        raise ValueError(f"edit row {row.row_id} cannot reference itself")
    if not row.source_property_id:
        raise ValueError(f"edit row {row.row_id} has no source_property_id")
    if not row.target_property_id:
        raise ValueError(f"edit row {row.row_id} has no target_property_id")

    source = source_rows.get(row.source_row_id)
    if source is None or source.stage != "native":
        raise ValueError(
            f"edit row {row.row_id} source_row_id must reference a native row: "
            f"{row.source_row_id}"
        )
    for field_name in (
        "model_backbone",
        "context_id",
        "seed",
        "design",
        "conditioning",
    ):
        if getattr(row, field_name) != getattr(source, field_name):
            raise ValueError(f"edit row {row.row_id} must match source {field_name}")
    if row.source_property_id != source.property_id:
        raise ValueError(
            f"edit row {row.row_id} source_property_id does not match source"
        )
    if row.target_property_id != row.property_id:
        raise ValueError(
            f"edit row {row.row_id} target_property_id does not match target"
        )
    return source


def calculate_eligibility(
    manifest: list[ManifestRow],
    scores: dict[str, ScoringRow],
    source_rows: dict[str, ManifestRow],
) -> EligibilityStats:
    """Calculate source-valid edit eligibility using a strict native-row join"""
    n_attempted = len(manifest)
    n_eligible = 0

    for row in manifest:
        source = _validate_edit_source(row, source_rows)
        # a failed source render is conservatively ineligible even if scores are malformed
        source_score = scores.get(source.row_id) if source.status == "success" else None
        if source_score is not None and source_score.target_present:
            n_eligible += 1

    rate = n_eligible / n_attempted if n_attempted > 0 else 0.0

    return EligibilityStats(
        n_attempted=n_attempted,
        n_eligible=n_eligible,
        rate=rate,
    )


def analyze_image_validity(
    manifest_path: Path,
    scores_path: Path,
    gate_threshold: float = 0.90,
    blinding_mapping_path: Path | None = None,
    unblinded_scores_output: Path | None = None,
    required_model_property_arms: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Analyze bakeoff validity after optional authorized unblinding.

    The 90% gate uses target presence only. Visibility and ambiguity are
    secondary diagnostics and do not change the gate decision.
    """
    manifest = load_manifest_jsonl(manifest_path)
    manifest_by_id = {row.row_id: row for row in manifest}
    native_rows = [row for row in manifest if row.stage == "native"]
    edit_rows = [row for row in manifest if row.stage == "edit"]
    bakeoff_rows = [
        row
        for row in native_rows
        if row.design == "bakeoff" and row.conditioning == "direct_text"
    ]
    if blinding_mapping_path is None:
        scores = load_scoring_csv(scores_path)
        expected_score_ids = {row.row_id for row in manifest if row.status == "success"}
    else:
        scores = import_blinded_scores(
            scores_path,
            blinding_mapping_path,
            manifest_path,
            unblinded_scores_output,
        )
        expected_score_ids = {
            row.row_id for row in bakeoff_rows if row.status == "success"
        }
    score_ids = set(scores)

    missing_scores = expected_score_ids - score_ids
    if missing_scores:
        sample = sorted(missing_scores)[:5]
        raise ValueError(
            f"manifest contains {len(missing_scores)} successful row(s) without scores: "
            f"{sample}{' ...' if len(missing_scores) > 5 else ''}"
        )
    unknown_scores = score_ids - expected_score_ids
    if unknown_scores:
        sample = sorted(unknown_scores)[:5]
        raise ValueError(
            f"scoring CSV contains {len(unknown_scores)} row(s) not in manifest "
            f"successes: {sample}{' ...' if len(unknown_scores) > 5 else ''}"
        )

    for row in edit_rows:
        _validate_edit_source(row, manifest_by_id)

    # audit rows diagnose prompt wording but never inflate candidate bakeoff counts
    overall = calculate_validity(bakeoff_rows, scores, gate_threshold)
    per_model = calculate_per_model_validity(bakeoff_rows, scores, gate_threshold)
    per_property = calculate_per_property_validity(bakeoff_rows, scores, gate_threshold)
    per_model_property: dict[str, dict[str, ValidityStats]] = {}
    for model in sorted({row.model_backbone for row in bakeoff_rows}):
        model_rows = [row for row in bakeoff_rows if row.model_backbone == model]
        per_model_property[model] = calculate_per_property_validity(
            model_rows, scores, gate_threshold
        )

    eligibility = (
        calculate_eligibility(edit_rows, scores, manifest_by_id) if edit_rows else None
    )
    scored_rows = [manifest_by_id[row_id] for row_id in scores]
    n_ambiguous = sum(scores[row.row_id].prompt_ambiguous for row in scored_rows)
    n_present = sum(scores[row.row_id].target_present for row in scored_rows)
    n_invisible = sum(
        scores[row.row_id].target_present and not scores[row.row_id].target_visible
        for row in scored_rows
    )

    def stats_dict(stats: ValidityStats) -> dict[str, Any]:
        return {
            "n_total": stats.n_total,
            "n_valid": stats.n_valid,
            "rate": stats.rate,
            "ci_lower": stats.ci_lower,
            "ci_upper": stats.ci_upper,
            "passes_gate": stats.passes_gate,
        }

    failures = [row for row in manifest if row.status == "error"]
    bakeoff_failures = [row for row in bakeoff_rows if row.status == "error"]
    observed_arms = {
        (model, prop)
        for model, properties in per_model_property.items()
        for prop in properties
    }
    arm_contract = (
        required_model_property_arms
        if required_model_property_arms is not None
        else {
            (model, prop)
            for model in {row.model_backbone for row in bakeoff_rows}
            for prop in {row.property_id for row in bakeoff_rows}
        }
    )
    unexpected_arms = observed_arms - arm_contract
    if unexpected_arms:
        raise ValueError(
            "manifest contains model-property arms outside the required contract: "
            f"{sorted(unexpected_arms)}"
        )
    failed_arms = []
    for model, prop in sorted(arm_contract):
        stats = per_model_property.get(model, {}).get(prop)
        if stats is None:
            failed_arms.append(
                {
                    "model_backbone": model,
                    "property_id": prop,
                    "n_total": 0,
                    "n_valid": 0,
                    "rate": 0.0,
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                    "passes_gate": False,
                    "missing": True,
                }
            )
        elif not stats.passes_gate:
            failed_arms.append(
                {
                    "model_backbone": model,
                    "property_id": prop,
                    **stats_dict(stats),
                    "missing": False,
                }
            )
    required_arm_count = len(arm_contract)
    return {
        "gate_contract": {
            "design": "bakeoff",
            "conditioning": "direct_text",
            "criterion": "target_present",
            "threshold": gate_threshold,
            "secondary_diagnostics": ["target_visible", "prompt_ambiguous"],
        },
        "overall": stats_dict(overall),
        "qualification_decision": {
            "passes_gate": required_arm_count > 0 and not failed_arms,
            "required_model_property_arms": required_arm_count,
            "failed_arms": failed_arms,
        },
        "per_model": {model: stats_dict(stats) for model, stats in per_model.items()},
        "per_property": {
            prop: stats_dict(stats) for prop, stats in per_property.items()
        },
        "per_model_property": {
            model: {prop: stats_dict(stats) for prop, stats in properties.items()}
            for model, properties in per_model_property.items()
        },
        "render_failures": {
            "n_total": len(failures),
            "n_bakeoff": len(bakeoff_failures),
            "by_model": {
                model: sum(row.model_backbone == model for row in bakeoff_failures)
                for model in sorted({row.model_backbone for row in bakeoff_rows})
            },
            "by_property": {
                prop: sum(row.property_id == prop for row in bakeoff_failures)
                for prop in sorted({row.property_id for row in bakeoff_rows})
            },
        },
        "ambiguity": {
            "n_total": len(scored_rows),
            "n_ambiguous": n_ambiguous,
            "rate": n_ambiguous / len(scored_rows) if scored_rows else 0.0,
        },
        "visibility": {
            "n_total": n_present,
            "n_invisible": n_invisible,
            "rate": n_invisible / n_present if n_present else 0.0,
        },
        "eligibility": (
            {
                "n_attempted": eligibility.n_attempted,
                "n_eligible": eligibility.n_eligible,
                "rate": eligibility.rate,
            }
            if eligibility
            else None
        ),
    }


def write_summary_csv(results: dict[str, Any], output_path: Path) -> None:
    """
    Write human-readable summary CSV.

    Args:
        results: analysis results dictionary
        output_path: path to output CSV
    """
    rows = []

    # overall
    overall = results["overall"]
    rows.append(
        {
            "group_type": "overall",
            "group_id": "all",
            "n_total": overall["n_total"],
            "n_valid": overall["n_valid"],
            "rate": f"{overall['rate']:.4f}",
            "ci_lower": f"{overall['ci_lower']:.4f}",
            "ci_upper": f"{overall['ci_upper']:.4f}",
            "passes_gate": str(overall["passes_gate"]),
        }
    )

    # per-model
    for model, stats in results["per_model"].items():
        rows.append(
            {
                "group_type": "model",
                "group_id": model,
                "n_total": stats["n_total"],
                "n_valid": stats["n_valid"],
                "rate": f"{stats['rate']:.4f}",
                "ci_lower": f"{stats['ci_lower']:.4f}",
                "ci_upper": f"{stats['ci_upper']:.4f}",
                "passes_gate": str(stats["passes_gate"]),
            }
        )

    # model-property cells prevent pooled properties from hiding a weak arm
    for model, properties in results.get("per_model_property", {}).items():
        for prop, stats in properties.items():
            rows.append(
                {
                    "group_type": "model_property",
                    "group_id": f"{model}:{prop}",
                    "n_total": stats["n_total"],
                    "n_valid": stats["n_valid"],
                    "rate": f"{stats['rate']:.4f}",
                    "ci_lower": f"{stats['ci_lower']:.4f}",
                    "ci_upper": f"{stats['ci_upper']:.4f}",
                    "passes_gate": str(stats["passes_gate"]),
                }
            )

    # per-property
    for prop, stats in results["per_property"].items():
        rows.append(
            {
                "group_type": "property",
                "group_id": prop,
                "n_total": stats["n_total"],
                "n_valid": stats["n_valid"],
                "rate": f"{stats['rate']:.4f}",
                "ci_lower": f"{stats['ci_lower']:.4f}",
                "ci_upper": f"{stats['ci_upper']:.4f}",
                "passes_gate": str(stats["passes_gate"]),
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "group_type",
            "group_id",
            "n_total",
            "n_valid",
            "rate",
            "ci_lower",
            "ci_upper",
            "passes_gate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_results_json(results: dict[str, Any], output_path: Path) -> None:
    """
    Write machine-readable JSON results.

    Args:
        results: analysis results dictionary
        output_path: path to output JSON
    """
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
