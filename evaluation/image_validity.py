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
    image_path: str
    source_row_id: str | None


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
        "image_path",
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

            source_row_id = obj.get("source_row_id")
            if source_row_id is not None and (
                not isinstance(source_row_id, str) or not source_row_id.strip()
            ):
                raise ValueError(
                    f"manifest line {line_num} field source_row_id "
                    "must be a nonblank string when present"
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
                    image_path=obj["image_path"].strip(),
                    source_row_id=(
                        source_row_id.strip() if source_row_id is not None else None
                    ),
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
        if row.row_id not in scores:
            raise ValueError(f"manifest row {row.row_id} has no corresponding score")

        score = scores[row.row_id]
        if score.target_present:
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


def calculate_eligibility(
    manifest: list[ManifestRow],
    scores: dict[str, ScoringRow],
) -> EligibilityStats:
    """
    Calculate edit eligibility statistics for edit stages.

    An edit row is eligible if its source_row_id exists and the source
    target is present (scored target_present=true).

    Args:
        manifest: list of manifest rows (should be edit stage only)
        scores: mapping from row_id to score

    Returns:
        EligibilityStats with attempted and eligible counts
    """
    n_attempted = len(manifest)
    n_eligible = 0

    for row in manifest:
        if row.stage != "edit":
            raise ValueError(f"eligibility row {row.row_id} must have stage edit")
        if not row.source_row_id:
            raise ValueError(f"edit row {row.row_id} has no source_row_id")
        if row.source_row_id == row.row_id:
            raise ValueError(f"edit row {row.row_id} cannot reference itself")
        if row.source_row_id not in scores:
            raise ValueError(
                f"edit row {row.row_id} references source without a score: "
                f"{row.source_row_id}"
            )

        source_score = scores[row.source_row_id]
        if source_score.target_present:
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
) -> dict[str, Any]:
    """
    Run full validity analysis and return structured results.

    Args:
        manifest_path: path to manifest JSONL
        scores_path: path to scoring CSV
        gate_threshold: minimum observed rate to pass native gate

    Returns:
        dictionary with overall, per_model, per_property stats,
        plus separate ambiguity and visibility reports

    Raises:
        ValueError: if score CSV contains rows not in the manifest
    """
    scores = load_scoring_csv(scores_path)
    manifest = load_manifest_jsonl(manifest_path)

    # validate score coverage: all manifest rows must have scores, no extra scores
    manifest_ids = {r.row_id for r in manifest}
    score_ids = set(scores.keys())

    missing_scores = manifest_ids - score_ids
    if missing_scores:
        sample = sorted(missing_scores)[:5]
        raise ValueError(
            f"manifest contains {len(missing_scores)} row(s) without scores: "
            f"{sample}{' ...' if len(missing_scores) > 5 else ''}"
        )

    unknown_scores = score_ids - manifest_ids
    if unknown_scores:
        sample = sorted(unknown_scores)[:5]
        raise ValueError(
            f"scoring CSV contains {len(unknown_scores)} row(s) not in manifest: "
            f"{sample}{' ...' if len(unknown_scores) > 5 else ''}"
        )

    # only native rows contribute to the validity gate
    native_rows = [r for r in manifest if r.stage == "native"]
    edit_rows = [r for r in manifest if r.stage == "edit"]

    # corrupt source joins must not appear as ordinary ineligible attempts
    native_ids = {row.row_id for row in native_rows}
    for row in edit_rows:
        if not row.source_row_id:
            raise ValueError(f"edit row {row.row_id} has no source_row_id")
        if row.source_row_id not in native_ids:
            raise ValueError(
                f"edit row {row.row_id} source_row_id must reference a native row: "
                f"{row.source_row_id}"
            )

    # native validity (for gate decision)
    overall = calculate_validity(native_rows, scores, gate_threshold)

    # per-model validity (native only)
    per_model = calculate_per_model_validity(native_rows, scores, gate_threshold)

    # per-property validity (native only)
    per_property = calculate_per_property_validity(native_rows, scores, gate_threshold)

    eligibility = None
    if edit_rows:
        eligibility = calculate_eligibility(edit_rows, scores)

    # ambiguity and visibility counts (across all manifest rows)
    n_ambiguous = sum(1 for r in manifest if scores[r.row_id].prompt_ambiguous)
    n_present_all = sum(1 for r in manifest if scores[r.row_id].target_present)
    n_invisible = sum(
        1
        for r in manifest
        if scores[r.row_id].target_present and not scores[r.row_id].target_visible
    )

    return {
        "overall": {
            "n_total": overall.n_total,
            "n_valid": overall.n_valid,
            "rate": overall.rate,
            "ci_lower": overall.ci_lower,
            "ci_upper": overall.ci_upper,
            "passes_gate": overall.passes_gate,
        },
        "per_model": {
            model: {
                "n_total": stats.n_total,
                "n_valid": stats.n_valid,
                "rate": stats.rate,
                "ci_lower": stats.ci_lower,
                "ci_upper": stats.ci_upper,
                "passes_gate": stats.passes_gate,
            }
            for model, stats in per_model.items()
        },
        "per_property": {
            prop: {
                "n_total": stats.n_total,
                "n_valid": stats.n_valid,
                "rate": stats.rate,
                "ci_lower": stats.ci_lower,
                "ci_upper": stats.ci_upper,
                "passes_gate": stats.passes_gate,
            }
            for prop, stats in per_property.items()
        },
        "ambiguity": {
            "n_total": len(manifest),
            "n_ambiguous": n_ambiguous,
            "rate": n_ambiguous / len(manifest) if manifest else 0.0,
        },
        "visibility": {
            "n_total": n_present_all,
            "n_invisible": n_invisible,
            "rate": n_invisible / n_present_all if n_present_all > 0 else 0.0,
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
