"""
Property-mask feature construction for ridge baselines.

Ensures plain and pairwise ridge use the same column order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class InteractionColumn:
    left_pid: int
    right_pid: int
    left_property: str
    right_property: str
    left_category: str
    right_category: str


def cross_category_interactions(properties) -> tuple[InteractionColumn, ...]:
    """
    Return canonical cross-category property pairs in pid order.

    Iterates left_pid then right_pid, both ascending. Includes a pair only when
    left_pid < right_pid AND the properties belong to DIFFERENT categories.
    Does not include same-category pairs.
    """
    columns = []
    n_props = properties.n_properties

    for left_pid in range(n_props):
        left_cid = properties.pid_to_cid[left_pid]
        left_property = properties.pid_to_property[left_pid]
        left_category = properties.cid_to_category[left_cid]

        for right_pid in range(left_pid + 1, n_props):
            right_cid = properties.pid_to_cid[right_pid]

            # only include cross-category pairs
            if left_cid == right_cid:
                continue

            right_property = properties.pid_to_property[right_pid]
            right_category = properties.cid_to_category[right_cid]

            columns.append(
                InteractionColumn(
                    left_pid=left_pid,
                    right_pid=right_pid,
                    left_property=left_property,
                    right_property=right_property,
                    left_category=left_category,
                    right_category=right_category,
                )
            )

    return tuple(columns)


def pairwise_design(
    mask: torch.Tensor,
    columns: Sequence[InteractionColumn],
) -> torch.Tensor:
    """
    Return [26 main effects | 276 cross-category products], shape (n, 302).

    The mask input is shape (n, 26). Main effects are the 26 mask columns directly.
    Each interaction column value = mask[:, left_pid] * mask[:, right_pid].
    No intercept appended (fit_ridge from run_baselines.py appends it).
    """
    n = mask.size(0)
    n_main = mask.size(1)
    n_interactions = len(columns)

    # pre-allocate output
    design = torch.zeros(n, n_main + n_interactions, dtype=mask.dtype, device=mask.device)

    # main effects (first 26 columns)
    design[:, :n_main] = mask

    # interaction columns
    for i, col in enumerate(columns):
        design[:, n_main + i] = mask[:, col.left_pid] * mask[:, col.right_pid]

    return design


def design_fingerprint(columns: Sequence[InteractionColumn]) -> dict[str, object]:
    """
    Return ordered column metadata and a SHA-256 hash of column definitions.

    Includes: list of column dicts (with all 6 fields), sha256 over JSON of the
    sorted column list, n_main_effects, n_interactions, n_total.
    """
    column_dicts = [asdict(col) for col in columns]

    # sha256 over sorted JSON representation
    canonical_json = json.dumps(column_dicts, sort_keys=True, ensure_ascii=True)
    sha256_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    n_main_effects = 26  # hardcoded for current dataset
    n_interactions = len(columns)
    n_total = n_main_effects + n_interactions

    return {
        "columns": column_dicts,
        "sha256": sha256_hash,
        "n_main_effects": n_main_effects,
        "n_interactions": n_interactions,
        "n_total": n_total,
    }
