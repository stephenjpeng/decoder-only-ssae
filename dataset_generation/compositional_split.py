"""
Systematic train / holdout splits for compositional generalization.

A *concept combination* is one choice per category (full tuple). Holdout prompts
are disjoint from training on this tuple — no holdout row appears in the train
set. This does not change SSAE training code; point `folder_path` at the train
folder only, and use the holdout folder for evaluation after embeddings exist.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def read_categories(path: Path | str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected non-empty category dict in {path}")
    for k, v in data.items():
        if not isinstance(v, list) or not v:
            raise ValueError(f"Category {k!r} must be a non-empty list of property strings")
    return data


def category_order(categories: Mapping[str, Any]) -> List[str]:
    return list(categories.keys())


def combo_tuple_key(choices: Mapping[str, str], keys: Sequence[str]) -> Tuple[str, ...]:
    return tuple(choices[k] for k in keys)


def choices_to_prompt(choices: Mapping[str, str], keys: Sequence[str]) -> str:
    return ", ".join(choices[k] for k in keys)


def enumerate_full_combos(categories: Mapping[str, List[str]]) -> List[Dict[str, str]]:
    keys = category_order(categories)
    tuples = itertools.product(*(categories[k] for k in keys))
    return [dict(zip(keys, t)) for t in tuples]


def _property_locations(
    categories: Mapping[str, list[str]],
) -> dict[str, tuple[str, int]]:
    """Map each property phrase to (category_name, within-category index)."""
    locations = {}
    for cat_name, props in categories.items():
        for idx, prop in enumerate(props):
            locations[prop] = (cat_name, idx)
    return locations


def _validate_holdout_pair(
    categories: Mapping[str, list[str]],
    pair: tuple[str, str],
) -> tuple[tuple[str, str], tuple[str, str]]:
    """
    Resolve and validate two unique values from different categories.
    Returns ((cat_a, phrase_a), (cat_b, phrase_b)).
    Raises ValueError for: unknown phrase, duplicate phrases, same-category pair.
    """
    phrase_a, phrase_b = pair

    # check for duplicates
    if phrase_a == phrase_b:
        raise ValueError(f"Duplicate property in holdout pair: {phrase_a!r}")

    locations = _property_locations(categories)

    # check both phrases exist
    if phrase_a not in locations:
        raise ValueError(f"Unknown property {phrase_a!r} in holdout pair")
    if phrase_b not in locations:
        raise ValueError(f"Unknown property {phrase_b!r} in holdout pair")

    cat_a, _ = locations[phrase_a]
    cat_b, _ = locations[phrase_b]

    # check different categories
    if cat_a == cat_b:
        raise ValueError(
            f"Holdout pair must be from different categories; both {phrase_a!r} "
            f"and {phrase_b!r} are from {cat_a!r}"
        )

    return ((cat_a, phrase_a), (cat_b, phrase_b))


def _split_by_value_pair(
    all_combos: Sequence[dict[str, str]],
    pair: tuple[str, str],
) -> tuple[list[int], list[int]]:
    """
    Put every tuple containing BOTH values into holdout; rest to train.
    Returns (train_indices, holdout_indices).
    """
    phrase_a, phrase_b = pair
    train_idx = []
    holdout_idx = []

    for i, combo in enumerate(all_combos):
        values = set(combo.values())
        if phrase_a in values and phrase_b in values:
            holdout_idx.append(i)
        else:
            train_idx.append(i)

    return train_idx, holdout_idx


def _shuffle_split_indices(
    n_total: int, n_holdout: int, seed: int
) -> Tuple[List[int], List[int]]:
    if n_holdout <= 0:
        return list(range(n_total)), []
    if n_holdout >= n_total:
        raise ValueError(
            f"n_holdout ({n_holdout}) must be strictly less than n_total ({n_total}) "
            "so the training set is non-empty."
        )
    rng = random.Random(seed)
    order = list(range(n_total))
    rng.shuffle(order)
    holdout_idx = order[:n_holdout]
    train_idx = order[n_holdout:]
    return train_idx, holdout_idx


def _subsample(candidates: List[int], max_n: int | None, seed: int, salt: int) -> List[int]:
    if max_n is None or len(candidates) <= max_n:
        return candidates
    rng = random.Random(seed + salt)
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    return shuffled[:max_n]


def build_disjoint_split(
    categories: Mapping[str, List[str]],
    *,
    holdout_fraction: float | None = None,
    n_holdout: int | None = None,
    max_train_prompts: int | None = None,
    max_holdout_prompts: int | None = None,
    holdout_value_pair: tuple[str, str] | None = None,
    seed: int = 0,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any]]:
    """
    Split full Cartesian product into train / holdout by disjoint full tuples.

    Exactly one of holdout_fraction or n_holdout must be set (unless both imply
    n_holdout == 0 via fraction 0), OR holdout_value_pair is set.

    When holdout_value_pair is set, every tuple containing BOTH values goes to
    holdout; the rest to train. max_holdout_prompts is rejected in this mode.
    """
    keys = category_order(categories)
    all_combos = enumerate_full_combos(categories)
    n_total = len(all_combos)

    # validate mutual exclusivity
    if holdout_value_pair is not None:
        if max_holdout_prompts is not None:
            raise ValueError(
                "max_holdout_prompts cannot be used with holdout_value_pair; "
                "the pair determines the exact holdout set."
            )
        if holdout_fraction is not None or n_holdout is not None:
            raise ValueError(
                "Pass only holdout_value_pair, not holdout_fraction or n_holdout."
            )
    elif holdout_fraction is not None and n_holdout is not None:
        raise ValueError("Pass only one of holdout_fraction or n_holdout.")

    # determine split strategy
    if holdout_value_pair is not None:
        # leave-value-pair-out split
        validated_pair = _validate_holdout_pair(categories, holdout_value_pair)
        (cat_a, phrase_a), (cat_b, phrase_b) = validated_pair
        train_idx, holdout_idx = _split_by_value_pair(all_combos, holdout_value_pair)
        split_strategy = "leave_value_pair_out"
    else:
        # random shuffle split
        if holdout_fraction is not None:
            n_h = int(round(holdout_fraction * n_total))
        elif n_holdout is not None:
            n_h = n_holdout
        else:
            n_h = 0
        train_idx, holdout_idx = _shuffle_split_indices(n_total, n_h, seed)
        split_strategy = "disjoint_full_tuple_shuffle"

    train_choices = [all_combos[i] for i in train_idx]
    holdout_choices = [all_combos[i] for i in holdout_idx]

    train_keys_set = {combo_tuple_key(c, keys) for c in train_choices}
    holdout_keys_set = {combo_tuple_key(c, keys) for c in holdout_choices}
    overlap = train_keys_set & holdout_keys_set
    if overlap:
        raise RuntimeError(f"Internal error: overlapping tuples between train and holdout: {overlap}")

    if max_train_prompts is not None and len(train_choices) > max_train_prompts:
        # Subsample train combos while keeping holdout fixed; still disjoint.
        kept_train_idx = _subsample(
            list(range(len(train_choices))), max_train_prompts, seed, salt=101
        )
        train_choices = [train_choices[i] for i in sorted(kept_train_idx)]

    if max_holdout_prompts is not None and len(holdout_choices) > max_holdout_prompts:
        kept_h_idx = _subsample(
            list(range(len(holdout_choices))), max_holdout_prompts, seed, salt=202
        )
        holdout_choices = [holdout_choices[i] for i in sorted(kept_h_idx)]

    # build base stats
    stats = {
        "n_categories": len(keys),
        "category_order": keys,
        "n_total_full_factorial": n_total,
        "split_strategy": split_strategy,
        "n_train_written": len(train_choices),
        "n_holdout_written": len(holdout_choices),
        "seed": seed,
        "max_train_prompts": max_train_prompts,
        "max_holdout_prompts": max_holdout_prompts,
    }

    # add strategy-specific stats
    if holdout_value_pair is not None:
        # pair-specific stats
        (cat_a, phrase_a), (cat_b, phrase_b) = validated_pair
        stats["holdout_value_pair"] = [phrase_a, phrase_b]
        stats["holdout_pair_categories"] = [cat_a, cat_b]

        # count pair occurrences
        n_holdout_pair_matches = sum(
            1 for c in holdout_choices
            if phrase_a in c.values() and phrase_b in c.values()
        )
        n_train_pair_matches = sum(
            1 for c in train_choices
            if phrase_a in c.values() and phrase_b in c.values()
        )
        first_value_train_count = sum(1 for c in train_choices if phrase_a in c.values())
        second_value_train_count = sum(1 for c in train_choices if phrase_b in c.values())

        # expected holdout count: product of sizes of other categories
        size_a = len(categories[cat_a])
        size_b = len(categories[cat_b])
        n_expected = n_total // (size_a * size_b)

        stats["n_expected_pair_tuples"] = n_expected
        stats["n_holdout_pair_matches"] = n_holdout_pair_matches
        stats["n_train_pair_matches"] = n_train_pair_matches
        stats["first_value_train_count"] = first_value_train_count
        stats["second_value_train_count"] = second_value_train_count
        stats["all_holdout_rows_contain_pair"] = n_holdout_pair_matches == len(holdout_choices)
        stats["no_train_rows_contain_pair"] = n_train_pair_matches == 0
        stats["individual_values_remain_in_train"] = (
            first_value_train_count > 0 and second_value_train_count > 0
        )
    else:
        # random split stats
        stats["n_holdout_requested"] = n_h
        stats["holdout_fraction"] = holdout_fraction
        stats["n_holdout_param"] = n_holdout
    return train_choices, holdout_choices, stats


def format_prompt_entry(
    tid: int, choices: Mapping[str, str], keys: Sequence[str]
) -> Dict[str, Any]:
    return {
        "id": tid,
        "prompt": choices_to_prompt(choices, keys),
        "choices": {k: choices[k] for k in keys},
        "tuple_key": list(combo_tuple_key(choices, keys)),
    }


def write_split_folder(
    folder: Path,
    combos: List[Dict[str, str]],
    categories: Mapping[str, List[str]],
    *,
    split_label: str,
    copy_properties_same_from: Path | None = None,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    keys = category_order(categories)

    prompts = [format_prompt_entry(i, c, keys) for i, c in enumerate(combos)]

    with open(folder / "properties.json", "w", encoding="utf-8") as f:
        json.dump(dict(categories), f, indent=4, ensure_ascii=False)

    with open(folder / "prompts.json", "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=4, ensure_ascii=False)

    meta = {
        "split": split_label,
        "n_prompts": len(prompts),
        "manifest_ref": "split_manifest.json (parent directory)",
    }
    with open(folder / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    if copy_properties_same_from is not None:
        dest = folder / "properties_same.json"
        shutil.copy2(copy_properties_same_from, dest)

    tuples_path = folder / f"{split_label}_tuples.json"
    with open(tuples_path, "w", encoding="utf-8") as f:
        json.dump([list(combo_tuple_key(c, keys)) for c in combos], f, indent=2)


def write_compositional_split(
    categories_json: Path | str,
    output_root: Path | str,
    *,
    holdout_fraction: float | None = None,
    n_holdout: int | None = None,
    max_train_prompts: int | None = None,
    max_holdout_prompts: int | None = None,
    holdout_value_pair: tuple[str, str] | None = None,
    seed: int = 0,
    shuffle_train_ids: bool = False,
    properties_same_json: Path | str | None = None,
) -> Dict[str, Any]:
    """
    Write ``output_root/train`` and ``output_root/holdout`` datasets.

    Training uses only ``train/`` (prompts.json + properties.json + embds after
    you run embedding extraction). Holdout tuples never appear in train prompts.

    When holdout_value_pair is set, every tuple containing BOTH values goes to
    holdout; the rest to train.
    """
    categories_path = Path(categories_json)
    root = Path(output_root)
    categories = read_categories(categories_path)

    train_combos, holdout_combos, stats = build_disjoint_split(
        categories,
        holdout_fraction=holdout_fraction,
        n_holdout=n_holdout,
        max_train_prompts=max_train_prompts,
        max_holdout_prompts=max_holdout_prompts,
        holdout_value_pair=holdout_value_pair,
        seed=seed,
    )

    keys = category_order(categories)
    if shuffle_train_ids:
        rng = random.Random(seed + 991)
        order = list(range(len(train_combos)))
        rng.shuffle(order)
        train_combos = [train_combos[i] for i in order]

    psame = Path(properties_same_json) if properties_same_json else None
    if psame is not None and not psame.is_file():
        raise FileNotFoundError(f"properties_same.json not found: {psame}")

    manifest = {
        "categories_source": str(categories_path.resolve()),
        **stats,
        "train_dir": str((root / "train").resolve()),
        "holdout_dir": str((root / "holdout").resolve()),
    }

    write_split_folder(
        root / "train",
        train_combos,
        categories,
        split_label="train",
        copy_properties_same_from=psame,
    )
    write_split_folder(
        root / "holdout",
        holdout_combos,
        categories,
        split_label="holdout",
        copy_properties_same_from=psame,
    )

    # Disjointness certificate (tuple-level)
    train_keys = {combo_tuple_key(c, keys) for c in train_combos}
    hold_keys = {combo_tuple_key(c, keys) for c in holdout_combos}
    manifest["n_train_unique_tuples"] = len(train_keys)
    manifest["n_holdout_unique_tuples"] = len(hold_keys)
    manifest["tuple_intersection_size"] = len(train_keys & hold_keys)
    manifest["holdout_disjoint_from_train"] = len(train_keys & hold_keys) == 0

    with open(root / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create train/ and holdout/ prompt folders with disjoint full concept "
            "combinations for compositional generalization experiments."
        )
    )
    p.add_argument(
        "--categories_json",
        type=Path,
        required=True,
        help="Path to categories_with_properties.json (category -> list of phrases).",
    )
    p.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Directory to create train/ and holdout/ under.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--holdout_fraction",
        type=float,
        help="Fraction of full factorial assigned to holdout (e.g. 0.1).",
    )
    g.add_argument(
        "--n_holdout",
        type=int,
        help="Exact number of holdout tuples (disjoint from train).",
    )
    g.add_argument(
        "--holdout_value_pair",
        nargs=2,
        metavar=("VALUE_A", "VALUE_B"),
        help="Two property values from different categories; tuples containing BOTH go to holdout.",
    )
    p.add_argument(
        "--max_train_prompts",
        type=int,
        default=None,
        help="After split, randomly subsample train to at most this many prompts.",
    )
    p.add_argument(
        "--max_holdout_prompts",
        type=int,
        default=None,
        help="After split, randomly subsample holdout to at most this many prompts.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--shuffle_train_order",
        action="store_true",
        help="Shuffle train prompt order (ids 0..n-1) after split; holdout unchanged.",
    )
    p.add_argument(
        "--properties_same_json",
        type=Path,
        default=None,
        help="If set, copy this file into train/ and holdout/ as properties_same.json.",
    )
    return p.parse_args(argv)


def load_tuple_keys_from_prompts(folder: Path | str) -> set[Tuple[str, ...]]:
    """Load ``tuple_key`` entries from ``prompts.json`` (written by this module)."""
    folder = Path(folder)
    with open(folder / "prompts.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
    keys = set()
    for p in prompts:
        if "tuple_key" not in p:
            raise KeyError(
                f"Prompt id {p.get('id')} in {folder} has no 'tuple_key'; "
                "use prompts produced by compositional_split or add tuple_key fields."
            )
        keys.add(tuple(p["tuple_key"]))
    return keys


def verify_disjoint_train_holdout_folders(train_dir: Path | str, holdout_dir: Path | str) -> None:
    """Raise if any full tuple appears in both train and holdout ``prompts.json`` files."""
    train_keys = load_tuple_keys_from_prompts(train_dir)
    holdout_keys = load_tuple_keys_from_prompts(holdout_dir)
    overlap = train_keys & holdout_keys
    if overlap:
        sample = next(iter(overlap))
        raise ValueError(
            f"Train/holdout overlap: {len(overlap)} shared tuples (e.g. {sample!r})."
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    holdout_fraction = args.holdout_fraction
    n_holdout = args.n_holdout
    holdout_value_pair = tuple(args.holdout_value_pair) if args.holdout_value_pair else None
    manifest = write_compositional_split(
        args.categories_json,
        args.output_root,
        holdout_fraction=holdout_fraction,
        n_holdout=n_holdout,
        max_train_prompts=args.max_train_prompts,
        max_holdout_prompts=args.max_holdout_prompts,
        holdout_value_pair=holdout_value_pair,
        seed=args.seed,
        shuffle_train_ids=args.shuffle_train_order,
        properties_same_json=args.properties_same_json,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
