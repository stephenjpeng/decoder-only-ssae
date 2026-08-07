"""E2 — null-space characterisation of the one-hot compositional design (AUG-03).

Everything here is closed-form and depends only on the *category structure*, not on any
trained weights, embeddings or images. That is the point: the ambiguity it measures is a
property of the experimental design, so it bounds what *any* additive method — ridge, the
one-layer SSAE, mean-arithmetic — can possibly identify, and it is knowable before a
single GPU-hour is spent.

The design
----------
Each prompt selects exactly one value from each of ``C`` categories. Stack the one-hot
indicators into ``M in {0,1}^{n x p}`` (``p = sum_c n_c`` properties) and append an
intercept column, giving the augmented design ``A = [M | 1] in R^{n x (p+1)}``. This is
exactly what ``baselines/run_baselines.py::fit_ridge`` builds.

``A`` is rank-deficient by construction. Within each category the indicators sum to the
all-ones vector, so for every category ``c``::

    sum_{k in c} e_k  -  1  =  0

That is ``C`` independent linear dependencies among the ``p+1`` columns, hence::

    rank(A) = p + 1 - C          null dim = C

For the SD3.5 prompt grid (``p = 26``, ``C = 7``): **rank 20, null space 7**, matching the
proposal's E2 statement.

Why it matters for C2/C3
------------------------
Write a fitted solution as ``V in R^{(p+1) x d}`` (rows = concept vectors ``v_k``, last row
= intercept ``b``). Any ``V' = V + N Z`` with ``N`` a basis of null(``A``) and ``Z``
arbitrary produces *identical predictions on every representable prompt*. So:

* **Identified:** within-category contrasts ``v_k - v_k'`` for ``k, k'`` in the same
  category, and the full prediction ``a^T V`` for any valid one-hot ``a``.
* **Unidentified:** the absolute level of each concept vector, the intercept, and — the
  case the paper cares about — *the result of zeroing a block*.

Zeroing the block for concept ``k`` means evaluating at ``a - e_k``, which has no active
value in ``k``'s category. That point is **outside the row space of the design**: it is not
a valid prompt, no training example ever looked like it, and its predicted value moves
freely with the null-space component. Deletion is therefore arbitrary by construction,
while *replacement* (``a - e_k + e_k'``, still a valid prompt) is fully identified.

This is the formal reason the proposal insists deletion and replacement be reported
separately and never averaged, and the reason E4 proposes the category-marginal operator:
moving a block to its category marginal keeps the edit inside the row space, so the result
is invariant to the null-space offset.

Usage
-----
    python -m analysis.nullspace \\
        --categories dataset_generation/prompts/input/categories_with_properties.json \\
        --output_dir results/analysis/e2_nullspace

Writes ``e2_nullspace.json`` (machine-readable), ``null_basis.npy``, a
``run_manifest.json``, and ``e2_deletion_decomposition.png`` — the identified-vs-arbitrary
decomposition of a deletion, which is the single figure E2 owes the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trainings.utils.run_manifest import sha256_file, write_run_manifest


# --------------------------------------------------------------------------- design


def build_design(categories: dict[str, list[str]]) -> tuple[np.ndarray, list[str], list[int]]:
    """Full-factorial augmented design ``A = [M | 1]``.

    Returns ``(A, property_names, property_to_category)``. Rows enumerate the complete
    cartesian product, which is the right object for a *structural* rank statement: a
    subsample can only lose rank, never gain it, so the full grid gives the best case.
    """
    cat_names = list(categories)
    prop_names: list[str] = []
    prop_cat: list[int] = []
    for ci, c in enumerate(cat_names):
        for v in categories[c]:
            prop_names.append(v)
            prop_cat.append(ci)

    sizes = [len(categories[c]) for c in cat_names]
    n_rows = int(np.prod(sizes))
    p = len(prop_names)

    # Column offset of each category's block.
    offsets = np.cumsum([0] + sizes[:-1])

    M = np.zeros((n_rows, p), dtype=np.float64)
    # Mixed-radix enumeration; no need to materialise the prompt strings.
    for r in range(n_rows):
        rem = r
        for ci in range(len(sizes) - 1, -1, -1):
            choice = rem % sizes[ci]
            rem //= sizes[ci]
            M[r, offsets[ci] + choice] = 1.0

    A = np.concatenate([M, np.ones((n_rows, 1))], axis=1)
    return A, prop_names, prop_cat


def analytic_expectations(categories: dict[str, list[str]]) -> dict:
    p = sum(len(v) for v in categories.values())
    c = len(categories)
    return {
        "n_properties": p,
        "n_categories": c,
        "n_design_columns": p + 1,
        "expected_rank": p + 1 - c,
        "expected_null_dim": c,
        "n_full_factorial_rows": int(np.prod([len(v) for v in categories.values()])),
    }


# ----------------------------------------------------------------------- null space


def null_basis(A: np.ndarray, tol: float | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Orthonormal basis for null(A) via SVD. Returns ``(N, singular_values, rank)``.

    ``N`` has shape ``(n_cols, null_dim)``; its columns span the directions in coefficient
    space that leave every prediction on a valid prompt unchanged.
    """
    _, s, vt = np.linalg.svd(A, full_matrices=True)
    if tol is None:
        tol = max(A.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    rank = int((s > tol).sum())
    N = vt[rank:].T
    return N, s, rank


def structural_null_basis(prop_cat: list[int], n_categories: int) -> np.ndarray:
    """The *interpretable* null basis: one vector per category, ``sum_{k in c} e_k - 1``.

    Numerically equivalent to :func:`null_basis` up to an orthogonal change of basis, but
    each column has a plain-English meaning, which is what makes the deletion decomposition
    readable.
    """
    p = len(prop_cat)
    N = np.zeros((p + 1, n_categories))
    for ci in range(n_categories):
        for k, c in enumerate(prop_cat):
            if c == ci:
                N[k, ci] = 1.0
        N[p, ci] = -1.0
    return N


def subspace_agreement(N1: np.ndarray, N2: np.ndarray) -> float:
    """Largest principal angle (deg) between two subspaces; ~0 means they coincide."""
    q1, _ = np.linalg.qr(N1)
    q2, _ = np.linalg.qr(N2)
    s = np.linalg.svd(q1.T @ q2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return float(np.degrees(np.arccos(s.min())))


# ------------------------------------------------------------------ edit operators


def edit_vectors(
    prop_cat: list[int], target_pid: int, replacement_pid: int | None
) -> dict[str, np.ndarray]:
    """Coefficient-space difference vectors for the three edit operators.

    An edit maps a prompt indicator ``a`` to ``a + delta``. The predicted change is
    ``delta^T V``, so identifiability of the *edit* is identifiability of ``delta``: an edit
    is arbitrary exactly to the extent ``delta`` has a component in null(A)^perp... no —
    precisely the reverse. ``delta^T V`` is invariant under ``V -> V + N Z`` iff
    ``delta^T N = 0``, i.e. iff ``delta`` is orthogonal to the null space.
    """
    p = len(prop_cat)
    cat = prop_cat[target_pid]
    members = [k for k, c in enumerate(prop_cat) if c == cat]

    # Deletion: remove the active value, leave the category empty. Leaves the row space.
    delete = np.zeros(p + 1)
    delete[target_pid] = -1.0

    ops = {"deletion": delete}

    if replacement_pid is not None:
        # Replacement: still exactly one value active in the category. Stays in the row space.
        replace = np.zeros(p + 1)
        replace[target_pid] = -1.0
        replace[replacement_pid] = 1.0
        ops["replacement"] = replace

    # E4's category-marginal operator: replace the one-hot with the (here uniform) category
    # marginal. The block still sums to 1, so this also stays in the row space.
    marginal = np.zeros(p + 1)
    marginal[target_pid] = -1.0
    for k in members:
        marginal[k] += 1.0 / len(members)
    ops["category_marginal"] = marginal

    return ops


def deletion_ambiguity_closed_form(category_sizes: list[int]) -> list[float]:
    r"""Closed form for the arbitrary fraction of a deletion, per category.

    With the structural basis ``n_c = 1_c - e_intercept``, the Gram matrix is
    ``G = diag(m) + J`` (the ``+J`` because every basis vector shares the intercept
    coordinate). For ``delta = -e_k`` with ``k`` in category ``t`` we get
    ``N^T delta = -e_t``, so

    .. math::
        \|P_{null}\,\delta\|^2 = (G^{-1})_{tt}
                                = \frac{1}{m_t} - \frac{1/m_t^2}{1 + \sum_c 1/m_c}

    by Sherman-Morrison. Since ``\|delta\| = 1``, that square root *is* the arbitrary
    fraction.

    The consequence is the useful part: the arbitrary fraction is **decreasing in category
    size**. Deleting a value from a binary category is far less identified than deleting one
    from an eight-way category, purely because of the design. See ``run()`` for the E3
    prediction this licenses.
    """
    inv_sum = sum(1.0 / m for m in category_sizes)
    return [
        float(np.sqrt(1.0 / m - (1.0 / m**2) / (1.0 + inv_sum))) for m in category_sizes
    ]


def decompose_edit(delta: np.ndarray, N: np.ndarray) -> dict:
    """Split an edit into its identified and arbitrary parts.

    ``P_null delta`` is the component whose predicted effect can be changed at will by
    moving along the null space without altering any training-representable prediction.
    """
    q, _ = np.linalg.qr(N)
    proj = q @ (q.T @ delta)
    resid = delta - proj
    norm = float(np.linalg.norm(delta))
    a_norm = float(np.linalg.norm(proj))
    i_norm = float(np.linalg.norm(resid))
    return {
        "norm": norm,
        "arbitrary_norm": a_norm,
        "identified_norm": i_norm,
        # Norm fraction: the one with the closed form below. Note it does NOT sum to 1
        # with the identified fraction -- the two components are orthogonal, so their
        # *squares* add, not their norms.
        "arbitrary_fraction": (a_norm / norm) if norm > 0 else 0.0,
        # Energy fraction: this is the additive decomposition, and the one to plot.
        "arbitrary_energy_fraction": (a_norm**2 / norm**2) if norm > 0 else 0.0,
        "identified_energy_fraction": (i_norm**2 / norm**2) if norm > 0 else 0.0,
        "is_identified": bool(np.allclose(proj, 0.0, atol=1e-10)),
    }


# ----------------------------------------------------------------------------- plot


def plot_decomposition(results: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ops = list(results["edit_decomposition"])
    # Squared norms, because the identified and arbitrary parts are orthogonal: their
    # energies add to ||delta||^2 while their norms do not.
    ident = [results["edit_decomposition"][o]["identified_norm"] ** 2 for o in ops]
    arb = [results["edit_decomposition"][o]["arbitrary_norm"] ** 2 for o in ops]
    labels = [o.replace("_", " ") for o in ops]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.0, 4.2))

    x = np.arange(len(ops))
    ax0.bar(x, ident, 0.6, label="identified", color="#2563eb")
    ax0.bar(x, arb, 0.6, bottom=ident, label="arbitrary (null-space)", color="#b45309")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=12)
    ax0.set_ylabel(r"$\|\delta\|^2$ (orthogonal decomposition)")
    ax0.set_title(
        f"Edit decomposition — target '{results['target_property']}'\n"
        f"(design rank {results['rank']}, null dim {results['null_dim']})",
        fontsize=10,
    )
    ax0.legend(fontsize=9)
    ax0.grid(axis="y", alpha=0.3)

    for xi, o in enumerate(ops):
        d = results["edit_decomposition"][o]
        ax0.text(
            xi,
            d["norm"] ** 2,
            f"{d['arbitrary_energy_fraction']:.0%} arbitrary",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    s = np.asarray(results["singular_values"])
    ax1.semilogy(np.arange(1, s.size + 1), np.maximum(s, 1e-18), "o-", ms=4, color="#374151")
    ax1.axvline(results["rank"] + 0.5, color="#b45309", ls="--", lw=1.2)
    ax1.text(
        results["rank"] + 0.7,
        s[0],
        f"rank = {results['rank']}\nnull dim = {results['null_dim']}",
        fontsize=9,
        va="top",
        color="#b45309",
    )
    ax1.set_xlabel("singular value index")
    ax1.set_ylabel(r"$\sigma_i$ of $A=[M\,|\,\mathbf{1}]$")
    ax1.set_title("Design matrix spectrum", fontsize=10)
    ax1.grid(alpha=0.3)

    pc = results["per_category_deletion_ambiguity"]
    order = sorted(pc, key=lambda k: pc[k]["n_values"])
    sizes = [pc[k]["n_values"] for k in order]
    fracs = [pc[k]["deletion_arbitrary_energy_fraction"] for k in order]
    ax2.bar(np.arange(len(order)), fracs, 0.6, color="#b45309")
    for xi, f in enumerate(fracs):
        ax2.text(xi, f + 0.015, f"{f:.0%}", ha="center", fontsize=8)
    ax2.set_xticks(np.arange(len(order)))
    ax2.set_xticklabels([f"{k}\n(m={m})" for k, m in zip(order, sizes)], fontsize=8)
    ax2.set_ylabel(r"arbitrary energy fraction of $\delta_{\rm delete}$")
    ax2.set_ylim(0, 1)
    ax2.axhline(0.0, color="#2563eb", lw=1.2, ls="--")
    ax2.text(
        len(order) - 0.4, 0.03, "replacement = 0 for all",
        ha="right", fontsize=8, color="#2563eb",
    )
    ax2.set_title(
        "Deletion ambiguity is set by category size\n(E3 prediction: divergence orders the same way)",
        fontsize=10,
    )
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main


def run(
    categories_path: Path,
    output_dir: Path,
    target_property: str | None,
    replacement_property: str | None,
) -> dict:
    categories = json.loads(Path(categories_path).read_text())
    expect = analytic_expectations(categories)

    A, prop_names, prop_cat = build_design(categories)
    N_svd, svals, rank = null_basis(A)
    N_struct = structural_null_basis(prop_cat, expect["n_categories"])

    # Default target: the censoring use case from the proposal.
    if target_property is None:
        target_property = "holding a gun" if "holding a gun" in prop_names else prop_names[0]
    if target_property not in prop_names:
        raise SystemExit(f"--target_property {target_property!r} not in {prop_names}")
    target_pid = prop_names.index(target_property)

    if replacement_property is None:
        same_cat = [
            k for k, c in enumerate(prop_cat) if c == prop_cat[target_pid] and k != target_pid
        ]
        # 'gun -> coffee' is the proposal's on-manifold counterfactual.
        pref = [k for k in same_cat if prop_names[k] == "holding a coffee"]
        replacement_pid = (pref or same_cat)[0] if same_cat else None
    else:
        if replacement_property not in prop_names:
            raise SystemExit(f"--replacement_property {replacement_property!r} unknown")
        replacement_pid = prop_names.index(replacement_property)

    ops = edit_vectors(prop_cat, target_pid, replacement_pid)
    decomposition = {name: decompose_edit(d, N_svd) for name, d in ops.items()}

    # Per-category deletion ambiguity: numeric projection cross-checked against the
    # Sherman-Morrison closed form. Any disagreement means one of them is wrong.
    cat_names = list(categories)
    cat_sizes = [len(categories[c]) for c in cat_names]
    closed_form = deletion_ambiguity_closed_form(cat_sizes)
    per_category = {}
    for ci, cname in enumerate(cat_names):
        pid = prop_cat.index(ci)
        d = decompose_edit(edit_vectors(prop_cat, pid, None)["deletion"], N_svd)
        per_category[cname] = {
            "n_values": cat_sizes[ci],
            "deletion_arbitrary_fraction": d["arbitrary_fraction"],
            "deletion_arbitrary_energy_fraction": d["arbitrary_energy_fraction"],
            "deletion_arbitrary_fraction_closed_form": closed_form[ci],
            "agrees_with_closed_form": bool(
                abs(d["arbitrary_fraction"] - closed_form[ci]) < 1e-9
            ),
        }
    if not all(v["agrees_with_closed_form"] for v in per_category.values()):
        raise RuntimeError(
            "numeric null-space projection disagrees with the closed form; "
            "one of the two derivations is wrong"
        )

    ranked = sorted(per_category.items(), key=lambda kv: -kv[1]["deletion_arbitrary_fraction"])

    results = {
        "analysis": "E2_nullspace",
        "categories_path": str(categories_path),
        "categories_sha256": sha256_file(categories_path),
        "category_names": list(categories),
        "category_sizes": {k: len(v) for k, v in categories.items()},
        "property_names": prop_names,
        "property_to_category": prop_cat,
        "expected": expect,
        "design_shape": list(A.shape),
        "rank": rank,
        "null_dim": int(A.shape[1] - rank),
        "rank_matches_theory": rank == expect["expected_rank"],
        "null_dim_matches_theory": (A.shape[1] - rank) == expect["expected_null_dim"],
        "singular_values": [float(v) for v in svals],
        "structural_vs_svd_null_max_principal_angle_deg": subspace_agreement(N_struct, N_svd),
        "target_property": target_property,
        "target_pid": target_pid,
        "replacement_property": prop_names[replacement_pid] if replacement_pid is not None else None,
        "edit_decomposition": decomposition,
        "per_category_deletion_ambiguity": per_category,
        "e3_prediction": {
            "statement": (
                "The arbitrary (null-space) fraction of a deletion depends only on the "
                "target's category size and is decreasing in it. E3 measures between-model "
                "divergence (SSAE-L1 erasure vs ridge erasure on the same tuple and seed) as "
                "its direct test of C3. That divergence is caused by the two fits landing on "
                "different null-space offsets, so it should be ORDERED BY CATEGORY SIZE: "
                "largest for binary categories, smallest for the 8-value 'situation'."
            ),
            "ordering_most_to_least_arbitrary": [
                {"category": k, "n_values": v["n_values"],
                 "arbitrary_fraction": v["deletion_arbitrary_fraction"]}
                for k, v in ranked
            ],
            "applies_to_proposal_targets": {
                "and a hat": per_category.get("hat", {}).get("deletion_arbitrary_fraction"),
                "holding a gun": per_category.get("action", {}).get("deletion_arbitrary_fraction"),
                "at the beach": per_category.get("situation", {}).get("deletion_arbitrary_fraction"),
            },
            "falsifiable_because": (
                "the three E3 targets sit in categories of size 2, 4 and 8 respectively, so "
                "the prediction is a strict ordering hat > gun > beach on deletion divergence, "
                "with NO corresponding ordering on replacement divergence (replacement is "
                "fully identified for every category size)."
            ),
        },
        "interpretation": {
            "deletion": (
                "a - e_k has no active value in the target category, so it lies outside "
                "the design's row space; its prediction moves freely along the null space "
                "and the edit is not identified by the data."
            ),
            "replacement": (
                "a - e_k + e_k' remains a valid one-hot prompt, so the edit is orthogonal "
                "to the null space and fully identified. This is why the proposal's primary "
                "linear-representation test uses replacement, not deletion."
            ),
            "category_marginal": (
                "moving the block to its category marginal keeps the block sum at 1 and "
                "therefore stays in the row space — E4's constructive fix for the deletion "
                "ambiguity."
            ),
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "e2_nullspace.json").write_text(json.dumps(results, indent=2))
    np.save(output_dir / "null_basis_svd.npy", N_svd)
    np.save(output_dir / "null_basis_structural.npy", N_struct)
    plot_decomposition(results, output_dir / "e2_deletion_decomposition.png")

    write_run_manifest(
        output_dir,
        run_kind="analysis",
        config={
            "analysis": "E2_nullspace",
            "categories_path": str(categories_path),
            "target_property": target_property,
            "replacement_property": results["replacement_property"],
        },
        extra={
            "categories_sha256": results["categories_sha256"],
            "rank": rank,
            "null_dim": results["null_dim"],
            "design_shape": results["design_shape"],
            "note": (
                "Closed-form; depends only on the category structure. No checkpoints, "
                "embeddings or renders involved."
            ),
        },
    )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--categories",
        type=Path,
        default=Path("dataset_generation/prompts/input/categories_with_properties.json"),
    )
    ap.add_argument("--output_dir", type=Path, default=Path("results/analysis/e2_nullspace"))
    ap.add_argument("--target_property", type=str, default=None)
    ap.add_argument("--replacement_property", type=str, default=None)
    args = ap.parse_args()

    r = run(args.categories, args.output_dir, args.target_property, args.replacement_property)

    print(f"design           : {r['design_shape'][0]} x {r['design_shape'][1]}  (= [M | 1])")
    print(f"properties       : {r['expected']['n_properties']} in {r['expected']['n_categories']} categories")
    print(f"rank             : {r['rank']}  (theory {r['expected']['expected_rank']})  -> {'OK' if r['rank_matches_theory'] else 'MISMATCH'}")
    print(f"null dim         : {r['null_dim']}  (theory {r['expected']['expected_null_dim']})  -> {'OK' if r['null_dim_matches_theory'] else 'MISMATCH'}")
    print(
        "structural basis : max principal angle vs SVD basis = "
        f"{r['structural_vs_svd_null_max_principal_angle_deg']:.3e} deg"
    )
    print(f"\ntarget '{r['target_property']}' -> replacement '{r['replacement_property']}'")
    for name, d in r["edit_decomposition"].items():
        flag = "IDENTIFIED" if d["is_identified"] else "ARBITRARY"
        print(
            f"  {name:20s} |delta|={d['norm']:.4f}  identified={d['identified_norm']:.4f}  "
            f"arbitrary={d['arbitrary_norm']:.4f}  ({d['arbitrary_fraction']:.1%})  {flag}"
        )

    print("\ndeletion ambiguity by category (closed form verified against numeric projection):")
    pc = r["per_category_deletion_ambiguity"]
    for k in sorted(pc, key=lambda k: -pc[k]["deletion_arbitrary_fraction"]):
        v = pc[k]
        print(
            f"  {k:12s} m={v['n_values']:2d}  "
            f"arbitrary norm = {v['deletion_arbitrary_fraction']:.4f}  "
            f"energy = {v['deletion_arbitrary_energy_fraction']:.1%}"
        )
    print("\nE3 prediction:", r["e3_prediction"]["statement"])


if __name__ == "__main__":
    main()
