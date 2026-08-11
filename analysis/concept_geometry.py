"""E5 — anatomy of the learned concept subspaces, measured from the weights.

Motivation
----------
E7's capacity axis has now been derived three times and got a different answer each time:
parameter count (proposal), 26 concept directions + intercept (F6), and the reachable affine
hull of 19 (F9, which the h=19 run then vindicated). Every one of those was an argument about
what the model *can* express, not a measurement of what it *does*.

This script measures it. For the tied-code decoder the block contribution is
``v_k = W_k σ(y_k)``, rank 1 by construction at any depth, so two things are checkable
directly from a checkpoint:

* **Used fraction** ``‖W_k σ̂_k‖² / ‖W_k‖²_F`` with ``σ̂_k`` the unit-normalised code. The
  prediction is ≈ 1/``n_repeat``: only one direction out of ``n_repeat`` columns is ever
  exercised, so the remaining columns are unconstrained. At ``n_repeat = 10`` that means ~90%
  of the head's parameters are structurally dead, which is exactly why L1's 26.1M parameters
  behave like 2.0M.
* **Concept-vector rank.** The 26 ``v_k`` live in ``R^d``, but only differences of valid
  one-hot patterns are ever evaluated, so the *identifiable* span is ``p − C = 19``. Reporting
  both the raw rank of ``{v_k}`` and the rank of the within-category contrast set settles the
  hull argument empirically.

Also computes the decorrelation diagnostic the earlier draft leaned on: the Gram matrix of
``{v_k}`` and mean |cos| within- vs cross-category, testing the unconstrained-feature-model
prediction ``⟨y_k1, y_k2⟩ ≈ 0``.

Usage
-----
    python -m analysis.concept_geometry \\
        --checkpoints results/topk_sweep/topk_100000_L1 results/topk_sweep/topk_100000_L2_h19 \\
        --output_dir results/analysis/e5_geometry

CPU only, seconds per checkpoint: it reads weights, never embeddings or images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.io import load_decoder_checkpoint
from trainings.utils.run_manifest import checkpoint_fingerprint, write_run_manifest


def leaky(t: torch.Tensor, slope: float = 0.2) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(t, negative_slope=slope)


def participation_ratio(s: np.ndarray) -> float:
    """Effective rank as the participation ratio of the singular values.

    ``(Σ s²)² / Σ s⁴``. Equals 1 for a rank-1 spectrum and n for n equal singular values;
    less brittle than counting values above an arbitrary threshold.
    """
    s2 = s.astype(np.float64) ** 2
    denom = float((s2**2).sum())
    return float((s2.sum() ** 2) / denom) if denom > 0 else 0.0


def analyse_checkpoint(ckpt: Path, device: str = "cpu") -> dict:
    decoder, tp, train_ds = load_decoder_checkpoint(ckpt, device=device)
    decoder.eval()

    model_name = tp["model_name"]
    if model_name != "model_avg_feature":
        raise SystemExit(f"E5 as written targets the tied-code decoder; got {model_name!r}")

    n_properties = int(tp["n_properties"])
    n_repeat = int(tp["n_repeat"])
    num_layers = int(tp.get("num_layers", 1) or 1)

    Y = decoder.Y.weight.detach().cpu().to(torch.float64)  # (p+1, n_repeat)
    codes = leaky(Y[1:])  # (p, n_repeat)
    pad_code = leaky(Y[0])

    props = train_ds.properties
    prop_cat = [int(props.pid_to_cid[k]) for k in range(n_properties)]
    names = [props.pid_to_property[k] for k in range(n_properties)]

    # --- first-layer weight block per property ------------------------------------
    first = decoder.linear if isinstance(decoder.linear, torch.nn.Linear) else decoder.linear[0]
    W1 = first.weight.detach().cpu().to(torch.float64)  # (out, p*n_repeat)

    used_fraction = []
    for k in range(n_properties):
        Wk = W1[:, k * n_repeat : (k + 1) * n_repeat]
        c = codes[k]
        cn = c / (c.norm() + 1e-30)
        num = float((Wk @ cn).norm() ** 2)
        den = float((Wk**2).sum())
        used_fraction.append(num / den if den > 0 else 0.0)

    result = {
        "checkpoint": str(ckpt),
        "model_name": model_name,
        "num_layers": num_layers,
        "n_repeat": n_repeat,
        "n_properties": n_properties,
        "n_categories": len(set(prop_cat)),
        "dim_output": int(tp["dim_output"]),
        "truncate_embds_topk": tp.get("truncate_embds_topk"),
        "padding_code_absmax": float(Y[0].abs().max()),
        "used_fraction_mean": float(np.mean(used_fraction)),
        "used_fraction_min": float(np.min(used_fraction)),
        "used_fraction_max": float(np.max(used_fraction)),
        "used_fraction_predicted": 1.0 / n_repeat,
        "used_fraction_per_property": dict(zip(names, map(float, used_fraction))),
    }

    # --- concept vectors, only meaningful for the additive one-layer model ---------
    if num_layers == 1:
        d = W1.shape[0]
        V = torch.empty((n_properties, d), dtype=torch.float64)
        for k in range(n_properties):
            Wk = W1[:, k * n_repeat : (k + 1) * n_repeat]
            V[k] = Wk @ codes[k] - Wk @ pad_code
        Vn = V.numpy()

        s_raw = np.linalg.svd(Vn, compute_uv=False)
        rank_raw = int(np.linalg.matrix_rank(Vn))

        # Within-category contrasts: the only directions the data can identify.
        contrasts = []
        for c in sorted(set(prop_cat)):
            members = [k for k, cc in enumerate(prop_cat) if cc == c]
            for j in members[1:]:
                contrasts.append(Vn[members[0]] - Vn[j])
        Cm = np.stack(contrasts)
        s_con = np.linalg.svd(Cm, compute_uv=False)

        # Gram / decorrelation
        Vu = Vn / (np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-30)
        G = Vu @ Vu.T
        within, cross = [], []
        for i in range(n_properties):
            for j in range(i + 1, n_properties):
                (within if prop_cat[i] == prop_cat[j] else cross).append(abs(float(G[i, j])))

        result.update({
            "concept_vectors": {
                "rank_raw": rank_raw,
                "effective_rank_participation_ratio": participation_ratio(s_raw),
                "singular_values_top10": [float(x) for x in s_raw[:10]],
                "contrast_set_rank": int(np.linalg.matrix_rank(Cm)),
                "contrast_effective_rank": participation_ratio(s_con),
                "identifiable_dim_theory": n_properties - len(set(prop_cat)),
                "rank_matches_identifiable_theory":
                    int(np.linalg.matrix_rank(Cm)) == n_properties - len(set(prop_cat)),
            },
            "decorrelation": {
                "mean_abs_cos_within_category": float(np.mean(within)) if within else float("nan"),
                "mean_abs_cos_cross_category": float(np.mean(cross)) if cross else float("nan"),
                "max_abs_cos_offdiag": float(
                    max(abs(G[i, j]) for i in range(n_properties) for j in range(n_properties) if i != j)
                ),
            },
        })
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    ap.add_argument("--output_dir", type=Path, default=Path("results/analysis/e5_geometry"))
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    out = []
    for c in args.checkpoints:
        if not (c / "model.pt").is_file():
            print(f"skip {c} (no model.pt)")
            continue
        r = analyse_checkpoint(c, args.device)
        out.append(r)
        cv = r.get("concept_vectors")
        print(f"\n=== {c.name}  (L{r['num_layers']}, n_repeat={r['n_repeat']}, k={r['truncate_embds_topk']})")
        print(f"  used fraction  mean {r['used_fraction_mean']:.4f}  "
              f"(predicted 1/n_repeat = {r['used_fraction_predicted']:.4f})  "
              f"range [{r['used_fraction_min']:.4f}, {r['used_fraction_max']:.4f}]")
        if cv:
            print(f"  concept vectors: raw rank {cv['rank_raw']}, "
                  f"effective rank {cv['effective_rank_participation_ratio']:.2f}")
            print(f"  contrast set:    rank {cv['contrast_set_rank']} "
                  f"(theory {cv['identifiable_dim_theory']}) -> "
                  f"{'MATCH' if cv['rank_matches_identifiable_theory'] else 'MISMATCH'}")
            dc = r["decorrelation"]
            print(f"  mean |cos| within-category {dc['mean_abs_cos_within_category']:.4f}  "
                  f"cross-category {dc['mean_abs_cos_cross_category']:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "e5_geometry.json").write_text(json.dumps(out, indent=2))
    write_run_manifest(
        args.output_dir,
        run_kind="analysis",
        config={"analysis": "E5_concept_geometry",
                "checkpoints": [str(c) for c in args.checkpoints]},
        model_fingerprint={str(c): checkpoint_fingerprint(c) for c in args.checkpoints
                           if (c / "model.pt").is_file()},
        extra={"n_analysed": len(out)},
    )
    print(f"\nwrote {args.output_dir / 'e5_geometry.json'}")


if __name__ == "__main__":
    main()
