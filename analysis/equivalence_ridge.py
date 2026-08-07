"""E1 — weight-level SSAE-L1 <-> ridge equivalence (AUG-03).

Why this is a *weight*-level test
---------------------------------
For the tied-code decoder (``model_avg_feature``) at one layer, the model is provably an
additive linear function of the property indicators. The forward pass is::

    embds = Y(mask_arange)            # (B, p, n_repeat), row 0 is a zero padding row
    embds = LeakyReLU(0.2)(embds)     # elementwise
    x_hat = Linear(embds.flatten(1))  # (B, d)

An inactive property indexes the embedding's ``padding_idx=0`` row. *When that row is
zero* — the intended case — ``LeakyReLU(0) = 0``, an inactive block contributes nothing,
and::

    x_hat_i = b + sum_{k active in i} W_k sigma(y_k)   with   v_k := W_k sigma(y_k)

.. warning::
   **The padding row is not always zero.** ``nn.Embedding(padding_idx=...)`` zeroes that
   row at init and is supposed to zero its gradient, but the MPS backend does not honour
   the gradient masking, so any checkpoint trained on Apple silicon has a *trained*
   padding row. See ``CONTEXT/research-notes/`` (finding F4).

   The model stays additive either way. Writing ``y_0`` for the padding row::

       x_hat_i = sum_k a_k W_k sigma(y_k) + sum_k (1 - a_k) W_k sigma(y_0) + b
               = sum_k a_k [W_k (sigma(y_k) - sigma(y_0))] + [b + sum_k W_k sigma(y_0)]

   so the correct effective quantities are::

       v_k = W_k (sigma(y_k) - sigma(y_0))        b_eff = b + sum_k W_k sigma(y_0)

   which reduce to the intended formulas when ``y_0 = 0``. This script always uses the
   general form and records whether the padding row was zero.

``v_k`` is a fixed vector in ``R^d`` per property. SSAE-L1 is therefore *the same
hypothesis class* as ridge on 26 indicators plus intercept — not merely similar to it.
Any observed difference between the two must come from the fitting procedure (Adam +
implicit regularisation vs closed-form ridge at some lambda) or from the unidentified
null-space offset, and this script separates those two causes.

That distinction is the quantitative core of C2/C3: E3 measures how much SSAE's erasure
diverges from ridge's erasure, and the null-space offset measured here is what should
predict that divergence.

What is compared
----------------
Only *identified* quantities are compared directly (see ``analysis/nullspace.py``):

* within-category contrasts ``v_k - v_k'`` — per-contrast cosine and norm ratio;
* the effective ``lambda*`` minimising the total squared contrast discrepancy;
* the function-level agreement of ``x_hat`` on holdout.

The *unidentified* part is reported separately as the null-space offset
``||P_null(V_ssae - V_ridge)||_F / ||V_ssae||_F``. A large value there with small contrast
discrepancy is the expected outcome, and it is the honest way to say "these are the same
model, fitted to different arbitrary gauges".

Usage
-----
    python -m analysis.equivalence_ridge \\
        --checkpoint results/topk_sweep/topk_100000_L1 \\
        --train_folder results/compositional_split/train \\
        --holdout_folder results/compositional_split/holdout \\
        --output_dir results/analysis/e1_equivalence

Runs on CPU. Peak memory is dominated by the training matrix (n_prompts x top-k).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.io import h5_dataset_for_folder, load_decoder_checkpoint
from trainings.utils.run_manifest import checkpoint_fingerprint, write_run_manifest

DEFAULT_LAMBDAS = (
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 1e4,
)


# ------------------------------------------------------------------ SSAE extraction


def extract_concept_vectors(decoder, tp) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(V, b)`` with ``V`` of shape ``(p, d)`` and ``b`` of shape ``(d,)``.

    Only valid for a one-layer tied-code decoder; anything else is rejected loudly rather
    than silently producing a meaningless linearisation.
    """
    model_name = tp["model_name"]
    num_layers = int(tp.get("num_layers", 1) or 1)

    if model_name != "model_avg_feature":
        raise SystemExit(
            f"E1 is defined for the tied-code decoder 'model_avg_feature'; got "
            f"'{model_name}'. For untied codes the per-prompt Y rows are not a single "
            f"concept vector and the equivalence statement does not apply."
        )
    if num_layers != 1:
        raise SystemExit(
            f"E1 requires num_layers=1 (SSAE-L1); got {num_layers}. A deeper head is not "
            f"additive in the indicators, so no exact v_k exists."
        )

    linear = decoder.linear
    if not isinstance(linear, torch.nn.Linear):
        raise SystemExit(f"expected a bare nn.Linear head, got {type(linear).__name__}")

    n_properties = int(tp["n_properties"])
    n_repeat = int(tp["n_repeat"])

    W = linear.weight.detach().cpu().to(torch.float64)  # (d, p*n_repeat)
    b = linear.bias.detach().cpu().to(torch.float64)  # (d,)

    if W.shape[1] != n_properties * n_repeat:
        raise SystemExit(
            f"head input {W.shape[1]} != n_properties*n_repeat "
            f"({n_properties}*{n_repeat}); block layout assumption is wrong"
        )

    # Embedding row k+1 holds property k's code; row 0 is the padding row used by every
    # inactive property.
    Y = decoder.Y.weight.detach().cpu().to(torch.float64)  # (p+1, n_repeat)
    pad_is_zero = bool(torch.allclose(Y[0], torch.zeros_like(Y[0]), atol=1e-12))

    sig = lambda t: torch.nn.functional.leaky_relu(t, negative_slope=0.2)
    code_pad = sig(Y[0])  # (n_repeat,)
    codes = sig(Y[1:])  # (p, n_repeat)

    # General form: an inactive block contributes W_k sigma(y_0), not zero. Subtracting
    # that baseline from every block and folding the total into the intercept recovers an
    # exactly equivalent additive model. Identical to the naive formula when y_0 = 0.
    V = torch.empty((n_properties, W.shape[0]), dtype=torch.float64)
    b_eff = b.clone()
    for k in range(n_properties):
        Wk = W[:, k * n_repeat : (k + 1) * n_repeat]  # (d, n_repeat)
        pad_contrib = Wk @ code_pad
        V[k] = Wk @ codes[k] - pad_contrib
        b_eff = b_eff + pad_contrib

    return V, b_eff, pad_is_zero


def verify_linearisation(decoder, tp, V, b, mask_reduced, n_check=8) -> float:
    """Sanity check: does ``b + sum_k a_k v_k`` reproduce the decoder's own forward pass?

    If this is not ~machine precision, the block-layout assumption above is wrong and every
    number downstream is meaningless.
    """
    device = next(decoder.parameters()).device
    n_properties = int(tp["n_properties"])
    n_repeat = int(tp["n_repeat"])

    rng = np.random.default_rng(0)
    idx = rng.choice(len(mask_reduced), size=min(n_check, len(mask_reduced)), replace=False)

    max_rel = 0.0
    with torch.no_grad():
        for i in idx:
            a = mask_reduced[i].to(torch.float64)
            arange = (torch.arange(1, n_properties + 1) * mask_reduced[i].cpu().long()).to(device)
            emb = decoder.Y(arange.unsqueeze(0))
            emb = decoder.activation(emb).reshape(1, n_properties * n_repeat)
            direct = decoder.linear(emb).squeeze(0).detach().cpu().to(torch.float64)

            linearised = b + (a.cpu().unsqueeze(0) @ V).squeeze(0)
            denom = direct.norm().item() or 1.0
            max_rel = max(max_rel, ((direct - linearised).norm() / denom).item())
    return max_rel


# --------------------------------------------------------------------------- ridge


def fit_ridge_np(M: np.ndarray, X: np.ndarray, lam: float) -> np.ndarray:
    """Ridge with intercept, matching ``baselines/run_baselines.py::fit_ridge``.

    Returns ``(p+1, d)``: rows 0..p-1 are ``beta_k``, row ``p`` is the intercept.
    """
    n = M.shape[0]
    A = np.concatenate([M, np.ones((n, 1))], axis=1)
    m = A.shape[1]
    G = A.T @ A + lam * np.eye(m)
    return np.linalg.solve(G, A.T @ X)


# ---------------------------------------------------------------------- comparisons


def within_category_contrasts(prop_cat: list[int]) -> list[tuple[int, int]]:
    """All ordered-by-index pairs inside each category — the identified contrasts."""
    pairs = []
    for c in sorted(set(prop_cat)):
        members = [k for k, cc in enumerate(prop_cat) if cc == c]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.append((members[i], members[j]))
    return pairs


def contrast_matrix(V: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([V[i] - V[j] for i, j in pairs], axis=0)


def cosine_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    num = (A * B).sum(axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return num / np.maximum(den, 1e-30)


def null_projector(prop_cat: list[int], n_categories: int) -> np.ndarray:
    """Orthogonal projector onto null(A) in ``(p+1)``-dim coefficient space."""
    from analysis.nullspace import structural_null_basis

    N = structural_null_basis(prop_cat, n_categories)
    Q, _ = np.linalg.qr(N)
    return Q @ Q.T


# ----------------------------------------------------------------------------- main


def run(
    checkpoint: Path,
    train_folder: Path,
    holdout_folder: Path,
    output_dir: Path,
    lambdas: tuple[float, ...],
    device: str,
) -> dict:
    decoder, tp, train_ds = load_decoder_checkpoint(checkpoint, device=device)
    decoder.eval()

    V_ssae_t, b_ssae_t, pad_is_zero = extract_concept_vectors(decoder, tp)
    mask_reduced = train_ds.mask_reduced
    if not pad_is_zero:
        print(
            "WARNING: this checkpoint's Y padding row is non-zero (likely trained on the "
            "MPS backend, which does not honour nn.Embedding padding_idx gradient "
            "masking). Using the general v_k = W_k(sigma(y_k) - sigma(y_0)) form."
        )

    lin_err = verify_linearisation(decoder, tp, V_ssae_t, b_ssae_t, mask_reduced)
    if lin_err > 1e-5:
        raise SystemExit(
            f"linearisation check failed (max relative deviation {lin_err:.3e}); the "
            f"v_k = W_k sigma(y_k) decomposition does not reproduce the decoder's forward "
            f"pass, so E1's premise is broken for this checkpoint"
        )

    V_ssae = V_ssae_t.numpy()
    b_ssae = b_ssae_t.numpy()
    V_ssae_aug = np.concatenate([V_ssae, b_ssae[None, :]], axis=0)  # (p+1, d)

    props = train_ds.properties
    n_properties = int(tp["n_properties"])
    prop_cat = [int(props.pid_to_cid[k]) for k in range(n_properties)]
    n_categories = len(set(prop_cat))
    prop_names = [props.pid_to_property[k] for k in range(n_properties)]

    # Training design + targets, in the same normalised truncated space the SSAE fits.
    X_list, M_list = [], []
    for i in range(len(train_ds)):
        x, m = train_ds[i]
        X_list.append(x.numpy())
        M_list.append(m.numpy())
    X = np.stack(X_list).astype(np.float64)
    M = np.stack(M_list).astype(np.float64)
    del X_list, M_list

    pairs = within_category_contrasts(prop_cat)
    D_ssae = contrast_matrix(V_ssae, pairs)

    # --- lambda sweep on the identified contrasts only
    sweep = []
    best = None
    for lam in lambdas:
        Wr = fit_ridge_np(M, X, lam)
        D_r = contrast_matrix(Wr[:n_properties], pairs)
        disc = float(((D_ssae - D_r) ** 2).sum())
        rel = float(np.sqrt(disc) / max(np.linalg.norm(D_ssae), 1e-30))
        rec = {
            "lambda": float(lam),
            "contrast_sq_discrepancy": disc,
            "relative_contrast_discrepancy": rel,
            "mean_contrast_cosine": float(cosine_rows(D_ssae, D_r).mean()),
        }
        sweep.append(rec)
        if best is None or disc < best[0]:
            best = (disc, lam, Wr)

    _, lam_star, W_star = best
    V_ridge = W_star[:n_properties]
    b_ridge = W_star[n_properties]

    # --- per-contrast diagnostics at lambda*
    D_star = contrast_matrix(V_ridge, pairs)
    cos = cosine_rows(D_ssae, D_star)
    nr = np.linalg.norm(D_ssae, axis=1) / np.maximum(np.linalg.norm(D_star, axis=1), 1e-30)
    per_contrast = [
        {
            "pair": [prop_names[i], prop_names[j]],
            "category": int(prop_cat[i]),
            "cosine": float(c),
            "norm_ratio_ssae_over_ridge": float(r),
        }
        for (i, j), c, r in zip(pairs, cos, nr)
    ]

    # --- intercept
    b_cos = float(
        b_ssae @ b_ridge / max(np.linalg.norm(b_ssae) * np.linalg.norm(b_ridge), 1e-30)
    )

    # --- null-space offset: the size of the erasure ambiguity (the C2 number)
    P = null_projector(prop_cat, n_categories)
    Delta = V_ssae_aug - W_star  # (p+1, d)
    proj = P @ Delta
    resid = Delta - proj
    fro = lambda A: float(np.linalg.norm(A))
    null_offset = {
        "delta_frobenius": fro(Delta),
        "null_component_frobenius": fro(proj),
        "identified_component_frobenius": fro(resid),
        "null_offset_over_V_norm": fro(proj) / max(fro(V_ssae_aug), 1e-30),
        "identified_offset_over_V_norm": fro(resid) / max(fro(V_ssae_aug), 1e-30),
        "null_share_of_delta_energy": (fro(proj) ** 2) / max(fro(Delta) ** 2, 1e-30),
    }

    # --- function-level agreement on holdout
    holdout_ds = h5_dataset_for_folder(checkpoint, holdout_folder)
    Xh, Mh = [], []
    for i in range(len(holdout_ds)):
        x, m = holdout_ds[i]
        Xh.append(x.numpy())
        Mh.append(m.numpy())
    Xh = np.stack(Xh).astype(np.float64)
    Mh = np.stack(Mh).astype(np.float64)

    Ah = np.concatenate([Mh, np.ones((Mh.shape[0], 1))], axis=1)
    pred_ssae = Ah @ V_ssae_aug
    pred_ridge = Ah @ W_star

    diff = pred_ssae - pred_ridge
    row_rel = np.linalg.norm(diff, axis=1) / np.maximum(np.linalg.norm(pred_ssae, axis=1), 1e-30)
    var = Xh.var(axis=0).sum()
    fvu = lambda P_: float(((Xh - P_) ** 2).sum(axis=1).mean() / max(var, 1e-30))

    function_level = {
        "n_holdout": int(Xh.shape[0]),
        "max_relative_deviation": float(row_rel.max()),
        "mean_relative_deviation": float(row_rel.mean()),
        "mse_ssae": float(((Xh - pred_ssae) ** 2).mean()),
        "mse_ridge_at_lambda_star": float(((Xh - pred_ridge) ** 2).mean()),
        "fvu_ssae": fvu(pred_ssae),
        "fvu_ridge_at_lambda_star": fvu(pred_ridge),
        "paired_win_rate_ssae_better": float(
            (((Xh - pred_ssae) ** 2).mean(axis=1) < ((Xh - pred_ridge) ** 2).mean(axis=1)).mean()
        ),
    }

    results = {
        "analysis": "E1_equivalence_ridge",
        "checkpoint": str(checkpoint),
        "train_folder": str(train_folder),
        "holdout_folder": str(holdout_folder),
        "model_name": tp["model_name"],
        "num_layers": int(tp.get("num_layers", 1) or 1),
        "n_repeat": int(tp["n_repeat"]),
        "n_properties": n_properties,
        "n_categories": n_categories,
        "dim_output": int(V_ssae.shape[1]),
        "n_train": int(X.shape[0]),
        "padding_row_is_zero": pad_is_zero,
        "padding_row_note": (
            "zero padding row: v_k = W_k sigma(y_k) as intended"
            if pad_is_zero
            else "NON-ZERO padding row (MPS gradient-masking bug); used the general form "
            "v_k = W_k(sigma(y_k) - sigma(y_0)), b_eff = b + sum_k W_k sigma(y_0)"
        ),
        "linearisation_max_relative_deviation": lin_err,
        "lambda_grid": list(map(float, lambdas)),
        "lambda_sweep": sweep,
        "lambda_star": float(lam_star),
        "lambda_star_at_grid_edge": bool(
            lam_star == min(lambdas) or lam_star == max(lambdas)
        ),
        "contrast_summary": {
            "n_contrasts": len(pairs),
            "mean_cosine": float(cos.mean()),
            "min_cosine": float(cos.min()),
            "mean_norm_ratio": float(nr.mean()),
            "relative_discrepancy": float(
                np.linalg.norm(D_ssae - D_star) / max(np.linalg.norm(D_ssae), 1e-30)
            ),
        },
        "per_contrast": per_contrast,
        "intercept_cosine": b_cos,
        "null_space_offset": null_offset,
        "function_level": function_level,
        "reading": (
            "High contrast cosines with a large null_offset_over_V_norm is the predicted "
            "outcome: SSAE-L1 and ridge are the same function on representable prompts but "
            "sit at different points of the 7-dim unidentified gauge. That offset is what "
            "E3's between-model erasure divergence should track, and it should NOT appear "
            "in replacement edits."
        ),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "e1_equivalence.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(
        output_dir / "e1_solutions.npz",
        V_ssae_aug=V_ssae_aug.astype(np.float32),
        W_ridge_star=W_star.astype(np.float32),
        lambda_star=np.array(lam_star),
    )
    _plot(results, output_dir / "e1_equivalence.png")

    write_run_manifest(
        output_dir,
        run_kind="analysis",
        config={
            "analysis": "E1_equivalence_ridge",
            "checkpoint": str(checkpoint),
            "lambda_grid": list(map(float, lambdas)),
        },
        seed=tp.get("seed"),
        dataset=train_ds,
        extra_datasets={"holdout": holdout_ds},
        model_fingerprint=checkpoint_fingerprint(checkpoint),
        extra={
            "lambda_star": float(lam_star),
            "contrast_mean_cosine": results["contrast_summary"]["mean_cosine"],
            "null_offset_over_V_norm": null_offset["null_offset_over_V_norm"],
        },
    )
    return results


def _plot(r: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4.2))

    lams = [s["lambda"] for s in r["lambda_sweep"]]
    disc = [s["relative_contrast_discrepancy"] for s in r["lambda_sweep"]]
    ax0.loglog(lams, disc, "o-", color="#2563eb")
    ax0.axvline(r["lambda_star"], color="#b45309", ls="--")
    ax0.text(r["lambda_star"], max(disc), f"  λ*={r['lambda_star']:g}", color="#b45309", fontsize=9)
    ax0.set_xlabel("ridge λ")
    ax0.set_ylabel("relative contrast discrepancy")
    ax0.set_title("Which ridge does SSAE-L1 match?", fontsize=10)
    ax0.grid(alpha=0.3, which="both")

    cosines = [c["cosine"] for c in r["per_contrast"]]
    ax1.hist(cosines, bins=min(20, max(5, len(cosines) // 2)), color="#2563eb")
    ax1.axvline(1.0, color="#374151", ls="--", lw=1)
    ax1.set_xlabel("cosine(Δv SSAE, Δv ridge(λ*))")
    ax1.set_ylabel("within-category contrasts")
    ax1.set_title(
        f"Identified contrasts agree\nmean {r['contrast_summary']['mean_cosine']:.4f}, "
        f"min {r['contrast_summary']['min_cosine']:.4f}",
        fontsize=10,
    )
    ax1.grid(alpha=0.3)

    ns = r["null_space_offset"]
    parts = [ns["identified_component_frobenius"] ** 2, ns["null_component_frobenius"] ** 2]
    ax2.bar([0], [parts[0]], 0.6, color="#2563eb", label="identified")
    ax2.bar([0], [parts[1]], 0.6, bottom=[parts[0]], color="#b45309", label="arbitrary (null)")
    ax2.set_xticks([0])
    ax2.set_xticklabels([r"$\|V_{\rm SSAE}-V_{\rm ridge}\|_F^2$"])
    ax2.set_title(
        "Where the two solutions differ\n"
        f"null share {ns['null_share_of_delta_energy']:.1%}, "
        f"offset/‖V‖ {ns['null_offset_over_V_norm']:.3g}",
        fontsize=10,
    )
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--train_folder", type=Path, required=True)
    ap.add_argument("--holdout_folder", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, default=Path("results/analysis/e1_equivalence"))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--lambdas",
        type=str,
        default=",".join(f"{v:g}" for v in DEFAULT_LAMBDAS),
        help="Comma-separated ridge λ grid.",
    )
    args = ap.parse_args()

    lambdas = tuple(float(v) for v in args.lambdas.split(",") if v.strip())
    r = run(
        args.checkpoint,
        args.train_folder,
        args.holdout_folder,
        args.output_dir,
        lambdas,
        args.device,
    )

    print(f"linearisation check   : max rel dev {r['linearisation_max_relative_deviation']:.3e}")
    print(f"lambda*               : {r['lambda_star']:g}"
          + ("  [AT GRID EDGE - widen --lambdas]" if r["lambda_star_at_grid_edge"] else ""))
    cs = r["contrast_summary"]
    print(f"contrasts (n={cs['n_contrasts']})     : mean cos {cs['mean_cosine']:.6f}, "
          f"min {cs['min_cosine']:.6f}, rel discrepancy {cs['relative_discrepancy']:.4e}")
    print(f"intercept cosine      : {r['intercept_cosine']:.6f}")
    ns = r["null_space_offset"]
    print(f"null-space offset     : ‖P_null Δ‖/‖V‖ = {ns['null_offset_over_V_norm']:.4e}  "
          f"({ns['null_share_of_delta_energy']:.1%} of the difference energy)")
    fl = r["function_level"]
    print(f"holdout x_hat agreement: max rel dev {fl['max_relative_deviation']:.4e}, "
          f"mean {fl['mean_relative_deviation']:.4e}")
    print(f"holdout FVU           : SSAE {fl['fvu_ssae']:.6f} vs ridge(λ*) "
          f"{fl['fvu_ridge_at_lambda_star']:.6f}")


if __name__ == "__main__":
    main()
