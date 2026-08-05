"""Training-set reconstruction metrics for the SSAE and matched linear baselines.

The SSAE reconstructs each training prompt from its own trained ``Y`` row
(``decoder(batch_size=1, batch_idx=idx)``); ridge and mean-arithmetic reconstruct from
the same binary attribute mask fed to the holdout benchmark. Comparing all three on the
same training rows gives the "train-set gap" that the compositional holdout benchmark
cannot: if the SSAE beats ridge here by the same margin as on the holdout, the current
composition method (per-property block means) is not discarding interaction structure;
if the gap is much larger on train, it is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from baselines.run_baselines import (
    fit_mean_arithmetic,
    fit_ridge,
    predict_linear,
    predict_mean_arithmetic,
)
from evaluation.io import load_decoder_checkpoint


def _stack_dataset(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ms = [], []
    for i in range(len(dataset)):
        x, m = dataset[i]
        xs.append(x)
        ms.append(m.float())
    return torch.stack(xs, dim=0), torch.stack(ms, dim=0)


def _pair_mse_cos(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    mse = torch.nn.functional.mse_loss(pred, target, reduction="mean").item()
    cos = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()
    return float(mse), float(cos)


@torch.no_grad()
def reconstruction_metrics(
    checkpoint_dir: Path | str,
    *,
    device: str | None = None,
    compare_baselines: bool = True,
    ridge_lambda: float = 1e-2,
) -> dict:
    """
    Per-prompt training-set reconstruction. When ``compare_baselines`` is set (default),
    also fits ridge (λ = ``ridge_lambda``) and mean-arithmetic on the same dataset
    and reports their per-row MSE/cosine.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    decoder, _, dataset = load_decoder_checkpoint(checkpoint_dir, device=device)

    decoder.eval()
    decoder = decoder.to(dev)

    ssae_mse: list[float] = []
    ssae_cos: list[float] = []
    for idx in range(len(dataset)):
        target, _ = dataset[idx]
        target = target.unsqueeze(0).to(dev).float()
        pred = decoder(batch_size=1, batch_idx=idx)
        m, c = _pair_mse_cos(pred, target)
        ssae_mse.append(m)
        ssae_cos.append(c)

    out: dict = {
        "n_prompts": len(dataset),
        "ridge_lambda": ridge_lambda if compare_baselines else None,
        "ssae_true_Y": {
            "mse_mean": float(sum(ssae_mse) / len(ssae_mse)),
            "cosine_mean": float(sum(ssae_cos) / len(ssae_cos)),
            "per_index_mse": ssae_mse,
            "per_index_cosine": ssae_cos,
        },
    }

    if compare_baselines:
        X, M = _stack_dataset(dataset)
        X_dev = X.to(dev).float()
        M_dev = M.to(dev).float()
        mu, deltas = fit_mean_arithmetic(X_dev, M_dev)
        W = fit_ridge(M_dev, X_dev, ridge_lambda)

        pred_ma = predict_mean_arithmetic(mu, deltas, M_dev)
        pred_ridge = predict_linear(M_dev, W)

        ma_mse: list[float] = []
        ma_cos: list[float] = []
        rg_mse: list[float] = []
        rg_cos: list[float] = []
        for i in range(len(dataset)):
            tgt = X_dev[i:i+1]
            m1, c1 = _pair_mse_cos(pred_ma[i:i+1], tgt)
            m2, c2 = _pair_mse_cos(pred_ridge[i:i+1], tgt)
            ma_mse.append(m1); ma_cos.append(c1)
            rg_mse.append(m2); rg_cos.append(c2)

        out["mean_arithmetic"] = {
            "mse_mean": float(sum(ma_mse) / len(ma_mse)),
            "cosine_mean": float(sum(ma_cos) / len(ma_cos)),
            "per_index_mse": ma_mse,
            "per_index_cosine": ma_cos,
        }
        out["ridge_embed"] = {
            "mse_mean": float(sum(rg_mse) / len(rg_mse)),
            "cosine_mean": float(sum(rg_cos) / len(rg_cos)),
            "per_index_mse": rg_mse,
            "per_index_cosine": rg_cos,
        }
        # Paired win rates (SSAE < baseline per row).
        n = len(dataset)
        out["paired_ssae_lt_ridge"] = {
            "n": n,
            "wins": sum(1 for a, b in zip(ssae_mse, rg_mse) if a < b),
        }
        out["paired_ssae_lt_mean"] = {
            "n": n,
            "wins": sum(1 for a, b in zip(ssae_mse, ma_mse) if a < b),
        }

    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="MSE/cosine reconstruction for each training prompt index. "
                    "SSAE uses the trained Y row; ridge and mean-arithmetic are fitted "
                    "on the same dataset and evaluated on the same rows for comparison."
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_json", type=Path, default=None)
    p.add_argument("--ridge_lambda", type=float, default=1e-2)
    p.add_argument("--skip_baselines", action="store_true",
                   help="Report only the SSAE (original behaviour).")
    args = p.parse_args()

    out = reconstruction_metrics(
        args.checkpoint,
        device=args.device,
        compare_baselines=not args.skip_baselines,
        ridge_lambda=args.ridge_lambda,
    )

    # Print a compact summary; full arrays go to JSON.
    ssae = out["ssae_true_Y"]
    print(f"SSAE (true Y):    MSE={ssae['mse_mean']:.6f}  cos={ssae['cosine_mean']:.6f}  n={out['n_prompts']}")
    if "ridge_embed" in out:
        r = out["ridge_embed"]; m = out["mean_arithmetic"]
        print(f"Ridge (λ={out['ridge_lambda']}): MSE={r['mse_mean']:.6f}  cos={r['cosine_mean']:.6f}")
        print(f"Mean-arithmetic:  MSE={m['mse_mean']:.6f}  cos={m['cosine_mean']:.6f}")
        pw_r = out["paired_ssae_lt_ridge"]; pw_m = out["paired_ssae_lt_mean"]
        print(f"Paired: SSAE<Ridge = {pw_r['wins']}/{pw_r['n']}  SSAE<Mean = {pw_m['wins']}/{pw_m['n']}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
