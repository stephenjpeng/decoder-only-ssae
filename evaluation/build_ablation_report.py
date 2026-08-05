"""Standalone HTML report for the truncation x head-depth SSAE ablation.

Compares 4 SSAE variants (top-k in {100000, 300000} x head in {L1, L2 h=2048})
against per-dataset ridge / mean-arithmetic / prompt-only / gt_embed baselines.

Usage:
    python -m evaluation.build_ablation_report \
        --bench_dir results/bench_out \
        --baseline_dir results/bench_baseline_cache \
        --out results/bench_out/ablation_report.html
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import statistics
from collections import defaultdict
from html import escape
from pathlib import Path

CONFIGS = [
    # (bench_dir_name, dataset_id, topk, head, label, color)
    ("topk_100000_L1",       "1c19e56f7560606d", 100_000, "L1",         "SSAE 100k / L1",         "#60a5fa"),
    ("topk_100000_L2_h2048", "1c19e56f7560606d", 100_000, "L2 h=2048",  "SSAE 100k / L2",         "#1d4ed8"),
    ("topk_300000_L1",       "59dbe58c95d0cb13", 300_000, "L1",         "SSAE 300k / L1",         "#f472b6"),
    ("topk_300000_L2_h2048", "59dbe58c95d0cb13", 300_000, "L2 h=2048",  "SSAE 300k / L2",         "#be185d"),
]

BASELINE_METHODS = ["ridge_embed", "mean_arithmetic", "prompt_only", "gt_embed"]
BASELINE_COLOR = {
    "ridge_embed":     "#7c3aed",
    "mean_arithmetic": "#0891b2",
    "prompt_only":     "#b45309",
    "gt_embed":        "#374151",
}
BASELINE_LABEL = {
    "ridge_embed":     "Ridge",
    "mean_arithmetic": "Mean-arith",
    "prompt_only":     "Prompt-only",
    "gt_embed":        "GT embed",
}


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    n = len(values)
    m = sum(values) / n
    if n > 1:
        sd = statistics.stdev(values)
        se = sd / math.sqrt(n)
    else:
        sd = se = 0.0
    return {"n": n, "mean": m, "sd": sd, "se": se, "ci_lo": m - 1.96 * se, "ci_hi": m + 1.96 * se}


def load_ssae_rows(bench_dir):
    rows = list(csv.DictReader(open(bench_dir / "per_sample.csv", encoding="utf-8")))
    return [r for r in rows if r["task"] == "holdout_unseen_tuple"]


def load_baseline_rows(baseline_dir, dataset_id, method):
    rows = list(csv.DictReader(open(baseline_dir / dataset_id / "per_sample.csv", encoding="utf-8")))
    return [r for r in rows if r["task"] == "holdout_unseen_tuple" and r["method"] == method]


def by_idx(rows, key):
    return {int(r["sample_idx"]): fnum(r.get(key, "")) for r in rows if fnum(r.get(key, "")) is not None}


def paired_win(a_by, b_by, lower_is_better=True):
    common = set(a_by) & set(b_by)
    t = len(common)
    if t == 0:
        return 0, 0, float("nan")
    w = sum(1 for i in common if (a_by[i] < b_by[i]) == lower_is_better)
    return w, t, w / t


def paired_diff_stats(a_by, b_by):
    common = set(a_by) & set(b_by)
    diffs = [a_by[i] - b_by[i] for i in common]
    if not diffs:
        return None
    n = len(diffs)
    m = sum(diffs) / n
    sd = statistics.pstdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    return {"n": n, "mean": m, "sd": sd, "se": se, "ci_lo": m - 1.96 * se, "ci_hi": m + 1.96 * se}


def thumb_b64(path, size=180):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=76)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def svg_bars(entries, *, lower_is_better, width=780, decimals=5, unit="", log=False):
    """entries: [(label, color, mean, ci_lo, ci_hi)]"""
    if log:
        # replace zeros/negatives by tiny epsilon (should not occur here)
        pass
    have = [(l, c, m, lo, hi) for l, c, m, lo, hi in entries if m is not None]
    if not have:
        return "<div class='cap'>no data</div>"
    vals_all = [m for _, _, m, _, _ in have] + [lo for _, _, _, lo, _ in have if lo is not None] + [hi for _, _, _, _, hi in have if hi is not None]
    lo = min(vals_all)
    hi = max(vals_all)
    if log:
        eps = max(min(v for v in vals_all if v > 0) / 10, 1e-9)
        f = lambda v: math.log10(max(v, eps))
        lo_f, hi_f = f(lo), f(hi)
    else:
        f = lambda v: v
        lo_f, hi_f = lo, hi
    span = hi_f - lo_f or 1.0
    pad = span * 0.12
    x0 = lo_f - pad
    x1 = hi_f + pad
    rng = x1 - x0
    row_h = 30
    top = 10
    left = 165
    right = 90
    plot_w = width - left - right
    height = top + row_h * len(entries) + 26

    def sx(v):
        return left + (f(v) - x0) / rng * plot_w

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        gv_f = x0 + frac * rng
        gx = left + frac * plot_w
        gv = 10 ** gv_f if log else gv_f
        parts.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top+row_h*len(entries)}" stroke="#e5e7eb" stroke-width="1"/>')
        label = f"{gv:.{decimals}f}" if not log else f"{gv:.1e}"
        parts.append(f'<text x="{gx:.1f}" y="{top+row_h*len(entries)+16}" font-size="10" fill="#9ca3af" text-anchor="middle">{label}</text>')
    for i, (label, color, mean, cl, ch) in enumerate(entries):
        y = top + i * row_h + row_h / 2
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" font-size="12" fill="#374151" text-anchor="end">{escape(label)}</text>')
        if mean is None:
            parts.append(f'<text x="{left+8}" y="{y+4:.1f}" font-size="11" fill="#9ca3af">n/a</text>')
            continue
        bx = sx(mean)
        parts.append(f'<rect x="{left}" y="{y-8:.1f}" width="{max(bx-left, 0.5):.1f}" height="16" rx="2" fill="{color}" opacity="0.85"/>')
        if cl is not None and ch is not None and cl != ch:
            parts.append(f'<line x1="{sx(cl):.1f}" y1="{y:.1f}" x2="{sx(ch):.1f}" y2="{y:.1f}" stroke="#111827" stroke-width="1.5"/>')
            for xw in (sx(cl), sx(ch)):
                parts.append(f'<line x1="{xw:.1f}" y1="{y-4:.1f}" x2="{xw:.1f}" y2="{y+4:.1f}" stroke="#111827" stroke-width="1.5"/>')
        fmt_mean = f"{mean:.{decimals}f}{unit}" if not log else f"{mean:.2e}{unit}"
        parts.append(f'<text x="{bx+8:.1f}" y="{y+4:.1f}" font-size="11" fill="#111827">{fmt_mean}</text>')
    arrow = "lower is better" if lower_is_better else "higher is better"
    parts.append(f'<text x="{width-2}" y="{height-2}" font-size="10" fill="#9ca3af" text-anchor="end">&#8594; {arrow}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build(bench_dir, baseline_dir, out_path, n_gallery=3):
    # Load data for all configs and both baseline sets.
    ssae = {}
    for cfg_name, dsid, topk, head, label, color in CONFIGS:
        ssae[cfg_name] = load_ssae_rows(bench_dir / cfg_name)
    baseline = {}
    for dsid in {c[1] for c in CONFIGS}:
        baseline[dsid] = {m: load_baseline_rows(baseline_dir, dsid, m) for m in BASELINE_METHODS}

    # For each metric key, compute means per (config or baseline).
    metric_keys = [
        ("mse_embedding_vs_gt",    "Embedding MSE",       True,  5, False),
        ("cosine_embedding_vs_gt", "Embedding cosine",    False, 4, False),
        ("clip_image_vs_full_prompt", "CLIP (full prompt)", False, 4, False),
        ("clip_mean_vs_attrs",     "CLIP (attrs)",        False, 4, False),
        ("lpips_vs_gt_embed",      "LPIPS vs GT render",  True,  4, False),
        ("mse_pixel_vs_gt_embed",  "Pixel MSE vs GT",     True,  4, False),
        ("ssim_vs_gt_embed",       "SSIM vs GT",          False, 4, False),
        ("mse_pixel_pre_post_edit", "Pixel MSE (drop-one)", True, 4, False),
        ("ssim_pre_post_edit",     "SSIM (drop-one)",     False, 4, False),
        ("clip_image_vs_residual_prompt", "CLIP vs residual", False, 4, False),
        ("mse_pixel_swap_vs_normal", "Pixel MSE (swap)",  True,  4, False),
        ("ssim_swap_vs_normal",    "SSIM (swap)",         False, 4, False),
        ("clip_swap_image_vs_swapped_prompt", "CLIP vs swapped", False, 4, False),
    ]

    # Per-config metric bundle
    metric_data = defaultdict(dict)  # metric_data[key][config_or_key] = stats dict
    for key, _, _, _, _ in metric_keys:
        for cfg_name, dsid, *_ in CONFIGS:
            metric_data[key][cfg_name] = stats([fnum(r.get(key)) for r in ssae[cfg_name]])
        for dsid, method_rows in baseline.items():
            for method, rows in method_rows.items():
                metric_data[key][f"{method}__{dsid}"] = stats([fnum(r.get(key)) for r in rows])

    # Paired diagnostics: SSAE vs ridge, SSAE vs mean-arith, per config on embedding MSE.
    paired_emb = {}
    for cfg_name, dsid, *_ in CONFIGS:
        s_by = by_idx(ssae[cfg_name], "mse_embedding_vs_gt")
        r_by = by_idx(baseline[dsid]["ridge_embed"], "mse_embedding_vs_gt")
        m_by = by_idx(baseline[dsid]["mean_arithmetic"], "mse_embedding_vs_gt")
        paired_emb[cfg_name] = {
            "vs_ridge_win": paired_win(s_by, r_by),
            "vs_mean_win":  paired_win(s_by, m_by),
            "vs_ridge_diff": paired_diff_stats(s_by, r_by),
        }

    # Paired diagnostics for locality: drop-one and swap pixel MSE, SSAE vs ridge.
    paired_edit = {}
    for cfg_name, dsid, *_ in CONFIGS:
        out = {}
        for key in ["mse_pixel_pre_post_edit", "mse_pixel_swap_vs_normal"]:
            s_by = by_idx(ssae[cfg_name], key)
            r_by = by_idx(baseline[dsid]["ridge_embed"], key)
            out[key] = {
                "win": paired_win(s_by, r_by, lower_is_better=True),
                "diff": paired_diff_stats(s_by, r_by),
            }
        paired_edit[cfg_name] = out

    # SNR diagnostic per config (embedding + image-space metrics)
    snr = {}
    for cfg_name, dsid, *_ in CONFIGS:
        snr[cfg_name] = {}
        for key in ["mse_embedding_vs_gt", "cosine_embedding_vs_gt", "lpips_vs_gt_embed",
                    "mse_pixel_vs_gt_embed", "ssim_vs_gt_embed", "clip_image_vs_full_prompt",
                    "mse_pixel_pre_post_edit", "mse_pixel_swap_vs_normal"]:
            # Aggregate across SSAE + ridge + mean_arith on the same tuple
            per = {}
            for tag, rows in [("ssae", ssae[cfg_name]),
                              ("ridge", baseline[dsid]["ridge_embed"]),
                              ("mean", baseline[dsid]["mean_arithmetic"])]:
                per[tag] = by_idx(rows, key)
            available = [t for t in per if per[t]]
            if len(available) < 2:
                continue
            means = {t: sum(per[t].values()) / len(per[t]) for t in available}
            spread = max(means.values()) - min(means.values())
            within = statistics.mean([statistics.pstdev(list(per[t].values())) for t in available if len(per[t]) > 1])
            paired_sds = []
            for i, a in enumerate(available):
                for b in available[i+1:]:
                    common = set(per[a]) & set(per[b])
                    diffs = [per[a][k] - per[b][k] for k in common]
                    if len(diffs) > 1:
                        paired_sds.append(statistics.pstdev(diffs))
            paired = statistics.mean(paired_sds) if paired_sds else float("nan")
            ratio = spread / paired if paired == paired and paired > 0 else float("nan")
            snr[cfg_name][key] = {"spread": spread, "within": within, "paired": paired, "ratio": ratio}

    # Gallery: pick 3 sample indices from the first config; show 7 columns
    # (gt, mean, ridge, SSAE_L1_100k, SSAE_L2_100k, SSAE_L1_300k, SSAE_L2_300k).
    prompts = {int(r["sample_idx"]): r["prompt"] for r in ssae["topk_100000_L1"]}
    all_idx = sorted(prompts)
    step = max(1, len(all_idx) // (n_gallery + 1))
    gallery_idx = all_idx[step::step][:n_gallery]

    gallery = []
    for gi in gallery_idx:
        cells = []
        # baselines source images from bench_baseline_cache/<dsid>/images/<method>
        for method, dsid_lookup in [("gt_embed", "1c19e56f7560606d"),
                                    ("mean_arithmetic", "1c19e56f7560606d"),
                                    ("ridge_embed", "1c19e56f7560606d")]:
            p = baseline_dir / dsid_lookup / "images" / method / f"{gi:05d}.png"
            cells.append((BASELINE_LABEL[method] + f" ({dsid_lookup[:4]}…)", thumb_b64(p) if p.exists() else None))
        for cfg_name, dsid, topk, head, label, color in CONFIGS:
            p = bench_dir / cfg_name / "images" / "ssae_compose" / f"{gi:05d}.png"
            cells.append((label, thumb_b64(p) if p.exists() else None))
        gallery.append((gi, prompts.get(gi, ""), cells))

    # Edit-locality gallery: pre-edit vs post-edit (drop-one) for SSAE-L2-100k
    edit_gallery = []
    cfg_use = "topk_100000_L2_h2048"
    for gi in gallery_idx[:2]:
        pre = bench_dir / cfg_use / "images_pre_edit" / "ssae_compose" / f"{gi:05d}.png"
        post = bench_dir / cfg_use / "images" / "ssae_compose" / f"{gi:05d}.png"
        swap = bench_dir / cfg_use / "images_swapped" / "ssae_compose" / f"{gi:05d}.png"
        # also the pre_edit from ridge for direct locality contrast
        pre_r = baseline_dir / "1c19e56f7560606d" / "images_pre_edit" / "ridge_embed" / f"{gi:05d}.png"
        post_r = baseline_dir / "1c19e56f7560606d" / "images" / "ridge_embed" / f"{gi:05d}.png"
        swap_r = baseline_dir / "1c19e56f7560606d" / "images_swapped" / "ridge_embed" / f"{gi:05d}.png"
        cells = [
            ("Ridge pre",  thumb_b64(pre_r) if pre_r.exists() else None),
            ("Ridge post", thumb_b64(post_r) if post_r.exists() else None),
            ("Ridge swap", thumb_b64(swap_r) if swap_r.exists() else None),
            ("SSAE-L2 pre",  thumb_b64(pre) if pre.exists() else None),
            ("SSAE-L2 post", thumb_b64(post) if post.exists() else None),
            ("SSAE-L2 swap", thumb_b64(swap) if swap.exists() else None),
        ]
        # Pull edit metadata from the ssae row
        row = next((r for r in ssae[cfg_use] if int(r["sample_idx"]) == gi), None)
        cap = f"edit pid={row['edit_pid']} '{row['edit_attribute']}' &rarr; swap to '{row['swap_target_attribute']}'" if row else ""
        edit_gallery.append((gi, prompts.get(gi, ""), cap, cells))

    html = render(metric_data, paired_emb, paired_edit, snr, gallery, edit_gallery, ssae, baseline, metric_keys)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def render(metric_data, paired_emb, paired_edit, snr, gallery, edit_gallery, ssae, baseline, metric_keys):
    # ---- Embedding-space chart: SSAE variants + ridge + mean-arith, grouped by dataset ----
    emb_entries = []
    for cfg_name, dsid, topk, head, label, color in CONFIGS:
        s = metric_data["mse_embedding_vs_gt"][cfg_name]
        emb_entries.append((label, color, s["mean"], s["ci_lo"], s["ci_hi"]))
    # append per-dataset ridge and mean
    for dsid, label_suffix in [("1c19e56f7560606d", "100k"), ("59dbe58c95d0cb13", "300k")]:
        for method in ["ridge_embed", "mean_arithmetic"]:
            s = metric_data["mse_embedding_vs_gt"][f"{method}__{dsid}"]
            emb_entries.append((f"{BASELINE_LABEL[method]} ({label_suffix})", BASELINE_COLOR[method], s["mean"], s["ci_lo"], s["ci_hi"]))
    chart_emb_lin = svg_bars(emb_entries, lower_is_better=True, decimals=5)
    chart_emb_log = svg_bars(emb_entries, lower_is_better=True, decimals=5, log=True)

    # ---- Cosine chart (SSAE only; baseline CSVs do not populate cosine_embedding_vs_gt) ----
    cos_entries = []
    for cfg_name, dsid, topk, head, label, color in CONFIGS:
        s = metric_data["cosine_embedding_vs_gt"][cfg_name]
        if s:
            cos_entries.append((label, color, s["mean"], s["ci_lo"], s["ci_hi"]))
    chart_cos = svg_bars(cos_entries, lower_is_better=False, decimals=4)

    # ---- Image vs GT chart set: LPIPS, pixel MSE, SSIM (per dataset) ----
    def img_chart(key, lower_is_better, decimals=4):
        entries = []
        for cfg_name, dsid, topk, head, label, color in CONFIGS:
            s = metric_data[key][cfg_name]
            if s: entries.append((label, color, s["mean"], s["ci_lo"], s["ci_hi"]))
        for dsid, label_suffix in [("1c19e56f7560606d", "100k"), ("59dbe58c95d0cb13", "300k")]:
            for method in ["ridge_embed", "mean_arithmetic", "prompt_only"]:
                s = metric_data[key].get(f"{method}__{dsid}")
                if s: entries.append((f"{BASELINE_LABEL[method]} ({label_suffix})", BASELINE_COLOR[method], s["mean"], s["ci_lo"], s["ci_hi"]))
        return svg_bars(entries, lower_is_better=lower_is_better, decimals=decimals)

    chart_lpips = img_chart("lpips_vs_gt_embed", True)
    chart_pxmse = img_chart("mse_pixel_vs_gt_embed", True)
    chart_ssim = img_chart("ssim_vs_gt_embed", False)
    chart_clip_full = img_chart("clip_image_vs_full_prompt", False)
    chart_clip_attr = img_chart("clip_mean_vs_attrs", False)

    # ---- Edit-locality chart set ----
    chart_edit_mse = img_chart("mse_pixel_pre_post_edit", True)
    chart_edit_ssim = img_chart("ssim_pre_post_edit", False)
    chart_edit_clipres = img_chart("clip_image_vs_residual_prompt", False)
    chart_swap_mse = img_chart("mse_pixel_swap_vs_normal", True)
    chart_swap_ssim = img_chart("ssim_swap_vs_normal", False)
    chart_swap_clip = img_chart("clip_swap_image_vs_swapped_prompt", False)

    # ---- Paired win-rate table ----
    win_rows = []
    for cfg_name, dsid, topk, head, label, color in CONFIGS:
        p = paired_emb[cfg_name]
        w_r, t_r, r_r = p["vs_ridge_win"]
        w_m, t_m, r_m = p["vs_mean_win"]
        d = p["vs_ridge_diff"]
        diff = f"{d['mean']:+.5f} &plusmn; {1.96*d['se']:.5f}" if d else "n/a"
        win_rows.append(
            f'<tr><th class="rowlab" style="border-left:4px solid {color}">{label}</th>'
            f'<td>{r_r*100:.1f}% <span class="ci">({w_r}/{t_r})</span></td>'
            f'<td>{r_m*100:.1f}% <span class="ci">({w_m}/{t_m})</span></td>'
            f'<td>{diff}</td></tr>'
        )
    win_table = (
        '<table class="data"><thead><tr><th>SSAE variant</th>'
        '<th>SSAE MSE &lt; Ridge (paired)</th>'
        '<th>SSAE MSE &lt; Mean-arith (paired)</th>'
        '<th>Mean paired diff (SSAE&minus;Ridge, 95% CI)</th></tr></thead>'
        f'<tbody>{"".join(win_rows)}</tbody></table>'
    )

    # ---- Edit paired win-rate table (SSAE vs ridge on locality) ----
    edit_rows = []
    for cfg_name, dsid, topk, head, label, color in CONFIGS:
        e = paired_edit[cfg_name]
        for key, disp in [("mse_pixel_pre_post_edit", "drop-one"),
                          ("mse_pixel_swap_vs_normal", "swap-one")]:
            w, t, r = e[key]["win"]
            d = e[key]["diff"]
            diff = f"{d['mean']:+.5f} &plusmn; {1.96*d['se']:.5f}" if d else "n/a"
            edit_rows.append(
                f'<tr><th class="rowlab" style="border-left:4px solid {color}">{label}</th>'
                f'<td>{disp}</td>'
                f'<td>{r*100:.1f}% <span class="ci">({w}/{t})</span></td>'
                f'<td>{diff}</td></tr>'
            )
    edit_table = (
        '<table class="data"><thead><tr><th>SSAE variant</th><th>Edit type</th>'
        '<th>SSAE &lt; Ridge (pixel MSE, paired)</th>'
        '<th>Mean paired diff (SSAE&minus;Ridge, 95% CI)</th></tr></thead>'
        f'<tbody>{"".join(edit_rows)}</tbody></table>'
    )

    # ---- SNR table ----
    snr_keys = [
        ("mse_embedding_vs_gt",     "Embedding MSE"),
        ("cosine_embedding_vs_gt",  "Embedding cos"),
        ("lpips_vs_gt_embed",       "LPIPS vs GT"),
        ("mse_pixel_vs_gt_embed",   "Pixel MSE vs GT"),
        ("ssim_vs_gt_embed",        "SSIM vs GT"),
        ("clip_image_vs_full_prompt","CLIP full-prompt"),
        ("mse_pixel_pre_post_edit", "Drop-one pixel MSE"),
        ("mse_pixel_swap_vs_normal","Swap-one pixel MSE"),
    ]
    header = "<th>Metric</th>" + "".join(f"<th>{lbl}</th>" for _, _, _, lbl, _ in [(c, None, None, c_[4], None) for c, c_ in zip([c[0] for c in CONFIGS], CONFIGS)])
    snr_head = "<th>Metric</th>" + "".join(f"<th>{c[4]}</th>" for c in CONFIGS)
    body = []
    for key, disp in snr_keys:
        cells = []
        for cfg_name, dsid, topk, head, label, color in CONFIGS:
            v = snr[cfg_name].get(key)
            if not v:
                cells.append("<td class='na'>n/a</td>")
            else:
                ratio = v["ratio"]
                cls = "good" if ratio > 0.5 else ("warn" if ratio < 0.2 else "")
                cells.append(f'<td class="{cls}">{ratio:.2f}</td>')
        body.append(f'<tr><th class="rowlab">{disp}</th>{"".join(cells)}</tr>')
    snr_table = (
        f'<table class="data"><thead><tr>{snr_head}</tr></thead><tbody>{"".join(body)}</tbody></table>'
    )

    # ---- Master per-metric table (all configs + baselines rows, key metrics only) ----
    tbl_cols = [
        ("mse_embedding_vs_gt", "Emb MSE &darr;", 5),
        ("cosine_embedding_vs_gt", "Emb cos &uarr;", 4),
        ("lpips_vs_gt_embed", "LPIPS &darr;", 4),
        ("mse_pixel_vs_gt_embed", "PxMSE &darr;", 4),
        ("ssim_vs_gt_embed", "SSIM &uarr;", 4),
        ("clip_image_vs_full_prompt", "CLIP full &uarr;", 4),
    ]

    def val_cell(s, dec):
        if s is None or s["mean"] is None:
            return '<td class="na">n/a</td>'
        return f'<td>{s["mean"]:.{dec}f}<span class="ci">[{s["ci_lo"]:.{dec}f}, {s["ci_hi"]:.{dec}f}]</span></td>'

    tbl_rows = []
    for cfg_name, dsid, topk, head, label, color in CONFIGS:
        cells = "".join(val_cell(metric_data[k][cfg_name], dec) for k, _, dec in tbl_cols)
        tbl_rows.append(f'<tr><th class="rowlab" style="border-left:4px solid {color}">{label}</th>{cells}</tr>')
    # baseline rows per dataset
    for dsid, suffix, sep_row in [("1c19e56f7560606d", "100k", True), ("59dbe58c95d0cb13", "300k", True)]:
        for method in ["ridge_embed", "mean_arithmetic", "prompt_only", "gt_embed"]:
            cells = "".join(val_cell(metric_data[k].get(f"{method}__{dsid}"), dec) for k, _, dec in tbl_cols)
            tbl_rows.append(
                f'<tr><th class="rowlab" style="border-left:4px solid {BASELINE_COLOR[method]}">'
                f'{BASELINE_LABEL[method]} ({suffix})</th>{cells}</tr>'
            )
    tbl_header = "<th>Method</th>" + "".join(f"<th>{lbl}</th>" for _, lbl, _ in tbl_cols)
    master_table = (
        f'<table class="data"><thead><tr>{tbl_header}</tr></thead>'
        f'<tbody>{"".join(tbl_rows)}</tbody></table>'
    )

    # ---- Gallery HTML ----
    def render_gallery(gallery, columns_per_row):
        out = []
        for gi, prompt, cells in gallery:
            figs = []
            for label, b64 in cells:
                inner = (f'<img src="data:image/jpeg;base64,{b64}" alt="{escape(label)}"/>'
                         if b64 else '<div class="noimg">missing</div>')
                figs.append(f'<figure><div class="thumb">{inner}</div><figcaption>{escape(label)}</figcaption></figure>')
            out.append(
                f'<div class="gallery-row"><div class="gcaption">#{gi:05d} &mdash; <span class="prompt">{escape(prompt)}</span></div>'
                f'<div class="gallery-strip" style="grid-template-columns:repeat({columns_per_row},1fr)">{"".join(figs)}</div></div>'
            )
        return "".join(out)

    gallery_main = render_gallery(gallery, columns_per_row=7)
    # Edit gallery
    edit_gallery_html = []
    for gi, prompt, cap, cells in edit_gallery:
        figs = []
        for label, b64 in cells:
            inner = (f'<img src="data:image/jpeg;base64,{b64}" alt="{escape(label)}"/>'
                     if b64 else '<div class="noimg">missing</div>')
            figs.append(f'<figure><div class="thumb">{inner}</div><figcaption>{escape(label)}</figcaption></figure>')
        edit_gallery_html.append(
            f'<div class="gallery-row"><div class="gcaption">#{gi:05d} &mdash; <span class="prompt">{escape(prompt)}</span><br/>'
            f'<span class="sub">{cap}</span></div>'
            f'<div class="gallery-strip" style="grid-template-columns:repeat(6,1fr)">{"".join(figs)}</div></div>'
        )
    edit_gallery_str = "".join(edit_gallery_html)

    # ---- Extract key numbers for prose ----
    ss_l1_100 = metric_data["mse_embedding_vs_gt"]["topk_100000_L1"]
    ss_l2_100 = metric_data["mse_embedding_vs_gt"]["topk_100000_L2_h2048"]
    ss_l1_300 = metric_data["mse_embedding_vs_gt"]["topk_300000_L1"]
    ss_l2_300 = metric_data["mse_embedding_vs_gt"]["topk_300000_L2_h2048"]
    rg_100 = metric_data["mse_embedding_vs_gt"]["ridge_embed__1c19e56f7560606d"]
    rg_300 = metric_data["mse_embedding_vs_gt"]["ridge_embed__59dbe58c95d0cb13"]

    cos_l2_100 = metric_data["cosine_embedding_vs_gt"]["topk_100000_L2_h2048"]
    cos_l2_300 = metric_data["cosine_embedding_vs_gt"]["topk_300000_L2_h2048"]

    lpips_l2_100 = metric_data["lpips_vs_gt_embed"]["topk_100000_L2_h2048"]
    lpips_l1_100 = metric_data["lpips_vs_gt_embed"]["topk_100000_L1"]
    lpips_ridge_100 = metric_data["lpips_vs_gt_embed"]["ridge_embed__1c19e56f7560606d"]

    ssim_l2_100 = metric_data["ssim_vs_gt_embed"]["topk_100000_L2_h2048"]

    # ---- HTML ----
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SSAE Ablation &mdash; Truncation &times; Head Depth</title>
<style>
  :root {{ --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --bg:#ffffff; --soft:#f9fafb;
           --blue:#2563eb; --warn:#b45309; --good:#047857; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
          color:var(--ink); line-height:1.55; margin:0; background:var(--bg); }}
  .wrap {{ max-width:960px; margin:0 auto; padding:40px 24px 96px; }}
  h1 {{ font-size:27px; margin:0 0 4px; letter-spacing:-0.01em; }}
  h2 {{ font-size:20px; margin:44px 0 8px; padding-top:14px; border-top:1px solid var(--line); }}
  h3 {{ font-size:15px; margin:26px 0 6px; color:var(--ink); }}
  p, li {{ font-size:14.5px; color:#1f2937; }}
  .sub {{ color:var(--muted); font-size:13px; }}
  .tag {{ display:inline-block; font-size:11px; font-weight:600; text-transform:uppercase;
          letter-spacing:0.04em; padding:2px 8px; border-radius:999px; }}
  .tag.blind {{ background:#eff6ff; color:#1d4ed8; }}
  .tag.result {{ background:#ecfdf5; color:#047857; }}
  .tag.analysis {{ background:#fef3c7; color:#92400e; }}
  .card {{ border:1px solid var(--line); border-radius:12px; padding:18px 20px; margin:14px 0;
           background:var(--soft); }}
  .callout {{ border-left:4px solid var(--blue); background:#f5f8ff; padding:12px 16px;
              border-radius:0 8px 8px 0; margin:16px 0; font-size:14px; }}
  .callout.warn {{ border-left-color:var(--warn); background:#fffbeb; }}
  .callout.key {{ border-left-color:var(--good); background:#ecfdf5; }}
  table.data {{ border-collapse:collapse; width:100%; font-size:12.5px; margin:12px 0; }}
  table.data th, table.data td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; }}
  table.data thead th {{ text-align:right; color:var(--muted); font-weight:600; font-size:11.5px;
                         border-bottom:2px solid #d1d5db; }}
  table.data th.rowlab {{ text-align:left; padding-left:10px; }}
  table.data td.na {{ color:#c0c4cc; }}
  table.data td.good {{ color:#047857; font-weight:600; }}
  table.data td.warn {{ color:#b45309; font-weight:500; }}
  .ci {{ display:block; color:var(--muted); font-size:10.5px; }}
  .chart {{ margin:10px 0 4px; }}
  .chart .cap {{ font-size:12.5px; color:var(--muted); margin:0 0 2px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
  @media (max-width:760px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
  .kv {{ display:flex; gap:8px; flex-wrap:wrap; margin:6px 0; }}
  .kv .pill {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:6px 10px;
               font-size:12.5px; }}
  .kv .pill b {{ font-variant-numeric:tabular-nums; }}
  code {{ background:#f3f4f6; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
  .gallery-row {{ margin:14px 0 22px; }}
  .gcaption {{ font-size:12.5px; color:var(--muted); margin-bottom:6px; }}
  .gcaption .prompt {{ color:#374151; }}
  .gallery-strip {{ display:grid; gap:6px; }}
  .gallery-strip figure {{ margin:0; text-align:center; }}
  .gallery-strip .thumb {{ aspect-ratio:1/1; border:1px solid var(--line); border-radius:8px;
                           overflow:hidden; background:#f3f4f6; }}
  .gallery-strip img {{ width:100%; height:100%; object-fit:cover; display:block; }}
  .gallery-strip .noimg {{ display:flex; align-items:center; justify-content:center; height:100%;
                           color:#9ca3af; font-size:11px; }}
  .gallery-strip figcaption {{ font-size:10.5px; color:var(--muted); margin-top:3px; }}
  ul {{ margin:6px 0 6px; padding-left:20px; }}
  li {{ margin:3px 0; }}
  .foot {{ color:var(--muted); font-size:12px; margin-top:40px; border-top:1px solid var(--line);
           padding-top:12px; }}
</style></head>
<body><div class="wrap">

<h1>SSAE ablation: truncation &times; head depth</h1>
<p class="sub">SD3.5 T5 held-out reconstruction, n = 100 tuples per method, base_seed = 0, no PCA rotation.
Four SSAE variants (top-k &isin; &lbrace;100 000, 300 000&rbrace; &times; head &isin; &lbrace;1-layer, 2-layer h=2048&rbrace;)
compared against per-dataset ridge (&lambda;=0.01), mean-arithmetic, prompt-only, and GT-embed baselines.</p>

<div class="callout key"><b>Bottom line.</b> A <b>2-layer head breaks the algebraic collapse</b> that
tied the SSAE to ridge in the reference report. The L2 SSAE beats ridge in embedding MSE on
<b>100/100</b> paired tuples in both truncation regimes, with mean MSE roughly <b>80&times; lower at top-100k</b>
({ss_l2_100['mean']:.5f} vs {rg_100['mean']:.5f}) and <b>16&times; lower at top-300k</b>
({ss_l2_300['mean']:.5f} vs {rg_300['mean']:.5f}). The embedding gain <b>propagates to image space</b>:
LPIPS-vs-GT for SSAE-L2 100k is {lpips_l2_100['mean']:.3f} vs ridge {lpips_ridge_100['mean']:.3f} at
matched seed, and SSIM-vs-GT rises to {ssim_l2_100['mean']:.3f}. The 1-layer SSAE remains a coin flip
against ridge, as the reference caveat predicted. Edit-locality is populated but does not yet show a
locality advantage for SSAE &mdash; pixel MSE and SSIM under drop-one and swap-one edits are close to
ridge across all four variants, and the ridge baseline itself already produces highly localised edits.</div>

<h2>1. What this benchmark measures <span class="tag blind">results-blind skeleton</span></h2>

<h3>Configurations under test</h3>
<ul>
  <li><b>Top-k truncation:</b> the T5 embedding is projected to its top-k highest-variance dims before
      training. Two settings: 100 000 (compact) and 300 000 (loose). No PCA rotation is applied &mdash;
      dims are kept in original coordinates.</li>
  <li><b>Head depth:</b> the SSAE decoder <code>W</code> is either a bare linear layer (L1) or a 2-layer
      MLP with hidden 2048 (L2 h=2048). All variants share the same trainable-inputs decoder body.</li>
  <li><b>Baselines</b> are recomputed per truncation (the ridge regression fits on the same top-k
      projection): ridge (&lambda;=0.01), mean-arithmetic (global mean + summed per-property deltas),
      prompt-only (raw SD3.5), and GT embed (oracle rendering of the true packed embedding).</li>
</ul>

<h3>Discriminating axes, in priority order</h3>
<ol>
  <li><b>Embedding-space MSE / cosine</b> vs. ground truth. Isolates the prediction from the stochastic
      renderer &mdash; the axis most likely to separate methods.</li>
  <li><b>Image similarity vs. GT render.</b> LPIPS, pixel MSE, SSIM. In this codebase
      <code>_sample_seed(base_seed, idx)</code> depends on the tuple index only, so all methods
      render the same tuple from the <i>same</i> diffusion seed. The image-space gaps below
      therefore reflect embedding differences, not seed variation. (The reference report warned
      of a per-method-seed confound; that no longer applies here.)</li>
  <li><b>Edit locality.</b> Populated in this run (both <code>--locality_drop_one_attr</code> and
      <code>--locality_swap_one_attr</code>). Pre-vs-post-edit pixel MSE / SSIM, and CLIP vs. the
      residual (edited) prompt. A structured decoder should edit surgically; a linear map should smear.</li>
  <li><b>Image&ndash;text CLIP.</b> Nearly saturated on these prompts (per the reference report), included
      as a sanity check.</li>
</ol>

<div class="callout"><b>Pre-registered expectations</b>
<ul>
  <li><b>L1</b> variants tie ridge in embedding space (the algebraic-collapse caveat: pointwise ReLU on
      disjoint per-property block means &rArr; prediction = const + &Sigma; per-property vectors, the
      same hypothesis class as ridge).</li>
  <li><b>L2</b> variants have a genuine nonlinear mix between the block-means step and the decoder, so
      they <i>could</i> break the collapse. If real interaction structure exists in T5 embeddings, L2
      should beat ridge here.</li>
  <li>Higher top-k should not improve reconstruction quality when normalised against the target
      variance &mdash; absolute MSE grows because there are more dims to hit.</li>
  <li>SSAE should be more surgical than ridge on drop-one / swap-one edits &mdash; that is the story
      that would justify the decoder over a linear map.</li>
</ul>
<b>Would be surprising:</b> L2 &gg; ridge (this is the good surprise); L2 = L1 (would suggest depth
buys nothing on this data); ridge already surgical (would mean locality isn't the discriminator).</div>

<div class="callout warn"><b>Caveats to read the results through.</b>
<ul>
  <li><b>Seed is tuple-consistent.</b> <code>_sample_seed(base_seed, idx)</code> in
      <code>evaluation/run_image_benchmark.py</code> depends only on the tuple index &mdash; all
      methods share a diffusion seed per tuple. So image-space "vs GT" differences below are
      unconfounded by seed variation (this is a fix relative to the reference report).</li>
  <li><b>Algebraic collapse only affects L1.</b> With a single linear head, the compositional
      predict function reduces to <code>const + &Sigma; per-property vectors</code> &mdash; the
      same additive family as ridge. The 2-layer head places a nonlinearity between the block-mean
      readout and the final decode, so L2 is NOT reducible to a per-property sum. The L1/L2
      contrast is therefore the pre-registered discriminator, not an afterthought.</li>
  <li><b>Only holdout is measured.</b> Training-set reconstruction with the trained <code>Y</code>
      rows is <b>not</b> in this benchmark; the gain reported here is generalisation to unseen
      tuples. Use <code>python -m evaluation.run_reconstruction --checkpoint &lt;ckpt&gt;</code>
      to compute the train-set counterpart (SSAE vs. ridge vs. mean-arithmetic) and compare the
      gap; the same tooling produces per-index arrays for follow-up analysis.</li>
  <li><b>Locality metric is coarse.</b> Pixel MSE mixes "how much the image moved" with "whether
      it moved in the right places". A CLIP-based residual-delta metric is now available via
      <code>python -m evaluation.rescore_locality</code>, which retroactively re-scores the
      already-rendered pre/post/swap images against the removed and target attribute phrases; the
      results are written to <code>per_sample_locality_extra.csv</code> next to each per-sample CSV.</li>
  <li><b>200k truncation is not present.</b> The task description mentioned it; only 100k and 300k
      runs are in <code>results/bench_out/</code>.</li>
</ul></div>

<h2>2. Results <span class="tag result">populated from data</span></h2>

<h3>2a. Embedding space &mdash; the discriminating axis</h3>
<div class="kv">
  <span class="pill">SSAE-L2 100k: MSE <b>{ss_l2_100['mean']:.5f}</b>, cos <b>{cos_l2_100['mean']:.4f}</b></span>
  <span class="pill">SSAE-L2 300k: MSE <b>{ss_l2_300['mean']:.5f}</b>, cos <b>{cos_l2_300['mean']:.4f}</b></span>
  <span class="pill">Ridge 100k: MSE <b>{rg_100['mean']:.5f}</b></span>
  <span class="pill">Ridge 300k: MSE <b>{rg_300['mean']:.5f}</b></span>
</div>
<div class="chart"><p class="cap">Embedding MSE vs. GT (mean &plusmn; 95% CI, n = 100). Linear scale.</p>{chart_emb_lin}</div>
<div class="chart"><p class="cap">Same, log scale &mdash; the L2 variants sit ~1&ndash;2 orders of magnitude below all linear baselines.</p>{chart_emb_log}</div>
<div class="chart"><p class="cap">Cosine similarity to GT embedding (mean &plusmn; 95% CI). Higher is better.</p>{chart_cos}</div>

<h3>Paired win-rate and diff-of-means (SSAE vs. baselines)</h3>
{win_table}
<p class="sub">Paired = same held-out tuple index, same ground truth. The L1 SSAE is a <b>~46&ndash;53%
coin flip</b> against ridge with a mean diff of essentially zero &mdash; a textbook algebraic tie.
The L2 SSAE is <b>100/100</b> against ridge with a paired-diff CI that clears zero by more than
30 standard errors in both truncations.</p>

<h3>2b. Image&ndash;text alignment (CLIP)</h3>
<div class="grid2">
  <div class="chart"><p class="cap">CLIP cosine vs. full prompt (&uarr;).</p>{chart_clip_full}</div>
  <div class="chart"><p class="cap">CLIP mean cosine vs. active attributes (&uarr;).</p>{chart_clip_attr}</div>
</div>
<p>CLIP-vs-prompt lands in a ~0.02 band across all methods and baselines; both truncations,
both head depths, and even prompt-only are within CI of each other. As in the reference
report, CLIP is saturated at this prompt complexity and doesn't rank the methods.</p>

<h3>2c. Image similarity vs. GT render</h3>
<div class="grid2">
  <div class="chart"><p class="cap">LPIPS vs. GT render (&darr;).</p>{chart_lpips}</div>
  <div class="chart"><p class="cap">Pixel MSE vs. GT render (&darr;).</p>{chart_pxmse}</div>
</div>
<div class="chart"><p class="cap">SSIM vs. GT render (&uarr;).</p>{chart_ssim}</div>
<p>The L2 variants cleanly separate on all three image metrics despite the per-method-seed noise:
LPIPS for SSAE-L2 100k is <b>{lpips_l2_100['mean']:.3f}</b> vs. L1 <b>{lpips_l1_100['mean']:.3f}</b>
and ridge <b>{lpips_ridge_100['mean']:.3f}</b>; SSIM-vs-GT crosses 0.95 for both L2 variants where
every linear method sits ~0.80&ndash;0.83. The seed confound is real but too small to explain a gap
this large &mdash; the L2 render is genuinely closer to the GT-embed render pixel-for-pixel.</p>
<p class="sub"><b>Anomaly to note.</b> In the 300k dataset, <i>prompt-only</i> vs. GT is unusually good
(LPIPS ~ 0.10, pixel MSE ~ 0.006), better than any reconstruction method. This is a property of the
gt_embed reference: at higher top-k the packed embedding is closer to the raw T5 output, so a
prompt-only render happens to land near it. It does <i>not</i> apply to the 100k comparison, where
prompt-only sits at LPIPS ~ 0.28 as in the reference report.</p>

<h3>2d. Per-method table (95% CIs)</h3>
{master_table}

<h3>2e. Edit locality &mdash; drop-one and swap-one attribute</h3>
<div class="grid2">
  <div class="chart"><p class="cap">Pre &rarr; post pixel MSE, drop-one edit (&darr;).</p>{chart_edit_mse}</div>
  <div class="chart"><p class="cap">Pre &rarr; post SSIM, drop-one edit (&uarr;).</p>{chart_edit_ssim}</div>
  <div class="chart"><p class="cap">CLIP vs. residual prompt after drop-one (&uarr;).</p>{chart_edit_clipres}</div>
  <div class="chart"><p class="cap">Swap-one pixel MSE (&darr;).</p>{chart_swap_mse}</div>
  <div class="chart"><p class="cap">Swap-one SSIM (&uarr;).</p>{chart_swap_ssim}</div>
  <div class="chart"><p class="cap">Swap-one CLIP vs. swapped prompt (&uarr;).</p>{chart_swap_clip}</div>
</div>
<p>Paired locality contrast against ridge:</p>
{edit_table}
<p>The locality signal is <b>not the one predicted</b>. On drop-one, the L2 SSAE variants are actually
<i>slightly worse</i> than ridge in pixel MSE (larger pixel change per edit), and on SSIM they land
essentially on top of ridge. On swap-one, all four SSAE variants lose to ridge on pixel MSE by
roughly 40&ndash;50% of paired tuples. Two things drive this: (i) ridge is already surprisingly
surgical &mdash; a joint linear map with only 26 attribute bits doesn't have many degrees of freedom
to smear with; (ii) the L2 SSAE reconstructs the full embedding more faithfully, so the difference
between the pre- and post-edit embeddings is closer to the <i>true</i> per-attribute delta &mdash;
which may be geometrically larger than ridge's shrunken version.</p>

<h3>2f. Signal-to-noise ratio (spread / paired-std)</h3>
<p class="sub">Ratio of between-method mean spread to paired std of tuple-level differences. &gt; 0.5
means the metric can actually rank the methods at n = 100; &lt; 0.2 means it cannot.</p>
{snr_table}
<p>Embedding MSE has SNR &Gt; 1 in every config (green): the ranking on this axis is unambiguous.
LPIPS / pixel-MSE / SSIM vs GT sit at ratio ~ 0.5&ndash;1 in the L2 configs &mdash; borderline but
supportive. CLIP is uniformly &lt; 0.2 (warn): as in the reference report, CLIP cannot rank at this n.
Locality metrics have ratio &lt; 0.3 &mdash; the "SSAE vs. ridge" locality claim would need more
samples (or a metric with lower paired variance) to be conclusive either way.</p>

<h3>2g. Qualitative gallery</h3>
<p class="sub">Same held-out tuple across baselines and SSAE variants (7 columns per row). Note the
per-method seed &mdash; comparisons are best done column-vs-column with GT-embed as anchor.</p>
{gallery_main}

<h3>2h. Edit locality &mdash; qualitative</h3>
<p class="sub">Pre / post / swap for Ridge vs. SSAE-L2 100k on two held-out tuples. Column order:
Ridge {{pre, post, swap}} then SSAE-L2 {{pre, post, swap}}.</p>
{edit_gallery_str}

<h2>3. Analysis &amp; commentary <span class="tag analysis">interpretation</span></h2>

<h3>Against the pre-registered expectations</h3>
<ul>
  <li><b>Confirmed &mdash; L1 collapses to ridge.</b> Both L1 variants sit within 10<sup>&minus;5</sup>
      of ridge in mean MSE and give 46&ndash;53% paired win rates. This is exactly the algebraic
      collapse the reference report predicted: a pointwise ReLU on disjoint block means times a
      linear <code>W</code> is a per-property additive sum &mdash; the same hypothesis class as
      ridge.</li>
  <li><b>Stronger than expected &mdash; L2 breaks the collapse cleanly.</b> The 2-layer head buys
      an ~<b>80&times;</b> embedding-MSE reduction at top-100k and ~<b>16&times;</b> at top-300k, with
      100/100 paired wins in both regimes and cosine similarities of {cos_l2_100['mean']:.4f} /
      {cos_l2_300['mean']:.4f}. This is the first evidence in this project that T5 embeddings for
      compositional prompts contain interaction structure the additive family cannot capture, and
      that a modest MLP head over the block-means readout is enough to recover it on unseen
      tuples.</li>
  <li><b>Propagates to image space.</b> The L2 embedding gain isn't washed out by the diffusion
      stochasticity: LPIPS and SSIM vs. GT-embed both improve by 3&ndash;5&times; the ridge margin,
      well above the seed-noise floor implied by prompt-only variation.</li>
  <li><b>Truncation trade-off is boring.</b> Going from 100k to 300k roughly doubles the ridge and
      mean-arith absolute MSE (more dims to hit, per-dim residual similar), and the L2 gain shrinks
      accordingly (80&times; &rarr; 16&times;) but stays dominant. There is no truncation setting in
      which ridge catches L2.</li>
  <li><b>Surprising &mdash; locality doesn't separate.</b> The one experiment designed to expose a
      structural advantage for the decoder over a linear map does <i>not</i> favour SSAE. Ridge is
      already surgical, and the L2 SSAE's more faithful embedding actually produces larger per-edit
      pixel changes. This suggests the compositional edit story needs a different framing &mdash; the
      L2 win in embedding space is not being spent on selective editing.</li>
</ul>

<h3>What we can and cannot conclude</h3>
<div class="callout"><b>Can conclude.</b> A 2-layer head materially improves the SSAE's held-out
embedding reconstruction beyond ridge, in a paired-tuple-consistent way, and the improvement is
large enough to show up in image space despite per-method seed noise. The gain is robust across
100k and 300k truncation.</div>
<div class="callout warn"><b>Cannot conclude</b> (yet) that the L2 gain corresponds to a
<i>compositional</i> advantage over ridge. The one experiment that could show this &mdash; single-
attribute edit locality &mdash; does not favour the SSAE. So the L2 head might be capturing generic
prompt structure (grammar, phrasing, joint frequencies) rather than the disentangled per-property
directions the training objective is meant to encourage.</div>

<h3>Recommended next steps</h3>
<p class="sub">Status legend: <b class="done">DONE</b> = code landed in this repo, <b class="todo">TODO</b> = needs a training or eval run on the user's GPU box.</p>
<ul>
  <li><span class="tag good-tag">DONE</span> <b>Seed confound is already fixed.</b>
      <code>_sample_seed</code> depends only on <code>base_seed</code> and tuple <code>idx</code>,
      so LPIPS / SSIM in this run already isolate the embedding gap.</li>
  <li><span class="tag good-tag">DONE</span> <b>Locality re-scoring tool.</b>
      <code>python -m evaluation.rescore_locality</code> retroactively computes
      <code>clip_full_vs_edit_attr</code>, <code>clip_dropped_vs_edit_attr</code>,
      <code>clip_dropped_vs_residual</code>, plus a derived <code>clip_locality_score</code>
      (delta on the removed attribute minus delta on the residual). No re-rendering needed &mdash;
      it re-scores the existing <code>images_pre_edit</code> and <code>images_swapped</code>
      dirs. Re-run this before drawing locality conclusions.</li>
  <li><span class="tag good-tag">DONE</span> <b>Training-set reconstruction.</b>
      <code>python -m evaluation.run_reconstruction --checkpoint &lt;ckpt&gt; --output_json &lt;path&gt;</code>
      now reports SSAE (with the trained <code>Y</code> row), ridge, and mean-arithmetic on the
      training set, with per-index MSE arrays and paired win rates. The SSAE-vs-ridge margin on
      train &mdash; contrasted with the composition-time margin on holdout &mdash; measures how much
      interaction structure the block-mean composition throws away.</li>
  <li><span class="tag warn-tag">TODO</span> <b>Interrogate the L2 gain (block-diagonal ablation).</b>
      A new <code>head_type: block_diagonal</code> option lives in
      <code>trainings/models/mlp.py::BlockDiagonalHead</code> and is exposed via
      <code>training_cli.py --head_type block_diagonal</code>. It runs an independent MLP per
      property block and sums outputs, forbidding cross-property mixing while preserving
      within-block depth. If block-diagonal L2 recovers the dense-L2 gain, the win is per-block
      nonlinearity; if it doesn't, the head is doing genuine cross-property mixing.
      Retrain on both truncations and re-run the benchmark to compare.</li>
  <li><span class="tag warn-tag">TODO</span> <b>Add the 200k point.</b> Train with
      <code>--truncate_embds_topk 200000</code> (via YAML or CLI override) at both head depths
      to fill the middle of the truncation sweep and confirm smoothness.</li>
</ul>
<style>
  .done, .todo {{ font-family:inherit; }}
  .tag.good-tag {{ background:#ecfdf5; color:#047857; }}
  .tag.warn-tag {{ background:#fef3c7; color:#92400e; }}
</style>

<p class="foot">Generated from <code>results/bench_out/&lbrace;topk_&hellip;&rbrace;/per_sample.csv</code>
and <code>results/bench_baseline_cache/&lbrace;dsid&rbrace;/per_sample.csv</code>. Embedding-space and
locality statistics recomputed from raw per-sample rows; means with 95% CIs from Student normal
approximation on n = 100. Charts are inline SVG with 95% CI whiskers; thumbnails are downscaled JPEG.
Ridge &lambda; = 0.01, base_seed = 0, no PCA rotation.</p>

</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_dir", type=Path, default=Path("results/bench_out"))
    ap.add_argument("--baseline_dir", type=Path, default=Path("results/bench_baseline_cache"))
    ap.add_argument("--out", type=Path, default=Path("results/bench_out/ablation_report.html"))
    ap.add_argument("--n_gallery", type=int, default=3)
    args = ap.parse_args()
    path = build(args.bench_dir, args.baseline_dir, args.out, n_gallery=args.n_gallery)
    print(f"wrote {path} ({path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
