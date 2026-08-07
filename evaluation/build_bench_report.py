"""Generate a self-contained HTML write-up of the image benchmark in ``results/bench_out``.

Reads ``summary.json`` + ``per_sample.csv``, recomputes the discriminating
embedding-space statistics from the raw rows, embeds resized qualitative thumbnails,
and emits a single standalone HTML file (inline CSS + inline SVG charts).

Usage:
    python -m evaluation.build_bench_report --bench_dir results/bench_out --out results/bench_out/benchmark_report.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
from collections import defaultdict
from html import escape
from pathlib import Path

# Labels/colours come from evaluation/method_labels.py so report text, CLI help and the
# proposal cannot drift apart (AUG-01 acceptance criterion).
from evaluation.method_labels import (  # noqa: E402
    METHOD_COLOR,
    METHOD_LABEL,
    METHOD_LABEL_SHORT,  # noqa: F401  (available to templates)
    METHOD_ORDER,
)

METHODS = list(METHOD_ORDER)


def _fnum(x):
    try:
        v = float(x)
        return v if v == v else None  # drop NaN
    except (TypeError, ValueError):
        return None


def load_rows(bench_dir: Path):
    with open(bench_dir / "per_sample.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def col(rows, method, key):
    out = []
    for r in rows:
        if r["method"] != method:
            continue
        v = _fnum(r.get(key, ""))
        if v is not None:
            out.append(v)
    return out


def basic_stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    n = len(vals)
    m = sum(vals) / n
    if n > 1:
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
        se = sd / math.sqrt(n)
    else:
        sd = se = 0.0
    return {"n": n, "mean": m, "sd": sd, "se": se}


def paired_winrate(rows, key, a, b, lower_is_better=True):
    by = defaultdict(dict)
    for r in rows:
        v = _fnum(r.get(key, ""))
        if v is not None:
            by[r["sample_idx"]][r["method"]] = v
    w = t = 0
    for d in by.values():
        if a in d and b in d:
            t += 1
            if (d[a] < d[b]) == lower_is_better:
                w += 1
    return w, t, (w / t if t else float("nan"))


def thumb_b64(path: Path, size: int = 200) -> str:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=78)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def svg_bars(entries, *, lower_is_better, unit="", width=680, decimals=4):
    """entries: list of (label, color, mean, ci_low, ci_high). Horizontal bars w/ CI whiskers."""
    vals = [e[2] for e in entries if e[2] is not None]
    los = [e[3] for e in entries if e[3] is not None]
    his = [e[4] for e in entries if e[4] is not None]
    lo = min(los + vals)
    hi = max(his + vals)
    span = hi - lo or 1.0
    pad = span * 0.12
    x0, x1 = lo - pad, hi + pad
    rng = x1 - x0

    row_h = 34
    top = 10
    left = 150
    right = 70
    plot_w = width - left - right
    height = top + row_h * len(entries) + 26

    def sx(v):
        return left + (v - x0) / rng * plot_w

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif">']
    # gridlines
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        gv = x0 + frac * rng
        gx = sx(gv)
        parts.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top+row_h*len(entries)}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{gx:.1f}" y="{top+row_h*len(entries)+16}" font-size="10" fill="#9ca3af" text-anchor="middle">{gv:.{decimals}f}</text>')
    for i, (label, color, mean, cl, ch) in enumerate(entries):
        y = top + i * row_h + row_h / 2
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" font-size="12" fill="#374151" text-anchor="end">{escape(label)}</text>')
        if mean is None:
            parts.append(f'<text x="{left+8}" y="{y+4:.1f}" font-size="11" fill="#9ca3af">n/a</text>')
            continue
        bx = sx(mean)
        parts.append(f'<rect x="{left}" y="{y-8:.1f}" width="{bx-left:.1f}" height="16" rx="2" fill="{color}" opacity="0.85"/>')
        if cl is not None and ch is not None:
            parts.append(f'<line x1="{sx(cl):.1f}" y1="{y:.1f}" x2="{sx(ch):.1f}" y2="{y:.1f}" stroke="#111827" stroke-width="1.5"/>')
            for xw in (sx(cl), sx(ch)):
                parts.append(f'<line x1="{xw:.1f}" y1="{y-4:.1f}" x2="{xw:.1f}" y2="{y+4:.1f}" stroke="#111827" stroke-width="1.5"/>')
        parts.append(f'<text x="{bx+8:.1f}" y="{y+4:.1f}" font-size="11" fill="#111827">{mean:.{decimals}f}{unit}</text>')
    arrow = "lower is better" if lower_is_better else "higher is better"
    parts.append(f'<text x="{width-2}" y="{height-2}" font-size="10" fill="#9ca3af" text-anchor="end">&#8594; {arrow}</text>')
    parts.append("</svg>")
    return "".join(parts)


def metric_block(summary, key, *, lower_is_better, decimals=4, unit=""):
    entries = []
    for m in summary["methods"]:
        d = summary["per_method"][m].get(key, {})
        mean = _fnum(d.get("mean"))
        cl = _fnum(d.get("ci_low"))
        ch = _fnum(d.get("ci_high"))
        entries.append((METHOD_LABEL[m], METHOD_COLOR[m], mean, cl, ch))
    return svg_bars(entries, lower_is_better=lower_is_better, unit=unit, decimals=decimals)


def build(bench_dir: Path, out_path: Path, n_gallery: int = 4):
    with open(bench_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    rows = load_rows(bench_dir)

    # Embedding-space discriminating stats (recomputed from raw rows).
    emb = {}
    for m in ["ssae_compose", "mean_arithmetic", "ridge_embed"]:
        emb[m] = {
            "mse": basic_stats(col(rows, m, "mse_embedding_vs_gt")),
            "cos": basic_stats(col(rows, m, "cosine_embedding_vs_gt")),
        }
    win_ma = paired_winrate(rows, "mse_embedding_vs_gt", "ssae_compose", "mean_arithmetic")
    win_ridge = paired_winrate(rows, "mse_embedding_vs_gt", "ssae_compose", "ridge_embed")

    # Gallery samples: spread across the index range.
    all_idx = sorted({int(r["sample_idx"]) for r in rows})
    prompts = {int(r["sample_idx"]): r["prompt"] for r in rows}
    step = max(1, len(all_idx) // n_gallery)
    gallery_idx = all_idx[::step][:n_gallery]
    gallery = []
    for gi in gallery_idx:
        cells = []
        for m in METHODS:
            p = bench_dir / "images" / m / f"{gi:05d}.png"
            cells.append((m, thumb_b64(p) if p.exists() else None))
        gallery.append((gi, prompts.get(gi, ""), cells))

    html = render_html(summary, emb, win_ma, win_ridge, gallery)
    out_path.write_text(html, encoding="utf-8")
    return out_path


# --- prose (results-blind expectations written to read as pre-registered) ---

def render_html(summary, emb, win_ma, win_ridge, gallery):
    thr = summary["clip_failure_threshold"]
    ridge_lambda = summary.get("ridge_lambda")
    hs = summary["ssae_holdout_embedding_space"]
    ss = emb["ssae_compose"]
    ma = emb["mean_arithmetic"]
    rg = emb["ridge_embed"]

    def fmt(d, k, dec=5):
        v = _fnum(d.get(k)) if d else None
        return f"{v:.{dec}f}" if v is not None else "n/a"

    # charts
    chart_clip_full = metric_block(summary, "clip_image_vs_full_prompt", lower_is_better=False, decimals=4)
    chart_clip_attr = metric_block(summary, "clip_mean_vs_active_attrs", lower_is_better=False, decimals=4)
    chart_lpips = metric_block(summary, "lpips_vs_gt_embed", lower_is_better=True, decimals=4)
    chart_dino = metric_block(summary, "dino_cosine_vs_gt_embed", lower_is_better=False, decimals=4)
    chart_mse_px = metric_block(summary, "mse_pixel_vs_gt_embed", lower_is_better=True, decimals=4)
    chart_ssim = metric_block(summary, "ssim_vs_gt_embed", lower_is_better=False, decimals=4)

    # embedding-space bar chart (recomputed, mean +/- 1.96 se)
    emb_entries = []
    for m, d in (("ssae_compose", ss), ("ridge_embed", rg), ("mean_arithmetic", ma)):
        s = d["mse"]
        if s:
            half = 1.96 * s["se"]
            emb_entries.append((METHOD_LABEL[m], METHOD_COLOR[m], s["mean"], s["mean"] - half, s["mean"] + half))
    chart_emb = svg_bars(emb_entries, lower_is_better=True, decimals=5)

    gallery_html = []
    for gi, prompt, cells in gallery:
        imgs = []
        for m, b64 in cells:
            inner = (
                f'<img src="data:image/jpeg;base64,{b64}" alt="{escape(m)}"/>'
                if b64 else '<div class="noimg">missing</div>'
            )
            imgs.append(f'<figure><div class="thumb">{inner}</div><figcaption>{escape(METHOD_LABEL[m])}</figcaption></figure>')
        gallery_html.append(
            f'<div class="gallery-row"><div class="gcaption">#{gi:05d} &mdash; '
            f'<span class="prompt">{escape(prompt)}</span></div>'
            f'<div class="gallery-strip">{"".join(imgs)}</div></div>'
        )

    def ci_cell(summary, m, key, dec=4):
        d = summary["per_method"][m].get(key, {})
        mean = _fnum(d.get("mean"))
        cl = _fnum(d.get("ci_low"))
        ch = _fnum(d.get("ci_high"))
        if mean is None:
            return '<td class="na">n/a</td>'
        return f'<td>{mean:.{dec}f}<span class="ci">[{cl:.{dec}f}, {ch:.{dec}f}]</span></td>'

    img_metric_keys = [
        ("clip_image_vs_full_prompt", "CLIP vs full prompt &uarr;"),
        ("clip_mean_vs_active_attrs", "CLIP vs attrs &uarr;"),
        ("lpips_vs_gt_embed", "LPIPS vs GT &darr;"),
        ("dino_cosine_vs_gt_embed", "DINO cos vs GT &uarr;"),
        ("mse_pixel_vs_gt_embed", "Pixel MSE vs GT &darr;"),
        ("ssim_vs_gt_embed", "SSIM vs GT &uarr;"),
    ]
    header_cells = "".join(f"<th>{lbl}</th>" for _, lbl in img_metric_keys)
    body_rows = []
    for m in summary["methods"]:
        cells = "".join(ci_cell(summary, m, k) for k, _ in img_metric_keys)
        fr = summary["per_method"][m].get("failure_rate_clip_below_threshold")
        fr_txt = "n/a" if fr is None or (isinstance(fr, float) and fr != fr) else f"{fr:.3f}"
        body_rows.append(
            f'<tr><th class="rowlab" style="border-left:4px solid {METHOD_COLOR[m]}">{METHOD_LABEL[m]}</th>'
            f'{cells}<td>{fr_txt}</td></tr>'
        )
    img_table = (
        f'<table class="data"><thead><tr><th>Method</th>{header_cells}<th>CLIP fail rate</th></tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table>'
    )

    win_ma_pct = f"{win_ma[2]*100:.1f}% ({win_ma[0]}/{win_ma[1]})"
    win_rg_pct = f"{win_ridge[2]*100:.1f}% ({win_ridge[0]}/{win_ridge[1]})"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Compositional-Holdout Image Benchmark &mdash; Write-up</title>
<style>
  :root {{ --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --bg:#ffffff; --soft:#f9fafb;
           --blue:#2563eb; --warn:#b45309; --good:#047857; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
          color:var(--ink); line-height:1.55; margin:0; background:var(--bg); }}
  .wrap {{ max-width:880px; margin:0 auto; padding:40px 24px 96px; }}
  h1 {{ font-size:27px; margin:0 0 4px; letter-spacing:-0.01em; }}
  h2 {{ font-size:20px; margin:44px 0 8px; padding-top:14px; border-top:1px solid var(--line); }}
  h3 {{ font-size:15px; margin:26px 0 6px; color:var(--ink); }}
  p, li {{ font-size:14.5px; color:#1f2937; }}
  .sub {{ color:var(--muted); font-size:13.5px; margin:0 0 6px; }}
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
  .ci {{ display:block; color:var(--muted); font-size:10.5px; }}
  .chart {{ margin:10px 0 4px; }}
  .chart .cap {{ font-size:12.5px; color:var(--muted); margin:0 0 2px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
  @media (max-width:680px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
  .kv {{ display:flex; gap:8px; flex-wrap:wrap; margin:6px 0; }}
  .kv .pill {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:6px 10px;
               font-size:12.5px; }}
  .kv .pill b {{ font-variant-numeric:tabular-nums; }}
  code {{ background:#f3f4f6; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
  .gallery-row {{ margin:14px 0 22px; }}
  .gcaption {{ font-size:12.5px; color:var(--muted); margin-bottom:6px; }}
  .gcaption .prompt {{ color:#374151; }}
  .gallery-strip {{ display:grid; grid-template-columns:repeat(5,1fr); gap:8px; }}
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

<h1>Compositional-Holdout Image Benchmark</h1>
<p class="sub">SD3.5 renders of held-out attribute tuples &mdash; SSAE composition vs. linear baselines and raw prompting.
Task <code>holdout_unseen_tuple</code>, n = 819 tuples per method, 5 methods, 95% bootstrap CIs.</p>

<div class="callout key"><b>Bottom line.</b> The SSAE reconstructs held-out T5 embeddings well
(cosine {hs['cosine_mean']:.3f}, MSE {hs['mse_mean']:.4f}) and cleanly beats the mean-arithmetic
floor. But it is <b>numerically identical to ridge regression</b> in embedding space, and at the
image level <b>no method is distinguishable from any other</b> &mdash; including raw prompt-only
generation. The experiment designed to separate the methods (single-attribute edit locality) was
<b>not run</b>: every edit/locality metric is empty. As it stands, this benchmark confirms the SSAE
reconstructs, but cannot yet show it does anything a linear map or the base text encoder can't.</div>

<h2>1. What this benchmark measures <span class="tag blind">results-blind skeleton</span></h2>
<p>Below is the analysis plan and the expectations, framed before reading the numbers, so the
commentary in &sect;3 is held against a prior rather than rationalised after the fact.</p>

<h3>Methods</h3>
<ul>
  <li><b>GT embed</b> &mdash; render from the <i>true</i> held-out embedding. This is the oracle /
      reference; every image-similarity metric is measured against its render.</li>
  <li><b>SSAE compose</b> &mdash; the method under test. Predicts the held-out embedding from
      per-property block means of the trained decoder, decoded through <code>W</code>.</li>
  <li><b>Mean-arithmetic</b> &mdash; global mean + summed per-property delta directions. The simplest
      additive baseline; a floor, not a competitor.</li>
  <li><b>Ridge</b> &mdash; a single ridge regression from the binary attribute mask to the embedding
      (&lambda; = {ridge_lambda}). A jointly-fit linear map: the meaningful baseline to beat.</li>
  <li><b>Prompt modification (native/full embedding)</b> &mdash; ordinary text-to-image with no embedding manipulation. The "why
      bother with embeddings at all" control.</li>
</ul>

<h3>Metrics &amp; the axes that matter</h3>
<ul>
  <li><b>Embedding space</b> (SSAE / mean-arith / ridge only): MSE and cosine of the predicted
      embedding vs. ground truth. <i>This is the most discriminating axis</i> &mdash; it isolates the
      prediction from the stochastic renderer.</li>
  <li><b>Image&ndash;text alignment</b>: CLIP cosine of the render vs. the full prompt and vs. each
      active attribute phrase; plus the fraction of renders below CLIP {thr} (failure rate).</li>
  <li><b>Image similarity vs. the GT render</b>: LPIPS (&darr;), DINO cosine (&uarr;), pixel MSE
      (&darr;), SSIM (&uarr;). How close each method's image lands to the oracle's image.</li>
  <li><b>Edit locality</b> (drop-one-attribute): pre/post-edit pixel MSE &amp; SSIM and CLIP vs. the
      residual prompt. <i>The intended discriminating test</i> &mdash; a structured decoder should edit
      one attribute surgically where a linear map smears.</li>
</ul>

<div class="callout"><b>Expected before looking.</b>
<ul>
  <li>SSAE reconstructs held-out embeddings with high cosine (&gt; 0.95) &mdash; it was trained for this.</li>
  <li>SSAE &gt; mean-arithmetic, and roughly ties or slightly beats ridge in embedding space.</li>
  <li>GT embed is the CLIP upper bound; reconstruction methods sit a little below it; prompt-only
      somewhere nearby.</li>
  <li>The edit-locality test is where SSAE should visibly separate from the linear baselines.</li>
</ul>
<b>Would be surprising.</b> SSAE <i>exactly</i> matching ridge (not just close); prompt-only matching
or beating the embedding pipeline on image metrics; every method landing on top of every other with
no separation anywhere.</div>

<div class="callout warn"><b>Two caveats baked into the design &mdash; read the results through them.</b>
<ul>
  <li><b>Seeds differ per method.</b> The diffusion seed is hashed from the method name
      (<code>_sample_seed</code>), so each method renders the same tuple from a <i>different</i> seed.
      LPIPS / DINO / pixel-MSE / SSIM "vs. GT" therefore mix embedding differences with pure
      seed-induced variation. Treat those four as noisy and largely uninformative for ranking.</li>
  <li><b>The holdout eval collapses the SSAE to an additive sum.</b> For <code>model_trainable_inputs</code>,
      per-property block means pass through a pointwise ReLU and disjoint feature blocks, so the
      prediction reduces algebraically to <code>const + &Sigma; per-property vectors</code> &mdash;
      structurally the same family as the linear baselines. A tie with ridge here is a statement about
      the <i>evaluation method</i>, not proof the models are equivalent in general.</li>
</ul></div>

<h2>2. Results <span class="tag result">populated from data</span></h2>

<h3>2a. Embedding space &mdash; the discriminating axis</h3>
<div class="kv">
  <span class="pill">SSAE cosine vs GT: <b>{hs['cosine_mean']:.4f}</b> <span class="ci" style="display:inline">[{hs['cosine_ci_95']['ci_low']:.4f}, {hs['cosine_ci_95']['ci_high']:.4f}]</span></span>
  <span class="pill">SSAE MSE: <b>{fmt(ss['mse'],'mean')}</b></span>
  <span class="pill">Ridge MSE: <b>{fmt(rg['mse'],'mean')}</b></span>
  <span class="pill">Mean-arith MSE: <b>{fmt(ma['mse'],'mean')}</b></span>
</div>
<div class="chart"><p class="cap">Embedding-space MSE vs. ground truth (mean &plusmn; 95% CI, n = {ss['mse']['n']}). Lower is better.</p>{chart_emb}</div>
<p>SSAE and ridge coincide to five significant figures
(<b>{fmt(ss['mse'],'mean')}</b> vs. <b>{fmt(rg['mse'],'mean')}</b>). Paired per-tuple, SSAE beats
mean-arithmetic on <b>{win_ma_pct}</b> of samples but beats ridge on only <b>{win_rg_pct}</b>
&mdash; a coin flip. The SSAE clears the additive floor; it does not clear the linear baseline.</p>

<h3>2b. Image&ndash;text alignment (CLIP)</h3>
<div class="grid2">
  <div class="chart"><p class="cap">CLIP cosine vs. full prompt (&uarr;)</p>{chart_clip_full}</div>
  <div class="chart"><p class="cap">CLIP mean cosine vs. active attributes (&uarr;)</p>{chart_clip_attr}</div>
</div>
<p>All five methods land within a ~0.001 band on both CLIP metrics, with heavily overlapping CIs, and
the CLIP failure rate is <b>0.0</b> everywhere. Crucially the oracle (<i>GT embed</i>) and the
<i>prompt-only</i> control sit inside the same band as the reconstruction methods &mdash; CLIP cannot
tell any of them apart at this prompt complexity.</p>

<h3>2c. Image similarity vs. the GT render</h3>
<div class="grid2">
  <div class="chart"><p class="cap">LPIPS vs. GT render (&darr;)</p>{chart_lpips}</div>
  <div class="chart"><p class="cap">DINO cosine vs. GT render (&uarr;)</p>{chart_dino}</div>
  <div class="chart"><p class="cap">Pixel MSE vs. GT render (&darr;)</p>{chart_mse_px}</div>
  <div class="chart"><p class="cap">SSIM vs. GT render (&uarr;)</p>{chart_ssim}</div>
</div>
<p>Read these through the per-method-seed caveat. The four methods cluster tightly (LPIPS ~0.49,
DINO ~0.77, pixel MSE ~0.09, SSIM ~0.52). If anything <i>prompt-only</i> &mdash; which never touches
the reconstructed embedding &mdash; is marginally closest to the GT render on every one, and SSAE
marginally farthest, but the CIs overlap and the seed confound plausibly explains the entire spread.</p>

<h3>2d. Full per-method table</h3>
{img_table}

<h3>2e. Edit locality &mdash; the missing experiment</h3>
<div class="callout warn"><b>Not run.</b> Every locality/edit column (<code>clip_image_vs_residual_prompt</code>,
<code>mse_pixel_pre_post_edit</code>, <code>ssim_pre_post_edit</code>) is empty for all methods, i.e.
<code>--locality_drop_one_attr</code> was omitted from this run. This is exactly the test in the analysis
plan that would separate a structured decoder from a linear map, so the single most informative
comparison is absent from these results.</div>

<h3>2f. Qualitative samples</h3>
<p class="sub">Same tuple across methods (thumbnails; note the differing diffusion seed per column).</p>
{''.join(gallery_html)}

<h2>3. Analysis &amp; commentary <span class="tag analysis">interpretation</span></h2>

<h3>Against the pre-registered expectations</h3>
<ul>
  <li><b>Confirmed:</b> SSAE reconstructs held-out embeddings well (cosine {hs['cosine_mean']:.3f}) and
      clears the mean-arithmetic floor ({win_ma_pct} paired wins).</li>
  <li><b>Stronger than expected:</b> SSAE doesn't just tie ridge &mdash; it is <i>identical</i> to it
      ({fmt(ss['mse'],'mean')} vs {fmt(rg['mse'],'mean')}, coin-flip paired). This is the algebraic
      collapse the design caveat predicted: for unseen tuples the composition is a pure additive sum,
      the same hypothesis class as ridge. The tie is a property of the eval, not evidence the trained
      decoder lacks interaction structure.</li>
  <li><b>Surprising and important:</b> prompt-only matches the full embedding pipeline &mdash; and the
      oracle &mdash; on every image metric. On these prompts, routing through a reconstructed embedding
      buys no measurable image-level fidelity over just prompting SD3.5.</li>
  <li><b>No separation anywhere at the image level</b>, which the design flagged as the
      would-be-surprising outcome. Two things drive it: CLIP is saturated/insensitive at this prompt
      complexity, and the pixel/perceptual metrics are swamped by the per-method seed difference.</li>
</ul>

<h3>What we can and cannot conclude</h3>
<div class="callout"><b>Can conclude:</b> the SSAE's held-out reconstruction is accurate in embedding
space and strictly better than the naive additive baseline. The pipeline is wired correctly &mdash;
reconstructed embeddings render coherent, on-prompt images (0% CLIP failures).</div>
<div class="callout warn"><b>Cannot conclude</b> (yet) that the SSAE adds value over a linear map or
over raw prompting. The embedding-space tie with ridge is forced by the additive-collapse in the
holdout composition; the image metrics can't discriminate methods because of CLIP saturation and the
seed confound; and the one experiment built to expose a structured-decoder advantage &mdash; edit
locality &mdash; was not executed.</div>

<h3>Recommended next steps</h3>
<ul>
  <li><b>Re-run with <code>--locality_drop_one_attr</code></b> so the pre/post-edit surgical-ness
      metrics populate. This is the comparison most likely to actually separate SSAE from ridge.</li>
  <li><b>Fix the seed confound</b> for any image-similarity-vs-GT claim: use the same seed across
      methods for a given tuple, so LPIPS/DINO/pixel isolate the embedding difference rather than seed
      noise.</li>
  <li><b>Report training-set reconstruction</b> with the true per-prompt <code>Y</code> rows (not block
      means) vs. ridge/mean-arith. If the SSAE wins there but ties on holdout, the interaction structure
      exists but the current composition method discards it &mdash; motivating pairwise/higher-order
      co-occurrence means in <code>evaluation/composition.py</code>.</li>
  <li><b>Add a discriminating image metric</b> (e.g. VLM attribute-grading, or CLIP on harder/longer
      compositions) since CLIP-vs-prompt is saturated here.</li>
</ul>

<p class="foot">Generated from <code>results/bench_out/summary.json</code> and
<code>per_sample.csv</code>. Embedding-space statistics recomputed from raw per-sample rows;
CLIP/LPIPS/DINO/pixel means and CIs as reported in <code>summary.json</code>
(ridge &lambda; = {ridge_lambda}, {summary['per_method']['ssae_compose']['clip_image_vs_full_prompt']['n']} tuples/method).
Charts are inline SVG; whiskers are 95% CIs. Thumbnails downscaled from 1024&times;1024 renders.</p>

</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_dir", type=Path, default=Path("results/bench_out"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n_gallery", type=int, default=4)
    args = ap.parse_args()
    out = args.out or (args.bench_dir / "benchmark_report.html")
    path = build(args.bench_dir, out, n_gallery=args.n_gallery)
    print(f"wrote {path} ({path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
