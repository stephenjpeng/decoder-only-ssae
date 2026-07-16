"""Streamlit viewer for evaluation.run_image_benchmark holdout output.

Run with: streamlit run viewer/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from viewer.aggregate import render_aggregate_tab
from viewer.data import (
    RunData,
    available_sample_ids,
    available_variants,
    discover_runs,
    edit_info_for,
    image_path,
    load_holdout_prompts,
    load_run,
    metrics_for,
    numeric_metric_columns,
    run_progress,
)

DEFAULT_HOLDOUT = "results/compositional_split/holdout"
DEFAULT_RESULTS_ROOT = "results"
DEFAULT_CAPTION_METRICS = [
    "clip_image_vs_full_prompt",
    "dino_cosine_vs_gt_embed",
    "lpips_vs_gt_embed",
    "ssim_vs_gt_embed",
]

st.set_page_config(page_title="Image benchmark viewer", layout="wide")


def _parse_run_entries(text: str) -> list[tuple[str, str]]:
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "=" in line:
            label, path = line.split("=", 1)
        else:
            label, path = Path(line).name, line
        entries.append((label.strip(), path.strip()))
    return entries


@st.cache_data
def _cached_holdout_prompts(holdout_folder: str) -> list[dict]:
    return load_holdout_prompts(Path(holdout_folder))


@st.cache_data
def _cached_run(label: str, output_dir: str) -> RunData:
    return load_run(label, Path(output_dir))


def _sidebar() -> tuple[list[dict], list[RunData], list[str], list[str], list[str], bool]:
    st.sidebar.header("Data sources")
    holdout_folder = st.sidebar.text_input("Holdout folder", value=DEFAULT_HOLDOUT)
    prompts = _cached_holdout_prompts(holdout_folder)

    discovered = discover_runs(DEFAULT_RESULTS_ROOT)
    picked = st.sidebar.multiselect(
        "Discovered benchmark runs", options=[str(p) for p in discovered]
    )
    extra_text = st.sidebar.text_area(
        "Additional runs (one per line, `label=path` or bare path)", value=""
    )
    if st.sidebar.button("Reload runs", help="Re-read per_sample.csv / summary.json / images from disk"):
        _cached_run.clear()
        st.rerun()

    run_entries = [(Path(p).name, p) for p in picked] + _parse_run_entries(extra_text)
    runs = [_cached_run(label, path) for label, path in run_entries]

    st.sidebar.header("Filters")
    all_methods: list[str] = []
    for run in runs:
        for m in run.methods:
            if m not in all_methods:
                all_methods.append(m)
    selected_methods = st.sidebar.multiselect(
        "Methods to show",
        options=all_methods,
        default=all_methods,
    )

    variant_options: list[str] = []
    for run in runs:
        for v in available_variants(run):
            if v not in variant_options:
                variant_options.append(v)
    default_variants = ["post"] if "post" in variant_options else variant_options[:1]
    variants = st.sidebar.multiselect(
        "Image variants (side by side)",
        options=variant_options,
        default=default_variants,
    )

    only_failures = st.sidebar.checkbox("Only show CLIP failures", value=False)

    st.sidebar.header("Metrics")
    all_metrics = sorted({c for r in runs for c in numeric_metric_columns(r)})
    caption_metrics = st.sidebar.multiselect(
        "Metrics shown under each thumbnail",
        options=all_metrics,
        default=[m for m in DEFAULT_CAPTION_METRICS if m in all_metrics],
    )

    return prompts, runs, caption_metrics, variants, selected_methods, only_failures


def _sample_ids_with_failures(runs: list[RunData]) -> set[int]:
    ids: set[int] = set()
    for run in runs:
        if run.per_sample.empty or "clip_fail" not in run.per_sample.columns:
            continue
        failing = run.per_sample.loc[run.per_sample["clip_fail"] == 1, "sample_idx"]
        ids.update(int(i) for i in failing)
    return ids


def _render_prompt_header(prompt_entry: dict) -> None:
    st.subheader(prompt_entry["prompt"])
    choices = prompt_entry.get("choices", {})
    st.caption(" | ".join(f"{k}: {v}" for k, v in choices.items()))


def _render_edit_details(runs: list[RunData], sample_id: int) -> None:
    """Show which attribute was dropped / swapped per run (if the run recorded any)."""
    blocks = []
    for run in runs:
        info = edit_info_for(run, sample_id)
        if not info:
            continue
        edit_attr = info.get("edit_attribute", "")
        swap_attr = info.get("swap_target_attribute", "")
        swapped_prompt = info.get("swapped_prompt", "")
        if not edit_attr and not swap_attr:
            continue
        lines = [f"**{run.label}** —"]
        if swap_attr:
            lines.append(f"swapped `{edit_attr}` → `{swap_attr}`")
            if swapped_prompt:
                lines.append(f"resulting prompt: _{swapped_prompt}_")
        else:
            lines.append(f"dropped `{edit_attr}`")
        blocks.append("  \n".join(lines))
    if not blocks:
        return
    with st.container(border=True):
        st.markdown("**Edit details**")
        for block in blocks:
            st.markdown(block)


def _metric_caption_lines(metrics: dict, caption_metrics: list[str]) -> list[str]:
    lines = []
    for m in caption_metrics:
        v = metrics.get(m)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            lines.append(f"{m}: {v:.3f}")
    return lines


def _progress_badge(run: RunData) -> str:
    p = run_progress(run)
    if p["complete"] and p["has_summary"]:
        return ""
    parts = []
    if not p["has_csv"]:
        parts.append("no per_sample.csv yet")
    elif p["expected_rows"]:
        parts.append(f"{p['n_rows']}/{p['expected_rows']} rows")
    else:
        parts.append(f"{p['n_rows']} rows")
    if not p["has_summary"]:
        parts.append("no summary.json")
    return " — in progress (" + ", ".join(parts) + ")"


def _render_grid(
    runs: list[RunData],
    sample_id: int,
    caption_metrics: list[str],
    variants: list[str],
    selected_methods: list[str],
) -> None:
    variants = variants or ["post"]
    for run in runs:
        methods = [m for m in run.methods if m in selected_methods]
        st.markdown(f"**{run.label}**{_progress_badge(run)}")
        if not methods:
            st.caption("(no methods selected for this run)")
            continue
        cols = st.columns(len(methods))
        for col, method in zip(cols, methods):
            with col:
                st.caption(method)
                if len(variants) == 1:
                    p = image_path(run, method, sample_id, variants[0])
                    if p.exists():
                        st.image(str(p), width="stretch")
                    else:
                        st.write("(no image)")
                else:
                    subcols = st.columns(len(variants))
                    for scol, variant in zip(subcols, variants):
                        with scol:
                            st.caption(variant)
                            p = image_path(run, method, sample_id, variant)
                            if p.exists():
                                st.image(str(p), width="stretch")
                            else:
                                st.write("(no image)")
                lines = _metric_caption_lines(
                    metrics_for(run, sample_id, method), caption_metrics
                )
                if lines:
                    st.caption("\n".join(lines))


def _render_full_metrics(
    runs: list[RunData], sample_id: int, selected_methods: list[str]
) -> None:
    with st.expander("Full metrics"):
        frames = []
        for run in runs:
            if run.per_sample.empty:
                continue
            rows = run.per_sample[
                (run.per_sample["sample_idx"] == sample_id)
                & (run.per_sample["method"].isin(selected_methods))
            ].copy()
            if rows.empty:
                continue
            rows.insert(0, "run", run.label)
            frames.append(rows)
        if frames:
            st.dataframe(pd.concat(frames, ignore_index=True))
        else:
            st.write("No rows for the current selection.")


_ARROW_SHORTCUT_JS = """
<script>
(function() {
  const doc = window.parent.document;
  if (doc.__arrowNavInstalled) return;
  doc.__arrowNavInstalled = true;
  doc.addEventListener('keydown', function(e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    const t = e.target;
    if (t && ['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName)) return;
    if (t && t.isContentEditable) return;
    if (doc.querySelector('div[role="listbox"]')) return;
    const label = e.key === 'ArrowLeft' ? 'Prev' : 'Next';
    for (const btn of doc.querySelectorAll('button')) {
      if (btn.innerText.trim() === label) {
        btn.click();
        e.preventDefault();
        return;
      }
    }
  });
})();
</script>
"""


def _render_navigator(candidate_ids: list[int], by_id: dict[int, dict]) -> int:
    n = len(candidate_ids)
    st.session_state.setdefault("prompt_position", 0)
    if st.session_state.prompt_position >= n:
        st.session_state.prompt_position = 0

    st.session_state["jump_prompt"] = candidate_ids[st.session_state.prompt_position]

    def _on_jump() -> None:
        st.session_state.prompt_position = candidate_ids.index(st.session_state.jump_prompt)

    def _prev() -> None:
        st.session_state.prompt_position = max(0, st.session_state.prompt_position - 1)

    def _next() -> None:
        st.session_state.prompt_position = min(n - 1, st.session_state.prompt_position + 1)

    cols = st.columns([1, 8, 1])
    with cols[0]:
        st.button("Prev", on_click=_prev, width="stretch", disabled=n <= 1)
    with cols[2]:
        st.button("Next", on_click=_next, width="stretch", disabled=n <= 1)
    with cols[1]:
        st.slider(
            "Prompt position",
            min_value=0,
            max_value=max(0, n - 1),
            key="prompt_position",
            label_visibility="collapsed",
            disabled=n <= 1,
        )

    st.selectbox(
        "Jump to prompt",
        options=candidate_ids,
        key="jump_prompt",
        on_change=_on_jump,
        format_func=lambda i: f"[{i}] {by_id[i]['prompt'][:80]}",
    )

    components.html(_ARROW_SHORTCUT_JS, height=0)
    return candidate_ids[st.session_state.prompt_position]


def _render_summary(runs: list[RunData]) -> None:
    with st.expander("Run summary (aggregate)"):
        for run in runs:
            st.markdown(f"**{run.label}**{_progress_badge(run)}")
            per_method = run.summary.get("per_method", {}) if run.summary else {}
            if not per_method:
                st.write("No summary.json found.")
                continue
            table = {}
            for method, metrics in per_method.items():
                row = {}
                for metric, value in metrics.items():
                    if isinstance(value, dict) and "mean" in value:
                        row[metric] = value["mean"]
                    else:
                        row[metric] = value
                table[method] = row
            st.dataframe(pd.DataFrame(table).T)


def _candidate_ids(
    prompts: list[dict], runs: list[RunData], only_failures: bool
) -> list[int]:
    """Holdout prompt ids that actually have data on disk across the loaded runs."""
    by_id = {p["id"]: p for p in prompts}
    available: set[int] = set()
    for run in runs:
        available |= available_sample_ids(run)
    candidate = [i for i in by_id if i in available]
    if only_failures:
        failing = _sample_ids_with_failures(runs)
        candidate = [i for i in candidate if i in failing]
    return candidate


def _render_browse_tab(
    prompts: list[dict],
    runs: list[RunData],
    caption_metrics: list[str],
    variants: list[str],
    selected_methods: list[str],
    only_failures: bool,
) -> None:
    by_id = {p["id"]: p for p in prompts}
    candidate_ids = _candidate_ids(prompts, runs, only_failures)
    if not candidate_ids:
        if only_failures:
            st.warning("No CLIP failures found across the loaded runs.")
        else:
            st.info(
                "No samples on disk yet for the loaded runs."
                " Use the Reload runs button in the sidebar once the benchmark has written some output."
            )
        return

    sample_id = _render_navigator(candidate_ids, by_id)
    _render_prompt_header(by_id[sample_id])
    _render_edit_details(runs, sample_id)
    _render_grid(runs, sample_id, caption_metrics, variants, selected_methods)
    _render_full_metrics(runs, sample_id, selected_methods)
    _render_summary(runs)


def main() -> None:
    st.title("Image benchmark viewer")
    prompts, runs, caption_metrics, variants, selected_methods, only_failures = _sidebar()

    if not prompts:
        st.info("Enter a valid holdout folder in the sidebar.")
        return
    if not runs:
        st.info("Select or add at least one benchmark output_dir in the sidebar.")
        return

    browse, aggregates = st.tabs(["Browse", "Aggregates"])
    with browse:
        _render_browse_tab(prompts, runs, caption_metrics, variants, selected_methods, only_failures)
    with aggregates:
        render_aggregate_tab(runs, selected_methods)


if __name__ == "__main__":
    main()
