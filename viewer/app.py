"""Streamlit viewer for evaluation.run_image_benchmark holdout output.

Run with: streamlit run viewer/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from viewer.data import (
    RunData,
    discover_runs,
    image_path,
    load_holdout_prompts,
    load_run,
    metrics_for,
    numeric_metric_columns,
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


def _sidebar() -> tuple[list[dict], list[RunData], list[str], bool]:
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

    run_entries = [(Path(p).name, p) for p in picked] + _parse_run_entries(extra_text)
    runs = [_cached_run(label, path) for label, path in run_entries]

    st.sidebar.header("Metrics")
    all_metrics = sorted({c for r in runs for c in numeric_metric_columns(r)})
    caption_metrics = st.sidebar.multiselect(
        "Metrics shown under each thumbnail",
        options=all_metrics,
        default=[m for m in DEFAULT_CAPTION_METRICS if m in all_metrics],
    )

    only_failures = st.sidebar.checkbox("Only show CLIP failures", value=False)
    return prompts, runs, caption_metrics, only_failures


def _sample_ids_with_failures(prompts: list[dict], runs: list[RunData]) -> set[int]:
    ids: set[int] = set()
    for run in runs:
        if "clip_fail" not in run.per_sample.columns:
            continue
        failing = run.per_sample.loc[run.per_sample["clip_fail"] == 1, "sample_idx"]
        ids.update(int(i) for i in failing)
    return ids


def _render_prompt_header(prompt_entry: dict) -> None:
    st.subheader(prompt_entry["prompt"])
    choices = prompt_entry.get("choices", {})
    st.caption(" | ".join(f"{k}: {v}" for k, v in choices.items()))


def _render_grid(runs: list[RunData], sample_id: int, caption_metrics: list[str]) -> None:
    for run in runs:
        st.markdown(f"**{run.label}**")
        cols = st.columns(len(run.methods) or 1)
        for col, method in zip(cols, run.methods):
            with col:
                st.caption(method)
                img_path = image_path(run, method, sample_id)
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.write("(no image)")
                metrics = metrics_for(run, sample_id, method)
                lines = [
                    f"{m}: {metrics[m]:.3f}" for m in caption_metrics if m in metrics
                ]
                if lines:
                    st.caption("\n".join(lines))


def _render_full_metrics(runs: list[RunData], sample_id: int) -> None:
    with st.expander("Full metrics"):
        frames = []
        for run in runs:
            rows = run.per_sample[run.per_sample["sample_idx"] == sample_id].copy()
            rows.insert(0, "run", run.label)
            frames.append(rows)
        if frames:
            st.dataframe(pd.concat(frames, ignore_index=True))
        else:
            st.write("No runs loaded.")


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
        st.button("Prev", on_click=_prev, use_container_width=True, disabled=n <= 1)
    with cols[2]:
        st.button("Next", on_click=_next, use_container_width=True, disabled=n <= 1)
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
            st.markdown(f"**{run.label}**")
            per_method = run.summary.get("per_method", {})
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


def main() -> None:
    st.title("Image benchmark viewer")
    prompts, runs, caption_metrics, only_failures = _sidebar()

    if not prompts:
        st.info("Enter a valid holdout folder in the sidebar.")
        return
    if not runs:
        st.info("Select or add at least one benchmark output_dir in the sidebar.")
        return

    by_id = {p["id"]: p for p in prompts}
    candidate_ids = list(by_id.keys())
    if only_failures:
        failing_ids = _sample_ids_with_failures(prompts, runs)
        candidate_ids = [i for i in candidate_ids if i in failing_ids]
        if not candidate_ids:
            st.warning("No CLIP failures found across the loaded runs.")
            return

    sample_id = _render_navigator(candidate_ids, by_id)

    _render_prompt_header(by_id[sample_id])
    _render_grid(runs, sample_id, caption_metrics)
    _render_full_metrics(runs, sample_id)
    _render_summary(runs)


if __name__ == "__main__":
    main()
