"""Altair-based quantitative-metrics plots for the loaded benchmark runs."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from viewer.data import RunData, numeric_metric_columns

# per_sample.csv can grow into the tens of thousands once multiple runs are loaded;
# Altair's default 5000-row guard would silently truncate.
alt.data_transformers.disable_max_rows()


_LOCALITY_METRICS = (
    "mse_pixel_pre_post_edit",
    "ssim_pre_post_edit",
    "clip_image_vs_residual_prompt",
    "mse_pixel_swap_vs_normal",
    "ssim_swap_vs_normal",
    "clip_swap_image_vs_swapped_prompt",
)


def _per_sample_long(runs: list[RunData], methods: list[str]) -> pd.DataFrame:
    frames = []
    for run in runs:
        if run.per_sample.empty:
            continue
        df = run.per_sample.copy()
        if methods:
            df = df[df["method"].isin(methods)]
        df.insert(0, "run", run.label)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _summary_long(runs: list[RunData], methods: list[str]) -> pd.DataFrame:
    """Flatten summary.json[per_method][method][metric] = {mean, ci_low, ci_high, n}."""
    rows = []
    for run in runs:
        per_method = run.summary.get("per_method", {}) if run.summary else {}
        for method, metrics in per_method.items():
            if methods and method not in methods:
                continue
            for metric, value in metrics.items():
                if not isinstance(value, dict) or "mean" not in value:
                    continue
                rows.append(
                    {
                        "run": run.label,
                        "method": method,
                        "metric": metric,
                        "mean": value.get("mean"),
                        "ci_low": value.get("ci_low"),
                        "ci_high": value.get("ci_high"),
                        "n": value.get("n"),
                    }
                )
    return pd.DataFrame(rows)


def _available_metrics(runs: list[RunData]) -> list[str]:
    return sorted({c for r in runs for c in numeric_metric_columns(r)})


def _pick_metric(runs: list[RunData], key: str, default_pref: list[str]) -> str | None:
    options = _available_metrics(runs)
    if not options:
        return None
    default = next((m for m in default_pref if m in options), options[0])
    return st.selectbox(
        "Metric",
        options=options,
        index=options.index(default),
        key=key,
    )


def _failure_strip(long_ps: pd.DataFrame) -> None:
    if "clip_fail" not in long_ps.columns:
        return
    fails = long_ps.groupby(["run", "method"])["clip_fail"].sum().reset_index(name="n_fail")
    if fails["n_fail"].sum() == 0:
        return
    chart = (
        alt.Chart(fails)
        .mark_bar()
        .encode(
            x=alt.X("method:N", title=None),
            y=alt.Y("n_fail:Q", title="clip_fail count"),
            color=alt.Color("run:N", title="run"),
            column=alt.Column("run:N", title=None) if fails["run"].nunique() > 1 else alt.value(None),
            tooltip=["run", "method", "n_fail"],
        )
        .properties(height=140)
    )
    st.altair_chart(chart, use_container_width=True)


def _bar_with_ci(summary_df: pd.DataFrame, metric: str) -> alt.Chart | None:
    df = summary_df[summary_df["metric"] == metric].dropna(subset=["mean"])
    if df.empty:
        return None
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("method:N", title=None),
            y=alt.Y("mean:Q", title=f"mean {metric}"),
            color=alt.Color("run:N", title="run"),
            xOffset="run:N",
            tooltip=["run", "method", "mean", "ci_low", "ci_high", "n"],
        )
    )
    errors = (
        alt.Chart(df)
        .mark_errorbar()
        .encode(
            x=alt.X("method:N"),
            y=alt.Y("ci_low:Q", title=""),
            y2="ci_high:Q",
            color=alt.Color("run:N"),
            xOffset="run:N",
        )
    )
    return (bars + errors).properties(height=280)


def _distribution(long_ps: pd.DataFrame, metric: str, style: str) -> alt.Chart | None:
    if metric not in long_ps.columns:
        return None
    cols = ["run", "method", metric]
    if "sample_idx" in long_ps.columns:
        cols.append("sample_idx")
    df = long_ps[cols].dropna(subset=[metric])
    if df.empty:
        return None
    base = alt.Chart(df).transform_calculate(jitter="random()")
    if style == "violin":
        return (
            base.transform_density(
                metric,
                as_=[metric, "density"],
                groupby=["run", "method"],
                extent=[float(df[metric].min()), float(df[metric].max())],
            )
            .mark_area(orient="horizontal", opacity=0.7)
            .encode(
                y=alt.Y(f"{metric}:Q"),
                x=alt.X("density:Q", stack="center", axis=None),
                color=alt.Color("run:N"),
                column=alt.Column("method:N", header=alt.Header(orient="bottom")),
            )
            .properties(width=90, height=280)
        )
    tooltip = [alt.Tooltip("run:N"), alt.Tooltip("method:N"), alt.Tooltip(f"{metric}:Q")]
    if "sample_idx" in df.columns:
        tooltip.append(alt.Tooltip("sample_idx:Q"))
    return (
        base.mark_circle(size=18, opacity=0.5)
        .encode(
            x=alt.X("jitter:Q", axis=None, scale=alt.Scale(domain=[-0.5, 1.5])),
            y=alt.Y(f"{metric}:Q"),
            color=alt.Color("run:N"),
            column=alt.Column("method:N", header=alt.Header(orient="bottom")),
            tooltip=tooltip,
        )
        .properties(width=90, height=280)
    )


def _scatter(long_ps: pd.DataFrame, x_metric: str, y_metric: str) -> alt.Chart | None:
    cols = [c for c in (x_metric, y_metric) if c in long_ps.columns]
    if len(cols) < 2:
        return None
    df = long_ps[["run", "method", "sample_idx", x_metric, y_metric]].dropna()
    if df.empty:
        return None
    return (
        alt.Chart(df)
        .mark_circle(size=30, opacity=0.6)
        .encode(
            x=alt.X(f"{x_metric}:Q"),
            y=alt.Y(f"{y_metric}:Q"),
            color=alt.Color("method:N"),
            column=alt.Column("run:N", header=alt.Header(orient="bottom")),
            tooltip=["run", "method", "sample_idx", x_metric, y_metric],
        )
        .properties(width=280, height=280)
        .interactive()
    )


def render_aggregate_tab(runs: list[RunData], methods: list[str]) -> None:
    if not runs:
        st.info("Load at least one run in the sidebar to see aggregates.")
        return
    long_ps = _per_sample_long(runs, methods)
    summary_df = _summary_long(runs, methods)
    if long_ps.empty and summary_df.empty:
        st.info(
            "Loaded runs have no per_sample.csv rows or summary.json entries yet."
            " Use the Reload runs button in the sidebar once the benchmark has written some output."
        )
        return

    st.subheader("CLIP failures")
    _failure_strip(long_ps)

    st.subheader("Per-method means (95% CI)")
    if summary_df.empty:
        st.caption("No summary.json data available for the loaded runs.")
    else:
        metric = _pick_metric(
            runs,
            key="agg_bar_metric",
            default_pref=[
                "clip_image_vs_full_prompt",
                "lpips_vs_gt_embed",
                "mse_pixel_vs_gt_embed",
            ],
        )
        if metric:
            chart = _bar_with_ci(summary_df, metric)
            if chart is None:
                st.caption(f"No summary CI data for `{metric}`.")
            else:
                st.altair_chart(chart, use_container_width=True)

    st.subheader("Per-sample distributions")
    if long_ps.empty:
        st.caption("No per_sample.csv rows loaded yet.")
    else:
        dist_metric = _pick_metric(
            runs,
            key="agg_dist_metric",
            default_pref=[
                "clip_image_vs_full_prompt",
                "lpips_vs_gt_embed",
                "mse_pixel_vs_gt_embed",
            ],
        )
        style = st.radio(
            "Distribution style", options=["strip", "violin"], horizontal=True, key="agg_dist_style"
        )
        if dist_metric:
            chart = _distribution(long_ps, dist_metric, style)
            if chart is None:
                st.caption(f"No non-null values for `{dist_metric}`.")
            else:
                st.altair_chart(chart, use_container_width=True)

    st.subheader("Two-metric scatter")
    if long_ps.empty:
        st.caption("No per_sample.csv rows loaded yet.")
    else:
        options = _available_metrics(runs)
        default_x = next(
            (m for m in ("mse_embedding_vs_gt", "clip_image_vs_full_prompt") if m in options),
            options[0] if options else None,
        )
        default_y = next(
            (
                m
                for m in ("clip_image_vs_full_prompt", "lpips_vs_gt_embed", "mse_pixel_vs_gt_embed")
                if m in options and m != default_x
            ),
            options[-1] if options else None,
        )
        col_x, col_y = st.columns(2)
        with col_x:
            x_metric = st.selectbox(
                "X metric",
                options=options,
                index=options.index(default_x) if default_x in options else 0,
                key="agg_scatter_x",
            )
        with col_y:
            y_metric = st.selectbox(
                "Y metric",
                options=options,
                index=options.index(default_y) if default_y in options else 0,
                key="agg_scatter_y",
            )
        if x_metric and y_metric and x_metric != y_metric:
            chart = _scatter(long_ps, x_metric, y_metric)
            if chart is None:
                st.caption("No overlapping non-null values for the chosen metrics.")
            else:
                st.altair_chart(chart, use_container_width=True)

    locality_present = [
        m for m in _LOCALITY_METRICS if m in long_ps.columns and long_ps[m].notna().any()
    ]
    if locality_present:
        st.subheader("Locality metrics")
        for metric in locality_present:
            chart = _distribution(long_ps, metric, "strip")
            if chart is not None:
                st.markdown(f"**{metric}**")
                st.altair_chart(chart, use_container_width=True)
