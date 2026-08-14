# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair>=5.5,<6",
#     "marimo==0.23.16",
#     "numpy>=1.26,<3",
#     "pandas>=2.2,<3",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # VLM metric comparison

    Compare all OpenAI vision-language model (VLM) judgments for the E3 image
    editing benchmark. Filters apply to every chart and table below.

    ## Success metric definitions

    For each image, the VLM returns two attribute-presence scores on a 0 to 1
    scale:

    - `edit_attribute_score`: presence of the original edited attribute
    - `swap_target_attribute_score`: presence of the requested replacement attribute

    The notebook derives `task_success` so higher is better for every experiment:

    | Experiment | Formula | Successful image |
    |---|---:|---|
    | Post | `min(requested_attribute_scores)` | Every requested attribute is present |
    | Delete | `1 - edit_attribute_score` | The original target is absent |
    | Swap | `min(swap_target_attribute_score, 1 - edit_attribute_score)` | The replacement is present and the original is absent |

    The minimum implements conjunction on the VLM's 0 to 1 scores. One missing
    required condition limits the image's task success.

    The other reported VLM metrics are direct judge outputs:

    - `match_full_prompt`: agreement with the variant-specific prompt
    - `non_target_preserved`: preservation of attributes not targeted by the edit

    Raw target-presence metrics remain available in the metric selector. Missing
    method and experiment combinations are unavailable judgments, not zero scores.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path
    from typing import Any

    import altair as alt
    import numpy as np
    import pandas as pd

    DATA_PATH = Path("results/vlm_e3_full/e3_vlm.jsonl")
    METHOD_ORDER = [
        "gt_embed",
        "prompt_only",
        "mean_arithmetic",
        "ridge_embed",
        "ssae_L1",
        "ssae_L2_h2048",
    ]
    METHOD_LABELS = {
        "gt_embed": "GT embed",
        "prompt_only": "Prompt only",
        "mean_arithmetic": "Mean arithmetic",
        "ridge_embed": "Ridge embed",
        "ssae_L1": "SSAE L1",
        "ssae_L2_h2048": "SSAE L2 h2048",
    }
    VARIANT_ORDER = ["deleted", "swapped", "post"]
    VARIANT_LABELS = {
        "deleted": "Delete",
        "swapped": "Swap",
        "post": "Post",
    }
    METRIC_LABELS = {
        "task_success": "Task success",
        "match_full_prompt": "Full-prompt match",
        "non_target_preserved": "Non-target preservation",
        "edit_attribute_score": "Original target presence (raw)",
        "swap_target_attribute_score": "Replacement target presence (raw)",
    }
    METRIC_HELP = {
        "task_success": "Higher is better for every experiment",
        "match_full_prompt": "Higher means the image better matches the judged prompt",
        "non_target_preserved": "Higher means non-target attributes are better preserved",
        "edit_attribute_score": (
            "Raw original-target presence. Lower is better for Delete and Swap; "
            "higher is better for Post"
        ),
        "swap_target_attribute_score": (
            "Raw replacement-target presence. Higher is better for Swap; "
            "the other experiments use this as an absence check"
        ),
    }

    def target_from_benchmark_dir(path: str) -> str:
        """Extract the edited target from a benchmark directory"""
        benchmark = Path(path)
        if benchmark.parent.parent.name.endswith("cache"):
            return benchmark.parent.name
        return benchmark.name.split("_", 1)[0]

    def method_from_record(method: str, benchmark_dir: str) -> str:
        """Give each SSAE run its own method label"""
        if method != "ssae_compose":
            return method
        run_name = Path(benchmark_dir).name
        if "_" not in run_name:
            return method
        return f"ssae_{run_name.split('_', 1)[1]}"

    def score_for_phrase(payload: dict[str, Any], phrase: str) -> float | None:
        """Read one exact phrase score from a VLM response"""
        if not phrase:
            return None
        for item in payload.get("attribute_scores", []):
            if item.get("phrase") == phrase:
                value = item.get("score")
                return None if value is None else float(value)
        return None

    def as_float(value: Any) -> float | None:
        """Convert an optional scalar to float"""
        return None if value is None or value == "" else float(value)

    def task_success_score(
        payload: dict[str, Any],
        *,
        variant: str,
        prompt: str,
        edit_attribute: str,
        swap_attribute: str,
    ) -> float | None:
        """Combine attribute-presence scores according to the edit objective"""
        edit_score = score_for_phrase(payload, edit_attribute)
        if variant == "deleted":
            return None if edit_score is None else 1.0 - edit_score
        if variant == "swapped":
            swap_score = score_for_phrase(payload, swap_attribute)
            if edit_score is None or swap_score is None:
                return None
            return min(swap_score, 1.0 - edit_score)
        if variant == "post":
            requested_attributes = [
                part.strip() for part in prompt.split(",") if part.strip()
            ]
            requested_scores = [
                score_for_phrase(payload, attribute)
                for attribute in requested_attributes
            ]
            if not requested_scores or any(score is None for score in requested_scores):
                return None
            return min(score for score in requested_scores if score is not None)
        raise ValueError(f"Unknown VLM image variant: {variant}")

    def load_vlm_results(path: Path) -> pd.DataFrame:
        """Load one flat row per judged image and derive task success"""
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                payload = record.get("vlm", {})
                edit_attribute = record.get("edit_attribute", "")
                swap_attribute = record.get("swap_target_attribute", "")
                variant = record["variant"]
                edit_score = score_for_phrase(payload, edit_attribute)
                swap_score = score_for_phrase(payload, swap_attribute)
                task_success = task_success_score(
                    payload,
                    variant=variant,
                    prompt=record.get("prompt_judged", ""),
                    edit_attribute=edit_attribute,
                    swap_attribute=swap_attribute,
                )
                rows.append(
                    {
                        "target": target_from_benchmark_dir(record["benchmark_dir"]),
                        "sample_idx": int(record["sample_idx"]),
                        "method": method_from_record(
                            record["method"], record["benchmark_dir"]
                        ),
                        "variant": variant,
                        "match_full_prompt": as_float(payload.get("match_full_prompt")),
                        "non_target_preserved": as_float(
                            payload.get("non_target_preserved")
                        ),
                        "edit_attribute_score": edit_score,
                        "swap_target_attribute_score": swap_score,
                        "task_success": task_success,
                        "prompt_judged": record.get("prompt_judged", ""),
                        "edit_attribute": edit_attribute,
                        "swap_target_attribute": swap_attribute,
                        "notes": payload.get("notes", ""),
                        "benchmark_dir": record["benchmark_dir"],
                        "image": record["image"],
                    }
                )
        frame = pd.DataFrame(rows)
        frame["method_label"] = frame["method"].map(METHOD_LABELS).fillna(
            frame["method"]
        )
        frame["variant_label"] = frame["variant"].map(VARIANT_LABELS)
        return frame

    def bootstrap_interval(
        values: np.ndarray,
        *,
        rng: np.random.Generator,
        n_boot: int = 2000,
    ) -> tuple[float, float]:
        """Return a percentile bootstrap interval for a sample mean"""
        clean = values[~np.isnan(values)]
        if len(clean) == 0:
            return float("nan"), float("nan")
        indices = rng.integers(0, len(clean), size=(n_boot, len(clean)))
        means = clean[indices].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        return float(low), float(high)

    def summarize_metric(
        frame: pd.DataFrame,
        metric: str,
        group_columns: list[str],
    ) -> pd.DataFrame:
        """Summarize one metric with deterministic bootstrap intervals"""
        rows: list[dict[str, Any]] = []
        rng = np.random.default_rng(20260813)
        for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            values = group[metric].dropna().to_numpy(dtype=float)
            low, high = bootstrap_interval(values, rng=rng)
            row = dict(zip(group_columns, key_values, strict=True))
            row.update(
                {
                    "score": float(values.mean()) if len(values) else float("nan"),
                    "ci_low": low,
                    "ci_high": high,
                    "n": int(len(values)),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def paired_method_differences(
        frame: pd.DataFrame,
        metric: str,
        baseline: str,
        methods: list[str],
        variants: list[str],
    ) -> pd.DataFrame:
        """Compare methods on matched target, sample, and experiment rows"""
        rows: list[dict[str, Any]] = []
        rng = np.random.default_rng(20260813)
        key_columns = ["target", "sample_idx", "variant"]
        baseline_rows = frame.loc[
            frame["method"].eq(baseline), key_columns + [metric]
        ].rename(columns={metric: "baseline_score"})
        for variant in variants:
            for method in methods:
                if method == baseline:
                    continue
                method_rows = frame.loc[
                    frame["method"].eq(method), key_columns + [metric]
                ].rename(columns={metric: "method_score"})
                matched = method_rows.merge(
                    baseline_rows,
                    on=key_columns,
                    how="inner",
                    validate="one_to_one",
                )
                matched = matched.loc[matched["variant"].eq(variant)].dropna()
                differences = (
                    matched["method_score"] - matched["baseline_score"]
                ).to_numpy(dtype=float)
                if not len(differences):
                    continue
                low, high = bootstrap_interval(differences, rng=rng)
                rows.append(
                    {
                        "variant": variant,
                        "variant_label": VARIANT_LABELS[variant],
                        "method": method,
                        "method_label": METHOD_LABELS.get(method, method),
                        "mean_difference": float(differences.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "n_pairs": int(len(differences)),
                    }
                )
        return pd.DataFrame(rows)

    def all_metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
        """Build a readable table of every VLM metric"""
        metrics = list(METRIC_LABELS)
        grouped = (
            frame.groupby(
                ["target", "variant", "variant_label", "method", "method_label"],
                dropna=False,
                observed=True,
            )[metrics]
            .mean()
            .reset_index()
        )
        counts = (
            frame.groupby(
                ["target", "variant", "variant_label", "method", "method_label"],
                dropna=False,
                observed=True,
            )
            .size()
            .rename("n")
            .reset_index()
        )
        output = counts.merge(
            grouped,
            on=["target", "variant", "variant_label", "method", "method_label"],
            validate="one_to_one",
        )
        return output[
            [
                "target",
                "variant_label",
                "method_label",
                "n",
                "task_success",
                "match_full_prompt",
                "non_target_preserved",
                "edit_attribute_score",
                "swap_target_attribute_score",
            ]
        ].rename(
            columns={
                "target": "Target",
                "variant_label": "Experiment",
                "method_label": "Method",
                "n": "N",
                "task_success": "Task success",
                "match_full_prompt": "Full-prompt match",
                "non_target_preserved": "Non-target preservation",
                "edit_attribute_score": "Original target presence",
                "swap_target_attribute_score": "Replacement target presence",
            }
        )

    return (
        DATA_PATH,
        METHOD_LABELS,
        METHOD_ORDER,
        METRIC_HELP,
        METRIC_LABELS,
        VARIANT_LABELS,
        VARIANT_ORDER,
        all_metric_summary,
        alt,
        load_vlm_results,
        paired_method_differences,
        pd,
        summarize_metric,
    )


@app.cell
def _(DATA_PATH, load_vlm_results):
    vlm_results = load_vlm_results(DATA_PATH)
    return (vlm_results,)


@app.cell
def _(
    METHOD_LABELS,
    METHOD_ORDER,
    METRIC_LABELS,
    VARIANT_LABELS,
    VARIANT_ORDER,
    mo,
    vlm_results,
):
    target_filter = mo.ui.dropdown(
        options=["All targets"] + sorted(vlm_results["target"].unique().tolist()),
        value="All targets",
        label="Target",
    )
    method_filter = mo.ui.multiselect(
        options={METHOD_LABELS[method]: method for method in METHOD_ORDER},
        value=[METHOD_LABELS[method] for method in METHOD_ORDER],
        label="Methods",
    )
    variant_filter = mo.ui.multiselect(
        options={VARIANT_LABELS[variant]: variant for variant in VARIANT_ORDER},
        value=[VARIANT_LABELS[variant] for variant in VARIANT_ORDER],
        label="Experiments",
    )
    metric_filter = mo.ui.dropdown(
        options={label: metric for metric, label in METRIC_LABELS.items()},
        value=METRIC_LABELS["task_success"],
        label="Metric",
    )
    baseline_filter = mo.ui.dropdown(
        options={METHOD_LABELS[method]: method for method in METHOD_ORDER},
        value=METHOD_LABELS["prompt_only"],
        label="Paired-comparison baseline",
    )
    show_intervals = mo.ui.checkbox(value=True, label="Show 95% bootstrap intervals")

    mo.vstack(
        [
            mo.hstack([target_filter, metric_filter, baseline_filter]),
            mo.hstack([method_filter, variant_filter]),
            show_intervals,
        ]
    )
    return (
        baseline_filter,
        method_filter,
        metric_filter,
        show_intervals,
        target_filter,
        variant_filter,
    )


@app.cell
def _(
    METRIC_HELP,
    METRIC_LABELS,
    method_filter,
    metric_filter,
    mo,
    target_filter,
    variant_filter,
    vlm_results,
):
    target_scoped_results = vlm_results.copy()
    if target_filter.value != "All targets":
        target_scoped_results = target_scoped_results.loc[
            target_scoped_results["target"].eq(target_filter.value)
        ]
    filtered_results = target_scoped_results.loc[
        target_scoped_results["method"].isin(method_filter.value)
        & target_scoped_results["variant"].isin(variant_filter.value)
    ].copy()
    selected_metric = metric_filter.value
    selected_metric_label = METRIC_LABELS[selected_metric]
    metric_guidance = METRIC_HELP[selected_metric]

    mo.md(
        f"**{selected_metric_label}:** {metric_guidance}. "
        f"The current filters include `{len(filtered_results):,}` judgments."
    )
    return (
        filtered_results,
        metric_guidance,
        selected_metric,
        selected_metric_label,
        target_scoped_results,
    )


@app.cell
def _(
    METHOD_LABELS,
    METHOD_ORDER,
    VARIANT_LABELS,
    VARIANT_ORDER,
    alt,
    filtered_results,
    mo,
    pd,
    selected_metric,
    selected_metric_label,
    show_intervals,
    summarize_metric,
):
    overview_summary = summarize_metric(
        filtered_results,
        selected_metric,
        ["method", "method_label", "variant", "variant_label"],
    )
    method_label_order = [METHOD_LABELS[method] for method in METHOD_ORDER]
    variant_label_order = [VARIANT_LABELS[variant] for variant in VARIANT_ORDER]

    overview_base = alt.Chart(overview_summary).encode(
        x=alt.X(
            "method_label:N",
            title=None,
            sort=method_label_order,
            axis=alt.Axis(labelAngle=-35),
        ),
        color=alt.Color(
            "method_label:N",
            title="Method",
            sort=method_label_order,
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("variant_label:N", title="Experiment"),
            alt.Tooltip("method_label:N", title="Method"),
            alt.Tooltip("score:Q", title="Mean", format=".3f"),
            alt.Tooltip("ci_low:Q", title="CI low", format=".3f"),
            alt.Tooltip("ci_high:Q", title="CI high", format=".3f"),
            alt.Tooltip("n:Q", title="N"),
        ],
    )
    overview_bars = overview_base.mark_bar(size=30).encode(
        y=alt.Y(
            "score:Q",
            title=selected_metric_label,
            scale=alt.Scale(domain=[0, 1]),
        )
    )
    overview_errors = overview_base.mark_rule(color="#333", strokeWidth=1.5).encode(
        y=alt.Y("ci_low:Q", title=selected_metric_label, scale=alt.Scale(domain=[0, 1])),
        y2="ci_high:Q",
    )
    overview_layers = (
        overview_bars + overview_errors
        if show_intervals.value
        else overview_bars
    ).properties(width=155, height=310)
    overview_chart = overview_layers.facet(
        column=alt.Column(
            "variant_label:N",
            title=None,
            sort=variant_label_order,
            header=alt.Header(labelFontSize=14),
        )
    )

    if overview_summary.empty:
        winner_table = pd.DataFrame(
            columns=["Experiment", "Best method", "Mean score"]
        )
    else:
        winner_table = (
            overview_summary.sort_values("score", ascending=False)
            .groupby("variant_label", sort=False, as_index=False)
            .first()[["variant_label", "method_label", "score"]]
            .rename(
                columns={
                    "variant_label": "Experiment",
                    "method_label": "Best method",
                    "score": "Mean score",
                }
            )
        )

    mo.vstack(
        [
            mo.md("## Overall method comparison"),
            mo.md(
                "Means are sample-weighted across the selected targets. "
                "Intervals resample judged images within each method and experiment."
            ),
            overview_chart,
            mo.md("### Highest mean among the selected methods"),
            mo.ui.table(winner_table.round(3), page_size=5),
        ]
    )
    return


@app.cell
def _(
    METHOD_LABELS,
    METHOD_ORDER,
    VARIANT_LABELS,
    VARIANT_ORDER,
    alt,
    filtered_results,
    mo,
    selected_metric,
    selected_metric_label,
    summarize_metric,
):
    target_summary = summarize_metric(
        filtered_results,
        selected_metric,
        ["target", "method", "method_label", "variant", "variant_label"],
    )
    target_base = alt.Chart(target_summary).encode(
        x=alt.X(
            "method_label:N",
            title=None,
            sort=[METHOD_LABELS[method] for method in METHOD_ORDER],
            axis=alt.Axis(labelAngle=-35),
        ),
        y=alt.Y("target:N", title="Target"),
    )
    target_rects = target_base.mark_rect().encode(
        color=alt.Color(
            "score:Q",
            title=selected_metric_label,
            scale=alt.Scale(domain=[0, 1], scheme="viridis"),
        ),
        tooltip=[
            alt.Tooltip("target:N", title="Target"),
            alt.Tooltip("variant_label:N", title="Experiment"),
            alt.Tooltip("method_label:N", title="Method"),
            alt.Tooltip("score:Q", title="Mean", format=".3f"),
            alt.Tooltip("n:Q", title="N"),
        ],
    )
    target_text = target_base.mark_text(fontSize=12).encode(
        text=alt.Text("score:Q", format=".2f"),
        color=alt.condition("datum.score > 0.55", alt.value("white"), alt.value("black")),
    )
    target_chart = (target_rects + target_text).facet(
        column=alt.Column(
            "variant_label:N",
            title=None,
            sort=[VARIANT_LABELS[variant] for variant in VARIANT_ORDER],
            header=alt.Header(labelFontSize=14),
        )
    ).properties(title=f"{selected_metric_label} by edited target")

    mo.vstack(
        [
            mo.md("## Target breakdown"),
            mo.md(
                "This view prevents a strong result on one edited concept from hiding "
                "a failure on another. Blank cells have no VLM judgments."
            ),
            target_chart,
        ]
    )
    return


@app.cell
def _(
    METHOD_LABELS,
    METHOD_ORDER,
    VARIANT_LABELS,
    VARIANT_ORDER,
    alt,
    baseline_filter,
    method_filter,
    metric_guidance,
    mo,
    paired_method_differences,
    selected_metric,
    selected_metric_label,
    show_intervals,
    target_scoped_results,
    variant_filter,
):
    paired_results = paired_method_differences(
        target_scoped_results,
        selected_metric,
        baseline_filter.value,
        method_filter.value,
        variant_filter.value,
    )
    paired_base = alt.Chart(paired_results).encode(
        y=alt.Y(
            "method_label:N",
            title=None,
            sort=[METHOD_LABELS[method] for method in METHOD_ORDER],
        ),
        tooltip=[
            alt.Tooltip("variant_label:N", title="Experiment"),
            alt.Tooltip("method_label:N", title="Method"),
            alt.Tooltip("mean_difference:Q", title="Mean difference", format="+.3f"),
            alt.Tooltip("ci_low:Q", title="CI low", format="+.3f"),
            alt.Tooltip("ci_high:Q", title="CI high", format="+.3f"),
            alt.Tooltip("n_pairs:Q", title="Matched pairs"),
        ],
    )
    paired_points = paired_base.mark_point(filled=True, size=85).encode(
        x=alt.X(
            "mean_difference:Q",
            title=f"{selected_metric_label}: method minus baseline",
        ),
        color=alt.Color("method_label:N", title="Method", legend=None),
    )
    paired_intervals = paired_base.mark_rule(strokeWidth=2).encode(
        x=alt.X("ci_low:Q", title=f"{selected_metric_label}: method minus baseline"),
        x2="ci_high:Q",
        color=alt.Color("method_label:N", title="Method", legend=None),
    )
    paired_zero = alt.Chart(paired_results).mark_rule(
        color="#777", strokeDash=[4, 4]
    ).encode(x=alt.datum(0))
    paired_layers = (
        paired_intervals + paired_points + paired_zero
        if show_intervals.value
        else paired_points + paired_zero
    ).properties(width=650, height=95)
    paired_chart = paired_layers.facet(
        row=alt.Row(
            "variant_label:N",
            title=None,
            sort=[VARIANT_LABELS[variant] for variant in VARIANT_ORDER],
            header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=14),
        )
    )
    baseline_label = METHOD_LABELS[baseline_filter.value]

    mo.vstack(
        [
            mo.md("## Paired differences"),
            mo.md(
                f"Each point is a method minus **{baseline_label}** on matched target, "
                f"sample, and experiment rows. {metric_guidance}. The baseline itself "
                "is omitted because its difference is always zero."
            ),
            paired_chart,
            mo.ui.table(paired_results.round(3), page_size=15),
        ]
    )
    return


@app.cell
def _(all_metric_summary, filtered_results, mo):
    complete_summary = all_metric_summary(filtered_results).round(3)
    mo.vstack(
        [
            mo.md("## All aggregate metrics"),
            mo.md(
                "Sort and filter this table to compare the raw VLM outputs. "
                "Task success is the only column whose direction is consistent across "
                "Delete, Swap, and Post."
            ),
            mo.ui.table(complete_summary, page_size=20),
        ]
    )
    return


@app.cell
def _(METHOD_LABELS, VARIANT_LABELS, mo, vlm_results):
    coverage = (
        vlm_results.groupby(["method", "variant"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=list(METHOD_LABELS), columns=list(VARIANT_LABELS), fill_value=0)
        .rename(index=METHOD_LABELS, columns=VARIANT_LABELS)
        .reset_index(names="Method")
    )
    target_coverage = (
        vlm_results.groupby("target", observed=True)
        .agg(
            judgments=("sample_idx", "size"),
            unique_samples=("sample_idx", "nunique"),
            methods=("method", "nunique"),
        )
        .reset_index()
        .rename(
            columns={
                "target": "Target",
                "judgments": "Judgments",
                "unique_samples": "Unique samples",
                "methods": "Methods",
            }
        )
    )
    duplicate_keys = int(
        vlm_results.duplicated(["target", "sample_idx", "method", "variant"]).sum()
    )
    missing_metric_cells = int(
        vlm_results[
            [
                "match_full_prompt",
                "non_target_preserved",
                "edit_attribute_score",
                "swap_target_attribute_score",
            ]
        ].isna().sum().sum()
    )

    mo.vstack(
        [
            mo.md("## Coverage and data checks"),
            mo.md(
                f"Loaded `{len(vlm_results):,}` judgments with `{duplicate_keys}` duplicate "
                f"target/sample/method/experiment keys and `{missing_metric_cells}` missing "
                "raw metric values. GT embed has only Post judgments by design."
            ),
            mo.hstack(
                [
                    mo.ui.table(coverage, page_size=10),
                    mo.ui.table(target_coverage, page_size=10),
                ]
            ),
        ]
    )
    return


@app.cell
def _(filtered_results, mo):
    sample_details = filtered_results[
        [
            "target",
            "sample_idx",
            "variant_label",
            "method_label",
            "task_success",
            "match_full_prompt",
            "non_target_preserved",
            "edit_attribute_score",
            "swap_target_attribute_score",
            "edit_attribute",
            "swap_target_attribute",
            "prompt_judged",
            "notes",
            "benchmark_dir",
            "image",
        ]
    ].rename(
        columns={
            "target": "Target",
            "sample_idx": "Sample",
            "variant_label": "Experiment",
            "method_label": "Method",
            "task_success": "Task success",
            "match_full_prompt": "Full-prompt match",
            "non_target_preserved": "Non-target preservation",
            "edit_attribute_score": "Original target presence",
            "swap_target_attribute_score": "Replacement target presence",
            "edit_attribute": "Original target",
            "swap_target_attribute": "Replacement target",
            "prompt_judged": "Prompt judged",
            "notes": "VLM notes",
            "benchmark_dir": "Benchmark directory",
            "image": "Image",
        }
    )
    numeric_columns = [
        "Task success",
        "Full-prompt match",
        "Non-target preservation",
        "Original target presence",
        "Replacement target presence",
    ]
    sample_details[numeric_columns] = sample_details[numeric_columns].round(3)

    mo.vstack(
        [
            mo.md("## Sample-level judgments"),
            mo.md(
                "Use the table filters to find low scores, compare methods on one sample, "
                "or inspect the judge notes behind an aggregate result."
            ),
            mo.ui.table(sample_details, page_size=15),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
