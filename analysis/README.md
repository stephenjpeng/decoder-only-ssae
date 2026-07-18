# Analysis scripts

Standalone diagnostics that do not modify the training pipeline. Both CLIs
load embeddings via `H5Dataset` with `truncate_embds_topk=None` and write a
two-panel figure plus a JSON of recommended `k` values into the dataset
folder.

## `elbow_topk` -- picking `truncate_embds_topk` from the range curve

Ranks embedding dims by `max(X) - min(X)` (the same criterion the training
pipeline already uses) and locates the elbow via three detectors: kneedle
(perpendicular distance to the endpoint chord), cumulative-explained-range
at threshold, and smoothed second-derivative peak.

```bash
python -m analysis.elbow_topk \
    --folder_path results/compositional_split/train/ \
    --threshold 0.95
```

## `elbow_pca` -- PCA-elbow variant (diagnostic only)

Runs the same three detectors on the PCA spectrum of the centred embedding
matrix (computed via the Gram trick). Compare the resulting `k` to the
range-based cutoff to see whether PCA truncation would be dramatically more
efficient. See `PROPOSAL.md` for the semantics discussion around actually
adopting PCA truncation.

```bash
python -m analysis.elbow_pca \
    --folder_path results/mps_run/ \
    --threshold 0.95
```

Tests for the three detectors live in `tests/test_elbow_detectors.py`.
