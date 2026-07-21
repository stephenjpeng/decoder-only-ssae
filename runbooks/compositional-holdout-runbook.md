# Compositional Holdout Runbook

End-to-end steps to generate prompts, extract embeddings, train, and evaluate the SSAE with a compositional (combinatorial) holdout split — i.e. whole property-combination tuples withheld from training, so evaluation tests generalization to unseen combinations rather than a random row-level split.

Normal (non-split) training via `generate_prompts.py` does not hold out any combinations — every prompt goes into training. Use this runbook only when you specifically want a compositional-generalization experiment.

## 1. Generate a disjoint train/holdout prompt split

Skip `generate_prompts.py`. Use `dataset_generation/compositional_split.py`, which enumerates the full Cartesian product of `categories_with_properties.json`, shuffles by seed, and carves out whole tuples for holdout with an explicit disjointness check.

```bash
python dataset_generation/compositional_split.py \
    --categories_json dataset_generation/prompts/input/categories_with_properties.json \
    --output_root results/compositional_split \
    --holdout_fraction 0.1 \
    --seed 0 \
    --properties_same_json dataset_generation/prompts/input/properties_same.json
```

- `--holdout_fraction` and `--n_holdout` are mutually exclusive (pick one).
- `--properties_same_json` is not required by the script but **is required downstream** — see Gotchas below. Pass it here so it's copied into both output folders automatically.
- Optional: `--max_train_prompts` / `--max_holdout_prompts` to cap dataset size after the split (subsampling only shrinks each side; never moves a tuple across the boundary).

Output: `results/compositional_split/train/` and `.../holdout/`, each with `prompts.json` (includes a `tuple_key` field per prompt), `properties.json`, `metadata.json`, `{split}_tuples.json`, and (if passed) `properties_same.json`. Also `results/compositional_split/split_manifest.json` — a disjointness certificate with tuple counts; check `holdout_disjoint_from_train: true` and eyeball `n_train_written`/`n_holdout_written` for sane category coverage, especially if any category has few property values.

## 2. Extract embeddings for both splits separately

Run `get_embeddings.py` twice, once per split, pointing `--out` at the split folders so their `properties.json`/`properties_same.json` are preserved (the script does not clobber an existing `properties.json`):

```bash
python get_embeddings.py --backbone <name> \
    --prompts results/compositional_split/train/prompts.json \
    --out results/compositional_split/train

python get_embeddings.py --backbone <name> \
    --prompts results/compositional_split/holdout/prompts.json \
    --out results/compositional_split/holdout
```

Output layout per folder: `prompts.json`, `properties.json`, `embds/manifest.json`, `embds/embds_<i>/{prompts.txt, <stream>.h5}`.

## 3. Train on the train split only

`training_cli.py` has no `--folder_path` flag — `folder_path` comes from the YAML. Copy `trainings/config/params_default.yaml` and set:

```yaml
dataloader:
  folder_path: "results/compositional_split/train"
```

Do not point `folder_path` at the holdout folder. Then:

```bash
python training_cli.py \
    --output_folder results/my_compositional_run \
    --path_yaml trainings/config/params_default.yaml \
    --overwrite_output True
```

Note `--overwrite_output` is `type=bool` in argparse — any non-empty string is truthy, so pass `True`/`False` literally.

## 4. Copy truncation/normalization sidecars to holdout

Holdout embeddings must use the same top-k truncation and normalization stats as train (`dim_output` is derived from the dataset, not the YAML). Run before any holdout evaluation:

```bash
python -m evaluation.run_copy_truncation \
    --train_folder results/compositional_split/train \
    --holdout_folder results/compositional_split/holdout
```

## 5. Run experiments against the holdout set

**Compositional embedding reconstruction** — predicts each holdout embedding from per-property block means learned during training, decoded through the trained `W`:

```bash
python -m evaluation.run_compositional_embeddings \
    --checkpoint results/my_compositional_run \
    --holdout_folder results/compositional_split/holdout \
    --output_json results/compositional_split/compositional_metrics.json
```

**Baselines** (mean-arithmetic / ridge / PCA) for comparison:

```bash
python -m baselines.run_baselines \
    --checkpoint results/my_compositional_run \
    --train_folder results/compositional_split/train \
    --holdout_folder results/compositional_split/holdout \
    --output_json results/compositional_split/baseline_metrics.json
```

**Image benchmark** (renders SD3 images, scores CLIP/LPIPS/DINO/pixel metrics):

```bash
python -m evaluation.run_image_benchmark \
    --checkpoint results/my_compositional_run \
    --holdout_folder results/compositional_split/holdout \
    --output_dir results/bench_out \
    --dino --locality_drop_one_attr --locality_swap_one_attr
```

Non-SSAE baselines (`gt_embed`, `mean_arithmetic`, `ridge_embed`, `prompt_only`) land in a shared cache at `results/bench_baseline_cache/<dataset_id>/`, keyed by (holdout `prompts.json`, training embeddings + mask, `--base_seed`, `--ridge_lambda`, SD3.5 pipeline fingerprint). The run folder holds SSAE outputs plus a `manifest.json` pointing at the cache. Comparing several checkpoints on the same holdout only re-renders SSAE:

```bash
# First run — populates baselines and SSAE-A
python -m evaluation.run_image_benchmark \
    --checkpoint results/ssae_A --holdout_folder results/compositional_split/holdout \
    --output_dir results/bench_out/ssae_A --dino --locality_drop_one_attr

# Second run — hits the cache; only ssae_compose is re-rendered
python -m evaluation.run_image_benchmark \
    --checkpoint results/ssae_B --holdout_folder results/compositional_split/holdout \
    --output_dir results/bench_out/ssae_B --dino --locality_drop_one_attr
```

Pre-warm the cache without any SSAE work via `--baselines_only`. Turn caching off entirely (legacy layout) with `--no_baseline_cache`. Locality flags grow an existing cache in place; changing `--ridge_lambda` or `--base_seed` (or any other key component) creates a fresh `dataset_id`. The MA/ridge fits also depend on the checkpoint's *training* data, so checkpoints trained on different splits do not share these baselines even against the same holdout. Optional per-row metrics (LPIPS, DINO) are captured at population time — if you'll want DINO, populate the cache with `--dino` set once.

For VRAM-constrained runs (typically untruncated `W`, which is multi-GB and OOMs when colocated with SD3.5), pass `--ssae_device cpu` to move the SSAE decoder off the SD3 device, and `--baseline_device cpu` to keep ridge/mean-arithmetic per-sample prediction off GPU as well. `--baseline_device` defaults to `--ssae_device`. `--ssae_device` is also accepted by `evaluation.run_compositional_embeddings` (Step 5, first block).

Both locality flags are independent and can be combined. For each sample with more than one active attribute, one active attribute is picked at random (seeded by `base_seed + idx` so the pick is reproducible), and the two flags each apply a different edit to that same attribute:

- `--locality_drop_one_attr`: zeros the attribute's mask bit for the embedding methods, or drops its phrase from the prompt for `prompt_only`. Writes pre-edit images to `<output_dir>/images_pre_edit/<method>/` and adds `mse_pixel_pre_post_edit` / `ssim_pre_post_edit` per sample (surgical-ness under removal), plus `clip_image_vs_residual_prompt` on the normal image.
- `--locality_swap_one_attr`: flips the attribute's mask bit to a random *different* property in the same category (e.g. blond → brunette), or substitutes the corresponding phrase in the prompt for `prompt_only`. Writes swap images to `<output_dir>/images_swapped/<method>/` and adds `mse_pixel_swap_vs_normal` / `ssim_swap_vs_normal` (surgical-ness under value swap) plus `clip_swap_image_vs_swapped_prompt` (did the swap image match the swapped prompt).
- Both flags share the same `edit_pid` per sample, so drop and swap results are directly comparable when run together. The chosen attribute, its phrase, and the swap target (if any) are recorded per row in `per_sample.csv` as `edit_pid`, `edit_attribute`, `swap_target_pid`, `swap_target_attribute`, `swapped_prompt`.

Categories with only one property (there's nothing to swap to) fall out of the swap test for those samples; they're still counted in the drop test.

Browse benchmark output side-by-side with the streamlit viewer:

```bash
streamlit run viewer/app.py
```

Sidebar defaults point at `results/compositional_split/holdout` and auto-discover benchmark runs under `results/`; override or add more via the sidebar inputs. Reads `per_sample.csv` / `summary.json` / `images/` directly, works on in-progress runs, and supports method filtering, image-variant switching (normal / pre-edit / swapped), aggregate plots, and an edit-details panel. Cache-backed runs (with a `manifest.json` pointing at a shared baseline cache) are handled transparently — baseline images and rows are pulled from the cache directory. The baseline cache directories themselves are excluded from run discovery. Moving or copying a run folder without its cache will lose the baseline images.

**Concept-strength / magnitude sensitivity** (works on either split; general diagnostic, not compositional-specific):

```bash
python -m evaluation.run_magnitude_sensitivity \
    --checkpoint results/my_compositional_run \
    --data_folder results/compositional_split/holdout \
    --output_dir results/mag_out \
    --concepts "holding a gun" \
    --magnitudes -10,-2,-1,0,1,2,5,10
```

The same table (with current defaults for `ridge_lambda`, `n_bootstrap`, etc.) is also in `README.md` under "Evaluation & benchmarks" — check there for anything that's drifted since this runbook was written.

## Gotchas

- **Missing `properties_same.json` at train time.** `H5Dataset` always constructs `SameId`, which unconditionally requires `<folder_path>/properties_same.json` (`trainings/dataloader/properties/same_id.py:101`) and hard-checks its keys match your categories exactly. `compositional_split.py` only writes this file if `--properties_same_json` is passed — otherwise training fails with `FileNotFoundError`. Fix: pass `--properties_same_json` when running the split (recommended, see Step 1), or copy an existing one manually into both `train/` and `holdout/`:
  ```bash
  cp dataset_generation/prompts/input/properties_same.json results/compositional_split/train/properties_same.json
  cp dataset_generation/prompts/input/properties_same.json results/compositional_split/holdout/properties_same.json
  ```
  An all-`false` example matching the current `categories_with_properties.json` keys (`hair`, `eyes`, `situation`, `t_shirt`, `hat`, `action`, `pose`) is checked in at `dataset_generation/prompts/input/properties_same.json` — `false` per category is a safe default meaning "never treat any prompt as same-id," for when you're not using the same-id feature. Only reuse it as-is if your categories match those keys exactly; otherwise regenerate it with the same schema (one key per category, `false` or a list of "same" property names — see `trainings/dataloader/properties/same_id.py` for how non-`false` values are consumed).
- **Missing embeddings manifest.** `MissingManifestError: No manifest.json at .../embds/manifest.json` means embeddings haven't been extracted yet for that folder, or `--out` pointed somewhere other than where `training_cli.py`'s YAML `folder_path` expects. Re-run `get_embeddings.py` with `--out` matching the YAML's `dataloader.folder_path`.
- **Never train on the holdout folder.** There is no code-level guard against this — it's enforced only by convention (point `folder_path` at `train/`, never `holdout/`).

## Interpreting `run_compositional_embeddings.py` output

Output is one JSON dict: `checkpoint`, `holdout_folder`, `model_name`, `n_holdout`, `mse_mean`, `cosine_mean`, then two large per-sample arrays `per_index_mse`/`per_index_cosine`. The summary scalars print *before* the arrays — if a run appears to have "no summary stats," it's almost always the per-sample arrays flooding terminal scrollback, not a missing value. Isolate the scalars directly:

```bash
python3 -c "import json; d=json.load(open('results/compositional_split/compositional_metrics.json')); print({k: d[k] for k in ('mse_mean','cosine_mean','n_holdout')})"
```

## Interpreting `run_baselines.py` output

```json
{
  "train_folder": "...",
  "holdout_folder": "...",
  "mean_arithmetic": {"mse_mean": ..., "cosine_mean": ...},
  "ridge": {"mse_mean": ..., "cosine_mean": ..., "lambda": ...},
  "pca": {"mse_mean": ..., "cosine_mean": ..., "k": ...}
}
```

| Baseline | What it computes | How to read it |
|---|---|---|
| `mean_arithmetic` | Global mean + independently-fit per-property delta vectors (present-mean minus absent-mean), summed over active properties. | Simplest additive hypothesis: property effects are independent and just add. A floor, not a competitor. |
| `ridge` | Single ridge regression jointly fit from the binary property mask (+ intercept) to the embedding, on train; applied to holdout masks. | Still linear in the mask, but jointly fit rather than independently per property — a meaningfully stronger linear baseline than mean-arithmetic. |
| `pca` | Unsupervised PCA basis fit on train embeddings, then the **true holdout embedding** is projected onto that basis and reconstructed — it never uses property masks and it "sees" the answer. | Not a fair prediction baseline. It's a floor: the best reconstruction error achievable by any k-dim linear subspace given perfect knowledge of the target. Useful only as a reference ceiling on achievable performance, not a thing to "beat" in the usual sense. |

Metrics: `mse_mean` (lower is better, scale depends on embedding normalization) and `cosine_mean` (higher is better, robust to overall scale).

**Comparing the SSAE against these baselines is not a clean apples-to-apples test of expressiveness**, because of how `evaluation/composition.py` builds holdout predictions for `model_trainable_inputs`: property block means are averaged marginally per single property (never per co-occurring tuple), placed into disjoint per-property feature slices, then passed through a pointwise `ReLU` and a single shared linear decoder. Since ReLU is pointwise and blocks never overlap, the result collapses algebraically to `const + sum over active properties of (per-property vector)` — structurally identical to `mean_arithmetic`. **Note:** this collapse argument assumes the historical single-`Linear` decoder head (`model.num_layers = 1`). With `num_layers > 1` the head is a nonlinear MLP and no longer decomposes additively across property blocks, so this evaluation path becomes a strictly more expressive predictor of holdout compositions (and the tie with `ridge` is no longer forced by construction). This means:

- A near-tie between SSAE and `ridge` does **not** imply the two models are equivalent in general. It means the compositional-holdout evaluation method itself is incapable of expressing property interactions for unseen combinations, regardless of what the trained `Y`/`W` actually encode for combinations seen during training.
- To check whether the SSAE captures real interaction structure that this eval discards, compare **training-set reconstruction** (using actual per-prompt `Y` rows, not block means) against `ridge`/`mean_arithmetic` fit on the same train data. If the SSAE clearly wins there, the interaction structure exists but doesn't currently transfer to holdout composition via this method.
- A genuinely more informative compositional predictor would use pairwise (or higher-order) co-occurrence means where available in train, rather than only single-property marginals — not currently implemented in `evaluation/composition.py`.
