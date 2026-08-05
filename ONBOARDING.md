# Onboarding: decoder-only-ssae

You're taking over as primary contributor. This doc is the map: what the code does, where the pieces live, what's on other branches, and what to watch out for. Read `README.md` for the paper-level pitch; this file is for the engineering hand-off.

## 1. What the project is

An implementation of **decoder-only Supervised Sparse Auto-Encoders (SSAEs)** for the SD3.5 T5 text-encoder embedding space. The model learns:

- a sparse feature matrix `Y ∈ R^{n_prompts × n_features}` (one row per training prompt), and
- a linear decoder `W` such that `x̂_i = W · σ(M_i ⊙ Y_i) + b`

where `M` is a **known** binary mask derived from the category/property structure of each prompt. Because features are pinned to properties via `M`, you can edit `Y`/`M` at inference and get semantically-targeted embedding edits, which then produce edited images through SD3.5.

There is no encoder. `Y` is a `nn.Parameter` (or `nn.Embedding` in the avg-feature variant) that is directly optimized against T5 targets.

## 2. Repo layout at a glance

```
generate_prompts.py                   entry: cartesian product of properties -> prompts.json
get_embeddings_large_turbo_many_h5.py entry: run SD3.5 T5 encoder, write per-prompt H5 files
training_cli.py                       thin argparse wrapper
trainable_inputs_all_clips.py         real training loop
dataset_generation/                   PromptsGenerator + category/property JSON
trainings/
  config/       config.py (YAML flattener + DI helper), params_default.yaml
  dataloader/   H5Dataset, Properties, SameId, mask construction, top-k truncation, MAX_MIN norm
  models/       three decoder variants + import_model registry
  utils/        Logger, LRScheduler, seed util, clip helper
inference/      SFDInference base + per-model subclasses, image_generator, notebook
config/         models.yaml (mostly unused)
docs/           GitHub Pages site (gh-pages via `github-page` branch merged to main)
```

## 3. The pipeline (what runs, in order)

```
categories_with_properties.json
        │  generate_prompts.py
        ▼
prompts.json + properties.json + metadata.json
        │  get_embeddings_large_turbo_many_h5.py (SD3.5, NF4-quantized T5)
        ▼
<folder>/embds/embds_<i>/{embds.h5, embds_pooled.h5, prompts.txt}
        │  training_cli.py -> trainable_inputs_all_clips.training(...)
        ▼
results/<run>/{model.pt, params.yaml, all_params.yaml, logs, plots}
        │  inference/notebooks/inference_and_testing_output_visuals.ipynb
        ▼
edited embeddings -> SD3.5 image generator -> PNGs
```

**Important:** `training_cli.py --overwrite_output` uses `argparse type=bool`, meaning any non-empty string is truthy. Pass the literal `True`/`False` shown in the README, or fix this properly with `action="store_true"`.

## 4. How training is wired (the one pattern to internalize)

Everything runs through `trainings/config/config.py::initialise_instance(Cls, tp)`:

```python
def initialise_instance(model, all_possible_args):
    sig = inspect.signature(model)
    args_model = list(sig.parameters.keys())
    filtered_kwargs = {k: v for k, v in all_possible_args.items() if k in args_model}
    return model(**filtered_kwargs)
```

The YAML is flattened into a single dict `tp` (grouping keys like `model.*`, `dataloader.*` are collapsed — **all leaf keys must be globally unique**). Every component (`H5Dataset`, `Decoder`, `LRScheduler`) is constructed by picking the subset of `tp` that matches its `__init__` signature.

Consequences:

- To add a new hyperparameter: add it to the YAML **and** add a same-named `__init__` arg on the consuming class. No other plumbing.
- **Typos are silent.** A misspelled `__init__` arg or YAML key falls back to the default; there is no validation. When something behaves unexpectedly, verify the arg name is actually being consumed.
- Derived shapes (`n_features`, `n_properties`, `tid_same`, `dim_output`, `n_prompts`) are written back into `tp` after `H5Dataset` is built, so the decoder sees them at construction time. See `trainable_inputs_all_clips.py:42-53` and the identical block in `inference/abstract.py:35-44` — **these are duplicated and must stay in sync.**

Training loop (`trainable_inputs_all_clips.py`):
- `decoder(batch_size, batch_idx)` — the decoder holds `Y` internally and slices by index; it does **not** take embeddings as input.
- MSE against the H5 target embedding, Adam, optional `LRScheduler`.
- Saves `model.pt` and `all_params.yaml` (the resolved `tp`).

## 5. Decoder variants (`trainings/models/`)

| File | Backing store for Y | Notes |
|---|---|---|
| `model_trainable_inputs.py` | `nn.Parameter(n_prompts, n_features)` | Most flexible. Uses `y_same` for shared-across-prompts features via `tid_same` indices. |
| `model_avg_feature.py` | `nn.Embedding` keyed by property id | Parameter-efficient; scales to more prompts. |
| `model_trainable_input_inv.py` | Param `Y` only; `W = (XᵀX + λI)⁻¹ XᵀY` closed-form | **Not registered in `import_model` — unreachable from the CLI.** |

`import_model` in `trainings/models/utils.py`:

```python
if model_name == "model_trainable_inputs":  ...
elif model_name == "model_avg_feature":     ...
else: raise Exception("model name do not exist.")
```

If you want to use the inverse variant, add the branch. Similarly for any new decoder.

All variants must implement `apply_mask(mask_reduced, batch_size)`, `forward(batch_size, batch_idx)`, and `get_rank_Y()`.

The decoder head is built by `trainings/models/mlp.py::build_head`. With `model.num_layers == 1` (default) it returns a plain `nn.Linear`, so old checkpoints keep the `linear.weight` / `linear.bias` keys. With `num_layers > 1` it returns an `nn.Sequential` of alternating `Linear`/activation blocks and expects `model.hidden_dims` (int broadcast, or list of length `num_layers - 1`). Both knobs are exposed on `training_cli.py` as `--num_layers` / `--hidden_dims` and land in `all_params.yaml`, so inference reconstructs the same shape without extra plumbing.

## 6. Data & masks

`trainings/dataloader/dataloader.py::H5Dataset` does more than you'd expect from the name:

1. Discovers `embds/embds_*/` folders and sorts numerically.
2. Detects whether `embds.h5` and/or `embds_pooled.h5` exist (the log messages here are misleading — the code uses whichever it finds, not "only pooled").
3. Builds `properties` (`Properties`) and `same_id` (`SameId`) helpers, which produce the per-prompt binary mask `mask_reduced` from `prompts.json` + `properties.json`.
4. Caches `indices_top_<k>.json` (top-k dims by max−min variance) and `embds_max/min_top_<k>.json` for `MAX_MIN` normalization. **Delete these files if you regenerate embeddings or change `truncate_embds_topk` semantics — they're stale-by-default.**
5. `denormalize()` is the inverse used at inference before feeding SD3.5.

`inference/abstract.py::SFDInference.overwrite_full_embedding` hard-codes SD3.5-specific shapes: `1×333×4096` for T5, `2048` for pooled CLIP, split on `[:-2048]` / `[-2048:]`. If the encoder or sequence length changes, this breaks.

## 7. Branches and current state

```
main                                                688f6d9  merge github-page
origin/github-page                                  3f575ce  adds docs/ (already merged)
origin/feature/additional-experiments-and-quant...  bea900e  large new subsystem, NOT merged
```

### `feature/additional-experiments-and-quantitative-metrics` (unmerged, ~2.9k LOC)

This is where the real research work is queued up. It adds three new top-level packages and touches `inference/image_generation/image_generator.py` and `requirements.txt`. Roughly:

- `dataset_generation/compositional_split.py` — disjoint train/holdout **full-tuple** split (compositional generalisation).
- `baselines/`
  - `run_baselines.py` — mean / ridge / PCA baselines against SSAE reconstruction.
  - `run_unsup_ae_holdout.py`, `run_unsup_sae_holdout.py` — unsupervised (sparse) autoencoders.
  - `unsup_ae.py`, `unsup_sparse_ae.py` — the models.
- `evaluation/`
  - `cli.py` — unified subcommand entry point (`python -m evaluation.cli ...`).
  - Embedding-space: `run_reconstruction.py`, `run_compositional_embeddings.py`, `decorrelation.py`, `run_decorrelation_plot.py`, `run_copy_truncation.py`.
  - Image-space: `run_image_benchmark.py` (SD3 + CLIP + LPIPS, optional DINOv2 and locality CLIP), `clip_scorer.py`, `clip_metrics.py`, `lpips_metric.py`, `dino_embed.py`.
  - Probes: `clip_linear_probe.py`, `run_clip_probe.py`.
  - VLM judge: `vlm_openai.py`, `run_vlm_openai_batch.py` (needs `OPENAI_API_KEY`).
  - Aggregation: `run_edit_type_summary.py`, `composition.py`, `bootstrap.py`, `locality.py`, `tensor_metrics.py`, `dataset_artifacts.py`, `sd3_pack.py`, `io.py`.
- `experiments/sweep_generate.py` — YAML grid generator for ablations.

The README on this branch documents the full "Evaluation & benchmarks" workflow. **You should probably plan to merge this early** — it's the natural next step and future PRs will conflict badly with it if it stays out.

There are no open PRs (only `#1` for github-page, already merged).

## 8. Dependencies and environment

- Python 3.10+, PyTorch 2.5.1 + CUDA 12.1 wheels pinned in `requirements.txt`.
- SD3.5 with NF4 T5 (via `bitsandbytes`); embedding extraction is GPU-bound.
- `xformers==0.0.28.post3` is pinned — brittle against Torch/CUDA upgrades.
- No test suite, no lint config, no CI. `black==25.1.0` is in requirements but not enforced.

## 9. How to contribute — practical checklist

1. **Set up a run end-to-end on a small config first.** Set `dataloader.simulated.simulated: True` in YAML to skip the H5 loading and iterate on model/training code in seconds.
2. **When adding a param**: add to YAML → add `__init__` arg with the same name → done. Grep for the key name to confirm it's actually consumed (silent-drop is the #1 footgun).
3. **When adding a decoder variant**: implement `apply_mask` / `forward(batch_size, batch_idx)` / `get_rank_Y`, then wire it into `trainings/models/utils.py::import_model`, then add a matching `inference/inference_model_<name>.py` subclassing `SFDInference`.
4. **Watch the two duplicated `tp`-derivation blocks** (`trainable_inputs_all_clips.py` and `inference/abstract.py`). If you add derived shapes to one, add them to the other. Extracting this into a shared helper is a good early cleanup.
5. **Regenerating embeddings**: delete the cached `indices_top_*.json` and `embds_max/min_*.json` in the prompts folder, otherwise the dataset will silently use stale top-k indices/normalization constants.
6. **Merging the evaluation branch**: rebase `feature/additional-experiments-and-quantitative-metrics` onto `main` (only `README.md`, `requirements.txt`, and `inference/image_generation/image_generator.py` should conflict). Do this before starting new work if possible.
7. **Follow existing style.** The codebase uses flat function-per-file modules, single-line comments, dataclass-free config-dicts, and `logger.print` for narration. Don't reflexively introduce Pydantic/Hydra/etc.

## 10. Known rough edges (candidate first PRs)

- `argparse type=bool` in `training_cli.py` — replace with `action="store_true"`.
- `import_model` doesn't register `model_trainable_input_inv` — either wire it up or delete the file.
- Duplicated `tp`-derivation between training and inference — extract.
- No tests. A single smoke test that trains for 1 epoch on `simulated=True` would prevent a lot of silent breakage from the `initialise_instance` pattern.
- `H5Dataset` is a 300-line class doing discovery, mask construction, caching, and normalization. Splitting caching/normalization out would help.
- SD3.5-specific magic numbers (`333`, `4096`, `2048`) in `inference/abstract.py::overwrite_full_embedding` should live next to the embedding extractor.
- `config/models.yaml` is present but doesn't appear to be referenced — verify and delete if dead.
- README + top-level scripts still have `dataset_generation_ouns/` and `prompts/your_directory/` placeholder paths; standardize.
