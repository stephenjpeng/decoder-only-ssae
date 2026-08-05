# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Decoder-only Supervised Sparse Auto-Encoders (SSAEs) for the SD3.5 T5 text-encoder embedding space. The pipeline is: define categories/properties → generate combinatorial prompts → extract T5 embeddings (H5) → train a decoder-only SSAE (learns sparse feature matrix `Y` + linear decoder `W`) → edit sparse features and reconstruct embeddings for image generation.

## Common commands

```bash
# Install (CUDA 12.1 wheels pinned in requirements.txt)
pip install -r requirements.txt

# 1. Generate prompts from dataset_generation/prompts/input/categories_with_properties.json
python generate_prompts.py

# 2. Extract T5 + pooled CLIP embeddings via SD3.5 (NF4-quantized). Configure paths at top of script.
python get_embeddings_large_turbo_many_h5.py

# 3. Train
python training_cli.py \
    --output_folder results/my_run \
    --path_yaml trainings/config/params_default.yaml \
    --overwrite_output True

# 4. Inference / image generation
jupyter notebook inference/notebooks/inference_and_testing_output_visuals.ipynb
```

There is no test suite, linter config, or CI in this repo.

## Architecture

### Pipeline stages
1. **Prompt generation** (`generate_prompts.py` → `dataset_generation/functions.py::PromptsGenerator`): cartesian product of properties across categories, shuffled and truncated. Emits `prompts.json` + `metadata.json`.
2. **Embedding extraction** (`get_embeddings_large_turbo_many_h5.py`): loads SD3.5 with NF4-quantized T5 encoder, writes one folder per prompt containing `embds.h5` (T5), `embds_pooled.h5` (CLIP), `prompts.txt`.
3. **Training** (`training_cli.py` → `trainable_inputs_all_clips.py::training`): the CLI is a thin wrapper. The real orchestration lives in `trainable_inputs_all_clips.training`.
4. **Inference** (`inference/`): `SFDInference` base + per-model subclasses (`inference_model_avg`, `inference_model_trainable_inputs`) load a trained checkpoint and expose `search_idx_prompt`, `get_x`, `overwrite_full_embedding`. `image_generation/image_generator.py` wraps SD3.5 for `generate_image_from_prompt` / `generate_image_from_embd`.

### Training data flow
`training()` in `trainable_inputs_all_clips.py`:
1. Reads YAML via `trainings/config/config.py::read_training_params_from_yaml`; every subsequent component is built with `initialise_instance(Cls, tp)`, which injects only the kwargs from `tp` that `Cls.__init__` accepts. This is the central pattern — any new component must declare its dependencies via `__init__` args matching keys in `tp`.
2. Builds `H5Dataset` (`trainings/dataloader/dataloader.py`). The dataset exposes `properties` (category/property indexing), `same_id` (shared-across-prompts features), `mask_reduced` (binary mask `M`), `dim_x`, and normalization.
3. Derives derived shapes (`n_features = n_properties * n_repeat`, `tid_same`, etc.) and stores them back into `tp` so downstream constructors see them.
4. Selects the decoder variant via `trainings/models/utils.py::import_model(model_name)`. **Note:** this registry currently maps only `model_trainable_inputs` and `model_avg_feature`; `model_trainable_input_inv` exists as a file but is not wired in.
5. Trains with Adam + optional `LRScheduler` (`trainings/utils/learning_rate_scheduler.py`). Loss is MSE between `decoder(batch_size, batch_idx)` and target embedding `y`. The decoder holds the entire trainable input `Y` internally — it takes indices, not inputs.

### Decoder variants (`trainings/models/`)
- `model_trainable_inputs.py`: `Y` is a full `nn.Parameter` matrix (one row per prompt). Supports `y_same` shared features. Most flexible.
- `model_avg_feature.py`: `Y` built from an `nn.Embedding` keyed by property id; more parameter-efficient.
- `model_trainable_input_inv.py`: learns only `Y`; `W` is closed-form ridge-regressed against `X`. **Not registered in `import_model`.**

All variants implement `apply_mask(mask, batch_size)`, `get_rank_Y()`, and a forward that reconstructs `x_hat = W · σ(M ⊙ Y_i) + b`.

The `W` head is built by `trainings/models/mlp.py::build_head` and controlled by `model.num_layers` (default `1`) and `model.hidden_dims` (int broadcast or list of length `num_layers - 1`). `num_layers=1` keeps a bare `nn.Linear` under `self.linear`, so state_dicts stay backward-compatible; `num_layers>1` swaps in an `nn.Sequential` of alternating `Linear`/activation blocks using the variant's own activation. Both are exposed as `--num_layers` / `--hidden_dims` on `training_cli.py` and override the YAML when set.

### Config surface
`trainings/config/params_default.yaml` is flat-ish but grouped (`model.*`, `dataloader.*`, `training.*`, `sparse_feature_design.*`). `read_training_params_from_yaml` flattens it into a single `tp` dict, so all keys must be globally unique. See README §Configuration Reference for individual parameters.

## Gotchas
- `training_cli.py --overwrite_output` uses `type=bool`, which in argparse means any non-empty string is truthy. Pass `True`/`False` verbatim as shown.
- Embeddings are loaded via `truncate_embds_topk` (top-k dims by variance) — `dim_output` is derived from the dataset, not the YAML.
- The `initialise_instance` pattern silently ignores unrecognized `tp` keys; a typo in a constructor arg name will not error, it will just use the default.
- `model_trainable_input_inv` is present but unreachable via `training_cli.py` without extending `import_model`.
