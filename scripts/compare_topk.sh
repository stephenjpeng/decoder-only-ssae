#!/usr/bin/env bash
# Sweep SSAE performance across truncate_embds_topk and num_layers.
#
# Trains the cross product TopK x Layers. For num_layers > 1, each hidden
# layer is hidden_dim wide (build_head broadcasts the single int).
#
# Pipeline:
#   1. (Optional) Regenerate the compositional split for train/ + holdout/.
#   2. (Optional) Extract SD3.5 text embeddings for both splits.
#   3. For each (topk, layers) pair:
#        - Write a per-run YAML with truncate_embds_topk = k and num_layers = L.
#        - Train the SSAE (unless --skip-training).
#        - Copy the training folder's truncation sidecars to holdout/.
#        - Run evaluation.run_compositional_embeddings; capture mse/cosine.
#        - Optionally run evaluation.run_image_benchmark (--run-image-benchmark).
#        - Optionally run evaluation.run_reconstruction (--run-train-reconstruction).
#   4. Aggregate all runs into <runs-root>/sweep_summary.csv.
#
# Existing splits and embeddings are reused unless the corresponding
# --recreate-* flag is set. Training always runs unless --skip-training is
# passed (useful with --run-image-benchmark for a benchmarks-only pass over
# previously trained checkpoints).
#
# The "no interaction terms" ablation is --head-type block_diagonal: each
# property block gets its own MLP and outputs are summed, forbidding
# cross-property mixing while preserving within-block depth.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/compare_topk.sh [options]

Sweep options:
  --topk LIST                Comma-separated TopK values. Default: 500,1000,2000,5000
  --layers LIST              Comma-separated head depths (>=1). Default: 1
  --seeds LIST               Comma-separated training seeds (E10). Default: 0
                             Seed 0 keeps the historical run tag; others get _s<seed>.
  --save-model-frequency N   Save a checkpoint every N epochs into <run>/checkpoints/
                             (E11 training-dynamics trajectories). Default: off.
  --num-workers N            DataLoader workers. Default: 1. Training at top-k=100k is
                             dataloader-bound (each item reads a 1.36M-dim float16 vector,
                             casts to float32 and fancy-indexes it), so a single worker
                             saturates one core while the GPU idles. 6 on an 8-vCPU box
                             cuts epoch time ~5x.
  --hidden-dim N             Width per hidden layer when layers > 1. Default: 1024
  --head-type TYPE           dense (default) or block_diagonal. block_diagonal ablates
                             cross-property mixing (independent MLP per property block).
  --pca-rotation             Rotate embeddings into a top-k PCA basis before truncation.

Split / data:
  --repo-root PATH           Repo root. Default: current dir.
  --categories PATH          categories_with_properties.json (relative or absolute).
  --properties-same PATH     properties_same.json (relative or absolute).
  --split-root PATH          Directory holding train/ and holdout/. Default: results/compositional_split
  --runs-root PATH           Root for per-run checkpoints + summary. Default: results/topk_sweep
  --holdout-fraction FLOAT   Fraction reserved as holdout. Default: 0.1
  --max-train-prompts N      Cap train prompts (0 = no cap). Default: 0
  --max-holdout-prompts N    Cap holdout prompts (0 = no cap). Default: 0
  --backbone NAME            Backbone in backbones/. Default: sd35_turbo_text_only
  --recreate-split           Rebuild the compositional split.
  --recreate-embeddings      Re-extract embeddings.

Training / eval toggles:
  --skip-training            Reuse existing checkpoints; only run eval / bench.
  --run-image-benchmark      Run evaluation.run_image_benchmark after each run.
  --run-train-reconstruction Run evaluation.run_reconstruction after each run.
  --rescore-locality         Run evaluation.rescore_locality once after the sweep.

Image benchmark:
  --bench-root PATH          Root for bench output. Default: results/bench_out
  --baseline-cache-root PATH Shared per-dataset baseline cache. Default: results/bench_baseline_cache
  --benchmark-max-samples N  Cap holdout samples per method (0 = no cap). Default: 0
  --benchmark-sd-device DEV  SD3.5 device. Default: cuda
  --benchmark-ssae-device DEV      SSAE decoder device. Default: same as sd device.
  --benchmark-baseline-device DEV  Ridge / mean-arith device. Default: same as ssae device.
  --locality-drop            Enable drop-one-attribute locality test.
  --locality-swap            Enable value-swap locality test.
  --benchmark-dino           Compute DINOv2 similarity vs GT-embed reference.
  --benchmark-simulated      Skip real SD3 rendering; write placeholder PNGs.

Other:
  --python BIN               Python interpreter. Default: python
  -h, --help                 Show this message.

Examples:
  # Default: 4 topk x 1 layer count = 4 runs
  scripts/compare_topk.sh

  # Sweep + image benchmark, drop-one locality
  scripts/compare_topk.sh --layers 1,2 --hidden-dim 1024 --run-image-benchmark --locality-drop

  # Ablate cross-property mixing at a single topk
  scripts/compare_topk.sh --topk 2000 --layers 2 --hidden-dim 2048 --head-type block_diagonal

  # Benchmark-only pass over existing checkpoints
  scripts/compare_topk.sh --topk 500,1000,2000,5000 --layers 1,2 --skip-training --run-image-benchmark
USAGE
}

# defaults
repo_root="$(pwd)"
categories="dataset_generation/prompts/input/categories_with_properties.json"
properties_same="dataset_generation/prompts/input/properties_same.json"
split_root="results/compositional_split"
runs_root="results/topk_sweep"
topk_csv="500,1000,2000,5000"
layers_csv="1"
seeds_csv="0"
save_model_frequency=""
num_workers=1
hidden_dim=1024
head_type="dense"
pca_rotation=0
holdout_fraction=0.1
max_train_prompts=0
max_holdout_prompts=0
backbone="sd35_turbo_text_only"
recreate_split=0
recreate_embeddings=0
skip_training=0
resume=0
run_image_benchmark=0
run_train_reconstruction=0
rescore_locality=0
bench_root="results/bench_out"
baseline_cache_root="results/bench_baseline_cache"
benchmark_max_samples=0
benchmark_sd_device="cuda"
benchmark_ssae_device=""
benchmark_baseline_device=""
locality_drop=0
locality_swap=0
benchmark_dino=0
benchmark_simulated=0
python_bin="python"

# argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-root)                repo_root="$2"; shift 2 ;;
        --categories)               categories="$2"; shift 2 ;;
        --properties-same)          properties_same="$2"; shift 2 ;;
        --split-root)               split_root="$2"; shift 2 ;;
        --runs-root)                runs_root="$2"; shift 2 ;;
        --topk)                     topk_csv="$2"; shift 2 ;;
        --layers)                   layers_csv="$2"; shift 2 ;;
        --seeds)                    seeds_csv="$2"; shift 2 ;;
        --save-model-frequency)     save_model_frequency="$2"; shift 2 ;;
        --num-workers)              num_workers="$2"; shift 2 ;;
        --hidden-dim)               hidden_dim="$2"; shift 2 ;;
        --head-type)                head_type="$2"; shift 2 ;;
        --pca-rotation)             pca_rotation=1; shift ;;
        --holdout-fraction)         holdout_fraction="$2"; shift 2 ;;
        --max-train-prompts)        max_train_prompts="$2"; shift 2 ;;
        --max-holdout-prompts)      max_holdout_prompts="$2"; shift 2 ;;
        --backbone)                 backbone="$2"; shift 2 ;;
        --recreate-split)           recreate_split=1; shift ;;
        --recreate-embeddings)      recreate_embeddings=1; shift ;;
        --skip-training)            skip_training=1; shift ;;
        --resume)                   resume=1; shift ;;
        --run-image-benchmark)      run_image_benchmark=1; shift ;;
        --run-train-reconstruction) run_train_reconstruction=1; shift ;;
        --rescore-locality)         rescore_locality=1; shift ;;
        --bench-root)               bench_root="$2"; shift 2 ;;
        --baseline-cache-root)      baseline_cache_root="$2"; shift 2 ;;
        --benchmark-max-samples)    benchmark_max_samples="$2"; shift 2 ;;
        --benchmark-sd-device)      benchmark_sd_device="$2"; shift 2 ;;
        --benchmark-ssae-device)    benchmark_ssae_device="$2"; shift 2 ;;
        --benchmark-baseline-device) benchmark_baseline_device="$2"; shift 2 ;;
        --locality-drop)            locality_drop=1; shift ;;
        --locality-swap)            locality_swap=1; shift ;;
        --benchmark-dino)           benchmark_dino=1; shift ;;
        --benchmark-simulated)      benchmark_simulated=1; shift ;;
        --python)                   python_bin="$2"; shift 2 ;;
        -h|--help)                  usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# validate head_type
case "$head_type" in
    dense|block_diagonal) ;;
    *) echo "--head-type must be 'dense' or 'block_diagonal' (got '$head_type')" >&2; exit 2 ;;
esac

# parse CSV lists
IFS=',' read -r -a topk_arr   <<< "$topk_csv"
IFS=',' read -r -a layers_arr <<< "$layers_csv"
IFS=',' read -r -a seeds_arr  <<< "$seeds_csv"

for L in "${layers_arr[@]}"; do
    if (( L < 1 )); then
        echo "layers must all be >= 1 (got $L)" >&2; exit 2
    fi
done
for L in "${layers_arr[@]}"; do
    if (( L > 1 )) && (( hidden_dim < 1 )); then
        echo "--hidden-dim must be >= 1 when layers contains values > 1" >&2; exit 2
    fi
done

# resolve paths under repo_root
repo_root="$(cd "$repo_root" && pwd)"
cd "$repo_root"

resolve_under() {
    # echo an absolute path; if arg is absolute, keep as-is
    local base="$1" p="$2"
    if [[ "$p" = /* ]]; then
        echo "$p"
    else
        echo "$base/$p"
    fi
}

categories=$(resolve_under "$repo_root" "$categories")
properties_same=$(resolve_under "$repo_root" "$properties_same")
split_root=$(resolve_under "$repo_root" "$split_root")
runs_root=$(resolve_under "$repo_root" "$runs_root")
bench_root=$(resolve_under "$repo_root" "$bench_root")
baseline_cache_root=$(resolve_under "$repo_root" "$baseline_cache_root")

if [[ ! -f "$categories" ]]; then
    echo "categories file not found: $categories" >&2
    echo "pass --categories <path> or run from repo root. repo_root=$repo_root" >&2
    exit 1
fi
if [[ ! -f "$properties_same" ]]; then
    echo "properties_same.json not found: $properties_same" >&2
    echo "pass --properties-same <path>. repo_root=$repo_root" >&2
    exit 1
fi

train_dir="$split_root/train"
holdout_dir="$split_root/holdout"
mkdir -p "$runs_root"

invoke_py() {
    # run python with the given argv, echo the command, exit on nonzero
    echo ">> $python_bin $*"
    "$python_bin" "$@"
}

# 1. compositional split
train_prompts="$train_dir/prompts.json"
if (( recreate_split )) || [[ ! -f "$train_prompts" ]]; then
    echo "== Building compositional split =="
    if [[ -d "$split_root" ]]; then rm -rf "$split_root"; fi
    split_args=(
        "dataset_generation/compositional_split.py"
        --categories_json "$categories"
        --output_root "$split_root"
        --holdout_fraction "$holdout_fraction"
        --properties_same_json "$properties_same"
    )
    if (( max_train_prompts   > 0 )); then split_args+=(--max_train_prompts   "$max_train_prompts"); fi
    if (( max_holdout_prompts > 0 )); then split_args+=(--max_holdout_prompts "$max_holdout_prompts"); fi
    invoke_py "${split_args[@]}"
else
    echo "split already at $split_root; use --recreate-split to rebuild."
fi

# the dataloader reads <split>/properties_same.json directly; older splits
# didn't copy it, so top it up now
for d in "$train_dir" "$holdout_dir"; do
    dst="$d/properties_same.json"
    if [[ ! -f "$dst" ]]; then
        cp "$properties_same" "$dst"
        echo "copied properties_same.json -> $dst"
    fi
done

# 2. embeddings
extract_embeddings() {
    local folder="$1"
    local prompts="$folder/prompts.json"
    if [[ ! -f "$prompts" ]]; then
        echo "missing $prompts" >&2; exit 1
    fi
    local manifest="$folder/embds/manifest.json"
    if (( ! recreate_embeddings )) && [[ -f "$manifest" ]]; then
        echo "embeddings already present under $folder; skipping (use --recreate-embeddings to rebuild)."
        return
    fi
    local embds_dir="$folder/embds"
    if [[ -d "$embds_dir" ]]; then rm -rf "$embds_dir"; fi
    invoke_py \
        get_embeddings.py \
        --backbone "$backbone" \
        --prompts "$prompts" \
        --out "$folder" \
        --categories "$categories"
}

echo "== Extracting embeddings (train) =="
extract_embeddings "$train_dir"
echo "== Extracting embeddings (holdout) =="
extract_embeddings "$holdout_dir"

# 3. sweep
summary_csv="$runs_root/sweep_summary.csv"
echo "topk,layers,hidden_dim,head_type,seed,mse_mean,cosine_mean,n_holdout,elapsed_sec,output_folder,bench_dir" > "$summary_csv"

if (( run_image_benchmark )); then mkdir -p "$bench_root"; fi

for k in "${topk_arr[@]}"; do
    for L in "${layers_arr[@]}"; do
      for seed in "${seeds_arr[@]}"; do
        if (( L == 1 )); then
            tag="topk_${k}_L1"
        else
            tag="topk_${k}_L${L}_h${hidden_dim}"
        fi
        if (( pca_rotation )); then tag="${tag}_pca"; fi
        if [[ "$head_type" == "block_diagonal" ]]; then tag="${tag}_blockdiag"; fi
        # Seed 0 keeps the historical tag so existing run dirs, bench dirs and the
        # report builders' hardcoded names keep resolving; extra seeds get a suffix.
        if (( seed != 0 )); then tag="${tag}_s${seed}"; fi
        echo "== $tag =="

        run_dir="$runs_root/$tag"
        yaml_path="$runs_root/$tag.yaml"
        metrics_out="$run_dir/holdout_compositional_metrics.json"

        # per-run YAML. num_layers / hidden_dims are also overridden on the CLI
        # below; keeping them in the YAML too keeps the file self-describing.
        if (( L == 1 )); then hidden_yaml="null"; else hidden_yaml="$hidden_dim"; fi
        if (( pca_rotation )); then pca_yaml="True"; else pca_yaml="False"; fi
        # E11 needs the checkpoint trajectory, not just the final weights.
        if [[ -n "$save_model_frequency" ]]; then save_model_yaml="$save_model_frequency"; else save_model_yaml="null"; fi

        # block_diagonal head is only implemented on model_trainable_inputs; dense sweeps
        # keep the parameter-efficient model_avg_feature default.
        if [[ "$head_type" == "block_diagonal" ]]; then model_name="model_trainable_inputs"; else model_name="model_avg_feature"; fi

        cat > "$yaml_path" <<YAML
training:
  model:
    model_name: "$model_name"
    using_blocs: False
    num_layers: $L
    hidden_dims: $hidden_yaml
    head_type: "$head_type"
  dataloader:
    folder_path: "$train_dir/"
    truncate_n_prompts: null
    truncate_embds_topk: $k
    pca_rotation: $pca_yaml
    add_property_is_the_same: True
    normalize: "MAX_MIN"
    num_workers: $num_workers
    simulated:
      simulated: False
      dim_clip_simulated: 100
  training:
    n_epochs: 100
    print_frequency: 1
    save_model_frequency: $save_model_yaml
    plot_frequency: 1
    seed: $seed
    batch_size: 16
    lr: 0.001
    beta1: 0.9
    beta2: 0.999
    lr_scheduler:
      lr_scheduler_type: LINEAR
      lr_scheduler_linear:
        lr_scheduler_lr_final_linear: 0.0001
  sparse_feature_design:
    n_repeat: 10
YAML

        elapsed=0
        # --resume: skip training if this specific config already has a checkpoint,
        # so partial sweeps (e.g. one crashed at bench) can pick up where they left off.
        if (( resume )) && [[ -f "$run_dir/model.pt" ]]; then
            echo "  --resume: checkpoint exists at $run_dir/model.pt, skipping training"
            skip_this=1
        else
            skip_this=0
        fi
        if (( skip_training || skip_this )); then
            if [[ ! -f "$run_dir/model.pt" ]]; then
                echo "missing checkpoint at $run_dir/model.pt; drop --skip-training or train it first." >&2
                exit 1
            fi
            (( skip_training )) && echo "  --skip-training set; reusing checkpoint at $run_dir"
        else
            train_args=(
                training_cli.py
                --output_folder "$run_dir"
                --path_yaml "$yaml_path"
                --overwrite_output True
                --num_layers "$L"
            )
            if (( L > 1 ));       then train_args+=(--hidden_dims "$hidden_dim"); fi
            if (( pca_rotation )); then train_args+=(--pca_rotation True); fi
            if [[ "$head_type" != "dense" ]]; then train_args+=(--head_type "$head_type"); fi

            t0=$(date +%s)
            invoke_py "${train_args[@]}"
            elapsed=$(( $(date +%s) - t0 ))
        fi

        # the first training run at each topk k writes indices_top_{k}.json +
        # embds_{max,min}_top_{k}.json into $train_dir. copy them to holdout so
        # downstream eval reuses the same coordinate selection + normalization.
        # cheap; run every time to stay in sync across layer sweeps.
        invoke_py \
            -m evaluation.run_copy_truncation \
            --train_folder "$train_dir" \
            --holdout_folder "$holdout_dir"

        invoke_py \
            -m evaluation.run_compositional_embeddings \
            --checkpoint "$run_dir" \
            --holdout_folder "$holdout_dir" \
            --ssae_device cpu \
            --output_json "$metrics_out"

        mse=""; cos=""; n_hold=""
        if [[ -f "$metrics_out" ]]; then
            # pull scalar fields with python so we don't need jq on the AWS host
            read -r mse cos n_hold < <("$python_bin" - "$metrics_out" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    obj = json.load(f)
def s(k):
    v = obj.get(k)
    return "" if v is None else str(v)
print(s("mse_mean"), s("cosine_mean"), s("n_holdout"))
PY
            )
        fi

        bench_dir=""
        if (( run_image_benchmark )); then
            bench_dir="$bench_root/$tag"
            # skip bench when resuming and a complete summary already exists
            if (( resume )) && [[ -f "$bench_dir/summary.json" ]]; then
                echo "  --resume: bench already done at $bench_dir, skipping"
            else
            echo "  -- image benchmark -> $bench_dir"
            bench_args=(
                -m evaluation.run_image_benchmark
                --checkpoint "$run_dir"
                --holdout_folder "$holdout_dir"
                --output_dir "$bench_dir"
                --baseline_cache_root "$baseline_cache_root"
                --sd_device "$benchmark_sd_device"
            )
            if [[ -n "$benchmark_ssae_device"     ]]; then bench_args+=(--ssae_device "$benchmark_ssae_device"); fi
            if [[ -n "$benchmark_baseline_device" ]]; then bench_args+=(--baseline_device "$benchmark_baseline_device"); fi
            if (( benchmark_max_samples > 0 )); then bench_args+=(--max_samples "$benchmark_max_samples"); fi
            if (( locality_drop ));       then bench_args+=(--locality_drop_one_attr); fi
            if (( locality_swap ));       then bench_args+=(--locality_swap_one_attr); fi
            if (( benchmark_dino ));      then bench_args+=(--dino); fi
            if (( benchmark_simulated )); then bench_args+=(--simulated); fi
            invoke_py "${bench_args[@]}"
            fi  # end resume-bench-skip else
        fi

        if (( run_train_reconstruction )); then
            recon_out="$run_dir/train_reconstruction.json"
            echo "  -- train-set reconstruction -> $recon_out"
            invoke_py \
                -m evaluation.run_reconstruction \
                --checkpoint "$run_dir" \
                --output_json "$recon_out"
        fi

        if (( L == 1 )); then hidden_col=""; else hidden_col="$hidden_dim"; fi
        echo "$k,$L,$hidden_col,$head_type,$seed,$mse,$cos,$n_hold,$elapsed,$run_dir,$bench_dir" >> "$summary_csv"
        echo "  topk=$k L=$L h=$hidden_col head=$head_type seed=$seed mse=$mse cosine=$cos elapsed=${elapsed}s bench=$bench_dir"
      done
    done
done

if (( rescore_locality )); then
    echo "== Rescoring locality across bench_out + baseline cache =="
    invoke_py \
        -m evaluation.rescore_locality \
        --bench_dir "$bench_root" \
        --baseline_dir "$baseline_cache_root"
fi

echo "== Sweep complete =="
column -s, -t < "$summary_csv" || cat "$summary_csv"
echo "summary: $summary_csv"
