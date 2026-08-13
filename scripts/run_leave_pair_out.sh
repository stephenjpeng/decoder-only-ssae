#!/usr/bin/env bash
# Fixed E8 leave-pair-out experiment driver.
#
# Usage:
#   scripts/run_leave_pair_out.sh \
#     --repo-root /path/to/repo \
#     --source-split-root /path/to/existing/compositional_split \
#     --output-root /path/to/results/leave_pair_out \
#     [--python /path/to/python] \
#     [--prepare-only] \
#     [--skip-training] \
#     [--resume]
#
# Requirements:
#   - The source split root must contain train/ and holdout/ with all 4096 tuples
#     and their extracted embeddings (embds/ dirs).
#   - GPU must be available for SSAE training unless --skip-training is set.
#
# The script writes a DONE marker only after all artifact checks pass.

set -euo pipefail

# ------------------------------------------------------------------ arg parsing

REPO_ROOT=""
SOURCE_SPLIT_ROOT=""
OUTPUT_ROOT=""
PYTHON="python"
PREPARE_ONLY=false
SKIP_TRAINING=false
RESUME=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-root)      REPO_ROOT="$2"; shift 2;;
        --source-split-root) SOURCE_SPLIT_ROOT="$2"; shift 2;;
        --output-root)    OUTPUT_ROOT="$2"; shift 2;;
        --python)         PYTHON="$2"; shift 2;;
        --prepare-only)   PREPARE_ONLY=true; shift;;
        --skip-training)  SKIP_TRAINING=true; shift;;
        --resume)         RESUME=true; shift;;
        *) echo "Unknown argument: $1" >&2; exit 1;;
    esac
done

if [[ -z "$REPO_ROOT" || -z "$SOURCE_SPLIT_ROOT" || -z "$OUTPUT_ROOT" ]]; then
    echo "ERROR: --repo-root, --source-split-root, and --output-root are required." >&2
    exit 1
fi

CATS_JSON="$REPO_ROOT/dataset_generation/prompts/input/categories_with_properties.json"
PROPS_SAME_JSON="$REPO_ROOT/dataset_generation/prompts/input/properties_same.json"

mkdir -p "$OUTPUT_ROOT"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ------------------------------------------------------------------ splits

build_pair_split() {
    local slug="$1"   # e.g. hat-and-gun
    local val_a="$2"
    local val_b="$3"
    local n_holdout="$4"
    local pair_root="$OUTPUT_ROOT/$slug/pair_split"
    local rand_root="$OUTPUT_ROOT/$slug/matched_random_split"

    # pair split
    if [[ -f "$pair_root/split_manifest.json" ]]; then
        log "pair split for $slug already exists; skipping creation"
    else
        log "building pair split for $slug ($val_a / $val_b)"
        "$PYTHON" -m dataset_generation.compositional_split \
            --categories_json "$CATS_JSON" \
            --output_root "$pair_root" \
            --holdout_value_pair "$val_a" "$val_b" \
            --seed 0 \
            --shuffle_train_order \
            --properties_same_json "$PROPS_SAME_JSON"
    fi

    # matched random control
    if [[ -f "$rand_root/split_manifest.json" ]]; then
        log "matched random split for $slug already exists; skipping creation"
    else
        log "building matched random split for $slug (n=$n_holdout)"
        "$PYTHON" -m dataset_generation.compositional_split \
            --categories_json "$CATS_JSON" \
            --output_root "$rand_root" \
            --n_holdout "$n_holdout" \
            --seed 0 \
            --shuffle_train_order \
            --properties_same_json "$PROPS_SAME_JSON"
    fi

    # verify holdout counts
    pair_n=$(python -c "import json; m=json.load(open('$pair_root/split_manifest.json')); print(m['n_holdout_written'])")
    rand_n=$(python -c "import json; m=json.load(open('$rand_root/split_manifest.json')); print(m['n_holdout_written'])")
    if [[ "$pair_n" -ne "$n_holdout" ]]; then
        echo "ERROR: $slug pair holdout has $pair_n rows, expected $n_holdout" >&2; exit 1
    fi
    if [[ "$rand_n" -ne "$n_holdout" ]]; then
        echo "ERROR: $slug random holdout has $rand_n rows, expected $n_holdout" >&2; exit 1
    fi
    log "$slug: pair=$pair_n holdout rows, random=$rand_n holdout rows — OK"
}

materialize_split() {
    local slug="$1"
    local split_type="$2"  # pair_split or matched_random_split
    local split_root="$OUTPUT_ROOT/$slug/$split_type"
    local resume_flag=""
    $RESUME && resume_flag="--resume"

    for partition in train holdout; do
        local dest="$split_root/$partition"
        if [[ -f "$dest/embedding_materialization_manifest.json" ]]; then
            log "$slug/$split_type/$partition embeddings already materialized"
            continue
        fi
        log "materializing $slug/$split_type/$partition embeddings"
        "$PYTHON" -m dataset_generation.materialize_embedding_split \
            --source_folder "$SOURCE_SPLIT_ROOT/train" \
            --source_folder "$SOURCE_SPLIT_ROOT/holdout" \
            --destination_folder "$dest" \
            --mode hardlink \
            $resume_flag
    done
}

# ------------------------------------------------------------------ training

train_split() {
    local slug="$1"
    local split_type="$2"   # pair_split or matched_random_split
    local split_root="$OUTPUT_ROOT/$slug/$split_type"
    local ckpt_root="$OUTPUT_ROOT/$slug/checkpoints"

    for seed in 0 1 2; do
        local run_name="${split_type}_h19_s${seed}"
        local ckpt="$ckpt_root/$run_name"
        if [[ -f "$ckpt/model.pt" ]]; then
            log "$run_name already trained; skipping"
            continue
        fi
        log "training $run_name"
        "$PYTHON" training_cli.py \
            --output_folder "$ckpt" \
            --path_yaml trainings/config/params_default.yaml \
            --overwrite_output True \
            --model_name model_avg_feature \
            --head_type dense \
            --num_layers 2 \
            --hidden_dims 19 \
            --n_repeat 10 \
            --truncate_embds_topk 100000 \
            --normalize MAX_MIN \
            --batch_size 16 \
            --lr 0.001 \
            --n_epochs 100 \
            --num_workers 6 \
            --seed "$seed" \
            --folder_path "$split_root/train/"
    done
}

copy_sidecars() {
    local slug="$1"
    local split_type="$2"
    local split_root="$OUTPUT_ROOT/$slug/$split_type"
    local ckpt_root="$OUTPUT_ROOT/$slug/checkpoints"

    # use s0 checkpoint as the sidecar source (all seeds share topk config)
    local ref_ckpt="$ckpt_root/${split_type}_h19_s0"
    log "copying top-k sidecars for $slug/$split_type"
    "$PYTHON" -m evaluation.run_copy_truncation \
        --checkpoint "$ref_ckpt" \
        --train_folder "$split_root/train" \
        --holdout_folder "$split_root/holdout"
}

run_compositional_embeddings() {
    local slug="$1"
    local split_type="$2"
    local split_root="$OUTPUT_ROOT/$slug/$split_type"
    local ckpt_root="$OUTPUT_ROOT/$slug/checkpoints"
    local metrics_root="$OUTPUT_ROOT/$slug/metrics"

    for seed in 0 1 2; do
        local run_name="${split_type}_h19_s${seed}"
        local ckpt="$ckpt_root/$run_name"
        local out="$metrics_root/$run_name"
        if [[ -f "$out/metrics.json" ]]; then
            log "$run_name compositional metrics already exist; skipping"
            continue
        fi
        log "running compositional embeddings for $run_name"
        "$PYTHON" -m evaluation.run_compositional_embeddings \
            --checkpoint "$ckpt" \
            --holdout_folder "$split_root/holdout" \
            --output_dir "$out"
    done
}

run_interaction_budget() {
    local slug="$1"
    local split_type="$2"
    local split_root="$OUTPUT_ROOT/$slug/$split_type"
    local ckpt_root="$OUTPUT_ROOT/$slug/checkpoints"
    local out_dir="$split_root/interaction_budget"

    if [[ -f "$out_dir/baseline_metrics.json" ]]; then
        log "$slug/$split_type interaction budget already exists; skipping"
        return
    fi
    log "running interaction budget for $slug/$split_type"
    "$PYTHON" -m baselines.run_interaction_budget \
        --checkpoint "$ckpt_root/${split_type}_h19_s0" \
        --train_folder "$split_root/train" \
        --holdout_folder "$split_root/holdout" \
        --output_dir "$out_dir"
}

# ------------------------------------------------------------------ main sequence

log "starting E8 leave-pair-out pipeline"

# step 1: build both pair splits and matched random controls
build_pair_split "hat-and-gun" "and a hat" "holding a gun" 512
build_pair_split "blond-and-blue-eyes" "A blond girl" "with blue eyes" 1024

if $PREPARE_ONLY; then
    log "prepare-only mode: stopping after split creation"
    exit 0
fi

# step 2: materialize raw embeddings
for slug in hat-and-gun blond-and-blue-eyes; do
    for split_type in pair_split matched_random_split; do
        materialize_split "$slug" "$split_type"
    done
done

if $SKIP_TRAINING; then
    log "skip-training mode: stopping before GPU training"
    exit 0
fi

# step 3: train h19 for seeds 0, 1, 2 on all four splits
for slug in hat-and-gun blond-and-blue-eyes; do
    for split_type in pair_split matched_random_split; do
        train_split "$slug" "$split_type"
    done
done

# step 4: copy train sidecars to holdout for each split
for slug in hat-and-gun blond-and-blue-eyes; do
    for split_type in pair_split matched_random_split; do
        copy_sidecars "$slug" "$split_type"
    done
done

# step 5: run compositional embedding metrics for all checkpoints
for slug in hat-and-gun blond-and-blue-eyes; do
    for split_type in pair_split matched_random_split; do
        run_compositional_embeddings "$slug" "$split_type"
    done
done

# step 6: run interaction budget baselines for all four splits
for slug in hat-and-gun blond-and-blue-eyes; do
    for split_type in pair_split matched_random_split; do
        run_interaction_budget "$slug" "$split_type"
    done
done

# step 7: run leave-pair analysis
log "running leave-pair analysis"
for slug in hat-and-gun blond-and-blue-eyes; do
    metrics_root="$OUTPUT_ROOT/$slug/metrics"
    pair_ssae_args=""
    rand_ssae_args=""
    for seed in 0 1 2; do
        pair_ssae_args="$pair_ssae_args --pair_ssae $metrics_root/pair_split_h19_s${seed}/metrics.json"
        rand_ssae_args="$rand_ssae_args --random_ssae $metrics_root/matched_random_split_h19_s${seed}/metrics.json"
    done

    "$PYTHON" -m analysis.leave_pair_out \
        --slug "$slug" \
        --pair_baseline "$OUTPUT_ROOT/$slug/pair_split/interaction_budget/baseline_metrics.json" \
        --random_baseline "$OUTPUT_ROOT/$slug/matched_random_split/interaction_budget/baseline_metrics.json" \
        $pair_ssae_args \
        $rand_ssae_args \
        --output_dir "$OUTPUT_ROOT/analysis/leave_pair_out/$slug"
done

# final check
log "all steps complete"
touch "$OUTPUT_ROOT/DONE"
log "wrote DONE marker at $OUTPUT_ROOT/DONE"
