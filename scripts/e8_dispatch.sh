#!/usr/bin/env bash
# E8 full pipeline dispatch script — runs on the EC2 box via tmux.
#
# Covers:
#   - Syncing new code and source embeddings from S3
#   - Creating the 4 leave-pair-out splits
#   - Materializing embeddings via hard-links
#   - Training 12 h19 SSAEs (3 seeds × 4 splits, topk=100k)
#   - Copying top-k sidecars train→holdout
#   - Running compositional embedding metrics (12 runs)
#   - Running plain+pairwise ridge interaction budget (4 splits)
#   - Running probe fits for 3 E3 targets (CPU, topk=300k)
#   - Rendering 3 probe E3 image benchmarks (GPU)
#   - Syncing all results to S3

set -euo pipefail

REPO="/home/ubuntu/decoder-only-ssae"
S3="s3://ssae-runs-sp"
PYTHON="$REPO/.venv/bin/python"
CATS_JSON="$REPO/dataset_generation/prompts/input/categories_with_properties.json"
PROPS_SAME="$REPO/dataset_generation/prompts/input/properties_same.json"
RESULTS="$REPO/results"
LPO="$RESULTS/leave_pair_out"
PROBE_DIR="$RESULTS/probe_intervention"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------- code sync

log "pulling latest code"
cd "$REPO"
git fetch origin feature/additional-models
git reset --hard origin/feature/additional-models

# ---------------------------------------------------------------- source embeddings

log "syncing source embeddings from S3 (skip if present)"
mkdir -p "$RESULTS/compositional_split"
aws s3 sync "$S3/compositional_split/train/"    "$RESULTS/compositional_split/train/"   --no-progress
aws s3 sync "$S3/compositional_split/holdout/"  "$RESULTS/compositional_split/holdout/" --no-progress
log "source split: $(ls "$RESULTS/compositional_split/train/embds/" | wc -l) train dirs, $(ls "$RESULTS/compositional_split/holdout/embds/" | wc -l) holdout dirs"

# ---------------------------------------------------------------- create splits

make_splits() {
    local slug="$1"; local val_a="$2"; local val_b="$3"; local n_hold="$4"
    local pair_root="$LPO/$slug/pair_split"
    local rand_root="$LPO/$slug/matched_random_split"

    if [[ ! -f "$pair_root/split_manifest.json" ]]; then
        log "creating pair split: $slug"
        "$PYTHON" dataset_generation/compositional_split.py \
            --categories_json "$CATS_JSON" --output_root "$pair_root" \
            --holdout_value_pair "$val_a" "$val_b" \
            --seed 0 --shuffle_train_order --properties_same_json "$PROPS_SAME"
    else
        log "pair split $slug already exists"
    fi

    if [[ ! -f "$rand_root/split_manifest.json" ]]; then
        log "creating matched random split: $slug (n=$n_hold)"
        "$PYTHON" dataset_generation/compositional_split.py \
            --categories_json "$CATS_JSON" --output_root "$rand_root" \
            --n_holdout "$n_hold" --seed 0 --shuffle_train_order \
            --properties_same_json "$PROPS_SAME"
    else
        log "random split $slug already exists"
    fi

    # verify counts
    local pair_n rand_n
    pair_n=$("$PYTHON" -c "import json; m=json.load(open('$pair_root/split_manifest.json')); print(m['n_holdout_written'])")
    rand_n=$("$PYTHON" -c "import json; m=json.load(open('$rand_root/split_manifest.json')); print(m['n_holdout_written'])")
    [[ "$pair_n" == "$n_hold" ]] || { echo "ERROR: $slug pair holdout=$pair_n expected=$n_hold" >&2; exit 1; }
    [[ "$rand_n" == "$n_hold" ]] || { echo "ERROR: $slug rand holdout=$rand_n expected=$n_hold" >&2; exit 1; }
    log "$slug: pair=$pair_n, rand=$rand_n holdout rows — OK"
}

mkdir -p "$LPO"
make_splits "hat-and-gun"         "and a hat"  "holding a gun" 512
make_splits "blond-and-blue-eyes" "A blond girl" "with blue eyes" 1024

# ---------------------------------------------------------------- materialize embeddings

materialize() {
    local slug="$1"; local split_type="$2"
    local split_root="$LPO/$slug/$split_type"
    for part in train holdout; do
        local dest="$split_root/$part"
        if [[ -f "$dest/embedding_materialization_manifest.json" ]]; then
            log "$slug/$split_type/$part already materialized"
        else
            log "materializing $slug/$split_type/$part"
            "$PYTHON" -m dataset_generation.materialize_embedding_split \
                --source_folder "$RESULTS/compositional_split/train" \
                --source_folder "$RESULTS/compositional_split/holdout" \
                --destination_folder "$dest" --mode hardlink
        fi
    done
}

for slug in hat-and-gun blond-and-blue-eyes; do
    for st in pair_split matched_random_split; do
        materialize "$slug" "$st"
    done
done

# ---------------------------------------------------------------- write training YAMLs

write_yaml() {
    local folder_path="$1"; local seed="$2"; local out="$3"
    mkdir -p "$(dirname "$out")"
    cat > "$out" <<YAML
training:
  model:
    model_name: "model_avg_feature"
    using_blocs: False
    num_layers: 2
    hidden_dims: 19
    head_type: "dense"
  dataloader:
    folder_path: "$folder_path"
    truncate_n_prompts: null
    truncate_embds_topk: 100000
    pca_rotation: False
    add_property_is_the_same: True
    normalize: "MAX_MIN"
    num_workers: 6
    simulated:
      simulated: False
      dim_clip_simulated: 100
  training:
    n_epochs: 100
    print_frequency: 1
    save_model_frequency: null
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
}

# ---------------------------------------------------------------- train h19

train_run() {
    local slug="$1"; local st="$2"; local seed="$3"
    local run_name="${st}_h19_s${seed}"
    local ckpt="$LPO/$slug/checkpoints/$run_name"
    local yaml="$LPO/$slug/yamls/${run_name}.yaml"

    if [[ -f "$ckpt/model.pt" ]]; then
        log "$slug/$run_name already trained"
        return
    fi
    mkdir -p "$LPO/$slug/yamls"
    write_yaml "$LPO/$slug/$st/train/" "$seed" "$yaml"
    log "training $slug/$run_name"
    "$PYTHON" training_cli.py \
        --output_folder "$ckpt" --path_yaml "$yaml" \
        --overwrite_output True \
        --num_layers 2 --hidden_dims 19 --head_type dense
    log "done training $slug/$run_name"
}

for slug in hat-and-gun blond-and-blue-eyes; do
    for st in pair_split matched_random_split; do
        for seed in 0 1 2; do
            train_run "$slug" "$st" "$seed"
        done
    done
done

# ---------------------------------------------------------------- copy sidecars train→holdout

copy_sidecars() {
    local slug="$1"; local st="$2"
    local ckpt="$LPO/$slug/checkpoints/${st}_h19_s0"
    local train_dir="$LPO/$slug/$st/train"
    local hold_dir="$LPO/$slug/$st/holdout"
    log "copying sidecars for $slug/$st"
    "$PYTHON" -m evaluation.run_copy_truncation \
        --checkpoint "$ckpt" \
        --train_folder "$train_dir" \
        --holdout_folder "$hold_dir"
}

for slug in hat-and-gun blond-and-blue-eyes; do
    for st in pair_split matched_random_split; do
        copy_sidecars "$slug" "$st"
    done
done

# ---------------------------------------------------------------- compositional embedding metrics

comp_metrics() {
    local slug="$1"; local st="$2"; local seed="$3"
    local run_name="${st}_h19_s${seed}"
    local ckpt="$LPO/$slug/checkpoints/$run_name"
    local out="$LPO/$slug/metrics/$run_name"
    if [[ -f "$out/metrics.json" ]]; then
        log "$slug/$run_name metrics already exist"
        return
    fi
    log "computing compositional metrics $slug/$run_name"
    "$PYTHON" -m evaluation.run_compositional_embeddings \
        --checkpoint "$ckpt" \
        --holdout_folder "$LPO/$slug/$st/holdout" \
        --output_dir "$out"
}

for slug in hat-and-gun blond-and-blue-eyes; do
    for st in pair_split matched_random_split; do
        for seed in 0 1 2; do
            comp_metrics "$slug" "$st" "$seed"
        done
    done
done

# ---------------------------------------------------------------- interaction budget (plain + pairwise ridge)

run_ib() {
    local slug="$1"; local st="$2"
    local ckpt="$LPO/$slug/checkpoints/${st}_h19_s0"
    local out="$LPO/$slug/$st/interaction_budget"
    if [[ -f "$out/baseline_metrics.json" ]]; then
        log "$slug/$st interaction budget already exists"
        return
    fi
    log "running interaction budget $slug/$st"
    "$PYTHON" -m baselines.run_interaction_budget \
        --checkpoint "$ckpt" \
        --train_folder "$LPO/$slug/$st/train" \
        --holdout_folder "$LPO/$slug/$st/holdout" \
        --output_dir "$out"
}

for slug in hat-and-gun blond-and-blue-eyes; do
    for st in pair_split matched_random_split; do
        run_ib "$slug" "$st"
    done
done

# also run on the standard random-tuple split (E2 primary experiment)
RAND_IB="$RESULTS/interaction_budget/random_tuple_100k"
if [[ ! -f "$RAND_IB/baseline_metrics.json" ]]; then
    log "running interaction budget on standard random split (100k topk)"
    # need a 100k checkpoint — use topk_100000_L2_h19 if present, else topk_100000_L1
    STD_CKPT="$RESULTS/topk_sweep/topk_100000_L2_h19"
    if [[ ! -d "$STD_CKPT" ]]; then
        log "syncing topk_100000_L2_h19 from S3"
        mkdir -p "$STD_CKPT"
        aws s3 sync "$S3/results/topk_sweep/topk_100000_L2_h19/" "$STD_CKPT/" --no-progress
    fi
    "$PYTHON" -m baselines.run_interaction_budget \
        --checkpoint "$STD_CKPT" \
        --train_folder "$RESULTS/compositional_split/train" \
        --holdout_folder "$RESULTS/compositional_split/holdout" \
        --output_dir "$RAND_IB"
fi

# ---------------------------------------------------------------- probe fits (CPU, topk=300k)

# sync the 300k h2048 checkpoint (needed for probe fit dataset config)
CKPT_300K="$RESULTS/topk_sweep/topk_300000_L2_h2048"
if [[ ! -d "$CKPT_300K" ]]; then
    log "syncing topk_300000_L2_h2048 from S3"
    mkdir -p "$CKPT_300K"
    aws s3 sync "$S3/results/topk_sweep/topk_300000_L2_h2048/" "$CKPT_300K/" --no-progress
fi

fit_probe() {
    local slug="$1"; local target="$2"; local replacement="$3"
    local out="$PROBE_DIR/$slug"
    if [[ -f "$out/probe_artifact.pt" ]]; then
        log "probe artifact $slug already exists"
        return
    fi
    log "fitting probe artifact: $slug"
    "$PYTHON" -m evaluation.run_probe_intervention_fit \
        --checkpoint "$CKPT_300K" \
        --train_folder "$RESULTS/compositional_split/train" \
        --target_property "$target" \
        --replacement_property "$replacement" \
        --output_dir "$out" \
        --device cpu --seed 0
}

fit_probe "gun-to-coffee"       "holding a gun"  "holding a coffee"
fit_probe "hat-to-baseball-cap" "and a hat"      "and a baseball cap"
fit_probe "beach-to-cafe"       "at the beach"   "sitting at a cafe"

# ---------------------------------------------------------------- probe E3 image benchmark

run_probe_bench() {
    local concept="$1"; local target="$2"; local replacement="$3"; local probe_slug="$4"
    local out="$RESULTS/bench_e3_probe/${concept}_probe"
    if [[ -f "$out/summary.json" ]]; then
        log "probe bench $concept already exists"
        return
    fi
    log "rendering probe E3 bench: $concept"
    "$PYTHON" -m evaluation.run_image_benchmark \
        --checkpoint "$CKPT_300K" \
        --holdout_folder "$RESULTS/compositional_split/holdout" \
        --output_dir "$out" \
        --methods gt_embed,linear_probe_direction \
        --no_baseline_cache \
        --target_property "$target" \
        --replacement_property "$replacement" \
        --probe_artifact "$PROBE_DIR/$probe_slug/probe_artifact.pt" \
        --locality_drop_one_attr --locality_swap_one_attr \
        --max_matched 60 --fill_policy train_mean --dino \
        --sd_device cuda
}

run_probe_bench "gun"   "holding a gun"  "holding a coffee"  "gun-to-coffee"
run_probe_bench "hat"   "and a hat"      "and a baseball cap" "hat-to-baseball-cap"
run_probe_bench "beach" "at the beach"   "sitting at a cafe"  "beach-to-cafe"

# ---------------------------------------------------------------- sync results to S3

log "syncing leave_pair_out results to S3"
aws s3 sync "$LPO/"          "$S3/results/leave_pair_out/"      --no-progress --exclude "embds/*"
aws s3 sync "$RESULTS/interaction_budget/" "$S3/results/interaction_budget/" --no-progress
aws s3 sync "$PROBE_DIR/"    "$S3/results/probe_intervention/"  --no-progress --exclude "*.pt"
# sync probe_artifact.pt separately (important artifact)
for slug in gun-to-coffee hat-to-baseball-cap beach-to-cafe; do
    [[ -f "$PROBE_DIR/$slug/probe_artifact.pt" ]] && \
        aws s3 cp "$PROBE_DIR/$slug/probe_artifact.pt" \
            "$S3/results/probe_intervention/$slug/probe_artifact.pt" --no-progress
done
aws s3 sync "$RESULTS/bench_e3_probe/" "$S3/results/bench_e3_probe/" --no-progress

log "all done — synced to S3"
date
