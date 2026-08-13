"""CLI for fitting a probe artifact for one target/replacement pair."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from evaluation.embedding_linear_probe import CategoricalLinearProbe, fit_probe
from evaluation.io import h5_dataset_for_folder
from trainings.utils.run_manifest import (
    checkpoint_fingerprint,
    dataset_fingerprint,
    write_run_manifest,
)


def load_and_validate_probe_artifact(path: Path, dataset) -> dict:
    """Load and validate probe artifact against dataset. Raise ValueError on mismatch.
    Check: schema_version, property_order, category_pid_groups, topk, normalize."""
    artifact = torch.load(path, map_location="cpu", weights_only=False)

    # schema version
    if artifact.get("schema_version") != 1:
        raise ValueError(
            f"artifact schema_version {artifact.get('schema_version')} != 1"
        )

    # property order
    expected_property_order = [
        dataset.properties.pid_to_property[pid]
        for pid in range(dataset.properties.n_properties)
    ]
    if artifact.get("property_order") != expected_property_order:
        raise ValueError("artifact property_order does not match dataset")

    # category_pid_groups
    expected_groups = [
        dataset.properties.cid_to_pids[cid]
        for cid in range(dataset.properties.n_categories)
    ]
    if artifact.get("category_pid_groups") != expected_groups:
        raise ValueError("artifact category_pid_groups does not match dataset")

    # topk
    if artifact.get("topk") != dataset.truncate_embds_topk:
        raise ValueError(
            f"artifact topk {artifact.get('topk')} != dataset topk {dataset.truncate_embds_topk}"
        )

    # normalize
    if artifact.get("normalize") != dataset.normalize:
        raise ValueError(
            f"artifact normalize {artifact.get('normalize')} != dataset normalize {dataset.normalize}"
        )

    return artifact


def main():
    parser = argparse.ArgumentParser(
        description="Fit a probe artifact for one target/replacement pair."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint directory")
    parser.add_argument("--train_folder", type=str, required=True, help="Path to training data folder")
    parser.add_argument("--target_property", type=str, required=True, help='Target property, e.g. "holding a gun"')
    parser.add_argument("--replacement_property", type=str, required=True, help='Replacement property, e.g. "holding a coffee"')
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (default: cpu)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--fit_fraction", type=float, default=0.8, help="Fraction of contexts for fitting (default: 0.8)")
    parser.add_argument(
        "--weight_decay_grid",
        type=str,
        default="0,1e-6,1e-5,1e-4,1e-3",
        help="Comma-separated weight decay values (default: 0,1e-6,1e-5,1e-4,1e-3)",
    )
    parser.add_argument(
        "--alpha_multipliers",
        type=str,
        default="0,0.25,0.5,0.75,1,1.5,2,3",
        help="Comma-separated alpha multipliers (default: 0,0.25,0.5,0.75,1,1.5,2,3)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_decay_grid = [float(x) for x in args.weight_decay_grid.split(",")]
    alpha_multipliers = [float(x) for x in args.alpha_multipliers.split(",")]

    # load training dataset
    print(f"Loading training dataset from {args.train_folder}...")
    dataset = h5_dataset_for_folder(args.checkpoint, args.train_folder, device="cpu")
    props = dataset.properties

    # stack all rows
    print("Stacking all embeddings...")
    n = len(dataset)
    X_all = torch.zeros((n, dataset.dim_x))
    M_all = torch.zeros((n, props.n_properties))
    for i in range(n):
        x, m = dataset[i]
        X_all[i] = x
        M_all[i] = m

    # find target and replacement pids
    if args.target_property not in props.property_to_pid:
        raise ValueError(f"target_property '{args.target_property}' not found in dataset")
    if args.replacement_property not in props.property_to_pid:
        raise ValueError(f"replacement_property '{args.replacement_property}' not found in dataset")

    target_pid = props.property_to_pid[args.target_property]
    replacement_pid = props.property_to_pid[args.replacement_property]
    target_cid = props.pid_to_cid[target_pid]

    print(f"Target: {args.target_property} (pid={target_pid}, cid={target_cid})")
    print(f"Replacement: {args.replacement_property} (pid={replacement_pid})")

    # context-key split
    print("Splitting by context...")
    context_keys = []
    for i in range(n):
        active_pids = [
            pid
            for pid in range(props.n_properties)
            if M_all[i, pid] > 0.5 and props.pid_to_cid[pid] != target_cid
        ]
        context_keys.append(tuple(sorted(active_pids)))

    # group row IDs by context_key
    context_to_rows = defaultdict(list)
    for i, ctx in enumerate(context_keys):
        context_to_rows[ctx].append(i)

    # shuffle contexts
    rng = random.Random(args.seed)
    ctx_keys_list = sorted(set(context_keys))
    rng.shuffle(ctx_keys_list)
    n_fit_ctx = int(round(args.fit_fraction * len(ctx_keys_list)))
    fit_ctx = set(ctx_keys_list[:n_fit_ctx])
    cal_ctx = set(ctx_keys_list[n_fit_ctx:])

    fit_ids = [i for i, ctx in enumerate(context_keys) if ctx in fit_ctx]
    cal_ids = [i for i, ctx in enumerate(context_keys) if ctx in cal_ctx]

    print(f"Fit contexts: {len(fit_ctx)}, Cal contexts: {len(cal_ctx)}")
    print(f"Fit rows: {len(fit_ids)}, Cal rows: {len(cal_ids)}")

    # require >= 25 calibration replacement pairs
    cal_pair_contexts = set()
    for ctx in cal_ctx:
        ctx_rows = context_to_rows[ctx]
        has_target = any(M_all[i, target_pid] > 0.5 for i in ctx_rows)
        has_replacement = any(M_all[i, replacement_pid] > 0.5 for i in ctx_rows)
        if has_target and has_replacement:
            cal_pair_contexts.add(ctx)

    if len(cal_pair_contexts) < 25:
        raise ValueError(
            f"Insufficient calibration pairs: {len(cal_pair_contexts)} < 25. "
            f"Increase fit_fraction or use a larger dataset."
        )

    print(f"Calibration contexts with both target and replacement: {len(cal_pair_contexts)}")

    # prepare fit and cal splits
    fit_X = X_all[fit_ids]
    fit_M = M_all[fit_ids]
    cal_X = X_all[cal_ids]
    cal_M = M_all[cal_ids]

    fit_mean = fit_X.mean(dim=0)

    # build category_pid_groups
    category_pid_groups = [
        props.cid_to_pids[cid] for cid in range(props.n_categories)
    ]

    # fit probe
    print("Fitting probe...")
    probe, fit_info = fit_probe(
        fit_X,
        fit_M,
        cal_X,
        cal_M,
        category_pid_groups,
        embedding_dim=dataset.dim_x,
        max_epochs=300,
        batch_size=32,
        lr=1e-3,
        patience=25,
        weight_decay_grid=weight_decay_grid,
        seed=args.seed,
        device=args.device,
    )

    print(f"Selected weight_decay: {fit_info['selected_weight_decay']}")
    print(f"Best cal accuracy: {fit_info['best_cal_accuracy']:.4f}")

    # compute directions
    w_target = probe.linear.weight.data[target_pid]
    w_replacement = probe.linear.weight.data[replacement_pid]
    d_replace = F.normalize(
        (w_replacement - w_target).unsqueeze(0), dim=1
    ).squeeze(0)

    alt_pids = [p for p in props.cid_to_pids[target_cid] if p != target_pid]
    w_alt_mean = probe.linear.weight.data[alt_pids].mean(dim=0)
    d_delete = F.normalize((w_target - w_alt_mean).unsqueeze(0), dim=1).squeeze(0)

    # calibrate replacement alpha
    print("Calibrating replacement alpha...")
    cal_X_centered = cal_X - fit_mean
    
    # collect cal pairs
    replacement_projections = []
    x_old_rows = []
    x_new_rows = []
    
    for ctx in cal_pair_contexts:
        ctx_rows = context_to_rows[ctx]
        cal_ctx_rows = [r for r in ctx_rows if r in cal_ids]
        
        old_rows = [r for r in cal_ctx_rows if M_all[r, target_pid] > 0.5]
        new_rows = [r for r in cal_ctx_rows if M_all[r, replacement_pid] > 0.5]
        
        if old_rows and new_rows:
            # use first of each for simplicity
            old_idx = cal_ids.index(old_rows[0])
            new_idx = cal_ids.index(new_rows[0])
            
            x_old = cal_X_centered[old_idx]
            x_new = cal_X_centered[new_idx]
            
            proj = torch.abs((x_new - x_old) @ d_replace).item()
            replacement_projections.append(proj)
            x_old_rows.append(old_idx)
            x_new_rows.append(new_idx)
    
    replacement_scale = torch.tensor(replacement_projections).median().item()
    print(f"Replacement scale: {replacement_scale:.4f}")
    
    # baseline nontarget accuracy
    x_old_tensor = cal_X_centered[x_old_rows]
    m_old_tensor = cal_M[x_old_rows]
    
    with torch.no_grad():
        logits_baseline = probe(x_old_tensor.to(args.device)).cpu()
        acc_baseline = probe.category_accuracy(logits_baseline, m_old_tensor)
        
        # nontarget categories
        nontarget_mask = torch.ones(props.n_categories, dtype=torch.bool)
        nontarget_mask[target_cid] = False
        baseline_nontarget_acc = acc_baseline[nontarget_mask].mean().item()
    
    print(f"Baseline nontarget accuracy: {baseline_nontarget_acc:.4f}")
    
    # grid search over alphas
    replacement_grid = []
    for mult in alpha_multipliers:
        alpha = replacement_scale * mult
        x_edited = x_old_tensor + alpha * d_replace.unsqueeze(0)
        
        with torch.no_grad():
            logits = probe(x_edited.to(args.device)).cpu()
            acc_per_cat = probe.category_accuracy(logits, m_old_tensor)
            
            nontarget_acc = acc_per_cat[nontarget_mask].mean().item()
            target_acc = acc_per_cat[target_cid].item()
            
            # probabilities
            target_cat_pids = props.cid_to_pids[target_cid]
            target_cat_logits = logits[:, target_cat_pids]
            probs = F.softmax(target_cat_logits, dim=1)
            
            target_local_idx = target_cat_pids.index(target_pid)
            replacement_local_idx = target_cat_pids.index(replacement_pid)
            
            old_target_prob = probs[:, target_local_idx].mean().item()
            replacement_prob = probs[:, replacement_local_idx].mean().item()
            
            # MSE to x_new
            x_new_tensor = cal_X_centered[x_new_rows]
            mse = F.mse_loss(x_edited, x_new_tensor).item()
            
            # out of range
            out_of_range_frac = ((x_edited < 0) | (x_edited > 1)).float().mean().item()
        
        replacement_grid.append({
            "alpha": alpha,
            "multiplier": mult,
            "target_acc": target_acc,
            "replacement_acc": replacement_prob,
            "old_target_prob": old_target_prob,
            "nontarget_acc": nontarget_acc,
            "mse_to_new": mse,
            "out_of_range_frac": out_of_range_frac,
        })
    
    # select best alpha for replacement
    viable_replace = [
        r for r in replacement_grid
        if r["nontarget_acc"] >= baseline_nontarget_acc - 0.01
    ]
    
    if viable_replace:
        # sort by replacement_acc descending, then mse ascending, then alpha ascending
        viable_replace.sort(
            key=lambda r: (-r["replacement_acc"], r["mse_to_new"], r["alpha"])
        )
        best_replace = viable_replace[0]
        replace_alpha = best_replace["alpha"]
        calibration_failed_replace = False
    else:
        replace_alpha = None
        calibration_failed_replace = True
    
    print(f"Replace alpha: {replace_alpha} (failed={calibration_failed_replace})")
    
    # calibrate deletion alpha
    print("Calibrating deletion alpha...")
    deletion_projections = []
    
    for i, old_idx in enumerate(x_old_rows):
        ctx = context_keys[cal_ids[old_idx]]
        ctx_rows = context_to_rows[ctx]
        cal_ctx_rows = [r for r in ctx_rows if r in cal_ids]
        
        # find rows in same context with different property in target category
        alt_rows = [
            r for r in cal_ctx_rows
            if M_all[r, target_pid] <= 0.5  # not target
            and any(M_all[r, p] > 0.5 for p in props.cid_to_pids[target_cid])  # has some property in target category
        ]
        
        if alt_rows:
            alt_indices = [cal_ids.index(r) for r in alt_rows]
            alternative_mean = cal_X_centered[alt_indices].mean(dim=0)
            
            x_old = cal_X_centered[old_idx]
            proj = torch.abs(((x_old - alternative_mean) * d_delete).sum()).item()
            deletion_projections.append(proj)
    
    if deletion_projections:
        deletion_scale = torch.tensor(deletion_projections).median().item()
    else:
        deletion_scale = 0.0
    
    print(f"Deletion scale: {deletion_scale:.4f}")
    
    # grid search for deletion
    deletion_grid = []
    for mult in alpha_multipliers:
        alpha = deletion_scale * mult
        x_edited = x_old_tensor - alpha * d_delete.unsqueeze(0)
        
        with torch.no_grad():
            logits = probe(x_edited.to(args.device)).cpu()
            acc_per_cat = probe.category_accuracy(logits, m_old_tensor)
            
            nontarget_acc = acc_per_cat[nontarget_mask].mean().item()
            
            # suppression rate: fraction where target is no longer top prediction
            target_cat_pids = props.cid_to_pids[target_cid]
            target_cat_logits = logits[:, target_cat_pids]
            pred_local = target_cat_logits.argmax(dim=1)
            target_local_idx = target_cat_pids.index(target_pid)
            suppression_rate = (pred_local != target_local_idx).float().mean().item()
            
            probs = F.softmax(target_cat_logits, dim=1)
            old_target_prob = probs[:, target_local_idx].mean().item()
            
            out_of_range_frac = ((x_edited < 0) | (x_edited > 1)).float().mean().item()
        
        deletion_grid.append({
            "alpha": alpha,
            "multiplier": mult,
            "suppression_rate": suppression_rate,
            "old_target_prob": old_target_prob,
            "nontarget_acc": nontarget_acc,
            "out_of_range_frac": out_of_range_frac,
        })
    
    # select best alpha for deletion
    viable_delete = [
        r for r in deletion_grid
        if r["nontarget_acc"] >= baseline_nontarget_acc - 0.01
    ]
    
    if viable_delete:
        # sort by suppression_rate descending, then old_target_prob ascending, then alpha ascending
        viable_delete.sort(
            key=lambda r: (-r["suppression_rate"], r["old_target_prob"], r["alpha"])
        )
        best_delete = viable_delete[0]
        delete_alpha = best_delete["alpha"]
        calibration_failed_delete = False
    else:
        delete_alpha = None
        calibration_failed_delete = True
    
    print(f"Delete alpha: {delete_alpha} (failed={calibration_failed_delete})")
    
    # write probe artifact
    property_order = [
        props.pid_to_property[pid] for pid in range(props.n_properties)
    ]
    
    artifact = {
        "schema_version": 1,
        "state_dict": probe.state_dict(),
        "fit_mean": fit_mean,
        "category_pid_groups": category_pid_groups,
        "property_order": property_order,
        "target_pid": target_pid,
        "replacement_pid": replacement_pid,
        "delete_direction": d_delete,
        "replace_direction": d_replace,
        "delete_alpha": delete_alpha,
        "replace_alpha": replace_alpha,
        "train_dataset_fingerprint": dataset_fingerprint(dataset),
        "topk": dataset.truncate_embds_topk,
        "normalize": dataset.normalize,
    }
    
    artifact_path = output_dir / "probe_artifact.pt"
    torch.save(artifact, artifact_path)
    print(f"Wrote {artifact_path}")
    
    # write fit summary
    with torch.no_grad():
        cal_logits = probe(cal_X_centered.to(args.device)).cpu()
        cal_acc_per_cat = probe.category_accuracy(cal_logits, cal_M)
    
    fit_summary = {
        "target_property": args.target_property,
        "target_pid": target_pid,
        "replacement_property": args.replacement_property,
        "replacement_pid": replacement_pid,
        "target_cid": target_cid,
        "n_fit_contexts": len(fit_ctx),
        "n_cal_contexts": len(cal_ctx),
        "n_fit_rows": len(fit_ids),
        "n_cal_rows": len(cal_ids),
        "n_calibration_pairs": len(cal_pair_contexts),
        "selected_weight_decay": fit_info["selected_weight_decay"],
        "best_cal_accuracy": fit_info["best_cal_accuracy"],
        "cal_accuracy_per_category": {
            props.cid_to_category[cid]: cal_acc_per_cat[cid].item()
            for cid in range(props.n_categories)
        },
        "replace_alpha": replace_alpha,
        "delete_alpha": delete_alpha,
        "calibration_failed_replace": calibration_failed_replace,
        "calibration_failed_delete": calibration_failed_delete,
        "replacement_scale": replacement_scale,
        "deletion_scale": deletion_scale,
        "delete_direction": d_delete.tolist(),
        "replace_direction": d_replace.tolist(),
    }
    
    summary_path = output_dir / "fit_summary.json"
    summary_path.write_text(json.dumps(fit_summary, indent=2))
    print(f"Wrote {summary_path}")
    
    # write calibration grids
    grid_path = output_dir / "calibration_grid.csv"
    with open(grid_path, "w") as f:
        f.write("operation,alpha,multiplier,metric,value\n")
        for r in replacement_grid:
            for key in ["target_acc", "replacement_acc", "old_target_prob", "nontarget_acc", "mse_to_new", "out_of_range_frac"]:
                f.write(f"replace,{r['alpha']},{r['multiplier']},{key},{r[key]}\n")
        for r in deletion_grid:
            for key in ["suppression_rate", "old_target_prob", "nontarget_acc", "out_of_range_frac"]:
                f.write(f"delete,{r['alpha']},{r['multiplier']},{key},{r[key]}\n")
    print(f"Wrote {grid_path}")
    
    # write run manifest
    manifest_path = write_run_manifest(
        output_dir,
        run_kind="probe_intervention_fit",
        seed=args.seed,
        config={
            "checkpoint": args.checkpoint,
            "train_folder": args.train_folder,
            "target_property": args.target_property,
            "replacement_property": args.replacement_property,
            "fit_fraction": args.fit_fraction,
            "weight_decay_grid": weight_decay_grid,
            "alpha_multipliers": alpha_multipliers,
        },
        dataset=dataset,
        model_fingerprint=checkpoint_fingerprint(args.checkpoint),
    )
    print(f"Wrote {manifest_path}")
    
    print("Done.")


if __name__ == "__main__":
    main()
