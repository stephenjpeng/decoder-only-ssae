# Research Proposal: Extended Model Benchmarking for Decoder-Only SSAEs

**Goal:** Extend Oun's existing decoder-only SSAE work (arXiv:2602.00924) by benchmarking the newly added model variants and backbones against the existing evaluation suite (reconstruction, compositional holdout, image-level, and magnitude-sensitivity metrics). The objective is a workshop paper submission within the next 2-3 months, built on existing infrastructure with expanded experimental scope for stronger empirical results.

**Team & time:** Stephen Peng (~5 hrs/week), Ouns El Harzli (~1 hr/week, advisory + co-authorship), through submission.
- Partners: limited engagement from other co-authors at NVIDIA / KAIST
- Expert advisory: TBD. Noah? Honestly could be anyone, Noah is an easy fit if we want it in AI Safety Lab

**Why this is a quick win:** Prompt generation, embedding extraction, training, and the full evaluation suite (reconstruction, holdout, image benchmark, magnitude sensitivity, VLM judging, CLIP probing) are already implemented end-to-end. Scaling up means running more configurations through the existing pipeline, not building anything new.

---

## Why this matters for BCG X and its clients

- **Controllable GenAI is an unmet enterprise need.** Clients deploying image/text generation currently rely on brittle prompt engineering to control brand, safety, and factual attributes; a sparse, editable feature space gives a training-free control layer over what the model expresses.
- **Interpretability is becoming a governance requirement, not a nice-to-have.** Regulators (EU AI Act), model-risk functions in FS, and brand-safety reviewers all need auditable answers to "what concepts drove this output," which opaque prompts can't provide but decomposed sparse features can.
- **Direct line to monetizable use cases.** Attribute-controlled creative production (marketing, retail catalog, media), synthetic data generation with steerable distributions, and brand-consistent content pipelines are all active BCG X delivery areas where this method plugs in directly.

---

## Timeline & target venues

The objective is a workshop paper submission cycle. Given today (mid-July 2026), the ICML 2026 workshop deadlines (Mechanistic Interpretability, Compositional Learning, Foundations of Deep Generative Models — arguably the best topical fit for this work) have already passed. That leaves a tiered venue plan:

| Tier | Venue | Est. paper deadline | Conference | Notes |
|---|---|---|---|---|
| **Primary** | NeurIPS 2026 workshops (Sydney / Paris / Atlanta) | ~Aug 29, 2026 (AoE, suggested) | Dec 11–13, 2026 | Accepted-workshop list just decided (Jul 11); confirm specific workshop CFPs in next 1–2 weeks. Watch for interpretability, UniReps/NeurReps, and generative-models workshops. |
| **Backup 1** | AAAI-27 workshops (Montréal) | ~Oct–Dec 2026 (inferred from AAAI-26 cadence) | Feb 16–23, 2027 | Sooner second shot if NeurIPS slips; workshop 2027 CFPs not yet published. |
| **Backup 2 / stretch** | ICLR 2027 workshops (Brazil) | ~Jan–Feb 2027 (inferred from ICLR-26 cadence) | Apr 24–28, 2027 | Better thematic fit than AAAI ("Learning Representations"); pushes timeline further. |
| **Next-cycle target** | ICML 2027 workshops | ~May 2027 (inferred) | ~Jul 2027 | Best topical home if this cycle slips or a v2/extended paper follows — Mech Interp + Compositional Learning + Foundations of Deep Generative Models trio historically co-locates here. |

Working plan: aim squarely at NeurIPS 2026 (Aug 29), hold AAAI-27 and ICLR 2027 as sequential fallbacks, flag ICML 2027 as the natural home for any follow-up.

---

## Resource request: compute — $10,000

### Experiment plan

| Workstream | Scope | Est. GPU-hours |
|---|---|---|
| Embedding extraction | Re-extract full prompt set (~74k train / 18k holdout, already generated) across 3 backbones (SD3.5 T5, Gemma-2-2B-it, generic HF causal LM) | ~250 |
| Training sweeps | 3 decoder variants x 3 backbones x hyperparameter grid (n_repeat, batch size, LR schedule) x 3 seeds for statistical robustness | ~200 |
| Image benchmark + magnitude sensitivity | Full holdout set, all variant/backbone combos, wider concept + magnitude coverage, CLIP probe + DINO similarity | ~450 |
| Deadline buffer | Reruns, failed configs, last-mile iteration before submission | ~150 |
| **Total** | | **~1,050 GPU-hours** |

### Pricing

Budgeted at on-demand (not spot) A100/H100-class rates (~$3-4/GPU-hr) rather than preemptible spot instances, since spot interruptions are a real risk during the final crunch before a deadline. That puts GPU compute at roughly **$3,000-4,200**.

The remaining **~$5,800-7,000** of the $10,000 ask is explicit headroom for: OpenAI vision-judge API calls at full holdout scale (~$300-500), multi-GPU runs if sweeps need to parallelize to hit the timeline, and unplanned re-runs — the single biggest risk to a "submission-ready in a few months" timeline with only 5+1 hrs/week of hands-on time is compute-bound iteration latency, not experiment design.

---

## Engineering backlog

### Elbow metric for `truncate_embds_topk` selection

**Status:** implemented on branch `feat/add-elbow-topk-analysis` (`analysis/elbow_topk.py`). Smoke-tested on `results/mps_run/` (1500 prompts x ~1.37M dims): kneedle returns k=1896, cumulative@0.95 returns k=922k (long tail dominates), 2nd-deriv returns k=5 (first sharp drop). The current hard-coded k=1000 is slightly conservative but sits in the same order of magnitude as the empirical kneedle answer. Follow-up (raise the YAML default to ~2000 or leave it) is still open.

**Problem.** `truncate_embds_topk` is hard-coded to 1000 in `params_default.yaml`. The underlying selection criterion in `get_indices_truncate_embds_topk()` is `diff = max(X, dim=0) - min(X, dim=0)`, sorted descending. Picking k=1000 is arbitrary; the elbow of that sorted diff curve is the principled cutoff where additional dimensions carry negligible range variance.

**Implementation: `analysis/elbow_topk.py`**

A standalone script (no changes to the training pipeline) that:

1. Loads the dataset with `truncate_embds_topk=None` to get the full sorted diff vector.
2. Applies three detection methods:
   - **Perpendicular distance (Kneedle-style):** normalize the curve to unit square, draw the chord from `(0, diff[0])` to `(N-1, diff[N-1])`, find k with maximum perpendicular distance to that chord. Parameter-free; works well when the curve is convex.
   - **Cumulative explained range:** find the smallest k where `cumsum(diff[:k]) / sum(diff) >= threshold` (default 0.95). Interpretable and easy to report in the paper.
   - **Second-derivative peak:** find k where the discrete second derivative of the sorted diff is maximized (steepest "bend"). Useful as a sanity check; may need light Gaussian smoothing first.
3. Outputs a two-panel matplotlib figure saved alongside the embeddings:
   - Top panel: sorted diff curve on log-y scale with vertical lines marking each method's k.
   - Bottom panel: cumulative explained-range curve with the threshold line.
4. Saves a JSON file with the three recommended k values so the result can be loaded into the YAML without re-running.

**CLI:**
```bash
python analysis/elbow_topk.py \
    --folder_path results/compositional_split/train/ \
    --threshold 0.95 \
    --out results/compositional_split/train/elbow_analysis.png
```

**Dependencies:** only `numpy`, `matplotlib`, `torch`, `h5py` — all already in `requirements.txt`. No new packages needed.

**Follow-up (optional).** If the perpendicular-distance k is stable across backbones/datasets, replace the hard-coded 1000 in `params_default.yaml` with the derived value and add a note in the paper that the truncation threshold was chosen empirically via the elbow criterion. If it varies, include the per-backbone k values as a table in the methods section.

### PCA-elbow variant (`analysis/elbow_pca.py`)

**Status:** implemented on the same branch. Diagnostic only — does not modify the training pipeline (a PCA truncation would need `H5Dataset` to store a `(K, d)` projection matrix and apply it in `__getitem__`, which is out of scope for this experiment).

**Method.** Loads full X, centres it, and gets the singular values via `eigh(X_c @ X_c.T)` — the Gram trick keeps memory to `n x n` rather than materialising V of shape `(d, d)`. Runs the same three elbow detectors on the sorted explained-variance-ratio curve.

**Smoke-test on `results/mps_run/`** (1500 prompts x 1.37M dims, so max_components = 1500):

| Method | k (components) | Cumulative explained variance |
|---|---|---|
| kneedle | **46** | 0.92 |
| cumulative @ 0.95 | 73 | 0.95 |
| second derivative | 5 | 0.43 |

**Interpretation.** PCA truncation dominates range-based top-K on this dataset by roughly 40x (k=46 vs. k=1896 for comparable information retention). Caveats before drawing paper-worthy conclusions:
- n_prompts caps the spectrum at 1500 here. On the full compositional_split train set (~74k prompts) the spectrum extends further and the elbow may shift; re-run with `--n_prompts_subsample` to gauge stability.
- This dataset has only 7 categories x 28 properties — low intrinsic dimensionality, so PCA has an easy job. More diverse prompts on other backbones (Gemma, generic HF LM) may need many more components.
- The pipeline change to actually adopt PCA truncation (projection matrix in dataloader, inverse projection in inference for image regeneration, storage format for the projection) is a meaningful lift — flag this as a follow-up experiment, not a drop-in swap.

**Run:**
```bash
python -m analysis.elbow_pca --folder_path results/mps_run/ --threshold 0.95
```

### PCA truncation as a training-time option (planned, not implemented)

**Goal.** Let the training pipeline optionally replace range-based top-K dim selection with a PCA projection. Both would live behind the same K-dim SSAE input, but PCA would learn on axes of maximum variance rather than raw dimensions.

#### Semantic decision (the crux)

The current top-K design is **partial editing**, not compression: `inference/abstract.py::overwrite_full_embedding` scatter-writes the SSAE's K-dim output into the corresponding indices of the source prompt's full-d embedding, leaving the other d - K dims untouched. That has no direct PCA analog because every original dim is a linear combination of every PC.

Two candidate semantics, in decreasing order of comparability to the current design:

- **Option A: residual add-back (recommended).** Split the source embedding into `x = P.T P (x - mean) + mean + residual`, where `residual = (I - P.T P) (x - mean)` lives in the null space of `P`. SSAE reconstructs the PCA-space K-dim vector; final output is `x_recon = P.T y + mean + residual_source`. This matches the "edit only what the SSAE controls, keep everything else from the source prompt" semantics of the current pipeline.
- **Option B: full replacement.** Discard the residual entirely: `x_recon = P.T y + mean`. Simpler and honest about lossiness, but changes the editing story from "steer K features" to "regenerate the whole embedding, lossy." Interesting as an ablation, probably not the default.

Recommend implementing (A) as the default with (B) available for ablations. This decision needs to be flagged in whichever paper section describes the truncation choice.

#### Config surface

New sibling keys in `dataloader.*`:

```yaml
dataloader:
  truncate_embds_topk: 1000        # existing; K
  truncate_embds_method: "range"   # new; "range" (default) | "pca"
  pca_semantics: "residual"        # new; "residual" (default) | "replace"; only used when method == "pca"
```

Keeping `truncate_embds_topk` as the K knob avoids proliferating names. Default `range` preserves backwards compatibility.

#### Storage layout (in `folder_path`)

- Range mode (unchanged): `indices_top_K.json`, `embds_max_top_K.json`, `embds_min_top_K.json`.
- PCA mode (new): `pca_top_K.npz` containing `mean` (d,), `components` (K, d), `singular_values` (K,), `explained_variance_ratio` (K,). Plus `embds_max_pca_K.json` / `embds_min_pca_K.json` for the post-projection MAX_MIN normalization stats.

Cache-and-reuse pattern mirrors `indices_top_K.json`: compute on first run, load thereafter.

#### Dataloader changes (`trainings/dataloader/dataloader.py`)

- `truncate_embds()` branches on `truncate_embds_method`.
- New `get_pca_projection()` that computes `(mean, P, sigma)` via the Gram trick (`eigvalsh(X_c @ X_c.T)`, top-K eigenvectors -> project back to component space with `P = V_topK.T @ X_c / sigma_topK`).
- `__getitem__` PCA path: `embds_pca = (embds - mean) @ P.T`, applied after the raw concat, before normalization. Cost is one `(K, d)` matmul per sample per batch — fine at K <= a few thousand.
- Normalization (`get_min_max_X`) already computes per-dim min/max on the truncated space; works unchanged as long as it runs after projection.

Cache the projection as a `torch.Tensor` on `self` at init so `__getitem__` doesn't reload it.

#### Inference changes (`inference/abstract.py`)

Two call sites to update:

- `overwrite_full_embedding` currently does `full_embd[indices] = embd_topk`. New PCA branch: `full_embd = P.T @ embd_topk + mean + residual_source` where `residual_source = full_embd_source - P.T @ (P @ (full_embd_source - mean)) - mean`. In `full_replace` mode drop the residual term.
- `get_full_embedding` reloads the source's full-d embedding regardless — no change needed there, it's already truncation-agnostic.

The training-run output folder should record which mode was used (already implicit in the YAML that's saved alongside checkpoints).

#### Compute plan for real training sets

`mps_run` has n=1500 prompts (Gram = 9 MB, trivial). `compositional_split/train/` has ~74k prompts:

- Gram matrix `X_c @ X_c.T` is 74k x 74k = ~22 GB float32. Borderline for a single node. Options: float64 half-precision the Gram alone (11 GB), or fall back to `torch.pca_lowrank(X_c, q=K, niter=4)` for K << n — much cheaper (O(n * d * K)) and empirically accurate to 3-4 decimal places on random-Gaussian tests.
- Getting `P` from Gram eigenvectors requires `V_topK.T @ X_centered` = `(K, n) @ (n, d)` = `(K, d)` matmul. For K=100, d=1.37M, that's a 137M-parameter output — 550 MB, one-time cost.
- Recommendation: default to `torch.pca_lowrank` with `niter=6`, exact Gram-trick path behind a `--exact` flag on the caching preprocessor.

#### Backwards compatibility

- Range-mode YAML runs unchanged: `truncate_embds_method` defaults to `"range"`.
- Existing checkpoints have no PCA metadata, so inference code should refuse to load a range-trained checkpoint with `truncate_embds_method: pca` (and vice versa). Simplest signal: presence/absence of `pca_top_K.npz` next to `indices_top_K.json`.

#### Validation plan (before training a real model)

1. **Round-trip reconstruction sanity.** For a held-out prompt, verify `MSE(x, P.T P (x - mean) + mean + residual) < 1e-6` (should be exact up to float32 noise).
2. **Variance-retained sanity.** Confirm `1 - explained_variance_at_K` matches the empirical MSE of the truncated + inverted reconstruction against the source, on holdout.
3. **Small training smoke test.** Retrain `model_avg_feature` on `mps_run` at K=46 (kneedle answer) with `truncate_embds_method: pca` for 5 epochs. Compare final training MSE and a couple of generated images against the range-truncated baseline at K=1000. If reconstruction quality is comparable at 20x smaller K, the option is worth carrying into the full sweep.

#### Open questions / risks

- **Normalization interaction.** MAX_MIN normalization on projected coords could be lopsided if a few PCs dominate; consider skipping normalization in PCA mode (or per-PC z-score).
- **Editability.** PCs are dense in the original embedding space, so "edit PC 3" doesn't correspond to a nameable concept the way "edit dim 42" arguably could. Only matters if the paper story leans on dim-level interpretability of the raw top-K.
- **Sweep cost.** Adding a `range` vs. `pca` axis to the existing (variant x backbone x seed) grid roughly doubles the training compute in the proposal — budget accordingly if this becomes a headline experiment.
