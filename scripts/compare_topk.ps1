<#
.SYNOPSIS
    Sweep 1-layer SSAE performance across truncate_embds_topk in {500, 1000, 2000, 5000}.

.DESCRIPTION
    Pipeline:
      1. (Optional) Regenerate the compositional split (prompts.json for train/ + holdout/).
      2. (Optional) Extract SD3.5 text embeddings for both splits.
      3. For each topk value:
           - Write a per-run YAML with truncate_embds_topk = k.
           - Train a 1-layer SSAE (num_layers=1, hidden_dims=null).
           - Copy the training folder's truncation sidecars to holdout/.
           - Run evaluation.run_compositional_embeddings; capture mse/cosine.
      4. Aggregate all runs into <RunsRoot>/sweep_summary.csv.

    Existing splits and embeddings are reused unless the corresponding
    -Recreate* switch is set. Training always runs (that's the sweep).

.PARAMETER RepoRoot
    Repository root; defaults to the current directory.

.PARAMETER Categories
    Path to categories_with_properties.json (relative to RepoRoot ok).

.PARAMETER SplitRoot
    Directory holding train/ and holdout/ subfolders.

.PARAMETER RunsRoot
    Where per-topk checkpoints and the summary CSV live.

.PARAMETER TopK
    Truncation values to sweep.

.PARAMETER HoldoutFraction
    Fraction of the full factorial reserved as holdout (compositional_split).

.PARAMETER MaxTrainPrompts
    Cap on train prompts after the split (optional).

.PARAMETER MaxHoldoutPrompts
    Cap on holdout prompts after the split (optional).

.PARAMETER Backbone
    Backbone name registered in backbones/. Default: sd35_turbo_text_only.

.PARAMETER RecreateSplit
    Rebuild the compositional split even if train/prompts.json exists.

.PARAMETER RecreateEmbeddings
    Re-extract embeddings even if <split>/embds/manifest.json exists.

.PARAMETER Python
    Python interpreter (default: python).

.EXAMPLE
    ./scripts/compare_topk.ps1

.EXAMPLE
    ./scripts/compare_topk.ps1 -RecreateSplit -RecreateEmbeddings -TopK 500,1000,2000,5000
#>

[CmdletBinding()]
param(
    [string]$RepoRoot = (Get-Location).Path,
    [string]$Categories = "dataset_generation/prompts/input/categories_with_properties.json",
    [string]$SplitRoot = "results/compositional_split",
    [string]$RunsRoot = "results/topk_sweep",
    [int[]]$TopK = @(500, 1000, 2000, 5000),
    [double]$HoldoutFraction = 0.1,
    [int]$MaxTrainPrompts = 0,
    [int]$MaxHoldoutPrompts = 0,
    [string]$Backbone = "sd35_turbo_text_only",
    [switch]$RecreateSplit,
    [switch]$RecreateEmbeddings,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$TrainDir   = Join-Path $SplitRoot "train"
$HoldoutDir = Join-Path $SplitRoot "holdout"
New-Item -ItemType Directory -Path $RunsRoot -Force | Out-Null

function ToPosix([string]$p) { return ($p -replace '\\', '/') }

function Invoke-Py {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Argv)
    Write-Host ">> $Python $($Argv -join ' ')" -ForegroundColor Cyan
    & $Python @Argv
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Python $($Argv -join ' ')" }
}

# 1. Compositional split -----------------------------------------------------
$trainPrompts = Join-Path $TrainDir "prompts.json"
if ($RecreateSplit -or -not (Test-Path $trainPrompts)) {
    Write-Host "== Building compositional split ==" -ForegroundColor Green
    if (Test-Path $SplitRoot) { Remove-Item -Recurse -Force $SplitRoot }
    $splitArgs = @(
        "dataset_generation/compositional_split.py",
        "--categories_json", $Categories,
        "--output_root", $SplitRoot,
        "--holdout_fraction", $HoldoutFraction
    )
    if ($MaxTrainPrompts   -gt 0) { $splitArgs += @("--max_train_prompts",   "$MaxTrainPrompts") }
    if ($MaxHoldoutPrompts -gt 0) { $splitArgs += @("--max_holdout_prompts", "$MaxHoldoutPrompts") }
    Invoke-Py @splitArgs
} else {
    Write-Host "Split already at $SplitRoot; use -RecreateSplit to rebuild." -ForegroundColor Yellow
}

# 2. Embeddings --------------------------------------------------------------
function Extract-Embeddings {
    param([string]$Folder)
    $prompts = Join-Path $Folder "prompts.json"
    if (-not (Test-Path $prompts)) { throw "Missing $prompts" }
    $manifest = Join-Path $Folder "embds/manifest.json"
    if (-not $RecreateEmbeddings.IsPresent -and (Test-Path $manifest)) {
        Write-Host "Embeddings already present under $Folder; skipping (use -RecreateEmbeddings to rebuild)." -ForegroundColor Yellow
        return
    }
    $embdsDir = Join-Path $Folder "embds"
    if (Test-Path $embdsDir) { Remove-Item -Recurse -Force $embdsDir }
    Invoke-Py @(
        "get_embeddings.py",
        "--backbone", $Backbone,
        "--prompts", $prompts,
        "--out", $Folder,
        "--categories", $Categories
    )
}

Write-Host "== Extracting embeddings (train) ==" -ForegroundColor Green
Extract-Embeddings $TrainDir
Write-Host "== Extracting embeddings (holdout) ==" -ForegroundColor Green
Extract-Embeddings $HoldoutDir

# 3. Sweep -------------------------------------------------------------------
$SummaryCsv = Join-Path $RunsRoot "sweep_summary.csv"
"topk,mse_mean,cosine_mean,n_holdout,elapsed_sec,output_folder" | Set-Content $SummaryCsv -Encoding utf8

foreach ($k in $TopK) {
    Write-Host "== topk = $k ==" -ForegroundColor Green
    $runDir     = Join-Path $RunsRoot ("topk_{0}" -f $k)
    $yamlPath   = Join-Path $RunsRoot ("topk_{0}.yaml" -f $k)
    $metricsOut = Join-Path $runDir "holdout_compositional_metrics.json"

    $trainDirPosix = ToPosix $TrainDir
@"
training:
  model:
    model_name: "model_avg_feature"
    using_blocs: False
    num_layers: 1
    hidden_dims: null
  dataloader:
    folder_path: "$trainDirPosix/"
    truncate_n_prompts: null
    truncate_embds_topk: $k
    add_property_is_the_same: True
    normalize: "MAX_MIN"
    num_workers: 1
    simulated:
      simulated: False
      dim_clip_simulated: 100
  training:
    n_epochs: 100
    print_frequency: 1
    save_model_frequency: null
    plot_frequency: 1
    seed: 0
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
"@ | Set-Content -Path $yamlPath -Encoding utf8

    $t0 = Get-Date
    Invoke-Py @(
        "training_cli.py",
        "--output_folder", $runDir,
        "--path_yaml", $yamlPath,
        "--overwrite_output", "True",
        "--num_layers", "1"
    )
    $elapsed = [int]((Get-Date) - $t0).TotalSeconds

    # The training run just wrote indices_top_{k}.json + embds_{max,min}_top_{k}.json
    # under $TrainDir. Copy them next to the holdout so the eval reuses the same
    # coordinate selection and per-dim normalization.
    Invoke-Py @(
        "-m", "evaluation.run_copy_truncation",
        "--train_folder",   $TrainDir,
        "--holdout_folder", $HoldoutDir
    )

    Invoke-Py @(
        "-m", "evaluation.run_compositional_embeddings",
        "--checkpoint",     $runDir,
        "--holdout_folder", $HoldoutDir,
        "--ssae_device",    "cpu",
        "--output_json",    $metricsOut
    )

    $mse = ""; $cos = ""; $nHold = ""
    if (Test-Path $metricsOut) {
        $obj = Get-Content $metricsOut -Raw | ConvertFrom-Json
        if ($null -ne $obj.mse_mean)    { $mse   = $obj.mse_mean }
        if ($null -ne $obj.cosine_mean) { $cos   = $obj.cosine_mean }
        if ($null -ne $obj.n_holdout)   { $nHold = $obj.n_holdout }
    }
    "$k,$mse,$cos,$nHold,$elapsed,$runDir" | Add-Content $SummaryCsv
    Write-Host ("  topk={0} mse={1} cosine={2} elapsed={3}s" -f $k, $mse, $cos, $elapsed) -ForegroundColor Green
}

Write-Host "== Sweep complete ==" -ForegroundColor Green
Import-Csv $SummaryCsv | Format-Table -AutoSize
Write-Host "Summary: $SummaryCsv"
