<#
.SYNOPSIS
    Sweep SSAE performance across truncate_embds_topk and num_layers.

.DESCRIPTION
    Trains the cross product TopK x Layers. For num_layers > 1, each hidden
    layer is HiddenDim wide (build_head broadcasts the single int).

    Pipeline:
      1. (Optional) Regenerate the compositional split (prompts.json for train/ + holdout/).
      2. (Optional) Extract SD3.5 text embeddings for both splits.
      3. For each (topk, layers) pair:
           - Write a per-run YAML with truncate_embds_topk = k and num_layers = L.
           - Train the SSAE (unless -SkipTraining).
           - Copy the training folder's truncation sidecars to holdout/.
           - Run evaluation.run_compositional_embeddings; capture mse/cosine.
           - Optionally run evaluation.run_image_benchmark (with -RunImageBenchmark).
      4. Aggregate all runs into <RunsRoot>/sweep_summary.csv.

    Existing splits and embeddings are reused unless the corresponding
    -Recreate* switch is set. Training always runs (that's the sweep), unless
    -SkipTraining is passed (useful with -RunImageBenchmark for a
    benchmarks-only pass over previously trained checkpoints).

    The image benchmark shares baselines across runs via the per-dataset cache
    at results/bench_baseline_cache/<dataset_id>/ — the first run populates
    baselines, subsequent runs render only ssae_compose.

.PARAMETER RepoRoot
    Repository root; defaults to the current directory.

.PARAMETER Categories
    Path to categories_with_properties.json (relative to RepoRoot ok).

.PARAMETER PropertiesSame
    Path to properties_same.json — copied into train/ and holdout/ by
    compositional_split so the H5Dataset can load it. Defaults to
    dataset_generation/prompts/input/properties_same.json.

.PARAMETER SplitRoot
    Directory holding train/ and holdout/ subfolders.

.PARAMETER RunsRoot
    Where per-run checkpoints and the summary CSV live.

.PARAMETER TopK
    Truncation values to sweep.

.PARAMETER Layers
    Decoder-head layer counts to sweep. 1 = single Linear (historical).
    N > 1 = N Linears with ReLU between; each hidden layer is HiddenDim wide.

.PARAMETER HiddenDim
    Width of each hidden layer when Layers > 1. Ignored for Layers == 1.

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

.PARAMETER SkipTraining
    Skip the training step. Existing run folders under <RunsRoot>/<tag>/ are
    assumed to hold a trained checkpoint. Meant to be combined with
    -RunImageBenchmark.

.PARAMETER RunImageBenchmark
    After (or instead of) training each (topk, layers) checkpoint, run
    evaluation.run_image_benchmark against the holdout. Baselines share the
    per-dataset cache at -BaselineCacheRoot.

.PARAMETER BenchRoot
    Root directory for image-benchmark output folders. Each run lands under
    <BenchRoot>/<tag>. Default: results/bench_out.

.PARAMETER BaselineCacheRoot
    Shared per-dataset baseline cache root passed to run_image_benchmark.
    Default: results/bench_baseline_cache.

.PARAMETER BenchmarkMaxSamples
    Cap on holdout samples rendered per method. 0 = no cap.

.PARAMETER BenchmarkSdDevice
    Device for the SD3.5 pipeline (default: cuda).

.PARAMETER BenchmarkSsaeDevice
    Device for the SSAE decoder. Empty = defaults to BenchmarkSdDevice.

.PARAMETER BenchmarkBaselineDevice
    Device for ridge / mean-arithmetic predictions. Empty = defaults to
    BenchmarkSsaeDevice.

.PARAMETER LocalityDrop
    Enable the drop-one-attribute locality test.

.PARAMETER LocalitySwap
    Enable the value-swap locality test.

.PARAMETER BenchmarkDino
    Compute DINOv2 cosine similarity vs the GT-embed reference image.

.PARAMETER BenchmarkSimulated
    Skip real SD3 rendering; write placeholder PNGs. For smoke-testing the
    pipeline end-to-end.

.PARAMETER Python
    Python interpreter (default: python).

.EXAMPLE
    # Default: 4 topk x 1 layer count = 4 runs
    ./scripts/compare_topk.ps1

.EXAMPLE
    # Sweep + image benchmark in one go
    ./scripts/compare_topk.ps1 -Layers 1,2 -HiddenDim 1024 -RunImageBenchmark -LocalityDrop

.EXAMPLE
    # Skip training; run image benchmarks over existing checkpoints
    ./scripts/compare_topk.ps1 -TopK 500,1000,2000,5000 -Layers 1,2 -SkipTraining -RunImageBenchmark
#>

[CmdletBinding()]
param(
    [string]$RepoRoot = (Get-Location).Path,
    [string]$Categories = "dataset_generation/prompts/input/categories_with_properties.json",
    [string]$PropertiesSame = "dataset_generation/prompts/input/properties_same.json",
    [string]$SplitRoot = "results/compositional_split",
    [string]$RunsRoot = "results/topk_sweep",
    [int[]]$TopK = @(500, 1000, 2000, 5000),
    [int[]]$Layers = @(1),
    [int]$HiddenDim = 1024,
    [double]$HoldoutFraction = 0.1,
    [int]$MaxTrainPrompts = 0,
    [int]$MaxHoldoutPrompts = 0,
    [string]$Backbone = "sd35_turbo_text_only",
    [switch]$RecreateSplit,
    [switch]$RecreateEmbeddings,
    [switch]$SkipTraining,
    [switch]$RunImageBenchmark,
    [string]$BenchRoot = "results/bench_out",
    [string]$BaselineCacheRoot = "results/bench_baseline_cache",
    [int]$BenchmarkMaxSamples = 0,
    [string]$BenchmarkSdDevice = "cuda",
    [string]$BenchmarkSsaeDevice = "",
    [string]$BenchmarkBaselineDevice = "",
    [switch]$LocalityDrop,
    [switch]$LocalitySwap,
    [switch]$BenchmarkDino,
    [switch]$BenchmarkSimulated,
    [string]$Python = "python"
)

if ($Layers | Where-Object { $_ -lt 1 }) { throw "Layers must all be >= 1" }
if (($Layers | Where-Object { $_ -gt 1 }) -and $HiddenDim -lt 1) {
    throw "HiddenDim must be >= 1 when Layers contains values > 1"
}

$ErrorActionPreference = "Stop"

function ToPosix([string]$p) { return ($p -replace '\\', '/') }

# Resolve $RepoRoot to an absolute POSIX path so every downstream Python call
# gets forward slashes only. On Windows, backslashed paths passed through the
# python launcher shim can get doubled up as "\\", which argparse then treats
# as a literal path containing a real backslash sequence.
$RepoRoot = ToPosix (Resolve-Path -LiteralPath $RepoRoot).Path
Set-Location -LiteralPath $RepoRoot

function Resolve-Under([string]$Base, [string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) { return (ToPosix $Path) }
    return (ToPosix (Join-Path $Base $Path))
}

$Categories        = Resolve-Under $RepoRoot $Categories
$PropertiesSame    = Resolve-Under $RepoRoot $PropertiesSame
$SplitRoot         = Resolve-Under $RepoRoot $SplitRoot
$RunsRoot          = Resolve-Under $RepoRoot $RunsRoot
$BenchRoot         = Resolve-Under $RepoRoot $BenchRoot
$BaselineCacheRoot = Resolve-Under $RepoRoot $BaselineCacheRoot

if (-not (Test-Path -LiteralPath $Categories)) {
    throw "Categories file not found: $Categories`n" +
          "Pass -Categories <path> or run from the repo root. RepoRoot is currently: $RepoRoot"
}
if (-not (Test-Path -LiteralPath $PropertiesSame)) {
    throw "properties_same.json not found: $PropertiesSame`n" +
          "Pass -PropertiesSame <path>. RepoRoot is currently: $RepoRoot"
}

$TrainDir   = ToPosix (Join-Path $SplitRoot "train")
$HoldoutDir = ToPosix (Join-Path $SplitRoot "holdout")
New-Item -ItemType Directory -Path $RunsRoot -Force | Out-Null

function Invoke-Py {
    # NOTE: Callers pass a single array (positional). `@(...)` is an array
    # literal, not a splat — using ValueFromRemainingArguments here would nest
    # the array and Python would see the whole arg list as one filename.
    param([Parameter(Mandatory=$true, Position=0)][string[]]$Argv)
    # POSIX-normalize every arg that looks like a path (any backslash present).
    $normalized = @()
    foreach ($a in $Argv) {
        if ($a -match '\\') { $normalized += (ToPosix $a) } else { $normalized += $a }
    }
    Write-Host ">> $Python $($normalized -join ' ')" -ForegroundColor Cyan
    & $Python @normalized
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Python $($normalized -join ' ')" }
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
        "--holdout_fraction", $HoldoutFraction,
        "--properties_same_json", $PropertiesSame
    )
    if ($MaxTrainPrompts   -gt 0) { $splitArgs += @("--max_train_prompts",   "$MaxTrainPrompts") }
    if ($MaxHoldoutPrompts -gt 0) { $splitArgs += @("--max_holdout_prompts", "$MaxHoldoutPrompts") }
    Invoke-Py $splitArgs
} else {
    Write-Host "Split already at $SplitRoot; use -RecreateSplit to rebuild." -ForegroundColor Yellow
}

# The dataloader reads <split>/properties_same.json directly. Older splits made
# before compositional_split copied it don't have this file — top it up now.
foreach ($d in @($TrainDir, $HoldoutDir)) {
    $dst = Join-Path $d "properties_same.json"
    if (-not (Test-Path -LiteralPath $dst)) {
        Copy-Item -LiteralPath $PropertiesSame -Destination $dst
        Write-Host "Copied properties_same.json -> $dst" -ForegroundColor Yellow
    }
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
$SummaryCsv = ToPosix (Join-Path $RunsRoot "sweep_summary.csv")
"topk,layers,hidden_dim,mse_mean,cosine_mean,n_holdout,elapsed_sec,output_folder,bench_dir" | Set-Content -Path $SummaryCsv -Encoding utf8

if ($RunImageBenchmark) { New-Item -ItemType Directory -Path $BenchRoot -Force | Out-Null }

# $TrainDir is already POSIX at this point.
$trainDirPosix = $TrainDir

foreach ($k in $TopK) {
    foreach ($L in $Layers) {
        $tag = if ($L -eq 1) { "topk_${k}_L1" } else { "topk_${k}_L${L}_h${HiddenDim}" }
        Write-Host "== $tag ==" -ForegroundColor Green

        $runDir     = ToPosix (Join-Path $RunsRoot $tag)
        $yamlPath   = ToPosix (Join-Path $RunsRoot "$tag.yaml")
        $metricsOut = ToPosix (Join-Path $runDir "holdout_compositional_metrics.json")

        # YAML head shape. `num_layers` and `hidden_dims` are also overridden by
        # the CLI below; keeping them in the YAML too keeps the file self-describing.
        $hiddenYaml = if ($L -eq 1) { "null" } else { "$HiddenDim" }
@"
training:
  model:
    model_name: "model_avg_feature"
    using_blocs: False
    num_layers: $L
    hidden_dims: $hiddenYaml
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

        $elapsed = 0
        if ($SkipTraining) {
            if (-not (Test-Path -LiteralPath (Join-Path $runDir "model.pt"))) {
                throw "Missing checkpoint at $runDir/model.pt; drop -SkipTraining or train it first."
            }
            Write-Host "  -SkipTraining set; reusing checkpoint at $runDir" -ForegroundColor Yellow
        } else {
            $trainArgs = @(
                "training_cli.py",
                "--output_folder", $runDir,
                "--path_yaml", $yamlPath,
                "--overwrite_output", "True",
                "--num_layers", "$L"
            )
            if ($L -gt 1) { $trainArgs += @("--hidden_dims", "$HiddenDim") }

            $t0 = Get-Date
            Invoke-Py $trainArgs
            $elapsed = [int]((Get-Date) - $t0).TotalSeconds
        }

        # The first training run at each topk k writes indices_top_{k}.json +
        # embds_{max,min}_top_{k}.json into $TrainDir. Copy them onto the holdout
        # so downstream eval reuses the same coordinate selection + normalization.
        # Cheap; run every time to stay in sync across layer sweeps.
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

        $benchDir = ""
        if ($RunImageBenchmark) {
            $benchDir = ToPosix (Join-Path $BenchRoot $tag)
            Write-Host "  -- image benchmark -> $benchDir" -ForegroundColor Cyan
            $benchArgs = @(
                "-m", "evaluation.run_image_benchmark",
                "--checkpoint",         $runDir,
                "--holdout_folder",     $HoldoutDir,
                "--output_dir",         $benchDir,
                "--baseline_cache_root", $BaselineCacheRoot,
                "--sd_device",          $BenchmarkSdDevice
            )
            if ($BenchmarkSsaeDevice)     { $benchArgs += @("--ssae_device",     $BenchmarkSsaeDevice) }
            if ($BenchmarkBaselineDevice) { $benchArgs += @("--baseline_device", $BenchmarkBaselineDevice) }
            if ($BenchmarkMaxSamples -gt 0) { $benchArgs += @("--max_samples", "$BenchmarkMaxSamples") }
            if ($LocalityDrop)       { $benchArgs += "--locality_drop_one_attr" }
            if ($LocalitySwap)       { $benchArgs += "--locality_swap_one_attr" }
            if ($BenchmarkDino)      { $benchArgs += "--dino" }
            if ($BenchmarkSimulated) { $benchArgs += "--simulated" }
            Invoke-Py $benchArgs
        }

        $hiddenCol = if ($L -eq 1) { "" } else { "$HiddenDim" }
        "$k,$L,$hiddenCol,$mse,$cos,$nHold,$elapsed,$runDir,$benchDir" | Add-Content $SummaryCsv
        Write-Host ("  topk={0} L={1} h={2} mse={3} cosine={4} elapsed={5}s bench={6}" -f $k, $L, $hiddenCol, $mse, $cos, $elapsed, $benchDir) -ForegroundColor Green
    }
}

Write-Host "== Sweep complete ==" -ForegroundColor Green
Import-Csv $SummaryCsv | Format-Table -AutoSize
Write-Host "Summary: $SummaryCsv"
