[CmdletBinding()]
param(
    [ValidateSet("all", "locomo", "longmemeval", "beam", "personamem", "personamem_v2", "perma")]
    [string[]]$Benchmark = @("all"),
    [switch]$DownloadData,
    [switch]$IncludeLargeScale,
    [switch]$CreateEvaluationEnvs
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$registryPath = Join-Path $repoRoot "benchmark_config\official_benchmarks.json"
$registry = Get-Content -Raw $registryPath | ConvertFrom-Json
$selected = if ($Benchmark -contains "all") { @($registry.benchmarks.psobject.Properties.Name) } else { $Benchmark }

function Require-Command([string]$Command) {
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "Required command '$Command' was not found on PATH."
    }
}

function Ensure-Checkout($Spec) {
    $path = Join-Path $repoRoot $Spec.checkout
    $parent = Split-Path -Parent $path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    if (-not (Test-Path (Join-Path $path ".git"))) {
        Write-Host "[clone] $($Spec.display_name)"
        & git clone --depth 1 $Spec.official_repository $path
        if ($LASTEXITCODE -ne 0) { throw "Clone failed: $($Spec.display_name)" }
    }
    & git -C $path fetch --depth 1 origin $Spec.pinned_commit
    if ($LASTEXITCODE -ne 0) { throw "Fetch failed: $($Spec.display_name) @ $($Spec.pinned_commit)" }
    & git -C $path checkout --detach $Spec.pinned_commit
    if ($LASTEXITCODE -ne 0) { throw "Checkout failed: $($Spec.display_name) @ $($Spec.pinned_commit)" }
}

function Download-Hf([string]$Dataset, [string]$RelativeDestination, [string[]]$Includes = @()) {
    $destination = Join-Path $repoRoot $RelativeDestination
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    # Large official releases can contain thousands of small files. A single worker
    # is slower but avoids anonymous-Hub 429 bursts and is safely resumable.
    $arguments = @("download", $Dataset, "--repo-type", "dataset", "--local-dir", $destination, "--max-workers", "1")
    if ($Includes.Count -gt 0) { $arguments += "--include"; $arguments += $Includes }
    Write-Host "[download] huggingface:$Dataset -> $RelativeDestination"
    & hf @arguments
    if ($LASTEXITCODE -ne 0) { throw "Hugging Face download failed: $Dataset" }
}

function Create-EvaluationEnv($Name, $Requirements) {
    $venv = Join-Path $repoRoot ".venv-benchmarks\$Name"
    if (-not (Test-Path $venv)) {
        Write-Host "[venv] $Name"
        & python -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed: $Name" }
    }
    $pip = Join-Path $venv "Scripts\python.exe"
    & $pip -m pip install --upgrade pip
    if ($env:OS -eq "Windows_NT" -and ($Name -in @("beam", "personamem"))) {
        & (Join-Path $PSScriptRoot "install_windows_evaluation_requirements.ps1") -Python $pip -Requirements (Join-Path $repoRoot $Requirements)
    } else {
        & $pip -m pip install -r (Join-Path $repoRoot $Requirements)
    }
    if ($LASTEXITCODE -ne 0) { throw "dependency install failed: $Name" }
}

Require-Command git
foreach ($name in $selected) { Ensure-Checkout $registry.benchmarks.$name }

if ($DownloadData) {
    Require-Command hf
    if ($selected -contains "longmemeval") {
        Download-Hf "xiaowu0162/longmemeval-cleaned" "datasets/official/longmemeval"
    }
    if ($selected -contains "personamem") {
        $files = @("README.md", "questions_32k.csv", "shared_contexts_32k.jsonl")
        if ($IncludeLargeScale) {
            $files = @("README.md", "questions_32k.csv", "questions_128k.csv", "questions_1M.csv", "shared_contexts_32k.jsonl", "shared_contexts_128k.jsonl", "shared_contexts_1M.jsonl")
        }
        Download-Hf "bowen-upenn/PersonaMem" "datasets/official/personamem" $files
    }
    if ($selected -contains "personamem_v2") {
        Download-Hf "bowen-upenn/ImplicitPersona" "datasets/official/personamem-v2"
    }
    if ($selected -contains "perma") {
        Download-Hf "ustclsc/PERMA" "datasets/official/perma"
    }
    if ($selected -contains "beam") {
        Write-Host "[data] BEAM conversations are already pinned in third_party/benchmarks/beam/chats."
        if ($IncludeLargeScale) {
            Download-Hf "Mohammadta/BEAM" "datasets/official/beam" @("README.md", "data/*")
            Download-Hf "Mohammadta/BEAM-10M" "datasets/official/beam-10m" @("README.md", "data/*")
        }
    }
}

if ($CreateEvaluationEnvs) {
    Require-Command python
    if ($selected -contains "longmemeval") {
        Create-EvaluationEnv "longmemeval" "third_party/benchmarks/longmemeval-official/requirements-lite.txt"
    }
    if ($selected -contains "beam") { Create-EvaluationEnv "beam" "third_party/benchmarks/beam/requirements.txt" }
    if ($selected -contains "personamem" -or $selected -contains "personamem_v2") {
        Create-EvaluationEnv "personamem" "third_party/benchmarks/personamem/requirements.txt"
    }
    if ($selected -contains "perma") { Create-EvaluationEnv "perma" "third_party/benchmarks/perma/requirements.txt" }
    if ($selected -contains "locomo") {
        Write-Warning "LoCoMo's upstream requirements.txt is a Linux conda export. No misleading Windows venv is created; use Linux/WSL for byte-for-byte upstream reproduction."
    }
}

Write-Host "Completed. Verify with: python scripts/benchmarks/verify_official_benchmarks.py"
