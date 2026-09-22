# Offline selection-window A/B sweep for LoCoMo-10 (retrieval-only, no LLM).
# Runs the selection-stage diagnostic for several (window_cap, token_bonus) pairs
# and stores one log + one JSON report per variant in benchmark_results/window_sweep/.
param(
    [int]$Limit = 0
)
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..\..")
New-Item -ItemType Directory -Force -Path benchmark_results\window_sweep | Out-Null
$variants = @(
    @{ cap = 24; bonus = 130 },
    @{ cap = 32; bonus = 130 },
    @{ cap = 40; bonus = 130 },
    @{ cap = 32; bonus = 200 },
    @{ cap = 48; bonus = 200 }
)
foreach ($v in $variants) {
    $tag = "cap$($v.cap)_b$($v.bonus)"
    $json = "benchmark_results/window_sweep/$tag.json"
    $log = "benchmark_results/window_sweep/$tag.log"
    $args2 = @(
        "scripts/diagnose_locomo_selection_stages.py",
        "--convs", "0,1,2,3,4,5,6,7,8,9",
        "--window-cap", "$($v.cap)",
        "--token-bonus", "$($v.bonus)",
        "--json", $json
    )
    if ($Limit -gt 0) { $args2 += @("--limit", "$Limit") }
    Write-Host "[sweep] running variant $tag ..."
    & python @args2 2>&1 | Out-File -FilePath $log -Encoding utf8
    Write-Host "[sweep] done $tag (exit $LASTEXITCODE, log $((Get-Item $log -ErrorAction SilentlyContinue).Length) bytes)"
}
Write-Host "[sweep] ALL VARIANTS DONE"
