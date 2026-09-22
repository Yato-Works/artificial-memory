[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Python,
    [Parameter(Mandatory = $true)]
    [string]$Requirements
)

$ErrorActionPreference = "Stop"

# The published BEAM and PersonaMem locks include Linux CUDA wheels (including
# nvidia-cufile and triton) which PyPI does not distribute for Windows.  Their
# official scorers can run on CPU; preserve every other upstream pin and let the
# Windows torch wheel supply the appropriate CPU runtime. This is intentionally an
# *evaluation* environment, not a claim of byte-identical GPU training reproduction.
$filtered = Get-Content $Requirements | Where-Object {
    $_ -and -not $_.TrimStart().StartsWith("#") -and
    $_ -notmatch "^(nvidia-|triton==)"
}

Write-Host "[windows-eval] Installing upstream requirements without Linux-only CUDA pins."
& $Python -m pip install @filtered
exit $LASTEXITCODE
