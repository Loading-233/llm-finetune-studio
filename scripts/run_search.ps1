param([string]$Until = "12:00")

# Launch short-run hyperparameter search until the given time.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root "outputs\search"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

& (Join-Path $root ".venv\Scripts\python.exe") `
    (Join-Path $root "scripts\overnight_search.py") `
    --until $Until `
    --serve-after `
    --threads 6 *>> (Join-Path $logDir "stdout.log")
