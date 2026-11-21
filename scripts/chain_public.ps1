param([string]$Until = "23:30", [double]$EvalRatio = 0.02)

# Waits for the running search to finish, then trains on the merged public dataset.
$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root "outputs\search_public"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$chainLog = Join-Path $logDir "chain.log"

function Write-Chain([string]$text) {
    Add-Content -Path $chainLog -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $text) -Encoding UTF8
}

Write-Chain "chain waiter started"

while (@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*overnight_search.py*" }).Count -gt 0) {
    Start-Sleep -Seconds 30
}

Write-Chain "previous search finished, starting public dataset search"

Start-Process -FilePath (Join-Path $root ".venv\Scripts\python.exe") `
    -ArgumentList @(
        (Join-Path $root "scripts\overnight_search.py"),
        "--until", $Until,
        "--serve-after",
        "--threads", "6",
        "--eval-ratio", "$EvalRatio",
        "--out", "outputs/search_public"
    ) `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logDir "stdout.log") `
    -RedirectStandardError (Join-Path $logDir "stderr.log")

Write-Chain "public dataset search launched"
