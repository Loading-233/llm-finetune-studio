param([string]$Until = "12:00")

# Search watchdog: restart the hyperparameter search if it died. ASCII only.

$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root "outputs\search"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$watchLog = Join-Path $logDir "watch.log"

function Write-Watch([string]$text) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $text
    Add-Content -Path $watchLog -Value $line -Encoding UTF8
}

$mutex = New-Object System.Threading.Mutex($false, "Local\LLM_Overnight_Search")
if (-not $mutex.WaitOne(0)) { exit 0 }

$parts = $Until.Split(":")
$stop = (Get-Date).Date.AddHours([int]$parts[0]).AddMinutes([int]$parts[1])
if ($stop -le (Get-Date)) { $stop = $stop.AddDays(1) }
Write-Watch ("search watchdog started, until {0}" -f $stop.ToString("yyyy-MM-dd HH:mm"))

while ((Get-Date) -lt $stop.AddMinutes(-2)) {
    $running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*overnight_search.py*" })

    if ($running.Count -eq 0) {
        Write-Watch "search not running, restarting"
        $python = Join-Path $root ".venv\Scripts\python.exe"
        $script = Join-Path $root "scripts\overnight_search.py"
        Start-Process -FilePath $python `
            -ArgumentList @($script, "--until", $Until, "--serve-after", "--threads", "6") `
            -WorkingDirectory $root `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logDir "watch_stdout.log") `
            -RedirectStandardError (Join-Path $logDir "watch_stderr.log")
        Start-Sleep -Seconds 90
    }
    Start-Sleep -Seconds 120
}

Write-Watch "search watchdog exit"
