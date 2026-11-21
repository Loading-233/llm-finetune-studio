# Training watchdog: checks every 2 minutes and restarts the trainer if it died.
# Stops at 07:00 next morning. ASCII only to avoid PowerShell 5 encoding issues.

$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root "outputs\overnight"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$watchLog = Join-Path $logDir "watch.log"

function Write-Watch([string]$text) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $text
    Add-Content -Path $watchLog -Value $line -Encoding UTF8
}

# skip if another watchdog instance is already running (named mutex)
$mutex = New-Object System.Threading.Mutex($false, "Local\LLM_Overnight_Watch")
if (-not $mutex.WaitOne(0)) { exit 0 }

$now = Get-Date
$stop = $now.Date.AddHours(7)
if ($stop -le $now) { $stop = $stop.AddDays(1) }
Write-Watch ("watchdog started, stop at {0}" -f $stop.ToString("yyyy-MM-dd HH:mm"))

while ((Get-Date) -lt $stop.AddMinutes(-2)) {
    $running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*overnight_train.py*" })

    if ($running.Count -eq 0) {
        Write-Watch "trainer not running, restarting"
        $python = Join-Path $root ".venv\Scripts\python.exe"
        $script = Join-Path $root "scripts\overnight_train.py"
        Start-Process -FilePath $python `
            -ArgumentList @($script, "--until", "07:00", "--serve-after", "--threads", "6", "--steps-per-round", "80") `
            -WorkingDirectory $root `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logDir "watch_stdout.log") `
            -RedirectStandardError (Join-Path $logDir "watch_stderr.log")
        Start-Sleep -Seconds 60
    }
    Start-Sleep -Seconds 120
}

Write-Watch "watchdog exit"
