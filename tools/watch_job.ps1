# 训练日志实时看板：把 outputs/jobs/<id>/train.log 的 JSON 事件格式化成进度条 + 指标
param(
    [string]$Job = "",
    [int]$Refresh = 2,
    [int]$History = 16,
    [switch]$Once
)

$root = Split-Path -Parent $PSScriptRoot
$jobsDir = Join-Path $root "outputs\jobs"
if (-not $Job) {
    $Job = (Get-ChildItem $jobsDir -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
}
if (-not $Job) { Write-Host "没有找到任何任务" -ForegroundColor Red; exit 1 }
$log = Join-Path $jobsDir "$Job\train.log"
if (-not (Test-Path $log)) { Write-Host "日志不存在：$log" -ForegroundColor Red; exit 1 }

try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }

function Read-Shared([string]$path) {
    try {
        $fs = [IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
        $sr = New-Object IO.StreamReader($fs, [Text.Encoding]::UTF8)
        $text = $sr.ReadToEnd()
        $sr.Close(); $fs.Close()
        return $text -split "`r?`n" | Where-Object { $_.Trim() }
    } catch { return @() }
}

function Format-Step($e) {
    $logs = $e.logs
    $pct = if ($e.max_steps) { $e.step / $e.max_steps } else { 0 }
    $filled = [int][Math]::Round($pct * 20)
    $bar = ("█" * $filled) + ("░" * (20 - $filled))
    $loss = if ($null -ne $logs.loss) { "{0:N4}" -f $logs.loss } else { "—" }
    $grad = if ($null -ne $logs.grad_norm) { "{0:N2}" -f $logs.grad_norm } else { "—" }
    $lr = if ($null -ne $logs.learning_rate) { "{0:E2}" -f $logs.learning_rate } else { "—" }
    $ep = if ($null -ne $logs.epoch) { "{0:N3}" -f $logs.epoch } else { "—" }
    "[{0}] {1} {2,5:N1}%  step {3}/{4}  loss {5}  grad {6}  lr {7}  epoch {8}" -f `
        $e.ts, $bar, ($pct * 100), $e.step, $e.max_steps, $loss, $grad, $lr, $ep
}

function Render {
    $events = @()
    foreach ($line in (Read-Shared $log)) {
        try { $events += ($line | ConvertFrom-Json) } catch { }
    }
    if (-not $events.Count) { Write-Host "日志里还没有可解析的事件…" -ForegroundColor DarkGray; return }

    $steps = @($events | Where-Object { $_.type -eq "step" })
    $last = $steps | Select-Object -Last 1
    $stages = @($events | Where-Object { $_.type -eq "stage" })

    $eta = "—"
    if ($steps.Count -ge 2 -and $last.max_steps) {
        $first = $steps[0]
        try {
            $t0 = [datetime]::ParseExact($first.ts, "HH:mm:ss", $null)
            $t1 = [datetime]::ParseExact($last.ts, "HH:mm:ss", $null)
            $per = ($t1 - $t0).TotalSeconds / [Math]::Max(1, ($last.step - $first.step))
            $left = ($last.max_steps - $last.step) * $per
            if ($left -gt 0) { $eta = "{0:N0} 分 {1:N0} 秒" -f [Math]::Floor($left / 60), ($left % 60) }
        } catch { }
    }

    Clear-Host
    Write-Host "任务 $Job" -ForegroundColor Cyan
    if ($last) {
        $pct = $last.step / [Math]::Max(1, $last.max_steps)
        $filled = [int][Math]::Round($pct * 40)
        $bar = ("█" * $filled) + ("░" * (40 - $filled))
        Write-Host ("  " + $bar + ("  {0:N1}%" -f ($pct * 100))) -ForegroundColor Green
        Write-Host ("  进度 {0}/{1}   剩余约 {2}   刷新 {3}s" -f $last.step, $last.max_steps, $eta, $Refresh) -ForegroundColor DarkGray
    }
    Write-Host ("-" * 100) -ForegroundColor DarkGray
    foreach ($e in @($steps | Select-Object -Last $History)) {
        Write-Host (Format-Step $e)
    }
    foreach ($e in @($stages | Select-Object -Last 4)) {
        Write-Host ("[{0}] » {1}" -f $e.ts, $e.text) -ForegroundColor Yellow
    }
    Write-Host ("-" * 100) -ForegroundColor DarkGray
    Write-Host "Ctrl+C 退出" -ForegroundColor DarkGray
}

Render
if (-not $Once) { while ($true) { Start-Sleep -Seconds $Refresh; Render } }
