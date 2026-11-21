param(
    [Parameter(Mandatory = $true)][string]$Job,
    [int]$Last = 10,
    [switch]$Once
)

$log = Join-Path $PSScriptRoot "..\outputs\jobs\$Job\train.log"
$log = [System.IO.Path]::GetFullPath($log)
if (-not (Test-Path $log)) { Write-Host "log not found: $log" -ForegroundColor Red; exit 1 }

$state = @{ FirstTs = $null; LastTs = $null; Count = 0; Max = 0 }

function Format-Line($line) {
    $o = $line | ConvertFrom-Json -ErrorAction SilentlyContinue
    if (-not $o) { return }
    if ($o.type -eq "step") {
        $ts = [datetime]::ParseExact($o.ts, "HH:mm:ss", $null)
        if ($state.FirstTs -eq $null) { $state.FirstTs = $ts }
        $state.LastTs = $ts
        $state.Count++
        $state.Max = [int]$o.max_steps
        $avg = 0
        if ($state.Count -gt 1) {
            $avg = ($state.LastTs - $state.FirstTs).TotalSeconds / ($state.Count - 1)
        }
        $left = ([int]$o.max_steps - [int]$o.step) * $avg
        $eta = $ts.AddSeconds($left).ToString("HH:mm")
        $lr = ('{0:0.0e+0}' -f [double]$o.logs.learning_rate)
        "{0}  step {1,3}/{2}  loss {3,6:N3}  lr {4,8}  epoch {5:N2}  eta {6,3}m (~{7})" -f `
            $o.ts, $o.step, $o.max_steps, $o.logs.loss, $lr, $o.logs.epoch, [math]::Round($left / 60), $eta
    }
    elseif ($o.type -eq "eval") {
        "{0}  [eval] loss {1:N3}" -f $o.ts, $o.logs.eval_loss
    }
    elseif ($o.type -eq "info") {
        "{0}  {1}" -f $o.ts, $o.text
    }
    elseif ($o.type -eq "done" -or $o.type -eq "error") {
        "{0}  [{1}] {2}" -f $o.ts, $o.type, $o.text
    }
}

if ($Once) {
    Get-Content $log -Tail $Last -Encoding UTF8 | ForEach-Object { Format-Line $_ }
    exit 0
}

Write-Host "watching job $Job (Ctrl+C to quit)" -ForegroundColor Cyan
Get-Content $log -Wait -Tail $Last -Encoding UTF8 | ForEach-Object { Format-Line $_ }
