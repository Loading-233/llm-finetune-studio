param(
    [string]$Job,
    [string]$Root = "D:\Documents\Trae-CN\temp_20261114_01\llm-finetune-studio"
)

# Force UTF-8 so Chinese log text is not garbled in PowerShell 5.1
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
try { $OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

if (-not $Job) {
    $latest = Get-ChildItem (Join-Path $Root "outputs\jobs") -Directory -ErrorAction SilentlyContinue |
              Sort-Object LastWriteTime -Descending |
              Select-Object -First 1
    if (-not $latest) { throw "No job directory under $Root\outputs\jobs" }
    $Job = $latest.Name
}

$log = Join-Path $Root "outputs\jobs\$Job\train.log"
if (-not (Test-Path $log)) { throw "Log not found: $log" }

Write-Host "job : $Job" -ForegroundColor Cyan
Write-Host "log : $log" -ForegroundColor DarkGray
Write-Host "exit: Ctrl+C" -ForegroundColor DarkGray
Write-Host ""

$started = Get-Date
$total = 0

function Format-Eta([datetime]$from, [int]$done, [int]$all) {
    if ($done -le 0 -or $all -le 0) { return "eta --:--" }
    $perStep = ((Get-Date) - $from).TotalMinutes / $done
    $eta = (Get-Date).AddMinutes($perStep * ($all - $done))
    return ("eta {0:HH:mm} ({1:N1} min/step)" -f $eta, $perStep)
}

Get-Content $log -Encoding UTF8 -Wait | ForEach-Object {
    if (-not $_.Trim()) { return }
    try { $o = $_ | ConvertFrom-Json } catch { return }

    switch ($o.type) {
        "stage" {
            Write-Host ("[{0}] {1}" -f $o.ts, $o.text) -ForegroundColor DarkCyan
        }
        "step" {
            $step = [int]$o.step
            $total = [int]$o.max_steps
            $loss = [double]$o.logs.loss
            $percent = if ($total -gt 0) { 100.0 * $step / $total } else { 0 }
            $line = "[{0}] {1,4}/{2} ({3,5:N1}%)  loss={4:N4}  {5}" -f `
                $o.ts, $step, $total, $percent, $loss, (Format-Eta $started $step $total)
            Write-Host $line -ForegroundColor Green
        }
        default {
            $text = $o.text
            if (-not $text) { $text = ($o | ConvertTo-Json -Compress) }
            Write-Host ("[{0}] {1}" -f $o.ts, $text) -ForegroundColor Yellow
        }
    }
}
