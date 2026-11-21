# 8765 常驻服务：进程退出后自动重启，日志追加到 outputs/serve_8765.log
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "outputs\serve_8765.log"

while ($true) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] starting gateway" | Out-File -FilePath $log -Append -Encoding UTF8
    Push-Location $root
    & $python (Join-Path $root "scripts\gateway.py") *>> $log
    Pop-Location
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] gateway exited, restart in 3s" | Out-File -FilePath $log -Append -Encoding UTF8
    Start-Sleep -Seconds 3
}
