# 通宵连续训练启动器：默认训练到次日 07:00，结束后自动拉起 Web 服务
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root "outputs\overnight"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

& (Join-Path $root ".venv\Scripts\python.exe") `
    (Join-Path $root "scripts\overnight_train.py") `
    --until 07:00 `
    --serve-after `
    --threads 6 `
    --steps-per-round 80 *>> (Join-Path $logDir "stdout.log")
