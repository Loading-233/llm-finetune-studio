param(
    [string]$Python = "python",
    [string]$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[1/2] 创建虚拟环境..." -ForegroundColor Cyan
    & $Python -m venv (Join-Path $root ".venv")
}

Write-Host "[2/2] 安装依赖（首次约 1-2 GB）..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip -i $Mirror
& $venvPython -m pip install -r (Join-Path $root "requirements.txt") -i $Mirror

Write-Host "完成。运行 scripts\run.ps1 启动服务。" -ForegroundColor Green
