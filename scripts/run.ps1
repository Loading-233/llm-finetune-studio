param(
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# 国内网络走 HuggingFace 镜像下载基座模型
if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }

Write-Host "启动 http://127.0.0.1:$Port" -ForegroundColor Cyan
& (Join-Path $root ".venv\Scripts\python.exe") -m uvicorn app.main:app `
    --host 127.0.0.1 --port $Port --app-dir $root
