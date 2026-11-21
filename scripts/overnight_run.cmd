@echo off
REM 通宵连续训练：默认跑到次日 07:00，结束后自动拉起 Web 服务
cd /d "%~dp0.."

if not exist "outputs\overnight" mkdir "outputs\overnight"

".venv\Scripts\python.exe" "scripts\overnight_train.py" ^
    --until 07:00 ^
    --serve-after ^
    --threads 6 ^
    --steps-per-round 80 ^
    --delete-task LLM_Overnight_Train ^
    >> "outputs\overnight\stdout.log" 2>> "outputs\overnight\stderr.log"
