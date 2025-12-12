@echo off
REM Deploy MM-Rec Training Server to Phoenix (Windows)

echo ================================================
echo Deploying MM-Rec Training Server to Phoenix
echo ================================================

set REMOTE_USER=onurbarlik@hotmail.com
set REMOTE_HOST=phoenix
set REMOTE_DIR=mm-rec-training
set LOCAL_DIR=%~dp0

echo.
echo [1/6] Creating remote directory...
ssh %REMOTE_USER%@%REMOTE_HOST% "mkdir %REMOTE_DIR% 2>nul"

echo.
echo [2/6] Syncing code to Phoenix...
scp -r ^
  "%LOCAL_DIR%mm_rec" ^
  "%LOCAL_DIR%server" ^
  "%LOCAL_DIR%client" ^
  "%LOCAL_DIR%configs" ^
  "%LOCAL_DIR%scripts" ^
  "%LOCAL_DIR%requirements.txt" ^
  %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%/

echo.
echo [3/6] Setting up Python environment...
ssh %REMOTE_USER%@%REMOTE_HOST% "cd %REMOTE_DIR% && python -m venv .venv"

echo.
echo [4/6] Installing PyTorch with CUDA...
ssh %REMOTE_USER%@%REMOTE_HOST% "cd %REMOTE_DIR% && .venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"

echo.
echo [5/6] Installing dependencies...
ssh %REMOTE_USER%@%REMOTE_HOST% "cd %REMOTE_DIR% && .venv\Scripts\pip install -r requirements.txt && .venv\Scripts\pip install -r server\requirements.txt"

echo.
echo [6/6] Starting server...
ssh %REMOTE_USER%@%REMOTE_HOST% "cd %REMOTE_DIR% && start /B .venv\Scripts\python server\train_server.py"

echo.
echo ================================================
echo Deployment Complete!
echo ================================================
echo.
echo Server running at: http://phoenix:8000
echo.
echo Next steps:
echo   1. Test: python client/train_client.py --server http://phoenix:8000 --action health
echo   2. Submit: python client/train_client.py --server http://phoenix:8000 --action submit --config configs/stage1_gpu.json
echo.
echo ================================================
