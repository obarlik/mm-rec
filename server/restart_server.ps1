# Server Restart Script (PowerShell)
# Kills old server, pulls code, and starts new server

Write-Host "🔄 Restarting server..." -ForegroundColor Cyan

# Get server directory
$ServerDir = Split-Path -Parent $PSScriptRoot
Set-Location $ServerDir

# Kill old server
Write-Host "⏹️  Stopping old server..." -ForegroundColor Yellow
Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like "*train_server.py*" } | ForEach-Object { 
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2

# Pull latest code
Write-Host "📥 Pulling latest code..." -ForegroundColor Yellow
git pull

# Build C++ extensions if needed
Write-Host "🔨 Building C++ extensions..." -ForegroundColor Yellow
if (Test-Path "mm_rec\cpp\setup.py") {
    Push-Location "mm_rec\cpp"
    python setup.py build_ext --inplace 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ C++ extensions built" -ForegroundColor Green
    }
    else {
        Write-Host "  ⚠️  C++ build failed, using Python fallback" -ForegroundColor Yellow
    }
    Pop-Location
}

# Activate conda and restart
Write-Host "🚀 Starting new server..." -ForegroundColor Green
conda activate mm-rec

# Start server in background
# CUDA_LAUNCH_BLOCKING removed for production speed
Start-Process -FilePath "python" -ArgumentList "server\train_server.py" -WindowStyle Hidden -RedirectStandardOutput "server.log" -RedirectStandardError "server_error.log"

Write-Host "✅ Server restarted!" -ForegroundColor Green
Write-Host "📝 Logs: Get-Content server.log -Wait" -ForegroundColor Cyan
