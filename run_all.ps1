# Unified startup script for NeuroVision - Backend + Frontend
# PowerShell version

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NeuroVision - Starting All Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Start backend in new window
Write-Host "[1/3] 🚀 Starting backend on port 8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd C:\Users\allur\brats7d_old\server; Write-Host '🚀 Starting NeuroVision Backend Server...' -ForegroundColor Green; Write-Host 'Port: 8000' -ForegroundColor Cyan; Write-Host ''; python -m uvicorn app:app --host 0.0.0.0 --port 8000"
) -WindowStyle Normal

# Wait a moment for backend to start
Start-Sleep -Seconds 2

# Start frontend in new window
Write-Host "[2/3] 🌐 Starting frontend on port 3000..." -ForegroundColor Blue
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd C:\Users\allur\brats7d_old\neurovision; Write-Host '🌐 Starting NeuroVision Frontend...' -ForegroundColor Blue; Write-Host 'Port: 3000' -ForegroundColor Cyan; Write-Host ''; npm run dev"
) -WindowStyle Normal

# Wait for services to start
Write-Host "[3/3] ⏳ Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Open browser
Write-Host "✅ Opening browser to http://localhost:3000..." -ForegroundColor Green
Start-Process "http://localhost:3000"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ✅ All services started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

