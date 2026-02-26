@echo off
REM Unified startup script for NeuroVision - Backend + Frontend
REM This script starts both the FastAPI backend and Next.js frontend

echo.
echo ========================================
echo   NeuroVision - Starting All Services
echo ========================================
echo.

REM Start backend in new PowerShell window
echo [1/3] 🚀 Starting backend on port 8000...
start "NeuroVision Backend" powershell -NoExit -Command "cd C:\Users\allur\brats7d_old\server; Write-Host '🚀 Starting NeuroVision Backend Server...' -ForegroundColor Green; Write-Host 'Port: 8000' -ForegroundColor Cyan; Write-Host ''; python -m uvicorn app:app --host 0.0.0.0 --port 8000"

REM Wait a moment for backend to start
timeout /t 2 /nobreak >nul

REM Start frontend in new PowerShell window
echo [2/3] 🌐 Starting frontend on port 3000...
start "NeuroVision Frontend" powershell -NoExit -Command "cd C:\Users\allur\brats7d_old\neurovision; Write-Host '🌐 Starting NeuroVision Frontend...' -ForegroundColor Blue; Write-Host 'Port: 3000' -ForegroundColor Cyan; Write-Host ''; npm run dev"

REM Wait for services to start
echo [3/3] ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Open browser
echo ✅ Opening browser to http://localhost:3000...
start http://localhost:3000

echo.
echo ========================================
echo   ✅ All services started!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit this window...
pause >nul

