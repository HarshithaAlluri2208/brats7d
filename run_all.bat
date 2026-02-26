@echo off
set PROJECT_ROOT=%~dp0

REM ========================================
REM NeuroVision Smart Startup Script
REM ========================================

echo.
echo ========================================
echo   NeuroVision - Starting All Services
echo ========================================
echo.

REM ===============================
REM CHECK NODE INSTALLATION
REM ===============================
where node >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Node.js not found.
    echo Please install Node.js 18+ from:
    echo https://nodejs.org
    pause
    exit /b
)

echo ✅ Node.js detected
node -v
npm -v
echo.

REM ===============================
REM INSTALL FRONTEND DEPENDENCIES
REM ===============================
echo 📦 Installing frontend dependencies...

cd "%PROJECT_ROOT%neurovision"

IF NOT EXIST node_modules (
    echo Running npm install...
    call npm install
) ELSE (
    echo node_modules already exists ✅
)

cd "%PROJECT_ROOT%"

REM ===============================
REM START BACKEND
REM ===============================
echo [1/3] 🚀 Starting backend on port 8000...

start "NeuroVision Backend" powershell -NoExit -Command ^
"cd '%PROJECT_ROOT%server'; ^
Write-Host '🚀 Starting NeuroVision Backend Server...' -ForegroundColor Green; ^
Write-Host 'Port: 8000' -ForegroundColor Cyan; ^
python -m uvicorn app:app --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak >nul

REM ===============================
REM START FRONTEND
REM ===============================
echo [2/3] 🌐 Starting frontend on port 3000...

start "NeuroVision Frontend" powershell -NoExit -Command ^
"cd '%PROJECT_ROOT%neurovision'; ^
Write-Host '🌐 Starting NeuroVision Frontend...' -ForegroundColor Blue; ^
Write-Host 'Port: 3000' -ForegroundColor Cyan; ^
npm run dev"

REM ===============================
REM WAIT + OPEN BROWSER
REM ===============================
echo [3/3] ⏳ Waiting for services...
timeout /t 10 /nobreak >nul

echo ✅ Opening browser...
start http://localhost:3000

echo.
echo ========================================
echo   ✅ All services started!
echo ========================================
echo.
echo Backend : http://localhost:8000
echo Frontend: http://localhost:3000
echo.

pause
