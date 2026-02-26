@echo off
REM Windows batch script to run the FastAPI server
REM Usage: run_server.bat [dev|prod]

setlocal

set MODE=%1
if "%MODE%"=="" set MODE=dev

echo Starting NeuroVision Inference Server in %MODE% mode...

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Using system Python.
)

REM Set environment variables
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1

REM Check if model checkpoint exists
if not exist "C:\Users\allur\brats7d_old\models\checkpoint_epoch50.pth" (
    echo WARNING: Model checkpoint not found at C:\Users\allur\brats7d_old\models\checkpoint_epoch50.pth
    echo Please ensure the checkpoint file exists before running inference.
)

if "%MODE%"=="dev" (
    echo Running in DEVELOPMENT mode...
    echo Server will reload on code changes.
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
) else if "%MODE%"=="prod" (
    echo Running in PRODUCTION mode...
    echo Using 4 workers for better performance.
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
) else (
    echo Invalid mode: %MODE%
    echo Usage: run_server.bat [dev|prod]
    exit /b 1
)

endlocal

