#!/bin/bash
# Linux/Unix shell script to run the FastAPI server
# Usage: ./run_server.sh [dev|prod]

set -e

MODE=${1:-dev}

echo "Starting NeuroVision Inference Server in $MODE mode..."

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Virtual environment not found. Using system Python."
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Check if model checkpoint exists
if [ ! -f "/path/to/brats7d_old/models/checkpoint_epoch50.pth" ] && [ ! -f "C:/Users/allur/brats7d_old/models/checkpoint_epoch50.pth" ]; then
    echo "WARNING: Model checkpoint not found."
    echo "Please ensure the checkpoint file exists before running inference."
    echo "Expected location: C:\\Users\\allur\\brats7d_old\\models\\checkpoint_epoch50.pth (Windows) or /path/to/brats7d_old/models/checkpoint_epoch50.pth (Linux)"
fi

# Run server based on mode
if [ "$MODE" == "dev" ]; then
    echo "Running in DEVELOPMENT mode..."
    echo "Server will reload on code changes."
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
elif [ "$MODE" == "prod" ]; then
    echo "Running in PRODUCTION mode..."
    echo "Using 4 workers for better performance."
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
else
    echo "Invalid mode: $MODE"
    echo "Usage: ./run_server.sh [dev|prod]"
    exit 1
fi

