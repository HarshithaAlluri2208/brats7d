# NeuroVision Startup Scripts

Quick start scripts to launch both backend and frontend services with a single command.

## Quick Start

**Double-click `run_all.bat`** to start everything!

## Available Scripts

### 1. `run_all.bat` (Windows Batch)
- Double-click to run
- Opens two PowerShell windows (backend + frontend)
- Automatically opens browser after 10 seconds
- **Recommended for most users**

### 2. `run_all.ps1` (PowerShell)
- Right-click → "Run with PowerShell"
- Same functionality as `.bat` file
- Better for PowerShell users

### 3. `npm run start:full` (Node.js)
- Requires `concurrently` package (installed automatically)
- Runs both servers in a single terminal
- Useful for development

## What Gets Started

1. **Backend Server** (Port 8000)
   - FastAPI inference server
   - Location: `C:\Users\allur\brats7d_old\server`
   - URL: http://localhost:8000

2. **Frontend Server** (Port 3000)
   - Next.js development server
   - Location: `C:\Users\allur\brats7d_old\neurovision`
   - URL: http://localhost:3000

## Prerequisites

- Python 3.10+ with dependencies installed (`pip install -r server/requirements.txt`)
- Node.js with npm
- Frontend dependencies installed (`cd neurovision && npm install`)

## Manual Start

If scripts don't work, start manually:

**Terminal 1 (Backend):**
```bash
cd C:\Users\allur\brats7d_old\server
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

**Terminal 2 (Frontend):**
```bash
cd C:\Users\allur\brats7d_old\neurovision
npm run dev
```

## Troubleshooting

- **Port already in use**: Close other applications using ports 3000 or 8000
- **Python not found**: Ensure Python is in your PATH
- **npm not found**: Install Node.js from nodejs.org
- **Dependencies missing**: Run `pip install -r server/requirements.txt` and `cd neurovision && npm install`

