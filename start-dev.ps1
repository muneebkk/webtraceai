# WebTrace AI Development Startup Script
# This script starts both the backend and frontend development servers

Write-Host "Starting WebTrace AI Development Environment..." -ForegroundColor Green

# Check execution policy
$executionPolicy = Get-ExecutionPolicy
if ($executionPolicy -eq "Restricted") {
    Write-Host "Execution policy is restricted. Please run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Red
    exit 1
}

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Check Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "Node.js not found. Please install Node.js 18+" -ForegroundColor Red
    exit 1
}

# Start Backend
Write-Host "Starting Backend Server..." -ForegroundColor Yellow
Set-Location "backend"

# Stop any running processes that might lock files
Write-Host "Stopping any running processes..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -eq "python" -or $_.ProcessName -eq "uvicorn"} | Stop-Process -Force -ErrorAction SilentlyContinue

# Wait a moment for processes to stop
Start-Sleep -Seconds 2

# Clean up existing venv if there are issues
if (Test-Path "venv") {
    Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
    try {
        Remove-Item -Recurse -Force "venv" -ErrorAction Stop
    } catch {
        Write-Host "Could not remove venv completely. Creating new one with different name..." -ForegroundColor Yellow
        if (Test-Path "venv_new") {
            Remove-Item -Recurse -Force "venv_new" -ErrorAction SilentlyContinue
        }
        Rename-Item "venv" "venv_old" -ErrorAction SilentlyContinue
    }
}

Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "Starting FastAPI server on http://localhost:8000" -ForegroundColor Green
$backendCommand = "cd '$PWD'; & 'venv\Scripts\Activate.ps1'; uvicorn main:app --reload --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCommand

Set-Location ".."

# Start Frontend
Write-Host "Starting Frontend Server..." -ForegroundColor Yellow
Set-Location "frontend"

if (-not (Test-Path "node_modules")) {
    Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
    npm install
}

Write-Host "Starting Vite development server on http://localhost:5173" -ForegroundColor Green
$frontendCommand = "cd '$PWD'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCommand

Set-Location ".."

Write-Host ""
Write-Host "WebTrace AI Development Environment Started!" -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "Backend API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop all servers" -ForegroundColor Yellow

while ($true) {
    Start-Sleep -Seconds 1
} 