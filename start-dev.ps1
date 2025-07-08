# WebTrace AI Development Startup Script
# This script starts both the backend and frontend development servers

Write-Host "🚀 Starting WebTrace AI Development Environment..." -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.11+ and try again." -ForegroundColor Red
    exit 1
}

# Check if Node.js is installed
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found. Please install Node.js 18+ and try again." -ForegroundColor Red
    exit 1
}

# Function to start backend
function Start-Backend {
    Write-Host "🔧 Starting Backend Server..." -ForegroundColor Yellow
    
    # Change to backend directory
    Set-Location "backend"
    
    # Check if virtual environment exists
    if (-not (Test-Path "venv")) {
        Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
        python -m venv venv
    }
    
    # Activate virtual environment
    Write-Host "🔌 Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
    
    # Install dependencies if requirements.txt is newer than venv
    if ((Get-Item "requirements.txt").LastWriteTime -gt (Get-Item "venv").LastWriteTime) {
        Write-Host "📥 Installing Python dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
    }
    
    # Start the backend server
    Write-Host "🚀 Starting FastAPI server on http://localhost:8000" -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; & 'venv\Scripts\Activate.ps1'; uvicorn main:app --reload --host 0.0.0.0 --port 8000"
    
    # Return to root directory
    Set-Location ".."
}

# Function to start frontend
function Start-Frontend {
    Write-Host "🎨 Starting Frontend Server..." -ForegroundColor Yellow
    
    # Change to frontend directory
    Set-Location "frontend"
    
    # Check if node_modules exists
    if (-not (Test-Path "node_modules")) {
        Write-Host "📥 Installing Node.js dependencies..." -ForegroundColor Yellow
        npm install
    }
    
    # Start the frontend server
    Write-Host "🚀 Starting Vite development server on http://localhost:5173" -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; npm run dev"
    
    # Return to root directory
    Set-Location ".."
}

# Start both servers
try {
    Start-Backend
    Start-Sleep -Seconds 3  # Give backend time to start
    
    Start-Frontend
    Start-Sleep -Seconds 3  # Give frontend time to start
    
    Write-Host ""
    Write-Host "🎉 WebTrace AI Development Environment Started!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📱 Frontend: http://localhost:5173" -ForegroundColor Cyan
    Write-Host "🔧 Backend API: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "📚 API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press Ctrl+C to stop all servers" -ForegroundColor Yellow
    
    # Keep the script running
    while ($true) {
        Start-Sleep -Seconds 1
    }
    
} catch {
    Write-Host "❌ Error starting development environment: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} 