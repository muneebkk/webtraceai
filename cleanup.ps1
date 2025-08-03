# Cleanup script to fix virtual environment issues

Write-Host "Cleaning up WebTrace AI environment..." -ForegroundColor Yellow

# Stop any running processes
Write-Host "Stopping running processes..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -eq "python" -or $_.ProcessName -eq "uvicorn" -or $_.ProcessName -eq "node"} | Stop-Process -Force -ErrorAction SilentlyContinue

Start-Sleep -Seconds 3

# Go to backend directory
Set-Location "backend"

# Try to remove the problematic venv
if (Test-Path "venv") {
    Write-Host "Removing problematic virtual environment..." -ForegroundColor Yellow
    try {
        Remove-Item -Recurse -Force "venv" -ErrorAction Stop
        Write-Host "Successfully removed venv" -ForegroundColor Green
    } catch {
        Write-Host "Could not remove venv. Renaming it..." -ForegroundColor Yellow
        if (Test-Path "venv_old") {
            Remove-Item -Recurse -Force "venv_old" -ErrorAction SilentlyContinue
        }
        Rename-Item "venv" "venv_old" -ErrorAction SilentlyContinue
        Write-Host "Renamed venv to venv_old" -ForegroundColor Green
    }
}

# Create fresh venv
Write-Host "Creating fresh virtual environment..." -ForegroundColor Yellow
python -m venv venv

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Set-Location ".."

Write-Host "Cleanup completed! You can now run .\start-dev.ps1" -ForegroundColor Green 