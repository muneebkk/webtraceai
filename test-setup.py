#!/usr/bin/env python3
"""
WebTrace AI Setup Test Script
This script verifies that all dependencies and components are properly installed.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("❌ Python 3.11+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_package(package_name, import_name=None):
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def check_file_exists(file_path):
    """Check if a file exists"""
    if Path(file_path).exists():
        print(f"✅ {file_path} exists")
        return True
    else:
        print(f"❌ {file_path} not found")
        return False

def check_directory_exists(dir_path):
    """Check if a directory exists"""
    if Path(dir_path).exists():
        print(f"✅ {dir_path} exists")
        return True
    else:
        print(f"❌ {dir_path} not found")
        return False

def main():
    print("🔍 WebTrace AI Setup Verification")
    print("=" * 40)
    
    all_good = True
    
    # Check Python version
    print("\n🐍 Python Environment:")
    if not check_python_version():
        all_good = False
    
    # Check required Python packages
    print("\n📦 Python Dependencies:")
    packages = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pillow", "PIL"),
        ("opencv-python", "cv2"),
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("beautifulsoup4", "bs4"),
        ("matplotlib", "matplotlib"),
        ("pydantic", "pydantic"),
        ("joblib", "joblib"),
    ]
    
    for package, import_name in packages:
        if not check_package(package, import_name):
            all_good = False
    
    # Check project structure
    print("\n📁 Project Structure:")
    files_to_check = [
        "backend/main.py",
        "backend/requirements.txt",
        "backend/app/__init__.py",
        "backend/app/routes.py",
        "backend/app/feature_extract.py",
        "backend/app/model_loader.py",
        "backend/app/utils.py",
        "frontend/package.json",
        "frontend/vite.config.js",
        "frontend/src/App.jsx",
        "frontend/src/main.jsx",
        "frontend/tailwind.config.js",
        "deployment/Dockerfile",
        "deployment/docker-compose.yml",
        "README.md",
        ".gitignore",
        "start-dev.ps1"
    ]
    
    for file_path in files_to_check:
        if not check_file_exists(file_path):
            all_good = False
    
    # Check directories
    dirs_to_check = [
        "backend/app",
        "frontend/src",
        "deployment"
    ]
    
    for dir_path in dirs_to_check:
        if not check_directory_exists(dir_path):
            all_good = False
    
    # Check if virtual environment exists
    print("\n🔧 Development Environment:")
    if Path("backend/venv").exists():
        print("✅ Backend virtual environment exists")
    else:
        print("⚠️  Backend virtual environment not found (run setup first)")
    
    if Path("frontend/node_modules").exists():
        print("✅ Frontend node_modules exists")
    else:
        print("⚠️  Frontend node_modules not found (run npm install first)")
    
    # Summary
    print("\n" + "=" * 40)
    if all_good:
        print("🎉 All checks passed! Your WebTrace AI setup is ready.")
        print("\n🚀 To start development:")
        print("   Windows: .\\start-dev.ps1")
        print("   Manual:  See README.md for instructions")
    else:
        print("❌ Some checks failed. Please review the issues above.")
        print("\n📖 Setup instructions:")
        print("   1. Install Python 3.11+")
        print("   2. Install Node.js 18+")
        print("   3. Run: cd backend && python -m venv venv")
        print("   4. Run: cd backend && venv\\Scripts\\activate && pip install -r requirements.txt")
        print("   5. Run: cd frontend && npm install")
        print("   6. Run: .\\start-dev.ps1")

if __name__ == "__main__":
    main() 