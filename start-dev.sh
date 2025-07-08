#!/bin/bash

# WebTrace AI Development Startup Script (Bash)
# This script starts both the backend and frontend development servers

echo "🚀 Starting WebTrace AI Development Environment..."

# Check if Python is installed
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1)
    echo "✅ Python found: $python_version"
else
    echo "❌ Python not found. Please install Python 3.11+ and try again."
    exit 1
fi

# Check if Node.js is installed
if command -v node &> /dev/null; then
    node_version=$(node --version 2>&1)
    echo "✅ Node.js found: $node_version"
else
    echo "❌ Node.js not found. Please install Node.js 18+ and try again."
    exit 1
fi

# Function to start backend
start_backend() {
    echo "🔧 Starting Backend Server..."
    
    # Change to backend directory
    cd backend
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    echo "🔌 Activating virtual environment..."
    source venv/bin/activate
    
    # Install dependencies if requirements.txt is newer than venv
    if [ requirements.txt -nt venv ]; then
        echo "📥 Installing Python dependencies..."
        pip install -r requirements.txt
    fi
    
    # Start the backend server
    echo "🚀 Starting FastAPI server on http://localhost:8000"
    uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    
    # Return to root directory
    cd ..
}

# Function to start frontend
start_frontend() {
    echo "🎨 Starting Frontend Server..."
    
    # Change to frontend directory
    cd frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo "📥 Installing Node.js dependencies..."
        npm install
    fi
    
    # Start the frontend server
    echo "🚀 Starting Vite development server on http://localhost:5173"
    npm run dev &
    FRONTEND_PID=$!
    
    # Return to root directory
    cd ..
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    echo "✅ Servers stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start both servers
start_backend
sleep 3  # Give backend time to start

start_frontend
sleep 3  # Give frontend time to start

echo ""
echo "🎉 WebTrace AI Development Environment Started!"
echo ""
echo "📱 Frontend: http://localhost:5173"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Keep the script running
wait 