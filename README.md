# WebTrace AI 🕵️‍♂️

**Detect if a website was built with AI tools** - Upload a screenshot and get instant analysis!

## ✨ What It Does

- 🔍 **AI Detection**: Tells you if a website was made by AI or humans
- 🛠️ **Tool Identification**: Shows which AI tool was likely used (Framer, Wix, Notion, etc.)
- 📊 **Confidence Score**: Gives you a percentage of how sure it is
- 🎨 **Easy Interface**: Just drag & drop a screenshot to analyze

## 🚀 Quick Start (3 Steps!)

### 1. **Prerequisites**
Make sure you have:
- ✅ Python 3.11+ installed
- ✅ Node.js 18+ installed

### 2. **Start Everything**

**Windows:**
```powershell
# Just run this one command!
.\start-dev.ps1
```

**Linux/Mac/Bash:**
```bash
# Make executable first time only
chmod +x start-dev.sh

# Then run
./start-dev.sh
```

**What this does:**
- ✅ Sets up Python virtual environment
- ✅ Installs all dependencies automatically
- ✅ Starts both backend and frontend servers
- ✅ Shows you all the URLs to access

### 3. **Use the App**
- 🌐 **Frontend**: http://localhost:5173
- 📚 **API Docs**: http://localhost:8000/docs

## 🛠️ Manual Setup (Alternative)

If you prefer to do it step by step:

### Backend Setup
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🎯 How to Use

1. **Open** http://localhost:5173
2. **Drag & drop** a website screenshot
3. **Get results** in seconds!

### Example Result:
```json
{
  "is_ai_generated": true,
  "confidence": 85,
  "predicted_tool": "Framer AI",
  "tool_probabilities": {
    "Framer AI": 85,
    "Wix ADI": 10,
    "Human": 5
  }
}
```

## 🔧 API Usage

### Analyze a Website
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "screenshot=@website.png" \
  -F "html_content=<html>...</html>" \
  -F "visible_text=Welcome to our website..."
```

### Get Supported Tools
```bash
curl "http://localhost:8000/api/tools"
```

## 🧠 How It Works

The system analyzes websites using **3 different approaches**:

### 1. **Visual Analysis** 👁️
- Colors and layout patterns
- Edge detection and textures
- White space analysis

### 2. **Code Structure** 🔧
- HTML tag patterns
- CSS class naming conventions
- Code complexity metrics

### 3. **Text Content** 📝
- Word patterns and readability
- Common AI-generated phrases
- Content structure analysis

## 🏗️ Project Structure

```
webtraceai/
├── backend/           # Python FastAPI server
│   ├── app/          # Main application code
│   ├── main.py       # Server entry point
│   └── requirements.txt
├── frontend/         # React app
│   ├── src/          # React components
│   └── package.json
├── start-dev.ps1     # One-click startup script
└── README.md         # This file
```

## 🐳 Production Deployment

### Using Docker
```bash
cd deployment
docker-compose up --build
```

Then visit: http://localhost
