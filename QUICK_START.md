# WebTrace AI - Quick Start Guide

## 🚀 Getting Started

### Prerequisites
- Python 3.8+ installed
- Node.js 16+ installed
- PowerShell (Windows) or Bash (Linux/Mac)

### Option 1: Use the Development Script (Recommended)

**Windows (PowerShell):**
```powershell
.\start-dev.ps1
```

**Linux/Mac (Bash):**
```bash
./start-dev.sh
```

This will automatically:
- Start the backend server on http://localhost:8000
- Start the frontend server on http://localhost:5173
- Open both in separate terminal windows
- Wait for both services to be ready

### Option 2: Manual Startup

**Start Backend:**
```powershell
cd backend
.\venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Start Frontend (in new terminal):**
```powershell
cd frontend
npm run dev
```

## 🎯 Using the Visualization Features

### 1. Access the Application
Open your browser and go to: **http://localhost:5173**

### 2. Upload Content
- **Screenshot**: Drag and drop or click to upload a website screenshot
- **Screenshot**: Upload a screenshot of the website (required)

### 3. Select Model
Choose from three AI models:
- **Original Random Forest**: Basic ensemble model
- **Improved Logistic Regression**: Optimized for accuracy
- **Custom Decision Tree**: Most interpretable

### 4. Analyze
Click **"Analyze Website"** to get predictions

### 5. Visualize (NEW!)
Click **"Visualize"** to see detailed breakdowns including:
- **Feature Importance**: Which elements mattered most
- **Decision Path**: Step-by-step decision process
- **Model Comparison**: How different models perform
- **Explanation**: Human-readable reasoning

## 🔍 Understanding the Visualizations

### Overview Tab
- High-level prediction summary
- Confidence visualization
- Key feature importance chart
- Human-readable explanation

### Features Tab
- Detailed feature importance analysis
- Top 8 most influential features
- Percentage contribution of each feature
- Interactive progress bars

### Decision Path Tab
- Step-by-step decision process
- Feature thresholds and comparisons
- Contribution levels (High/Medium/Low)
- Actual values vs. decision thresholds

### Model Comparison Tab
- Side-by-side model performance
- Accuracy comparisons
- Model descriptions and use cases
- Visual performance indicators

### Explanation Tab
- Detailed prediction rationale
- Key feature values
- Model-specific details
- Educational insights

## 🧪 Testing the System

### Test Scripts
```powershell
# Test visualization endpoint
cd backend
python test_visualization.py

# Run comprehensive demo
cd ..
python demo_visualization.py
```

### Sample Data
The system works with:
- **Website screenshots** (PNG, JPG)
- **Screenshot** (upload image file)

## 🔧 Troubleshooting

### Common Issues

**PowerShell Command Errors:**
- Use `;` instead of `&&` for command chaining
- Example: `cd backend; python script.py`

**Port Already in Use:**
- Backend (8000): Check if another server is running
- Frontend (5173): Check if another dev server is active

**Model Files Missing:**
- Ensure you're in the backend directory
- Check if model files exist: `model.pkl`, `improved_model.pkl`, `custom_tree_model.pkl`

**Visualization Not Loading:**
- Check browser console for errors
- Verify backend is running on port 8000
- Check network tab for API call failures

### Debug Information
- **Backend logs**: Show in the backend terminal window
- **Frontend logs**: Check browser console (F12)
- **API calls**: Monitor Network tab in browser dev tools

## 📚 Documentation

- **Model Visualization Guide**: `docs/MODEL_VISUALIZATION_GUIDE.md`
- **API Documentation**: http://localhost:8000/docs
- **Technical Details**: See individual component documentation

## 🎉 What's New

### Visualization Features
- **Interactive Model Breakdown**: See exactly how each model makes decisions
- **Feature Importance Analysis**: Understand which elements matter most
- **Decision Path Visualization**: Step-by-step decision process
- **Model Comparison**: Compare performance across different algorithms
- **Educational Explanations**: Human-readable insights

### Technical Improvements
- **Real-time Analysis**: Live feature extraction and processing
- **Multiple Model Support**: Three different AI approaches
- **Comprehensive Coverage**: Visual and structural analysis
- **Responsive Design**: Works on desktop and mobile

## 🚀 Next Steps

1. **Try Different Models**: Compare predictions across all three models
2. **Upload Various Content**: Test with different types of websites
3. **Explore Visualizations**: Use all tabs to understand the analysis
4. **Check API Documentation**: Visit http://localhost:8000/docs for technical details
5. **Read the Full Guide**: See `docs/MODEL_VISUALIZATION_GUIDE.md` for comprehensive information

---

**Happy Analyzing! 🎯**

The WebTrace AI system now provides unprecedented transparency into AI decision-making processes, helping you understand exactly how each model analyzes websites and arrives at predictions. 