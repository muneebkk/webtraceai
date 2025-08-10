# WebTrace AI - AI Website Detection System

Detect whether websites were created using AI tools or coded by humans using computer vision and machine learning.

## �� Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/webtraceai.git
   cd webtraceai
   ```

2. **Start the development environment**
   ```bash
   # Windows
   .\start-dev.ps1
   
   # Linux/Mac
   ./start-dev.sh
   ```

3. **Open your browser**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000

## ✨ Features

- 📸 **Screenshot Analysis** - Upload website screenshots
- �� **AI Detection** - 80-85% accuracy with visual features
- 🎯 **Real-time Results** - Instant predictions with confidence scores
- 🎨 **Modern UI** - Dark theme with drag & drop interface

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python, OpenCV, scikit-learn
- **Frontend**: React, Vite, Tailwind CSS
- **ML Model**: RandomForestClassifier with 19 visual features
- **Accuracy**: 80-85% on visual analysis

## 📊 Model Details

- **Algorithm**: RandomForestClassifier
- **Features**: 19 visual features (layout, color, texture, patterns)
- **Training Data**: 20+ samples per class
- **Response Time**: ~2-3 seconds

## 🔧 Development

```bash
# Test the model
cd .\backend\tests\
python test_model.py

# Test feature extraction
python test_feature_extractor.py

# Retrain model with new data
python improve_all_models.py
```

## 📈 Future Enhancements

- HTML/CSS code analysis
- Enhanced feature extraction
- Larger training dataset
- Ensemble model methods