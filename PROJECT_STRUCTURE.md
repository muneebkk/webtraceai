# WebTrace AI - Clean Project Structure

## 🎯 Project Goal
**Binary Classification**: Determine if a website screenshot was AI-generated or human-coded

## 📁 Simplified File Structure

```
webtraceai/
├── backend/                    # Python FastAPI backend
│   ├── app/
│   │   ├── feature_extract.py  # OpenCV feature extraction (19 features)
│   │   ├── model_loader.py     # ML model management
│   │   ├── routes.py           # API endpoints (/predict)
│   │   └── utils.py            # API utilities
│   ├── main.py                 # FastAPI app entry point
│   ├── requirements.txt        # Python dependencies
│   ├── test_feature_extractor.py  # Test feature extraction
│   └── train_simple_model.py   # Train ML model
├── dataset/                    # Website screenshots
│   ├── images/
│   │   ├── ai/                 # AI-generated websites
│   │   └── human/              # Human-coded websites
│   ├── labels.csv              # Dataset labels
│   └── README.md               # Data collection guide
├── frontend/                   # React frontend (existing)
├── start-dev.ps1               # Windows development script
├── start-dev.sh                # Linux/Mac development script
└── SIMPLIFIED_SETUP.md         # Project overview
```

## 🚀 Quick Start

### **Windows**:
```bash
.\start-dev.ps1
```

### **Linux/Mac**:
```bash
./start-dev.sh
```

## 👥 Team Responsibilities

### **Muneeb (Lead Developer)**
- **Feature Extraction**: `backend/app/feature_extract.py`
- **Model Training**: `backend/train_simple_model.py`
- **Testing**: `backend/test_feature_extractor.py`

### **Hassan Hadi (API Developer)**
- **FastAPI App**: `backend/main.py`
- **API Routes**: `backend/app/routes.py`
- **Model Loading**: `backend/app/model_loader.py`

### **Teammate A (Data Collection)**
- **Dataset**: `dataset/` folder
- **Manual screenshot collection**
- **Labels management**

## 📊 Features Extracted (19 total)

**Basic (4)**: `width`, `height`, `aspect_ratio`, `total_pixels`

**Color (5)**: `color_diversity_s`, `color_diversity_v`, `color_uniformity`, `avg_saturation`, `avg_brightness`

**Layout (5)**: `edge_density`, `contour_count`, `avg_contour_area`, `avg_contour_complexity`, `horizontal_vertical_ratio`

**Texture (5)**: `gradient_magnitude`, `texture_uniformity`, `local_variance`, `low_freq_energy`, `high_freq_energy`

## 🎯 Success Metrics
- **Dataset**: 100+ balanced samples
- **Model Accuracy**: > 80%
- **API**: Working /predict endpoint
- **Frontend**: Connected to backend

**Ready to start! 🚀** 