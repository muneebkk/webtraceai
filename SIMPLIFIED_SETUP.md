# WebTrace AI - Simplified Setup Guide

## 🎯 Project Goal
**Binary Classification**: Determine if a website screenshot was AI-generated or human-coded

## 📁 Clean File Structure

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
└── PROJECT_STRUCTURE.md        # Clean project overview
```

## 👥 Team Responsibilities

### **Muneeb (Lead Developer)**
**Focus**: OpenCV Image Extraction & AI Model Training
- `backend/app/feature_extract.py` - Extract image features using OpenCV
- `backend/app/model_loader.py` - ML model management
- `backend/train_simple_model.py` - Training script
- `backend/test_feature_extractor.py` - Test feature extraction

**Your Tasks**:
1. ✅ Learn OpenCV for image processing
2. ✅ Extract features from screenshots (19 features implemented)
3. Train binary classification model (AI vs Human)
4. Evaluate and optimize model performance

### **Hassan Hadi (API Developer)**
**Focus**: FastAPI Backend & Prediction Endpoint
- `backend/main.py` - FastAPI application
- `backend/app/routes.py` - API endpoints (/predict)
- `backend/app/utils.py` - API utilities

**Your Tasks**:
1. Learn FastAPI
2. Implement /predict endpoint
3. Handle image uploads
4. Connect to Muneeb's model

### **Teammate A (Data Specialist)**
**Focus**: Manual Dataset Collection
- `dataset/` - All dataset files

**Your Tasks**:
1. Collect 50+ AI-generated website screenshots manually
2. Collect 50+ human-coded website screenshots manually
3. Organize data in proper folders
4. Update labels.csv manually

## 🚀 Quick Start Commands

### **Easy Setup (Recommended)**:
```bash
# Windows
.\start-dev.ps1

# Linux/Mac
./start-dev.sh
```

**What this does:**
- ✅ Creates Python virtual environment automatically
- ✅ Installs all dependencies
- ✅ Starts both backend and frontend servers

### **Manual Setup**:

**For Muneeb (Model Training)**:
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python test_feature_extractor.py  # Test feature extraction
python train_simple_model.py      # Train your model
```

**For Hassan Hadi (API Development)**:
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python main.py  # Start API server
```

**For Teammate A (Data Collection)**:
```bash
# Manual data collection - no scripts needed
# Simply organize screenshots in dataset/images/ai/ and dataset/images/human/
# Update dataset/labels.csv manually
```

## 📸 Manual Dataset Collection

### **Folder Structure**:
```
dataset/
├── images/
│   ├── ai/                    # AI screenshots
│   │   ├── framer_001.png
│   │   ├── wix_001.png
│   │   └── ...
│   └── human/                 # Human screenshots
│       ├── human_001.png
│       ├── human_002.png
│       └── ...
└── labels.csv                 # Dataset labels
```

### **Naming Convention**:
- **AI websites**: `{tool}_{number}.png` (e.g., `framer_001.png`)
- **Human websites**: `human_{number}.png` (e.g., `human_001.png`)

### **Labels.csv Format**:
```csv
id,tool
framer_001,framer
wix_001,wix
human_001,human
human_002,human
```

## 🎯 Binary Classification Model

**Input**: Website screenshot
**Output**: 
- `is_ai_generated`: true/false
- `confidence`: 0.0-1.0
- `predicted_class`: "AI" or "Human"

**Features Extracted** (19 total):

**Basic Features (4)**:
- `width`, `height`, `aspect_ratio`, `total_pixels`

**Color Features (5)**:
- `color_diversity_s`, `color_diversity_v` - HSV color variation
- `color_uniformity` - Color distribution entropy
- `avg_saturation`, `avg_brightness` - Average color properties

**Layout Features (5)**:
- `edge_density` - Ratio of edge pixels
- `contour_count`, `avg_contour_area`, `avg_contour_complexity` - Shape analysis
- `horizontal_vertical_ratio` - Spatial distribution

**Texture Features (5)**:
- `gradient_magnitude` - Texture intensity
- `texture_uniformity` - Local Binary Pattern analysis
- `local_variance` - Neighborhood texture measure
- `low_freq_energy`, `high_freq_energy` - FFT frequency analysis

## 📊 Target Dataset

**Goal**: 100+ total samples
- **50+ AI-generated websites**
  - 15+ Framer websites
  - 15+ Wix ADI websites
  - 10+ Notion AI websites
  - 10+ Other AI tools
- **50+ Human-coded websites**
  - 25+ Professional developer websites
  - 25+ Personal/portfolio websites

## 🛠️ Development Workflow

### **Week 1**: Setup & Learning
- All team members set up development environment
- Learn assigned technologies
- Create initial dataset structure

### **Week 2**: Data Collection & Feature Extraction
- Teammate A: Start collecting screenshots
- Muneeb: Implement image feature extraction
- Hassan Hadi: Set up basic API structure

### **Week 3**: Model Training & API Integration
- Muneeb: Train initial model
- Hassan Hadi: Connect API to model
- Teammate A: Continue data collection

### **Week 4**: Testing & Optimization
- Muneeb: Optimize model performance
- Hassan Hadi: Test full API workflow
- Teammate A: Finalize dataset

## 📚 Key Learning Resources

### **Muneeb (OpenCV & ML)**:
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- **Note**: Using OpenCV instead of Pillow for all image processing

### **Hassan Hadi (FastAPI)**:
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [FastAPI File Uploads](https://fastapi.tiangolo.com/tutorial/file-upload/)

### **Teammate A (Data Collection)**:
- [Screenshot Best Practices](https://www.smashingmagazine.com/2011/03/how-to-conduct-usability-testing-for-accessible-web-design/)
- [Data Organization](https://www.dataquest.io/blog/data-science-project-structure/)

## 🎯 Success Metrics

- **Dataset**: 100+ balanced samples
- **Model Accuracy**: > 80%
- **API**: Working /predict endpoint
- **Frontend**: Connected to backend

**Ready to start! 🚀** 