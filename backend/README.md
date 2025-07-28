# Backend - WebTrace AI

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test feature extraction
python test_feature_extractor.py

# Start API server
python main.py
```

## 📁 Structure

- `main.py` - FastAPI app entry point
- `app/feature_extract.py` - OpenCV feature extraction (19 features)
- `app/model_loader.py` - ML model management
- `app/routes.py` - API endpoints (/predict)
- `app/utils.py` - API utilities
- `test_feature_extractor.py` - Test feature extraction
- `train_simple_model.py` - Train ML model

## 🧪 Testing

```bash
# Test with generated image
python test_feature_extractor.py

# Test with real image
python test_feature_extractor.py path/to/image.jpg
```

## 🤖 Training

```bash
# Train model (requires dataset)
python train_simple_model.py
``` 