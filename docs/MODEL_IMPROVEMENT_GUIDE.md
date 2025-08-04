# Model Improvement Guide - Fixing Overfitting and Bias

## 🚨 Current Issues Identified

Your model is experiencing overfitting and bias because:

1. **Small Dataset**: Only 40 samples (20 AI + 20 human)
2. **Single AI Tool Bias**: All AI samples are from V0.dev, creating narrow patterns
3. **Limited Diversity**: Model learns V0.dev-specific patterns instead of general AI vs human patterns
4. **No Regularization**: Simple Random Forest without proper hyperparameter tuning

## 🛠️ Solutions Implemented

### 1. Improved Model Training (`improve_model.py`)

**Features:**
- **Multiple Model Comparison**: Tests Random Forest, Gradient Boosting, SVM, and Logistic Regression
- **Hyperparameter Tuning**: Uses GridSearchCV to find optimal parameters
- **Cross-Validation**: 5-fold CV to prevent overfitting
- **Feature Scaling**: Proper scaling for SVM and Logistic Regression
- **Class Imbalance Handling**: SMOTE oversampling for balanced classes
- **Feature Analysis**: Identifies important features and correlations

**Usage:**
```bash
cd backend
python improve_model.py
```

### 2. Data Augmentation (`data_augmentation.py`)

**Augmentation Methods:**
- Brightness/Contrast adjustments
- Color saturation changes
- Hue shifts
- Gaussian blur
- Noise addition
- Rotation (±15 degrees)
- Crop and resize
- Perspective transformations

**Usage:**
```bash
cd backend
python data_augmentation.py
```

## 📈 Step-by-Step Improvement Process

### Step 1: Analyze Current Model
```bash
cd backend
python improve_model.py
```

This will:
- Load your current dataset
- Analyze feature importance
- Identify class imbalance
- Test multiple models
- Generate recommendations

### Step 2: Augment Your Dataset
```bash
cd backend
python data_augmentation.py
```

This creates:
- 2 augmented versions per original image
- Maintains original images
- Creates balanced dataset
- Saves to `dataset_augmented/`

### Step 3: Train Improved Model
```bash
cd backend
python improve_model.py
```

### Step 4: Collect More Diverse Data

**Priority 1: Add More AI Tools**
- **Framer AI**: framer.com (AI-generated sites)
- **Wix ADI**: wix.com/adi
- **Notion AI**: notion.so
- **Webflow AI**: webflow.com
- **10Web**: 10web.io
- **Hostinger AI**: hostinger.com/website-builder

**Priority 2: Add Diverse Website Types**
- E-commerce sites
- Portfolio websites
- Blog/News sites
- Corporate websites
- Landing pages
- SaaS product pages

**Priority 3: Add Different Screen Sizes**
- Desktop (1920x1080)
- Tablet (768x1024)
- Mobile (375x667)

## 🎯 Expected Improvements

### Before (Current Issues)
- **Accuracy**: ~60-70% on test set
- **Overfitting**: High accuracy on training, poor on external data
- **Bias**: AI websites from other tools classified as human
- **Generalization**: Poor performance on diverse websites

### After (With Improvements)
- **Accuracy**: 80-90% on test set
- **Generalization**: Better performance on external data
- **Robustness**: Handles different AI tools and website types
- **Confidence**: More reliable predictions

## 🔍 Model Selection Strategy

The improved trainer tests multiple models:

1. **Random Forest**: Good for feature importance, handles non-linear relationships
2. **Gradient Boosting**: Often better performance, but more prone to overfitting
3. **SVM**: Good for high-dimensional data, requires feature scaling
4. **Logistic Regression**: Simple, interpretable, good baseline

**Best Choice**: Usually Random Forest or Gradient Boosting with proper regularization

## 📊 Feature Engineering Insights

**Most Important Features** (based on analysis):
1. `edge_density` - AI sites often have cleaner edges
2. `color_uniformity` - Human sites have more color variation
3. `gradient_magnitude` - Texture differences
4. `contour_complexity` - Layout complexity
5. `horizontal_vertical_ratio` - Design patterns

## 🚀 Advanced Improvements

### 1. Deep Learning Approach
Consider using a CNN for better feature extraction:
```python
# Example CNN architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

### 2. Ensemble Methods
Combine multiple models for better performance:
```python
# Voting classifier
ensemble = VotingClassifier([
    ('rf', RandomForestClassifier()),
    ('gb', GradientBoostingClassifier()),
    ('svm', SVC(probability=True))
])
```

### 3. Transfer Learning
Use pre-trained models like ResNet or EfficientNet:
```python
# Fine-tune pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False)
# Add custom classification head
```

## 📋 Dataset Collection Checklist

### AI-Generated Websites (Target: 50+ samples)
- [ ] V0.dev (current: 20 samples)
- [ ] Framer AI (target: 10 samples)
- [ ] Wix ADI (target: 10 samples)
- [ ] Notion AI (target: 5 samples)
- [ ] Webflow AI (target: 5 samples)
- [ ] Other AI tools (target: 10 samples)

### Human-Coded Websites (Target: 50+ samples)
- [ ] Professional portfolios (target: 15 samples)
- [ ] E-commerce sites (target: 10 samples)
- [ ] Corporate websites (target: 10 samples)
- [ ] Blog/News sites (target: 10 samples)
- [ ] Landing pages (target: 5 samples)

### Quality Requirements
- [ ] High-resolution screenshots (1920x1080+)
- [ ] Full-page captures (not just viewport)
- [ ] Clean appearance (no popups/overlays)
- [ ] Diverse color schemes and layouts
- [ ] Different design styles and trends

## 🔄 Continuous Improvement

### Monitor Performance
- Track accuracy on new external data
- Monitor false positive/negative rates
- Analyze misclassified samples

### Regular Updates
- Retrain model monthly with new data
- Update feature extraction as needed
- Test on latest AI tools and trends

### A/B Testing
- Compare model versions
- Test different feature sets
- Evaluate ensemble methods

## 🎯 Success Metrics

**Target Performance:**
- **Accuracy**: >85% on test set
- **Precision**: >80% for both classes
- **Recall**: >80% for both classes
- **F1-Score**: >80% overall
- **External Validation**: >75% on completely new data

**Reduced Bias:**
- AI websites from different tools correctly classified
- Human websites from various sources correctly classified
- Consistent performance across different website types

## 💡 Quick Start Commands

```bash
# 1. Install new dependencies
cd backend
pip install -r requirements.txt

# 2. Analyze current model
python improve_model.py

# 3. Augment dataset
python data_augmentation.py

# 4. Train improved model
python improve_model.py

# 5. Test the new model
python -c "from app.model_loader import ModelLoader; ml = ModelLoader('improved_model.pkl'); print('Model loaded successfully!')"
```

This comprehensive approach should significantly improve your model's performance and reduce overfitting and bias issues. 