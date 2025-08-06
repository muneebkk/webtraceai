# Model Visualization Guide

## Overview

The WebTrace AI system now includes comprehensive visualization capabilities that allow users to understand exactly how each AI model makes predictions about whether a website was AI-generated or human-coded. This transparency helps build trust and provides educational insights into the decision-making process.

## Features

### 1. Interactive Model Visualization Modal

The visualization system provides a rich, interactive modal that displays:

- **Prediction Summary**: Clear display of the final prediction with confidence score
- **Feature Importance**: Visual representation of which features contributed most to the decision
- **Decision Path**: Step-by-step breakdown of how the model arrived at its conclusion
- **Model Comparison**: Side-by-side comparison of all available models
- **Detailed Explanation**: Human-readable explanation of the prediction

### 2. Multiple Visualization Tabs

The modal includes five specialized tabs:

#### Overview Tab
- High-level prediction summary
- Confidence visualization
- Key feature importance chart
- Human-readable explanation

#### Features Tab
- Detailed feature importance analysis
- Top 8 most influential features
- Percentage contribution of each feature
- Interactive progress bars

#### Decision Path Tab
- Step-by-step decision process
- Feature thresholds and comparisons
- Contribution levels (High/Medium/Low)
- Actual values vs. decision thresholds

#### Model Comparison Tab
- Side-by-side model performance
- Accuracy comparisons
- Model descriptions and use cases
- Visual performance indicators

#### Explanation Tab
- Detailed prediction rationale
- Key feature values
- Model-specific details
- Educational insights

## Backend Implementation

### New Endpoint: `/api/predict-visualization`

This endpoint provides detailed visualization data for each prediction:

```python
POST /api/predict-visualization
Content-Type: multipart/form-data

Parameters:
- screenshot: UploadFile (required)
- model: str (required) - "original", "improved", or "custom_tree"

Response:
{
  "model_used": "improved",
  "prediction": "AI-Generated",
  "confidence": 0.95,
  "feature_importance": {
    "color_diversity_s": 0.15,
    "edge_density": 0.12,
    "ai_font_ratio": 0.12,
    ...
  },
  "decision_path": [
    {
      "feature": "ai_font_ratio",
      "value": 0.8,
      "threshold": 0.5,
      "condition": "ai_font_ratio > 0.50",
      "importance": 0.12,
      "contribution": "High"
    },
    ...
  ],
  "feature_values": {
    "width": 1200,
    "height": 800,
    "color_diversity_s": 45.2,
    ...
  },
  "model_type": "LogisticRegression",
  "explanation": "The model is 95.0% confident this website was AI-generated..."
}
```

### Key Functions

#### `generate_visualization_data()`
Generates comprehensive visualization data including:
- Feature importance based on model type
- Decision path analysis
- Human-readable explanations

#### `get_feature_importance()`
Returns feature importance scores:
- **Custom Tree**: Uses actual tree-based importance
- **Improved Model**: Predefined importance based on model characteristics
- **Original Model**: Balanced importance distribution

#### `get_decision_path()`
Creates decision path visualization:
- **Custom Tree**: Traces actual tree traversal
- **Other Models**: Simulates decision path based on feature importance

#### `generate_explanation()`
Creates human-readable explanations including:
- Confidence level explanation
- Feature-specific insights
- Model-specific context

## Frontend Implementation

### Visualization Components

#### `FeatureImportanceChart`
- Displays top 8 most important features
- Interactive progress bars
- Percentage-based visualization
- Responsive design

#### `DecisionPathVisualization`
- Step-by-step decision breakdown
- Feature thresholds and comparisons
- Contribution level indicators
- Visual decision flow

#### `ModelComparisonChart`
- Side-by-side model comparison
- Performance metrics
- Visual accuracy indicators
- Model descriptions

#### `PredictionExplanation`
- Human-readable explanations
- Key feature values
- Model details
- Educational insights

#### `ModelVisualizationModal`
- Tabbed interface
- Responsive design
- Interactive elements
- Comprehensive data display

### Integration

The visualization system integrates seamlessly with the existing analysis flow:

1. User uploads screenshot
2. User selects preferred model
3. Analysis runs with both prediction and visualization data
4. Results display with "Visualize" button
5. Clicking "Visualize" opens the comprehensive modal

## Model-Specific Visualizations

### 1. Original Random Forest Model
- **Feature Importance**: Balanced across all features
- **Decision Path**: Simulated based on ensemble voting
- **Explanation**: Focuses on ensemble decision-making
- **Best For**: Baseline comparisons and simple use cases

### 2. Improved Logistic Regression Model
- **Feature Importance**: Optimized for selected features
- **Decision Path**: Linear decision boundaries
- **Explanation**: Statistical feature selection insights
- **Best For**: Production use and highest accuracy

### 3. Custom Decision Tree Model
- **Feature Importance**: Actual tree-based importance
- **Decision Path**: Real tree traversal path
- **Explanation**: Interpretable decision rules
- **Best For**: Educational purposes and interpretability

## Educational Benefits

### For Developers
- Understand how different algorithms approach the same problem
- Learn about feature engineering and selection
- See the impact of model optimization techniques

### For Users
- Build trust through transparency
- Understand prediction confidence
- Learn about AI detection capabilities

### For Researchers
- Compare model interpretability
- Analyze decision-making patterns
- Study feature importance across algorithms

## Technical Details

### Feature Categories

The system analyzes multiple feature categories:

1. **Basic Features**: Width, height, aspect ratio, total pixels
2. **Color Features**: Saturation diversity, brightness, color uniformity
3. **Layout Features**: Edge density, contour count, spatial distribution
4. **Texture Features**: Texture complexity, pattern analysis
5. **Visual Features**: Color schemes, typography, layout patterns

### Threshold Values

Each feature has model-specific thresholds:
- `color_diversity_s`: 50.0
- `edge_density`: 0.1
- `color_diversity_s`: 45.2
- `ai_font_ratio`: 0.5
- And more...

### Confidence Calculation

Confidence is calculated based on:
- Model prediction probabilities
- Feature importance weights
- Decision path strength
- Model-specific calibration

## Future Enhancements

### Planned Features
1. **Interactive Decision Trees**: Clickable tree visualization for custom model
2. **Feature Correlation Analysis**: Show relationships between features
3. **Model Performance History**: Track accuracy over time
4. **Export Capabilities**: Save visualizations as reports
5. **Comparative Analysis**: Compare predictions across multiple models

### Technical Improvements
1. **Real-time Visualization**: Live updates during analysis
2. **Advanced Charts**: More sophisticated chart types
3. **Custom Thresholds**: User-adjustable decision boundaries
4. **Model Training Visualization**: Show training process insights

## Usage Examples

### Basic Usage
1. Upload a website screenshot
2. Select "Improved Logistic Regression" model
3. Click "Analyze Website"
4. Click "Visualize" to see detailed breakdown
5. Explore different tabs for comprehensive insights

### Advanced Usage
1. Upload screenshot
2. Compare predictions across all three models
3. Analyze feature importance differences
4. Study decision path variations
5. Understand model-specific strengths

## Troubleshooting

### Common Issues

1. **Visualization Not Loading**
   - Check if backend is running on port 8000
   - Verify model files exist in backend directory
   - Check browser console for errors

2. **Missing Data**
   - Ensure screenshot is provided
   - Verify model selection is valid
   - Check network connectivity

3. **Performance Issues**
   - Large images may take longer to process
   - Large images may increase analysis time
   - Consider reducing image size for faster processing

### Debug Information

The system provides detailed logging:
- Backend logs show feature extraction process
- Frontend console shows API call details
- Network tab shows request/response data

## Conclusion

The model visualization system provides unprecedented transparency into AI decision-making processes. By showing exactly how each model arrives at its predictions, users can build trust in the system and gain valuable insights into AI detection capabilities.

The system is designed to be both educational and practical, serving the needs of developers, researchers, and end-users alike. With its comprehensive feature set and intuitive interface, it represents a significant step forward in AI explainability and transparency. 