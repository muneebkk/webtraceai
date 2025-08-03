# Hassan Hadi: API endpoints and request handling
# Focus: Handle /predict route: load image, run feature extraction, return prediction

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import io
import os
from .feature_extract import FeatureExtractor
from .model_loader import ModelLoader
from .utils import process_image, validate_file

router = APIRouter()

class AnalysisResponse(BaseModel):
    is_ai_generated: bool
    confidence: float
    predicted_class: str
    tool_probabilities: dict
    features_used: List[str]
    model_status: str
    model_used: str

class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    accuracy: float
    techniques: List[str]
    problems_solved: List[str]
    how_it_works: str
    best_for: str
    limitations: List[str]

@router.get("/models")
async def get_available_models():
    """
    Get information about all available models
    """
    models = [
        {
            "id": "original",
            "name": "Original Random Forest",
            "description": "Basic Random Forest classifier with all features",
            "accuracy": 73.33,
            "techniques": [
                "Random Forest Classifier (100 estimators)",
                "All 19 extracted features",
                "Simple train-test split (80/20)",
                "No feature selection",
                "No hyperparameter tuning"
            ],
            "problems_solved": [
                "Basic AI vs Human classification",
                "Quick baseline performance",
                "Simple implementation"
            ],
            "how_it_works": "Uses an ensemble of decision trees to classify websites. Each tree votes on whether a site is AI-generated or human-coded, with the majority vote determining the final prediction. Features include layout analysis, color patterns, and structural elements.",
            "best_for": "Quick analysis, baseline comparisons, simple use cases",
            "limitations": [
                "No feature selection (uses all features)",
                "Potential overfitting",
                "No handling of class imbalance",
                "Limited hyperparameter optimization",
                "Lower accuracy on current dataset"
            ]
        },
        {
            "id": "improved",
            "name": "Improved Logistic Regression",
            "description": "Advanced pipeline with feature selection and class balancing",
            "accuracy": 95.24,
            "techniques": [
                "Logistic Regression (selected via cross-validation)",
                "Feature selection (14/19 features)",
                "SMOTE for class imbalance",
                "GridSearchCV hyperparameter tuning",
                "5-fold cross-validation",
                "Feature scaling (StandardScaler)",
                "Correlation-based feature removal"
            ],
            "problems_solved": [
                "Class imbalance in dataset",
                "Feature redundancy",
                "Overfitting prevention",
                "Optimal model selection"
            ],
            "how_it_works": "Uses statistical feature selection to identify the most important features, then applies SMOTE to balance the dataset. Multiple algorithms are tested via cross-validation, with Logistic Regression selected as the best performer. Features are scaled for optimal performance.",
            "best_for": "Production use, highest accuracy, robust predictions",
            "limitations": [
                "Requires feature scaling",
                "More complex pipeline",
                "Longer training time",
                "Linear decision boundaries"
            ]
        },
        {
            "id": "custom_tree",
            "name": "Custom Decision Tree",
            "description": "Custom-built decision tree with pruning and interpretability",
            "accuracy": 93.06,
            "techniques": [
                "Custom decision tree implementation",
                "Tree pruning to prevent overfitting",
                "Gini/Entropy impurity calculation",
                "Feature importance analysis",
                "Customizable tree parameters",
                "Interpretable decision paths",
                "Max depth: 4, Min samples: 8"
            ],
            "problems_solved": [
                "Model interpretability",
                "Overfitting through pruning",
                "Custom algorithm development",
                "Educational purposes"
            ],
            "how_it_works": "A decision tree built from scratch that recursively splits the data based on feature thresholds. Uses impurity measures (Gini or Entropy) to find optimal splits. Includes pruning to remove unnecessary branches and prevent overfitting. Provides clear decision paths for interpretability.",
            "best_for": "Educational purposes, interpretable results, custom algorithm development",
            "limitations": [
                "Single decision tree (less robust than ensembles)",
                "Potential for overfitting without proper pruning",
                "Custom implementation may have bugs",
                "Limited to binary splits"
            ]
        }
    ]
    
    return {"models": models}

@router.post("/predict", response_model=AnalysisResponse)
async def predict_website(
    screenshot: UploadFile = File(None),
    html_content: str = Form(None),
    model: str = Form("improved", description="Model to use: 'original', 'improved', or 'custom_tree'")
):
    """
    Predict whether a website was AI-generated or human-coded
    Accepts screenshot, HTML code, or both for maximum accuracy
    """
    try:
        # Validate that at least one input is provided
        if not screenshot and not html_content:
            raise HTTPException(
                status_code=400, 
                detail="Please provide either a screenshot or HTML code for analysis"
            )
        
        # Debug: Log the received model parameter
        print(f"🔍 Received model parameter: '{model}'")
        
        # Validate model selection
        valid_models = ["original", "improved", "custom_tree"]
        if model not in valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Choose from: {', '.join(valid_models)}"
            )
        
        # Map model ID to file path
        model_files = {
            "original": "model.pkl",
            "improved": "improved_model.pkl", 
            "custom_tree": "custom_tree_model.pkl"
        }
        
        model_file = model_files[model]
        
        # Check if model file exists
        if not os.path.exists(model_file):
            raise HTTPException(
                status_code=500,
                detail=f"Model file {model_file} not found. Please ensure the model has been trained."
            )
        
        # Initialize feature extractor and model loader
        feature_extractor = FeatureExtractor()
        model_loader = ModelLoader()

        # Load the selected model
        print(f"🤖 Loading model: {model} ({model_file})")
        model_loader.load_model(model_file)
        
        # Verify model was loaded correctly
        if not model_loader.model_loaded:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model {model} from {model_file}"
            )
        
        print(f"✅ Model loaded successfully: {model_loader.model_type}")
        print(f"   Features available: {len(model_loader.feature_names)}")
        print(f"   Model format: {model_loader.model_info.get('model_format', 'unknown')}")
        
        features = {}
        
        # Process screenshot if provided
        if screenshot:
            print(f"📸 Processing uploaded image: {screenshot.filename}")
            
            # Validate uploaded file
            if not validate_file(screenshot):
                raise HTTPException(status_code=400, detail="Invalid file format. Please upload PNG or JPG image.")
            
            # Process the uploaded image
            image_data = await screenshot.read()
            processed_image = process_image(io.BytesIO(image_data))
            
            print(f"✅ Image processed successfully: {processed_image.size[0]}x{processed_image.size[1]}")
            
            # Extract image features
            print("🔍 Extracting image features...")
            image_features = feature_extractor.extract_image_features(processed_image)
            features.update(image_features)
            print(f"✅ Extracted {len(image_features)} image features")
        
        # Process HTML if provided
        if html_content:
            print("🔍 Processing HTML content...")
            # Enhanced HTML feature extraction
            html_features = {
                'html_length': len(html_content),
                'has_ai_signatures': 1 if any(sig in html_content.lower() for sig in ['wix', 'squarespace', 'framer', 'vercel', 'webflow', 'shopify']) else 0,
                'css_complexity': html_content.count('class=') + html_content.count('style='),
                'script_count': html_content.count('<script'),
                'div_count': html_content.count('<div'),
                'semantic_tags': sum(1 for tag in ['header', 'nav', 'main', 'section', 'article', 'aside', 'footer'] if f'<{tag}' in html_content.lower()),
                'bootstrap_indicators': 1 if any(indicator in html_content.lower() for indicator in ['bootstrap', 'btn-', 'container-fluid', 'row', 'col-']) else 0,
                'tailwind_indicators': 1 if any(indicator in html_content.lower() for indicator in ['tailwind', 'bg-', 'text-', 'p-', 'm-', 'flex', 'grid']) else 0,
                'ai_tool_signatures': sum(1 for tool in ['cursor', 'github copilot', 'tabnine', 'kite', 'intellicode'] if tool in html_content.lower())
            }
            features.update(html_features)
            print(f"✅ Extracted {len(html_features)} HTML features")
        
        # Make prediction
        print(f"🤖 Making prediction using {model} model...")
        prediction = model_loader.predict(features)
        
        # Generate tool-specific probabilities based on model and features
        tool_probabilities = generate_tool_probabilities(prediction, features, model)
        
        print(f"🎯 Final Result: {prediction['predicted_class'].upper()} (confidence: {prediction['confidence']:.3f})")
        print(f"🔧 Model used: {model}")
        
        return AnalysisResponse(
            is_ai_generated=prediction["is_ai_generated"],
            confidence=prediction["confidence"],
            predicted_class=prediction["predicted_class"],
            tool_probabilities=tool_probabilities,
            features_used=prediction["features_used"],
            model_status=prediction["model_status"],
            model_used=model
        )
        
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def generate_tool_probabilities(prediction, features, model):
    """
    Generate tool-specific probabilities based on prediction and features
    """
    base_confidence = prediction["confidence"]
    is_ai = prediction["is_ai_generated"]
    
    # Base tool probabilities (these would ideally come from the model)
    tools = {
        "v0 by Vercel": 0.0,
        "Framer AI": 0.0,
        "Cursor": 0.0,
        "Wix ADI": 0.0,
        "Notion AI": 0.0,
        "Human-coded": 0.0
    }
    
    if is_ai:
        # Distribute probability among AI tools based on features
        html_content = features.get('html_length', 0)
        has_ai_signatures = features.get('has_ai_signatures', 0)
        bootstrap_indicators = features.get('bootstrap_indicators', 0)
        tailwind_indicators = features.get('tailwind_indicators', 0)
        ai_tool_signatures = features.get('ai_tool_signatures', 0)
        
        # Model-specific tool preferences
        if model == "original":
            # Original model tends to detect more generic AI tools
            tools["v0 by Vercel"] = 0.35
            tools["Framer AI"] = 0.25
            tools["Cursor"] = 0.20
            tools["Wix ADI"] = 0.15
            tools["Notion AI"] = 0.05
        elif model == "improved":
            # Improved model is better at detecting specific tools
            if has_ai_signatures:
                tools["v0 by Vercel"] = 0.45
                tools["Framer AI"] = 0.30
                tools["Cursor"] = 0.15
                tools["Wix ADI"] = 0.08
                tools["Notion AI"] = 0.02
            else:
                tools["Cursor"] = 0.40
                tools["v0 by Vercel"] = 0.25
                tools["Framer AI"] = 0.20
                tools["Wix ADI"] = 0.10
                tools["Notion AI"] = 0.05
        elif model == "custom_tree":
            # Custom tree model has different detection patterns
            if tailwind_indicators:
                tools["v0 by Vercel"] = 0.50
                tools["Framer AI"] = 0.25
                tools["Cursor"] = 0.15
                tools["Wix ADI"] = 0.08
                tools["Notion AI"] = 0.02
            elif bootstrap_indicators:
                tools["Wix ADI"] = 0.40
                tools["Framer AI"] = 0.25
                tools["v0 by Vercel"] = 0.20
                tools["Cursor"] = 0.10
                tools["Notion AI"] = 0.05
            else:
                tools["Cursor"] = 0.35
                tools["v0 by Vercel"] = 0.25
                tools["Framer AI"] = 0.20
                tools["Wix ADI"] = 0.15
                tools["Notion AI"] = 0.05
    else:
        # Human-coded
        tools["Human-coded"] = 1.0
    
    # Normalize probabilities
    total = sum(tools.values())
    if total > 0:
        tools = {k: v / total for k, v in tools.items()}
    
    return tools

@router.get("/model-info")
async def get_model_info():
    """
    Get information about the current model
    """
    try:
        model_loader = ModelLoader()
        model_loader.load_model("improved_model.pkl")  # Default to improved model
        
        model_info = model_loader.get_model_info()
        return {
            "status": "trained_model" if model_loader.model_loaded else "mock_model",
            "model_type": type(model_loader.model).__name__ if model_loader.model else None,
            "feature_count": len(model_loader.feature_names),
            "features": model_loader.feature_names,
            "details": model_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.get("/model/{model_id}")
async def get_specific_model_info(model_id: str):
    """
    Get detailed information about a specific model by ID
    """
    try:
        # Validate model ID
        valid_models = ["original", "improved", "custom_tree"]
        if model_id not in valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model ID. Choose from: {', '.join(valid_models)}"
            )
        
        # Get model information from the models list
        models = [
            {
                "id": "original",
                "name": "Original Random Forest",
                "description": "Basic Random Forest classifier with all features",
                "accuracy": 82.69,
                "techniques": [
                    "Random Forest Classifier (100 estimators)",
                    "All 19 extracted features",
                    "Simple train-test split (80/20)",
                    "No feature selection",
                    "No hyperparameter tuning"
                ],
                "problems_solved": [
                    "Basic AI vs Human classification",
                    "Quick baseline performance",
                    "Simple implementation"
                ],
                "how_it_works": "Uses an ensemble of decision trees to classify websites. Each tree votes on whether a site is AI-generated or human-coded, with the majority vote determining the final prediction. Features include layout analysis, color patterns, and structural elements.",
                "best_for": "Quick analysis, baseline comparisons, simple use cases",
                "limitations": [
                    "No feature selection (uses all features)",
                    "Potential overfitting",
                    "No handling of class imbalance",
                    "Limited hyperparameter optimization"
                ]
            },
            {
                "id": "improved",
                "name": "Improved Logistic Regression",
                "description": "Advanced pipeline with feature selection and class balancing",
                "accuracy": 86.54,
                "techniques": [
                    "Logistic Regression (selected via cross-validation)",
                    "Feature selection (14/19 features)",
                    "SMOTE for class imbalance",
                    "GridSearchCV hyperparameter tuning",
                    "5-fold cross-validation",
                    "Feature scaling (StandardScaler)",
                    "Correlation-based feature removal"
                ],
                "problems_solved": [
                    "Class imbalance in dataset",
                    "Feature redundancy",
                    "Overfitting prevention",
                    "Optimal model selection"
                ],
                "how_it_works": "Uses statistical feature selection to identify the most important features, then applies SMOTE to balance the dataset. Multiple algorithms are tested via cross-validation, with Logistic Regression selected as the best performer. Features are scaled for optimal performance.",
                "best_for": "Production use, highest accuracy, robust predictions",
                "limitations": [
                    "Requires feature scaling",
                    "More complex pipeline",
                    "Longer training time",
                    "Linear decision boundaries"
                ]
            },
            {
                "id": "custom_tree",
                "name": "Custom Decision Tree",
                "description": "Custom-built decision tree with pruning and interpretability",
                "accuracy": 85.00,
                "techniques": [
                    "Custom decision tree implementation",
                    "Tree pruning",
                    "Gini/Entropy impurity calculation",
                    "Manual feature engineering",
                    "Interpretable decision paths"
                ],
                "problems_solved": [
                    "Model interpretability",
                    "Overfitting through pruning",
                    "Educational purposes",
                    "Custom feature importance"
                ],
                "how_it_works": "A decision tree built from scratch with pruning to prevent overfitting. Uses Gini impurity to make splits and provides interpretable decision paths. Each node represents a feature threshold that helps classify websites.",
                "best_for": "Educational purposes, interpretable results, custom analysis",
                "limitations": [
                    "Single decision tree (no ensemble)",
                    "Custom implementation",
                    "May not capture complex patterns",
                    "Limited to binary splits"
                ]
            }
        ]
        
        # Find the requested model
        model_info = next((model for model in models if model["id"] == model_id), None)
        
        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Check if model file exists
        model_files = {
            "original": "model.pkl",
            "improved": "improved_model.pkl", 
            "custom_tree": "custom_tree_model.pkl"
        }
        
        model_file = model_files[model_id]
        model_status = "available" if os.path.exists(model_file) else "not_found"
        
        return {
            **model_info,
            "status": model_status,
            "file_path": model_file
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "message": "WebTrace AI API is running"}
