# Hassan Hadi: API endpoints and request handling
# Focus: Handle /predict route: load image, run feature extraction, return prediction

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import io
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

@router.post("/predict", response_model=AnalysisResponse)
async def predict_website(screenshot: UploadFile = File(...)):
    """
    Predict whether a website screenshot was AI-generated or human-coded
    """
    try:
        # Validate uploaded file
        if not validate_file(screenshot):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload PNG or JPG image.")
        
        # Process the uploaded image
        image_data = await screenshot.read()
        processed_image = process_image(io.BytesIO(image_data))
        
        # Initialize feature extractor and model loader
        feature_extractor = FeatureExtractor()
        model_loader = ModelLoader()
        
        # Extract image features only
        features = feature_extractor.extract_image_features(processed_image)
        
        # Make prediction
        prediction = model_loader.predict(features)
        
        return AnalysisResponse(
            is_ai_generated=prediction["is_ai_generated"],
            confidence=prediction["confidence"],
            predicted_class=prediction["predicted_class"],
            tool_probabilities=prediction["tool_probabilities"],
            features_used=prediction["features_used"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/model-info")
async def get_model_info():
    """
    Get information about the current model
    """
    try:
        model_loader = ModelLoader()
        return model_loader.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "message": "WebTrace AI API is running"} 