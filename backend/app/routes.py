from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import io
from .feature_extract import FeatureExtractor
from .model_loader import ModelLoader
from .utils import process_image, validate_file

router = APIRouter()

class AnalysisRequest(BaseModel):
    html_content: Optional[str] = None
    visible_text: Optional[str] = None

class AnalysisResponse(BaseModel):
    is_ai_generated: bool
    confidence: float
    tool_probabilities: dict
    features_used: List[str]

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_website(
    screenshot: UploadFile = File(...),
    html_content: Optional[str] = None,
    visible_text: Optional[str] = None
):
    """
    Analyze a website screenshot to detect AI generation
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
        
        # Extract features
        features = feature_extractor.extract_all_features(
            image=processed_image,
            html_content=html_content,
            visible_text=visible_text
        )
        
        # Make prediction
        prediction = model_loader.predict(features)
        
        return AnalysisResponse(
            is_ai_generated=prediction["is_ai_generated"],
            confidence=prediction["confidence"],
            tool_probabilities=prediction["tool_probabilities"],
            features_used=prediction["features_used"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/tools")
async def get_supported_tools():
    """
    Get list of supported AI tools for detection
    """
    return {
        "supported_tools": [
            "Framer AI",
            "Wix ADI", 
            "Notion AI",
            "Durable",
            "Cursor",
            "v0 by Vercel",
            "ChatGPT HTML Generator",
            "Human"
        ]
    }

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