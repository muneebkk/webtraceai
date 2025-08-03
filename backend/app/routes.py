# Hassan Hadi: API endpoints and request handling
# Focus: Handle /predict route: load image, run feature extraction, return prediction

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
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
    model_status: str

@router.post("/predict", response_model=AnalysisResponse)
async def predict_website(
    screenshot: UploadFile = File(None),
    html_content: str = Form(None)
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
        
        # Initialize feature extractor and model loader
        feature_extractor = FeatureExtractor()
        model_loader = ModelLoader()
        
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
            # For now, we'll use basic HTML features
            # TODO: Implement enhanced HTML feature extraction
            html_features = {
                'html_length': len(html_content),
                'has_ai_signatures': 1 if any(sig in html_content.lower() for sig in ['wix', 'squarespace', 'framer', 'vercel']) else 0,
                'css_complexity': html_content.count('class=') + html_content.count('style='),
                'script_count': html_content.count('<script'),
                'div_count': html_content.count('<div'),
                'semantic_tags': sum(1 for tag in ['header', 'nav', 'main', 'section', 'article', 'aside', 'footer'] if f'<{tag}' in html_content.lower())
            }
            features.update(html_features)
            print(f"✅ Extracted {len(html_features)} HTML features")
        
        # Make prediction
        print("🤖 Making prediction...")
        prediction = model_loader.predict(features)
        
        print(f"🎯 Final Result: {prediction['predicted_class'].upper()} (confidence: {prediction['confidence']:.3f})")
        
        return AnalysisResponse(
            is_ai_generated=prediction["is_ai_generated"],
            confidence=prediction["confidence"],
            predicted_class=prediction["predicted_class"],
            tool_probabilities=prediction["tool_probabilities"],
            features_used=prediction["features_used"],
            model_status=prediction["model_status"]
        )
        
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
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