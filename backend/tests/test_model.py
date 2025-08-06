#!/usr/bin/env python3
"""
Test script to verify the trained model is working
"""

import cv2
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from feature_extract import FeatureExtractor
from model_loader import ModelLoader

def test_model():
    print("🧪 Testing WebTrace AI Model")
    print("=" * 40)
    
    # Initialize components
    extractor = FeatureExtractor()
    model_loader = ModelLoader()
    
    # Load the trained model first
    print("📂 Loading trained model...")
    model_loaded = model_loader.load_model("model.pkl")
    
    if not model_loaded:
        print("❌ Failed to load model. Make sure you have trained a model first.")
        print("💡 Run: python train_simple_model.py")
        return
    
    print("✅ Model loaded successfully!")
    print()
    
    # Test with a sample AI image
    ai_image_path = "../dataset/images/ai/V0_001.png"
    if os.path.exists(ai_image_path):
        print(f"📸 Testing with AI image: {ai_image_path}")
        img = cv2.imread(ai_image_path)
        features = extractor.extract_image_features(img)
        prediction = model_loader.predict(features)
        
        print(f"   Prediction: {prediction['predicted_class']}")
        print(f"   Confidence: {prediction['confidence']:.3f}")
        print(f"   Model Status: {prediction['model_status']}")
        print()
    else:
        print(f"❌ AI test image not found: {ai_image_path}")
    
    # Test with a sample human image
    human_image_path = "../dataset/images/human/human_001.png"
    if os.path.exists(human_image_path):
        print(f"📸 Testing with human image: {human_image_path}")
        img = cv2.imread(human_image_path)
        features = extractor.extract_image_features(img)
        prediction = model_loader.predict(features)
        
        print(f"   Prediction: {prediction['predicted_class']}")
        print(f"   Confidence: {prediction['confidence']:.3f}")
        print(f"   Model Status: {prediction['model_status']}")
        print()
    else:
        print(f"❌ Human test image not found: {human_image_path}")
    
    # Show model info
    print("📊 Model Information:")
    model_info = model_loader.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    test_model() 