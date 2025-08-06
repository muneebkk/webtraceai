#!/usr/bin/env python3
"""
Test script for the new visualization endpoint
"""

import requests
import json
import os

def test_visualization_endpoint():
    """Test the new /predict-visualization endpoint"""
    
    # Test data - simulate a simple image input
    # Note: This test requires an actual image file to be present
    test_image_path = "dataset/images/human/human_001.png"
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        print("Please ensure the test image exists before running this test.")
        return
    
    # Test all three models
    models = ["original", "improved", "custom_tree"]
    
    for model in models:
        print(f"\n🧪 Testing {model} model...")
        
        try:
            # Prepare form data with image file
            with open(test_image_path, 'rb') as f:
                files = {'screenshot': f}
                data = {'model': model}
            
            # Make request to visualization endpoint
            response = requests.post(
                "http://localhost:8000/api/predict-visualization",
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {model} model visualization successful!")
                print(f"   Prediction: {result['prediction']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Model Type: {result['model_type']}")
                print(f"   Features: {len(result['feature_values'])}")
                print(f"   Decision Steps: {len(result['decision_path'])}")
                print(f"   Top Features: {list(result['feature_importance'].keys())[:3]}")
            else:
                print(f"❌ {model} model failed with status {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ {model} model error: {str(e)}")
    
    print("\n🎉 Visualization endpoint test completed!")

if __name__ == "__main__":
    test_visualization_endpoint() 