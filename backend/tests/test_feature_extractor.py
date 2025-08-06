#!/usr/bin/env python3
"""
Test script for the feature extractor
Run this to test feature extraction on any image
"""

import cv2
import numpy as np
from PIL import Image
import io
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from feature_extract import FeatureExtractor

def create_test_image():
    """Create a simple test image for testing"""
    # Create a 400x300 test image with some patterns
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add some rectangles (simulating website elements)
    cv2.rectangle(img, (50, 50), (150, 100), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(img, (200, 50), (350, 100), (0, 255, 0), -1)  # Green rectangle
    cv2.rectangle(img, (50, 150), (350, 250), (0, 0, 255), -1)  # Red rectangle
    
    # Add some text-like patterns
    for i in range(0, 400, 20):
        cv2.line(img, (i, 0), (i, 300), (128, 128, 128), 1)
    
    return img

def test_with_sample_image():
    """Test feature extraction with a sample image"""
    print("🧪 Testing Feature Extractor")
    print("=" * 50)
    
    # Create test image
    test_img = create_test_image()
    
    # Convert to PIL Image (simulating upload)
    pil_img = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    
    # Convert to BytesIO (simulating file upload)
    img_buffer = io.BytesIO()
    pil_img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract features
    print("📊 Extracting features...")
    features = extractor.extract_image_features(img_buffer)
    
    # Display results
    print("\n📈 Extracted Features:")
    print("-" * 30)
    
    feature_names = extractor.get_feature_names()
    for name in feature_names:
        if name in features:
            value = features[name]
            print(f"{name:25}: {value:10.4f}")
    
    print(f"\n✅ Total features extracted: {len(features)}")
    
    # Test with numpy array directly
    print("\n🔄 Testing with numpy array...")
    features_np = extractor.extract_image_features(test_img)
    print(f"✅ Features from numpy array: {len(features_np)}")
    
    # Test with PIL Image directly
    print("\n🔄 Testing with PIL Image...")
    features_pil = extractor.extract_image_features(pil_img)
    print(f"✅ Features from PIL Image: {len(features_pil)}")
    
    return features

def test_with_real_image(image_path):
    """Test feature extraction with a real image file"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    print(f"🧪 Testing with real image: {image_path}")
    print("=" * 50)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        return None
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract features
    print("📊 Extracting features...")
    features = extractor.extract_image_features(img)
    
    # Display results
    print("\n📈 Extracted Features:")
    print("-" * 30)
    
    feature_names = extractor.get_feature_names()
    for name in feature_names:
        if name in features:
            value = features[name]
            print(f"{name:25}: {value:10.4f}")
    
    print(f"\n✅ Total features extracted: {len(features)}")
    
    return features

def main():
    print("🚀 WebTrace AI - Feature Extractor Testing")
    print("=" * 60)
    
    # Test with sample image
    test_with_sample_image()
    
    # Test with real image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_with_real_image(image_path)
    else:
        print("\n💡 To test with a real image, run:")
        print(f"   python test_feature_extractor.py <image_path>")

if __name__ == "__main__":
    main() 