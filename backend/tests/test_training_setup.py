#!/usr/bin/env python3
"""
Quick test script to verify training setup works
Tests feature extraction, dataset loading, and basic model training
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.feature_extract import FeatureExtractor
from custom_tree_model import CustomDecisionTree

def test_feature_extraction():
    """Test that feature extraction works"""
    print("🔍 Testing feature extraction...")
    
    extractor = FeatureExtractor()
    feature_names = extractor.get_meaningful_feature_names()
    print(f"  ✅ Found {len(feature_names)} meaningful features")
    
    # Test with a sample image if available
    test_image_path = "../dataset/images/ai"
    if os.path.exists(test_image_path):
        for filename in os.listdir(test_image_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(test_image_path, filename)
                try:
                    import cv2
                    img = cv2.imread(filepath)
                    if img is not None:
                        features = extractor.extract_meaningful_features(img)
                        print(f"  ✅ Successfully extracted {len(features)} features from {filename}")
                        return True
                except Exception as e:
                    print(f"  ❌ Error extracting features from {filename}: {e}")
                    return False
    
    print("  ⚠️  No test images found, but feature extraction setup looks good")
    return True

def test_dataset_loading():
    """Test that dataset loading works"""
    print("\n📁 Testing dataset loading...")
    
    extractor = FeatureExtractor()
    dataset_path = "../dataset"
    ai_path = os.path.join(dataset_path, "images", "ai")
    human_path = os.path.join(dataset_path, "images", "human")
    
    features_list = []
    labels = []
    
    # Load a few samples from each class
    max_samples = 5
    
    if os.path.exists(ai_path):
        print(f"  📂 Loading up to {max_samples} AI samples...")
        count = 0
        for filename in os.listdir(ai_path):
            if count >= max_samples:
                break
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(ai_path, filename)
                try:
                    import cv2
                    img = cv2.imread(filepath)
                    if img is not None:
                        features = extractor.extract_meaningful_features(img)
                        features_list.append(features)
                        labels.append(1)  # AI
                        count += 1
                        print(f"    ✅ Loaded AI: {filename}")
                except Exception as e:
                    print(f"    ❌ Error loading {filename}: {e}")
    
    if os.path.exists(human_path):
        print(f"  📂 Loading up to {max_samples} human samples...")
        count = 0
        for filename in os.listdir(human_path):
            if count >= max_samples:
                break
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(human_path, filename)
                try:
                    import cv2
                    img = cv2.imread(filepath)
                    if img is not None:
                        features = extractor.extract_meaningful_features(img)
                        features_list.append(features)
                        labels.append(0)  # Human
                        count += 1
                        print(f"    ✅ Loaded Human: {filename}")
                except Exception as e:
                    print(f"    ❌ Error loading {filename}: {e}")
    
    if len(features_list) < 4:
        print("  ❌ Not enough samples for testing")
        return False, None, None
    
    df = pd.DataFrame(features_list)
    labels = np.array(labels)
    
    print(f"  ✅ Successfully loaded {len(df)} samples ({np.sum(labels == 1)} AI, {np.sum(labels == 0)} Human)")
    return True, df, labels

def test_basic_models(df, labels):
    """Test that basic model training works"""
    print("\n🤖 Testing basic model training...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df.values, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"  📊 Training set: {len(X_train)} samples")
    print(f"  📊 Test set: {len(X_test)} samples")
    
    # Test Random Forest
    print("  🌲 Testing Random Forest...")
    try:
        rf = RandomForestClassifier(n_estimators=10, random_state=42)  # Small for quick test
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"    ✅ Random Forest accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"    ❌ Random Forest failed: {e}")
        return False
    
    # Test Logistic Regression
    print("  📈 Testing Logistic Regression...")
    try:
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"    ✅ Logistic Regression accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"    ❌ Logistic Regression failed: {e}")
        return False
    
    # Test Custom Tree
    print("  🌳 Testing Custom Tree...")
    try:
        custom_tree = CustomDecisionTree(max_depth=3, random_state=42)
        custom_tree.fit(X_train, y_train)
        y_pred = custom_tree.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"    ✅ Custom Tree accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"    ❌ Custom Tree failed: {e}")
        return False
    
    return True

def test_model_saving():
    """Test that model saving works"""
    print("\n💾 Testing model saving...")
    
    try:
        # Create a simple test model
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        X_test = np.random.rand(10, 5)
        y_test = np.random.randint(0, 2, 10)
        rf.fit(X_test, y_test)
        
        # Test saving
        model_data = {
            'model': rf,
            'feature_names': ['test1', 'test2', 'test3', 'test4', 'test5'],
            'model_type': 'RandomForestClassifier',
            'test': True
        }
        
        with open('test_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        # Test loading
        with open('test_model.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        
        print("  ✅ Model saving and loading works")
        
        # Clean up
        os.remove('test_model.pkl')
        return True
        
    except Exception as e:
        print(f"  ❌ Model saving failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running training setup tests...")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Feature extraction
    if test_feature_extraction():
        tests_passed += 1
    
    # Test 2: Dataset loading
    success, df, labels = test_dataset_loading()
    if success:
        tests_passed += 1
    
    # Test 3: Basic model training
    if success and test_basic_models(df, labels):
        tests_passed += 1
    
    # Test 4: Model saving
    if test_model_saving():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Training script should work.")
        print("🚀 You can now run: python retrain_models.py")
    else:
        print("❌ Some tests failed. Please fix issues before running full training.")
        print("🔧 Check the error messages above for details.")

if __name__ == "__main__":
    main() 