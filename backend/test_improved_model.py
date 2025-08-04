#!/usr/bin/env python3
"""
Test script for the improved model
"""

import os
import sys
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.model_loader import ModelLoader
from app.feature_extract import FeatureExtractor

def test_model_performance(model_path, dataset_path="../dataset"):
    """Test model performance on the dataset"""
    print(f"🧪 Testing model: {model_path}")
    print("=" * 50)
    
    # Load model
    model_loader = ModelLoader()
    if not model_loader.load_model(model_path):
        print("❌ Failed to load model")
        return
    
    # Load dataset
    extractor = FeatureExtractor()
    ai_path = os.path.join(dataset_path, "images", "ai")
    human_path = os.path.join(dataset_path, "images", "human")
    
    predictions = []
    true_labels = []
    file_names = []
    
    # Test AI images
    if os.path.exists(ai_path):
        print(f"📁 Testing AI images...")
        for filename in os.listdir(ai_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(ai_path, filename)
                try:
                    img = cv2.imread(filepath)
                    if img is not None:
                        features = extractor.extract_image_features(img)
                        prediction = model_loader.predict(features)
                        
                        predictions.append(1 if prediction['is_ai_generated'] else 0)
                        true_labels.append(1)  # AI = 1
                        file_names.append(f"ai/{filename}")
                        
                        status = "✅" if prediction['is_ai_generated'] else "❌"
                        print(f"  {status} {filename}: {prediction['predicted_class']} (conf: {prediction['confidence']:.3f})")
                except Exception as e:
                    print(f"  ❌ Error testing {filename}: {e}")
    
    # Test human images
    if os.path.exists(human_path):
        print(f"\n📁 Testing human images...")
        for filename in os.listdir(human_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(human_path, filename)
                try:
                    img = cv2.imread(filepath)
                    if img is not None:
                        features = extractor.extract_image_features(img)
                        prediction = model_loader.predict(features)
                        
                        predictions.append(1 if prediction['is_ai_generated'] else 0)
                        true_labels.append(0)  # Human = 0
                        file_names.append(f"human/{filename}")
                        
                        status = "✅" if not prediction['is_ai_generated'] else "❌"
                        print(f"  {status} {filename}: {prediction['predicted_class']} (conf: {prediction['confidence']:.3f})")
                except Exception as e:
                    print(f"  ❌ Error testing {filename}: {e}")
    
    if not predictions:
        print("❌ No test images found!")
        return
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f"\n📊 Performance Summary:")
    print(f"  Total samples: {len(predictions)}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(true_labels, predictions, target_names=['Human', 'AI']))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print(f"\n📊 Confusion Matrix:")
    print("     Predicted")
    print("      Human  AI")
    print(f"Actual Human  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"      AI      {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Detailed analysis
    print(f"\n🔍 Detailed Analysis:")
    
    # AI detection rate
    ai_correct = sum(1 for i, (pred, true) in enumerate(zip(predictions, true_labels)) 
                    if true == 1 and pred == 1)
    ai_total = sum(1 for label in true_labels if label == 1)
    ai_detection_rate = ai_correct / ai_total if ai_total > 0 else 0
    
    # Human detection rate
    human_correct = sum(1 for i, (pred, true) in enumerate(zip(predictions, true_labels)) 
                       if true == 0 and pred == 0)
    human_total = sum(1 for label in true_labels if label == 0)
    human_detection_rate = human_correct / human_total if human_total > 0 else 0
    
    print(f"  AI Detection Rate: {ai_detection_rate:.4f} ({ai_correct}/{ai_total})")
    print(f"  Human Detection Rate: {human_detection_rate:.4f} ({human_correct}/{human_total})")
    
    # False positives/negatives
    false_positives = sum(1 for i, (pred, true) in enumerate(zip(predictions, true_labels)) 
                         if true == 0 and pred == 1)
    false_negatives = sum(1 for i, (pred, true) in enumerate(zip(predictions, true_labels)) 
                         if true == 1 and pred == 0)
    
    print(f"  False Positives (Human → AI): {false_positives}")
    print(f"  False Negatives (AI → Human): {false_negatives}")
    
    # Show misclassified examples
    if false_positives > 0 or false_negatives > 0:
        print(f"\n⚠️  Misclassified Examples:")
        for i, (pred, true, filename) in enumerate(zip(predictions, true_labels, file_names)):
            if pred != true:
                error_type = "FP" if true == 0 and pred == 1 else "FN"
                print(f"  {error_type}: {filename}")
    
    return accuracy, predictions, true_labels

def compare_models():
    """Compare original and improved models"""
    print("🔄 Model Comparison")
    print("=" * 50)
    
    models_to_test = []
    
    # Check which models exist
    if os.path.exists("model.pkl"):
        models_to_test.append(("Original Model", "model.pkl"))
    
    if os.path.exists("improved_model.pkl"):
        models_to_test.append(("Improved Model", "improved_model.pkl"))
    
    if not models_to_test:
        print("❌ No models found to compare!")
        print("💡 Train models first:")
        print("   python train_simple_model.py")
        print("   python improve_model.py")
        return
    
    results = {}
    
    for model_name, model_path in models_to_test:
        print(f"\n🧪 Testing {model_name}...")
        accuracy, predictions, true_labels = test_model_performance(model_path)
        if accuracy is not None:
            results[model_name] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'true_labels': true_labels
            }
    
    # Compare results
    if len(results) > 1:
        print(f"\n🏆 Model Comparison Summary:")
        print("=" * 50)
        
        for model_name, result in results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            
            # Calculate additional metrics
            predictions = result['predictions']
            true_labels = result['true_labels']
            
            ai_correct = sum(1 for i, (pred, true) in enumerate(zip(predictions, true_labels)) 
                           if true == 1 and pred == 1)
            ai_total = sum(1 for label in true_labels if label == 1)
            ai_detection_rate = ai_correct / ai_total if ai_total > 0 else 0
            
            human_correct = sum(1 for i, (pred, true) in enumerate(zip(predictions, true_labels)) 
                              if true == 0 and pred == 0)
            human_total = sum(1 for label in true_labels if label == 0)
            human_detection_rate = human_correct / human_total if human_total > 0 else 0
            
            print(f"  AI Detection Rate: {ai_detection_rate:.4f}")
            print(f"  Human Detection Rate: {human_detection_rate:.4f}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

def main():
    print("🧪 WebTrace AI - Model Testing Tool")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # Test specific model
        model_path = sys.argv[1]
        test_model_performance(model_path)
    else:
        # Compare all available models
        compare_models()

if __name__ == "__main__":
    main() 