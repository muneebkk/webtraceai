#!/usr/bin/env python3
"""
Test script for the custom decision tree model
"""

import os
import sys
import cv2
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import custom tree model
try:
    from custom_tree_model import CustomDecisionTree
except ImportError:
    CustomDecisionTree = None

from app.feature_extract import FeatureExtractor

def test_custom_model(model_path="custom_tree_model.pkl"):
    """Test the custom tree model on the dataset"""
    print("🧪 Testing Custom Decision Tree Model...")
    print("=" * 50)
    
    # Load the model
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        feature_names = model_data['feature_names']
        extractor = model_data['extractor']
        
        print(f"✅ Model loaded successfully from {model_path}")
        print(f"   Model type: {model_data['model_type']}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Tree depth: {model_data['training_info']['tree_depth']}")
        print(f"   Total nodes: {model_data['training_info']['total_nodes']}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test on AI images
    print(f"\n📁 Testing AI images...")
    ai_path = "../dataset/images/ai"
    ai_predictions = []
    ai_true_labels = []
    ai_files = []
    
    if os.path.exists(ai_path):
        for filename in os.listdir(ai_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(ai_path, filename)
                try:
                    img = cv2.imread(filepath)
                    if img is not None and img.size > 0:
                        features = extractor.extract_image_features(img)
                        
                        # Make prediction
                        features_array = [features.get(name, 0) for name in feature_names]
                        X = np.array(features_array).reshape(1, -1)
                        
                        prediction = model.predict(X)[0]
                        probability = model.predict_proba(X)[0]
                        
                        ai_predictions.append(prediction)
                        ai_true_labels.append(1)  # 1 for AI
                        ai_files.append(filename)
                        
                        confidence = max(probability)
                        status = "✅" if prediction == 1 else "❌"
                        pred_class = "ai_generated" if prediction == 1 else "human_coded"
                        
                        print(f"  {status} {filename}: {pred_class} (conf: {confidence:.3f})")
                        
                except Exception as e:
                    print(f"  ❌ Error processing {filename}: {e}")
    
    # Test on human images
    print(f"\n📁 Testing human images...")
    human_path = "../dataset/images/human"
    human_predictions = []
    human_true_labels = []
    human_files = []
    
    if os.path.exists(human_path):
        for filename in os.listdir(human_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(human_path, filename)
                try:
                    img = cv2.imread(filepath)
                    if img is not None and img.size > 0:
                        features = extractor.extract_image_features(img)
                        
                        # Make prediction
                        features_array = [features.get(name, 0) for name in feature_names]
                        X = np.array(features_array).reshape(1, -1)
                        
                        prediction = model.predict(X)[0]
                        probability = model.predict_proba(X)[0]
                        
                        human_predictions.append(prediction)
                        human_true_labels.append(0)  # 0 for human
                        human_files.append(filename)
                        
                        confidence = max(probability)
                        status = "✅" if prediction == 0 else "❌"
                        pred_class = "human_coded" if prediction == 0 else "ai_generated"
                        
                        print(f"  {status} {filename}: {pred_class} (conf: {confidence:.3f})")
                        
                except Exception as e:
                    print(f"  ❌ Error processing {filename}: {e}")
    
    # Combine results
    all_predictions = ai_predictions + human_predictions
    all_true_labels = ai_true_labels + human_true_labels
    all_files = ai_files + human_files
    
    # Calculate metrics
    accuracy = accuracy_score(all_true_labels, all_predictions)
    
    print(f"\n📊 Performance Summary:")
    print(f"  Total samples: {len(all_predictions)}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(all_true_labels, all_predictions, target_names=['Human', 'AI']))
    
    # Confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    print(f"\n📊 Confusion Matrix:")
    print(f"     Predicted")
    print(f"      Human  AI")
    print(f"Actual Human  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"      AI      {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Detailed analysis
    ai_detection_rate = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    human_detection_rate = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    
    print(f"\n🔍 Detailed Analysis:")
    print(f"  AI Detection Rate: {ai_detection_rate:.4f} ({cm[1,1]}/{cm[1,0]+cm[1,1]})")
    print(f"  Human Detection Rate: {human_detection_rate:.4f} ({cm[0,0]}/{cm[0,0]+cm[0,1]})")
    print(f"  False Positives (Human → AI): {false_positives}")
    print(f"  False Negatives (AI → Human): {false_negatives}")
    
    # List misclassified examples
    misclassified = []
    for i, (pred, true, filename) in enumerate(zip(all_predictions, all_true_labels, all_files)):
        if pred != true:
            true_class = "AI" if true == 1 else "Human"
            pred_class = "AI" if pred == 1 else "Human"
            misclassified.append(f"{pred_class}→{true_class}: {filename}")
    
    if misclassified:
        print(f"\n⚠️  Misclassified Examples:")
        for example in misclassified:
            print(f"  {example}")
    else:
        print(f"\n🎉 No misclassifications!")
    
    # Feature importance analysis
    feature_importance = model.get_feature_importance()
    print(f"\n🔍 Feature Importance Analysis:")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print(f"  Top 5 most important features:")
    for feature, importance in sorted_features[:5]:
        print(f"    {feature}: {importance:.4f}")
    
    # Tree structure analysis
    print(f"\n🌳 Tree Structure Analysis:")
    print(f"  Tree depth: {model_data['training_info']['tree_depth']}")
    print(f"  Total nodes: {model_data['training_info']['total_nodes']}")
    print(f"  Leaf nodes: {model_data['training_info']['leaf_nodes']}")
    
    return {
        'accuracy': accuracy,
        'ai_detection_rate': ai_detection_rate,
        'human_detection_rate': human_detection_rate,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'feature_importance': feature_importance,
        'misclassified': misclassified
    }

def compare_models():
    """Compare all available models"""
    print("🏆 Model Comparison")
    print("=" * 50)
    
    models_to_test = [
        ("Original Model", "model.pkl"),
        ("Improved Model", "improved_model.pkl"),
        ("Custom Tree Model", "custom_tree_model.pkl")
    ]
    
    results = {}
    
    for model_name, model_path in models_to_test:
        if os.path.exists(model_path):
            print(f"\n🧪 Testing {model_name}...")
            try:
                result = test_custom_model(model_path)
                if result:
                    results[model_name] = result
            except Exception as e:
                print(f"❌ Error testing {model_name}: {e}")
        else:
            print(f"⚠️  Model file not found: {model_path}")
    
    # Summary comparison
    if results:
        print(f"\n🏆 Model Comparison Summary:")
        print("=" * 50)
        print(f"{'Model':<20} {'Accuracy':<10} {'AI Detection':<12} {'Human Detection':<15}")
        print("-" * 60)
        
        for model_name, result in results.items():
            print(f"{model_name:<20} {result['accuracy']:<10.4f} {result['ai_detection_rate']:<12.4f} {result['human_detection_rate']:<15.4f}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

if __name__ == "__main__":
    # Test custom model only
    test_custom_model()
    
    # Uncomment to compare all models
    # compare_models() 