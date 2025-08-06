#!/usr/bin/env python3
"""
Simple test for the custom decision tree model
"""

import os
import sys
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.feature_extract import FeatureExtractor
from custom_tree_model import CustomTreeTrainer

def test_custom_model():
    """Test the custom tree model by training and evaluating it"""
    print("🧪 Testing Custom Decision Tree Model...")
    print("=" * 50)
    
    # Create trainer and load dataset
    trainer = CustomTreeTrainer()
    df, labels, file_df = trainer.load_dataset()
    
    if df is None:
        print("❌ Failed to load dataset")
        return
    
    # Train model
    model = trainer.train_model(
        df.values, labels,
        max_depth=4,
        min_samples_split=8,
        min_samples_leaf=4,
        criterion='gini',
        prune=True
    )
    
    # Evaluate model
    results = trainer.evaluate_model(df.values, labels)
    
    # Test on individual samples
    print(f"\n🔍 Individual Sample Testing:")
    
    # Test AI samples
    ai_path = "../dataset/images/ai"
    ai_correct = 0
    ai_total = 0
    
    if os.path.exists(ai_path):
        print(f"\n📁 Testing AI images...")
        for filename in os.listdir(ai_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(ai_path, filename)
                try:
                    img = cv2.imread(filepath)
                    if img is not None and img.size > 0:
                        features = trainer.extractor.extract_image_features(img)
                        
                        # Make prediction
                        features_array = [features.get(name, 0) for name in trainer.feature_names]
                        X = np.array(features_array).reshape(1, -1)
                        
                        prediction = model.predict(X)[0]
                        probability = model.predict_proba(X)[0]
                        
                        ai_total += 1
                        if prediction == 1:  # Correctly predicted as AI
                            ai_correct += 1
                            status = "✅"
                        else:
                            status = "❌"
                        
                        confidence = max(probability)
                        pred_class = "ai_generated" if prediction == 1 else "human_coded"
                        
                        print(f"  {status} {filename}: {pred_class} (conf: {confidence:.3f})")
                        
                except Exception as e:
                    print(f"  ❌ Error processing {filename}: {e}")
    
    # Test human samples
    human_path = "../dataset/images/human"
    human_correct = 0
    human_total = 0
    
    if os.path.exists(human_path):
        print(f"\n📁 Testing human images...")
        for filename in os.listdir(human_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(human_path, filename)
                try:
                    img = cv2.imread(filepath)
                    if img is not None and img.size > 0:
                        features = trainer.extractor.extract_image_features(img)
                        
                        # Make prediction
                        features_array = [features.get(name, 0) for name in trainer.feature_names]
                        X = np.array(features_array).reshape(1, -1)
                        
                        prediction = model.predict(X)[0]
                        probability = model.predict_proba(X)[0]
                        
                        human_total += 1
                        if prediction == 0:  # Correctly predicted as human
                            human_correct += 1
                            status = "✅"
                        else:
                            status = "❌"
                        
                        confidence = max(probability)
                        pred_class = "human_coded" if prediction == 0 else "ai_generated"
                        
                        print(f"  {status} {filename}: {pred_class} (conf: {confidence:.3f})")
                        
                except Exception as e:
                    print(f"  ❌ Error processing {filename}: {e}")
    
    # Summary
    print(f"\n📊 Final Results:")
    print(f"  AI Detection: {ai_correct}/{ai_total} ({ai_correct/ai_total:.4f})")
    print(f"  Human Detection: {human_correct}/{human_total} ({human_correct/human_total:.4f})")
    print(f"  Overall Accuracy: {(ai_correct + human_correct)/(ai_total + human_total):.4f}")
    
    # Tree analysis
    print(f"\n🌳 Tree Analysis:")
    print(f"  Tree depth: {trainer._get_tree_depth(model.root)}")
    print(f"  Total nodes: {trainer._count_nodes(model.root)}")
    print(f"  Leaf nodes: {trainer._count_leaf_nodes(model.root)}")
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    print(f"\n🔍 Top 5 Feature Importance:")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:5]:
        print(f"  {feature}: {importance:.4f}")
    
    # Print tree structure (optional)
    print(f"\n🌳 Tree Structure:")
    model.print_tree()

if __name__ == "__main__":
    test_custom_model() 