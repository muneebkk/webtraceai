#!/usr/bin/env python3
"""
Simple training script for WebTrace AI
Use this once you have collected some dataset images
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import cv2

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.feature_extract import FeatureExtractor

class SimpleModelTrainer:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = self.extractor.get_feature_names()
    
    def load_dataset(self, dataset_path="../dataset"):
        """Load dataset from the dataset folder"""
        ai_path = os.path.join(dataset_path, "images", "ai")
        human_path = os.path.join(dataset_path, "images", "human")
        
        features_list = []
        labels = []
        
        # Load AI-generated images
        if os.path.exists(ai_path):
            print(f"📁 Loading AI images from: {ai_path}")
            for filename in os.listdir(ai_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepath = os.path.join(ai_path, filename)
                    try:
                        img = cv2.imread(filepath)
                        if img is not None:
                            features = self.extractor.extract_image_features(img)
                            features_list.append(features)
                            labels.append(1)  # 1 for AI-generated
                            print(f"  ✅ Loaded: {filename}")
                    except Exception as e:
                        print(f"  ❌ Error loading {filename}: {e}")
        
        # Load human-coded images
        if os.path.exists(human_path):
            print(f"📁 Loading human images from: {human_path}")
            for filename in os.listdir(human_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepath = os.path.join(human_path, filename)
                    try:
                        img = cv2.imread(filepath)
                        if img is not None:
                            features = self.extractor.extract_image_features(img)
                            features_list.append(features)
                            labels.append(0)  # 0 for human-coded
                            print(f"  ✅ Loaded: {filename}")
                    except Exception as e:
                        print(f"  ❌ Error loading {filename}: {e}")
        
        if not features_list:
            print("❌ No images found in dataset!")
            return None, None
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        labels = np.array(labels)
        
        print(f"\n📊 Dataset Summary:")
        print(f"  Total samples: {len(df)}")
        print(f"  AI-generated: {np.sum(labels == 1)}")
        print(f"  Human-coded: {np.sum(labels == 0)}")
        print(f"  Features: {len(self.feature_names)}")
        
        return df, labels
    
    def train_model(self, X, y):
        """Train the model"""
        print("\n🤖 Training Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📈 Model Performance:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Accuracy: {accuracy:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:25}: {row['importance']:.4f}")
        
        return accuracy
    
    def save_model(self, model_path="model.pkl"):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'extractor': self.extractor
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Model saved to: {model_path}")
    
    def predict_sample(self, image_path):
        """Test prediction on a sample image"""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return None
        
        features = self.extractor.extract_image_features(img)
        features_array = np.array([features[name] for name in self.feature_names])
        
        prediction = self.model.predict([features_array])[0]
        probability = self.model.predict_proba([features_array])[0]
        
        result = {
            'prediction': 'AI-generated' if prediction == 1 else 'Human-coded',
            'confidence': max(probability),
            'probabilities': {
                'human_coded': probability[0],
                'ai_generated': probability[1]
            }
        }
        
        return result

def main():
    print("🚀 WebTrace AI - Simple Model Trainer")
    print("=" * 50)
    
    trainer = SimpleModelTrainer()
    
    # Load dataset
    print("📂 Loading dataset...")
    X, y = trainer.load_dataset()
    
    if X is None:
        print("\n❌ No dataset found!")
        print("💡 Please add some images to:")
        print("   - dataset/images/ai/ (AI-generated websites)")
        print("   - dataset/images/human/ (Human-coded websites)")
        return
    
    # Train model
    accuracy = trainer.train_model(X, y)
    
    if accuracy > 0.6:  # Only save if accuracy is reasonable
        trainer.save_model()
        
        # Test with a sample if provided
        if len(sys.argv) > 1:
            test_image = sys.argv[1]
            print(f"\n🧪 Testing with: {test_image}")
            result = trainer.predict_sample(test_image)
            if result:
                print(f"  Prediction: {result['prediction']}")
                print(f"  Confidence: {result['confidence']:.4f}")
                print(f"  Probabilities: {result['probabilities']}")
    else:
        print("\n⚠️  Model accuracy is low. Consider:")
        print("   - Adding more training data")
        print("   - Improving feature extraction")
        print("   - Checking data quality")

if __name__ == "__main__":
    main() 