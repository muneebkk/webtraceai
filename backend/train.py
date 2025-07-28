# Muneeb: Main training script
# Focus: Train classifiers using scikit-learn (KNN, Decision Tree, etc.)
# Evaluate and save trained models using joblib

#!/usr/bin/env python3
"""
Training script for WebTrace AI model
Run this from the backend directory: python train.py
"""

import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.train_model import ModelTrainer

def main():
    print("🚀 WebTrace AI - Model Training")
    print("=" * 40)
    
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate()
    
    if results:
        print("\n✅ Training completed successfully!")
        print(f"📊 Total samples used: {results['n_samples']}")
        print(f"🎯 Model accuracy: {results['accuracy']:.3f}")
        print(f"📈 Cross-validation mean: {results['cv_mean']:.3f}")
        print(f"📁 Model saved to: models/webtrace_model.joblib")
        print(f"📁 Scaler saved to: models/webtrace_scaler.joblib")
        print(f"📊 Confusion matrix saved to: confusion_matrix.png")
        print(f"📊 Feature importance saved to: feature_importance.png")
    else:
        print("\n❌ Training failed! Check your dataset.")

if __name__ == "__main__":
    main() 