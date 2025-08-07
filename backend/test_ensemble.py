#!/usr/bin/env python3
"""
Test script to verify ensemble model compatibility
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from custom_tree_model import CustomDecisionTree

def test_ensemble_compatibility():
    print("🔍 Testing Ensemble Model Compatibility...")
    
    # Create synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 43)
    y = np.random.randint(0, 2, 100)
    
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Create the three models
    print("\n1️⃣ Creating Random Forest...")
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    print(f"   ✅ RF trained successfully")
    
    print("\n2️⃣ Creating Logistic Regression...")
    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)
    print(f"   ✅ LR trained successfully")
    
    print("\n3️⃣ Creating Custom Decision Tree...")
    dt = CustomDecisionTree(max_depth=5, random_state=42)
    dt.fit(X, y)
    print(f"   ✅ Custom DT trained successfully")
    
    # Test ensemble creation
    print("\n🎯 Creating Ensemble...")
    try:
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('lr', lr),
                ('dt', dt)
            ],
            voting='soft',
            weights=[0.4, 0.3, 0.3]
        )
        
        # Test ensemble training
        ensemble.fit(X, y)
        print(f"   ✅ Ensemble trained successfully")
        
        # Test ensemble prediction
        y_pred = ensemble.predict(X[:10])
        y_proba = ensemble.predict_proba(X[:10])
        print(f"   ✅ Ensemble predictions work")
        print(f"   📊 Predictions: {y_pred}")
        print(f"   📊 Probabilities shape: {y_proba.shape}")
        
        print("\n🎉 All ensemble tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Ensemble failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ensemble_compatibility()
    if success:
        print("\n✅ Ready to run the full training script!")
    else:
        print("\n❌ Need to fix ensemble issues first!") 