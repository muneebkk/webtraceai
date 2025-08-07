#!/usr/bin/env python3
"""
Debug script to test custom tree model with numpy arrays
"""

import numpy as np
import pandas as pd
from custom_tree_model import CustomTreeTrainer

def test_custom_tree():
    print("🔍 Testing Custom Tree with numpy arrays...")
    
    # Create synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 43)  # 100 samples, 43 features
    y = np.random.randint(0, 2, 100)  # Binary labels
    
    print(f"X shape: {X.shape}, type: {type(X)}")
    print(f"y shape: {y.shape}, type: {type(y)}")
    
    # Test with numpy arrays directly
    trainer = CustomTreeTrainer()
    
    try:
        model = trainer.train_model(
            X, y,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            criterion='gini',
            prune=True
        )
        print("✅ Custom tree training successful with numpy arrays!")
        
        # Test prediction
        y_pred = model.predict(X[:10])
        print(f"✅ Prediction successful: {y_pred}")
        
    except Exception as e:
        print(f"❌ Error with numpy arrays: {e}")
        import traceback
        traceback.print_exc()

def test_custom_tree_with_pandas():
    print("\n🔍 Testing Custom Tree with pandas DataFrames...")
    
    # Create synthetic data as pandas DataFrame
    np.random.seed(42)
    X_df = pd.DataFrame(np.random.rand(100, 43))
    y_series = pd.Series(np.random.randint(0, 2, 100))
    
    print(f"X_df shape: {X_df.shape}, type: {type(X_df)}")
    print(f"y_series shape: {y_series.shape}, type: {type(y_series)}")
    
    # Convert to numpy arrays
    X_array = X_df.values
    y_array = y_series.values
    
    print(f"X_array shape: {X_array.shape}, type: {type(X_array)}")
    print(f"y_array shape: {y_array.shape}, type: {type(y_array)}")
    
    # Test with converted arrays
    trainer = CustomTreeTrainer()
    
    try:
        model = trainer.train_model(
            X_array, y_array,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            criterion='gini',
            prune=True
        )
        print("✅ Custom tree training successful with converted arrays!")
        
        # Test prediction
        y_pred = model.predict(X_array[:10])
        print(f"✅ Prediction successful: {y_pred}")
        
    except Exception as e:
        print(f"❌ Error with converted arrays: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_custom_tree()
    test_custom_tree_with_pandas() 