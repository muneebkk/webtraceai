# Hassan Hadi: Model loading and prediction
# Focus: Load trained model and make predictions on new data

from typing import Dict, Any
import json
import pickle
import os

class ModelLoader:
    def __init__(self):
        self.model_loaded = False
        self.model = None
        self.feature_names = None
        self.extractor = None
        self.model_info = {
            "name": "WebTrace AI Model",
            "version": "1.0.0",
            "description": "AI-generated website detection model",
            "status": "not_loaded"
        }
        
        # Try to load model automatically
        self.load_model()
    
    def load_model(self, model_path="model.pkl"):
        """Load the trained model from pickle file"""
        try:
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                print("💡 Train a model first using: python train_simple_model.py")
                return False
            
            # Add current directory to Python path to handle import issues
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            # Load the trained model
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.extractor = model_data['extractor']
            self.model_loaded = True
            
            self.model_info.update({
                "status": "loaded",
                "model_type": type(self.model).__name__,
                "feature_count": len(self.feature_names),
                "features": self.feature_names
            })
            
            print(f"✅ Model loaded successfully from {model_path}")
            print(f"   Model type: {type(self.model).__name__}")
            print(f"   Features: {len(self.feature_names)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, features: Dict) -> Dict[str, Any]:
        """
        Make prediction based on extracted features
        Uses the trained model if available, otherwise falls back to mock
        """
        try:
            if not self.model_loaded or self.model is None:
                # Fallback to mock predictions if no model loaded
                print("⚠️  Using mock predictions (no trained model available)")
                return self._mock_predict(features)
            
            # Use the real trained model
            features_array = [features.get(name, 0) for name in self.feature_names]
            
            # Convert to numpy array and reshape for sklearn
            import numpy as np
            X = np.array(features_array).reshape(1, -1)
            
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            is_ai_generated = bool(prediction)
            confidence = max(probability)
            
            print(f"🤖 Model Prediction: {'AI-Generated' if is_ai_generated else 'Human-Coded'} (confidence: {confidence:.3f})")
            
            return {
                "is_ai_generated": is_ai_generated,
                "confidence": confidence,
                "predicted_class": "ai_generated" if is_ai_generated else "human_coded",
                "tool_probabilities": {
                    "ai_generated": probability[1],
                    "human_coded": probability[0]
                },
                "features_used": self.feature_names,
                "model_status": "trained_model"
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self._mock_predict(features)
    
    def _mock_predict(self, features: Dict) -> Dict[str, Any]:
        """Mock prediction logic for when no trained model is available"""
        # Enhanced heuristic using both visual and HTML features
        width = features.get('width', 800)
        height = features.get('height', 600)
        aspect_ratio = features.get('aspect_ratio', 1.33)
        
        # HTML-based indicators
        has_ai_signatures = features.get('has_ai_signatures', 0)
        html_length = features.get('html_length', 0)
        css_complexity = features.get('css_complexity', 0)
        
        # Enhanced prediction logic
        ai_score = 0
        
        # Visual indicators
        if aspect_ratio > 1.5:  # Wide screenshots might be AI-generated
            ai_score += 0.3
        if width < 500 or height < 300:  # Small images might be human-coded
            ai_score -= 0.2
        
        # HTML indicators
        if has_ai_signatures:
            ai_score += 0.4  # Strong indicator
        if html_length > 5000:  # Long HTML might be AI-generated
            ai_score += 0.1
        if css_complexity > 20:  # High CSS complexity might be AI
            ai_score += 0.1
        
        # Determine prediction
        is_ai_generated = ai_score > 0.2
        confidence = min(0.95, max(0.5, abs(ai_score) + 0.5))
        
        print(f"🎲 Mock Prediction: {'AI-Generated' if is_ai_generated else 'Human-Coded'} (confidence: {confidence:.3f})")
        
        return {
            "is_ai_generated": is_ai_generated,
            "confidence": confidence,
            "predicted_class": "ai_generated" if is_ai_generated else "human_coded",
            "tool_probabilities": {
                "ai_generated": confidence if is_ai_generated else 1 - confidence,
                "human_coded": 1 - confidence if is_ai_generated else confidence
            },
            "features_used": list(features.keys()),
            "model_status": "mock_model"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return self.model_info 