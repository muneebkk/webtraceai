# Hassan Hadi: Model loading and prediction
# Focus: Load trained model and make predictions on new data

from typing import Dict, Any
import json
import pickle
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Import custom tree model
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from custom_tree_model import CustomDecisionTree, TreeNode
except ImportError:
    CustomDecisionTree = None
    TreeNode = None

class ModelLoader:
    def __init__(self):
        self.model_loaded = False
        self.model = None
        self.feature_names = None
        self.extractor = None
        self.scaler = None
        self.best_score = 0
        self.model_type = None
        self.model_info = {
            "name": "WebTrace AI Model",
            "version": "1.0.0",
            "description": "AI-generated website detection model",
            "status": "not_loaded"
        }
        
        # Don't automatically load a model - wait for explicit load_model() call
    
    def load_model(self, model_path="model.pkl"):
        """Load the trained model from pickle file"""
        try:
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                print("💡 Train a model first using: python train_simple_model.py")
                return False
            
            # Reset model state to ensure clean loading
            self.model_loaded = False
            self.model = None
            self.feature_names = None
            self.extractor = None
            self.scaler = None
            self.best_score = 0
            self.model_type = None
            
            # Add current directory to Python path to handle import issues
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            # Add parent directory to path for custom tree model imports
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Load the trained model
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Handle different model formats
            if 'model_type' in model_data and model_data['model_type'] == 'CustomDecisionTree':
                # Custom tree model format
                self.model = self._reconstruct_custom_tree(model_data)
                self.feature_names = model_data['feature_names']
                self.extractor = model_data['extractor']
                self.scaler = model_data['scaler']
                self.best_score = 0
                self.model_type = 'CustomDecisionTree'
                self.model_loaded = True
                
                self.model_info.update({
                    "status": "loaded",
                    "model_type": self.model_type,
                    "feature_count": len(self.feature_names),
                    "features": self.feature_names,
                    "best_score": self.best_score,
                    "model_format": "custom_tree"
                })
            elif 'scaler' in model_data:
                # New improved model format
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
                self.extractor = model_data['extractor']
                self.scaler = model_data['scaler']
                self.best_score = model_data.get('best_score', 0)
                self.model_type = model_data.get('model_type', type(self.model).__name__)
                self.model_loaded = True
                
                self.model_info.update({
                    "status": "loaded",
                    "model_type": self.model_type,
                    "feature_count": len(self.feature_names),
                    "features": self.feature_names,
                    "best_score": self.best_score,
                    "model_format": "improved"
                })
            else:
                # Old model format
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
                self.extractor = model_data['extractor']
                self.scaler = None
                self.best_score = 0
                self.model_type = type(self.model).__name__
                self.model_loaded = True
                
                self.model_info.update({
                    "status": "loaded",
                    "model_type": self.model_type,
                    "feature_count": len(self.feature_names),
                    "features": self.feature_names,
                    "model_format": "legacy"
                })
            
            print(f"✅ Model loaded successfully from {model_path}")
            print(f"   Model type: {self.model_type}")
            print(f"   Features: {len(self.feature_names)}")
            if self.best_score > 0:
                print(f"   Best CV score: {self.best_score:.4f}")
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
            print(f"🤖 Making prediction with {self.model_type} model")
            features_array = [features.get(name, 0) for name in self.feature_names]
            
            # Convert to numpy array and reshape for sklearn
            import numpy as np
            X = np.array(features_array).reshape(1, -1)
            
            # Scale features if scaler is available (for SVM, Logistic Regression)
            # Only apply scaling if the model actually requires it
            if self.scaler is not None and isinstance(self.model, (SVC, LogisticRegression)):
                X = self.scaler.transform(X)
                print("   Applied feature scaling")
            
            # Custom tree model doesn't need scaling
            if CustomDecisionTree and isinstance(self.model, CustomDecisionTree):
                # Custom tree handles its own prediction logic
                print("   Using custom decision tree prediction")
            
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
    
    def _reconstruct_custom_tree(self, model_data):
        """Reconstruct custom tree model from serialized data"""
        if CustomDecisionTree is None:
            raise ImportError("CustomDecisionTree not available")
        
        # Create new custom tree with saved parameters
        params = model_data['model_params']
        model = CustomDecisionTree(
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            criterion=params['criterion'],
            prune=params['prune']
        )
        
        # Set the model attributes
        model.n_features = params['n_features']
        model.classes = np.array(params['classes'])
        model.n_classes = params['n_classes']
        model.feature_names = model_data['feature_names']
        
        # Reconstruct the tree structure
        model.root = self._reconstruct_tree_node(model_data['tree_structure'])
        
        return model
    
    def _reconstruct_tree_node(self, node_data):
        """Reconstruct tree node from serialized data"""
        if node_data is None:
            return None
        
        if node_data['is_leaf']:
            # Create leaf node
            node = TreeNode(
                is_leaf=True,
                prediction=node_data['prediction'],
                samples=node_data['samples'],
                depth=node_data['depth']
            )
        else:
            # Create internal node
            node = TreeNode(
                feature_idx=node_data['feature_idx'],
                threshold=node_data['threshold'],
                samples=node_data['samples'],
                depth=node_data['depth']
            )
            node.feature_name = node_data['feature_name']
            node.left = self._reconstruct_tree_node(node_data['left'])
            node.right = self._reconstruct_tree_node(node_data['right'])
        
        return node
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return self.model_info 