# Muneeb: Model loading and prediction
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
        self.feature_mask = None  # For improved model
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
            self.feature_mask = None
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
                # Create new extractor if not saved in model
                if 'extractor' in model_data:
                    self.extractor = model_data['extractor']
                else:
                    from .feature_extract import FeatureExtractor
                    self.extractor = FeatureExtractor()
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
            elif 'model_type' in model_data and model_data['model_type'] == 'VotingClassifier':
                # Ensemble model format
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
                # Create new extractor if not saved in model
                if 'extractor' in model_data:
                    self.extractor = model_data['extractor']
                else:
                    from .feature_extract import FeatureExtractor
                    self.extractor = FeatureExtractor()
                self.scaler = model_data.get('scaler', None)
                self.best_score = model_data.get('best_score', 0)
                self.model_type = 'VotingClassifier'
                self.model_loaded = True
                
                # Get ensemble details
                base_models = model_data.get('base_models', [])
                voting_method = model_data.get('voting_method', 'soft')
                weights = model_data.get('weights', [])
                
                self.model_info.update({
                    "status": "loaded",
                    "model_type": self.model_type,
                    "feature_count": len(self.feature_names),
                    "features": self.feature_names,
                    "best_score": self.best_score,
                    "model_format": "ensemble",
                    "base_models": base_models,
                    "voting_method": voting_method,
                    "weights": weights,
                    "description": f"Ensemble model combining {', '.join(base_models)} with {voting_method} voting"
                })
            elif 'scaler' in model_data:
                # New improved model format
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
                # Create new extractor if not saved in model
                if 'extractor' in model_data:
                    self.extractor = model_data['extractor']
                else:
                    from .feature_extract import FeatureExtractor
                    self.extractor = FeatureExtractor()
                self.scaler = model_data['scaler']
                self.feature_mask = model_data.get('feature_mask', None)  # Load feature mask
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
                # Create new extractor if not saved in model
                if 'extractor' in model_data:
                    self.extractor = model_data['extractor']
                else:
                    from .feature_extract import FeatureExtractor
                    self.extractor = FeatureExtractor()
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
            
            # Import numpy at the beginning
            import numpy as np
            
            # Use the real trained model
            print(f"🤖 Making prediction with {self.model_type} model")
            
            # Use features directly for all models (43 features)
            features_array = [features.get(name, 0) for name in self.feature_names]
            
            # Apply feature mask if available (for improved model)
            if hasattr(self, 'feature_mask') and self.feature_mask is not None:
                features_array = np.array(features_array)[self.feature_mask]
                print("   Applied feature mask (excluded font features)")
            
            # Convert to numpy array and reshape for sklearn
            X = np.array(features_array).reshape(1, -1)
            
            # Scale features if scaler is available (for SVM, Logistic Regression)
            # Only apply scaling if the model actually requires it
            if self.scaler is not None and isinstance(self.model, (SVC, LogisticRegression)):
                X = self.scaler.transform(X)
                print("   Applied feature scaling")
            
            # Handle ensemble model prediction
            if self.model_type == 'VotingClassifier':
                print("   Using ensemble voting classifier prediction")
                # Ensemble models handle their own scaling internally
                if self.scaler is not None:
                    X = self.scaler.transform(X)
                    print("   Applied feature scaling for ensemble")
            
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
        # Enhanced heuristic using meaningful visual features only
        # Avoid using basic properties like width, height, aspect_ratio, total_pixels
        
        ai_score = 0
        
        # Use meaningful visual features for prediction
        # Color features
        color_diversity_s = features.get('color_diversity_s', 0)
        color_diversity_v = features.get('color_diversity_v', 0)
        color_uniformity = features.get('color_uniformity', 0)
        avg_saturation = features.get('avg_saturation', 0)
        color_entropy = features.get('color_entropy', 0)
        
        # Layout features
        edge_density = features.get('edge_density', 0)
        contour_count = features.get('contour_count', 0)
        layout_symmetry = features.get('layout_symmetry', 0)
        alignment_score = features.get('alignment_score', 0)
        white_space_ratio = features.get('white_space_ratio', 0)
        content_density = features.get('content_density', 0)
        
        # Texture features
        texture_uniformity = features.get('texture_uniformity', 0)
        texture_contrast = features.get('texture_contrast', 0)
        texture_entropy = features.get('texture_entropy', 0)
        
        # Structural features
        border_regularity = features.get('border_regularity', 0)
        padding_consistency = features.get('padding_consistency', 0)
        component_alignment = features.get('component_alignment', 0)
        design_patterns = features.get('design_patterns', 0)
        
        # AI-generated websites tend to have:
        # 1. More uniform color schemes
        if color_uniformity > 5.0:
            ai_score += 0.2
        if color_entropy < 3.0:
            ai_score += 0.15
            
        # 2. Higher alignment and consistency
        if alignment_score > 100:
            ai_score += 0.2
        if padding_consistency > 0.8:
            ai_score += 0.15
        if component_alignment > 100:
            ai_score += 0.15
            
        # 3. More regular patterns
        if design_patterns > 5:
            ai_score += 0.2
        if border_regularity > 0.5:
            ai_score += 0.1
            
        # 4. Higher content density (AI tools often fill space)
        if content_density > 0.8:
            ai_score += 0.15
            
        # Human-coded websites tend to have:
        # 1. More texture variation
        if texture_entropy > 4.0:
            ai_score -= 0.2
        if texture_contrast > 100:
            ai_score -= 0.15
            
        # 2. More white space (human designers use breathing room)
        if white_space_ratio > 0.3:
            ai_score -= 0.2
            
        # 3. Less perfect symmetry (more organic layouts)
        if layout_symmetry < 0.7:
            ai_score -= 0.15
            
        # 4. More color diversity
        if color_diversity_s > 50:
            ai_score -= 0.15
        if color_diversity_v > 50:
            ai_score -= 0.15
        
        # Determine prediction
        is_ai_generated = ai_score > 0.1
        confidence = min(0.95, max(0.5, abs(ai_score) + 0.5))
        
        print(f"🎲 Mock Prediction: {'AI-Generated' if is_ai_generated else 'Human-Coded'} (confidence: {confidence:.3f})")
        print(f"   AI Score: {ai_score:.3f} (based on meaningful visual features)")
        
        return {
            "is_ai_generated": is_ai_generated,
            "confidence": confidence,
            "predicted_class": "ai_generated" if is_ai_generated else "human_coded",
            "tool_probabilities": {
                "ai_generated": confidence if is_ai_generated else 1 - confidence,
                "human_coded": 1 - confidence if is_ai_generated else confidence
            },
            "features_used": [k for k in features.keys() if k not in ['width', 'height', 'aspect_ratio', 'total_pixels']],
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