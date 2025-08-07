#!/usr/bin/env python3
"""
Fix Model Loading Issues for WebTrace AI
Repairs and validates all trained models
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import cv2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.feature_extract import FeatureExtractor
from custom_tree_model import CustomDecisionTree

class ModelFixer:
    """Fix and validate model loading issues"""
    
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.feature_names = self.extractor.get_feature_names()
        
    def fix_main_model(self):
        """Fix the main Random Forest model"""
        print("🔧 Fixing main model (Random Forest)...")
        
        if not os.path.exists("model.pkl"):
            print("  ❌ model.pkl not found")
            return False
        
        try:
            # Load the original model
            with open("model.pkl", 'rb') as f:
                model_data = pickle.load(f)
            
            # Create fixed model data
            fixed_model_data = {
                'model': model_data['model'],
                'feature_names': self.feature_names,
                'extractor': self.extractor,
                'scaler': model_data.get('scaler', None),
                'model_type': 'RandomForest'
            }
            
            # Save fixed model
            with open("model_fixed.pkl", 'wb') as f:
                pickle.dump(fixed_model_data, f)
            
            print("  ✅ Main model fixed and saved as model_fixed.pkl")
            return True
            
        except Exception as e:
            print(f"  ❌ Error fixing main model: {e}")
            return False
    
    def fix_improved_model(self):
        """Fix the improved Logistic Regression model"""
        print("🔧 Fixing improved model (Logistic Regression)...")
        
        if not os.path.exists("improved_model.pkl"):
            print("  ❌ improved_model.pkl not found")
            return False
        
        try:
            # Load the original model
            with open("improved_model.pkl", 'rb') as f:
                model_data = pickle.load(f)
            
            # Create fixed model data
            fixed_model_data = {
                'model': model_data['model'],
                'feature_names': self.feature_names,
                'extractor': self.extractor,
                'scaler': model_data['scaler'],
                'feature_mask': model_data.get('feature_mask', None),
                'model_type': 'LogisticRegression'
            }
            
            # Save fixed model
            with open("improved_model_fixed.pkl", 'wb') as f:
                pickle.dump(fixed_model_data, f)
            
            print("  ✅ Improved model fixed and saved as improved_model_fixed.pkl")
            return True
            
        except Exception as e:
            print(f"  ❌ Error fixing improved model: {e}")
            return False
    
    def test_fixed_models(self):
        """Test the fixed models"""
        print("\n🧪 Testing fixed models...")
        
        # Load dataset
        df, labels = self.load_dataset()
        if df is None:
            print("❌ Failed to load dataset")
            return
        
        results = {}
        
        # Test fixed main model
        if os.path.exists("model_fixed.pkl"):
            try:
                with open("model_fixed.pkl", 'rb') as f:
                    model_data = pickle.load(f)
                
                results['main_model_fixed'] = self.test_single_model(
                    'main_model_fixed', model_data, df.values, labels
                )
            except Exception as e:
                print(f"  ❌ Error testing fixed main model: {e}")
        
        # Test fixed improved model
        if os.path.exists("improved_model_fixed.pkl"):
            try:
                with open("improved_model_fixed.pkl", 'rb') as f:
                    model_data = pickle.load(f)
                
                results['improved_model_fixed'] = self.test_single_model(
                    'improved_model_fixed', model_data, df.values, labels
                )
            except Exception as e:
                print(f"  ❌ Error testing fixed improved model: {e}")
        
        # Test original custom tree model
        if os.path.exists("custom_tree_model.pkl"):
            try:
                with open("custom_tree_model.pkl", 'rb') as f:
                    model_data = pickle.load(f)
                
                # Reconstruct custom tree
                model = self._reconstruct_custom_tree(model_data)
                model_data_fixed = {
                    'model': model,
                    'feature_names': self.feature_names,
                    'extractor': self.extractor,
                    'scaler': None,
                    'type': 'CustomDecisionTree'
                }
                
                results['custom_tree_model'] = self.test_single_model(
                    'custom_tree_model', model_data_fixed, df.values, labels
                )
            except Exception as e:
                print(f"  ❌ Error testing custom tree model: {e}")
        
        # Compare results
        self.compare_results(results)
    
    def load_dataset(self, dataset_path="../dataset"):
        """Load dataset for testing"""
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
                        if img is not None and img.size > 0:
                            features = self.extractor.extract_image_features(img)
                            features_list.append(features)
                            labels.append(1)  # 1 for AI-generated
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
                        if img is not None and img.size > 0:
                            features = self.extractor.extract_image_features(img)
                            features_list.append(features)
                            labels.append(0)  # 0 for human-coded
                    except Exception as e:
                        print(f"  ❌ Error loading {filename}: {e}")
        
        if not features_list:
            print("❌ No valid images found in dataset!")
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
    
    def test_single_model(self, model_name: str, model_data: Dict, X: np.ndarray, y: np.ndarray) -> Dict:
        """Test a single model and return results"""
        print(f"\n🧪 Testing {model_name} ({model_data.get('type', model_data.get('model_type', 'Unknown'))})...")
        
        model = model_data['model']
        scaler = model_data.get('scaler', None)
        feature_mask = model_data.get('feature_mask', None)
        
        # Prepare features
        X_test = X.copy()
        
        # Apply feature mask if available
        if feature_mask is not None:
            X_test = X_test[:, feature_mask]
            print(f"  Applied feature mask: {X_test.shape[1]} features")
        
        # Apply scaling if available
        if scaler is not None:
            X_test = scaler.transform(X_test)
            print(f"  Applied feature scaling")
        
        # Make predictions
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            
            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            
            # Calculate per-class metrics
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Human class metrics
            human_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
            human_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
            human_f1 = 2 * (human_precision * human_recall) / (human_precision + human_recall) if (human_precision + human_recall) > 0 else 0
            
            results = {
                'model_name': model_name,
                'model_type': model_data.get('type', model_data.get('model_type', 'Unknown')),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'human_precision': human_precision,
                'human_recall': human_recall,
                'human_f1': human_f1,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  ✅ Accuracy: {accuracy:.4f}")
            print(f"  🎯 AI Precision: {precision:.4f}")
            print(f"  🎯 AI Recall: {recall:.4f}")
            print(f"  🎯 AI F1: {f1_score:.4f}")
            print(f"  👤 Human Precision: {human_precision:.4f}")
            print(f"  👤 Human Recall: {human_recall:.4f}")
            print(f"  👤 Human F1: {human_f1:.4f}")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Error testing model: {e}")
            return None
    
    def _reconstruct_custom_tree(self, model_data):
        """Reconstruct custom tree model from serialized data"""
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
        model.feature_names = self.feature_names
        
        # Reconstruct the tree structure
        model.root = self._reconstruct_tree_node(model_data['tree_structure'])
        
        return model
    
    def _reconstruct_tree_node(self, node_data):
        """Reconstruct tree node from serialized data"""
        if node_data is None:
            return None
        
        from custom_tree_model import TreeNode
        
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
    
    def compare_results(self, results: Dict):
        """Compare all model results"""
        print(f"\n📊 Model Comparison (Fixed):")
        print("=" * 70)
        
        comparison_data = []
        
        for model_name, result in results.items():
            if result is not None:
                comparison_data.append({
                    'Model': model_name,
                    'Type': result['model_type'],
                    'Accuracy': result['accuracy'],
                    'AI Precision': result['precision'],
                    'AI Recall': result['recall'],
                    'AI F1': result['f1_score'],
                    'Human Precision': result['human_precision'],
                    'Human Recall': result['human_recall'],
                    'Human F1': result['human_f1']
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print(comparison_df.to_string(index=False, float_format='%.4f'))
            
            # Find best model
            best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
            best_balanced = comparison_df.loc[(comparison_df['AI F1'] + comparison_df['Human F1']).idxmax()]
            
            print(f"\n🏆 Best Model by Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
            print(f"⚖️  Best Balanced Model: {best_balanced['Model']} (AI F1: {best_balanced['AI F1']:.4f}, Human F1: {best_balanced['Human F1']:.4f})")
    
    def run_fixes(self):
        """Run all model fixes"""
        print("🔧 Starting Model Fixes...")
        print("=" * 50)
        
        # Fix models
        main_fixed = self.fix_main_model()
        improved_fixed = self.fix_improved_model()
        
        if main_fixed or improved_fixed:
            # Test fixed models
            self.test_fixed_models()
            
            print(f"\n✅ Model fixes completed!")
            if main_fixed:
                print(f"  📁 Fixed main model: model_fixed.pkl")
            if improved_fixed:
                print(f"  📁 Fixed improved model: improved_model_fixed.pkl")
        else:
            print(f"\n❌ No models were fixed")

def main():
    """Main fixing function"""
    fixer = ModelFixer()
    fixer.run_fixes()

if __name__ == "__main__":
    main() 