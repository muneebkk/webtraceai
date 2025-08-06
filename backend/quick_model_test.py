#!/usr/bin/env python3
"""
Quick Model Testing Script for WebTrace AI
Fast testing without cross-validation for immediate results
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

class QuickModelTester:
    """Quick model testing without cross-validation"""
    
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.feature_names = self.extractor.get_feature_names()
        self.models = {}
        self.results = {}
        
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
    
    def load_models(self):
        """Load all trained models"""
        print("\n🤖 Loading trained models...")
        
        # Load main model (Random Forest)
        if os.path.exists("model.pkl"):
            try:
                with open("model.pkl", 'rb') as f:
                    model_data = pickle.load(f)
                self.models['main_model'] = {
                    'model': model_data['model'],
                    'feature_names': model_data['feature_names'],
                    'extractor': model_data['extractor'],
                    'scaler': model_data.get('scaler', None),
                    'type': 'RandomForest'
                }
                print("  ✅ Main model (Random Forest) loaded")
            except Exception as e:
                print(f"  ❌ Error loading main model: {e}")
        
        # Load improved model (Logistic Regression)
        if os.path.exists("improved_model.pkl"):
            try:
                with open("improved_model.pkl", 'rb') as f:
                    model_data = pickle.load(f)
                self.models['improved_model'] = {
                    'model': model_data['model'],
                    'feature_names': model_data['feature_names'],
                    'extractor': model_data['extractor'],
                    'scaler': model_data['scaler'],
                    'feature_mask': model_data.get('feature_mask', None),
                    'type': 'LogisticRegression'
                }
                print("  ✅ Improved model (Logistic Regression) loaded")
            except Exception as e:
                print(f"  ❌ Error loading improved model: {e}")
        
        # Load custom tree model
        if os.path.exists("custom_tree_model.pkl"):
            try:
                with open("custom_tree_model.pkl", 'rb') as f:
                    model_data = pickle.load(f)
                
                # Reconstruct custom tree
                from custom_tree_model import CustomDecisionTree, TreeNode
                
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
                
                self.models['custom_tree_model'] = {
                    'model': model,
                    'feature_names': model_data['feature_names'],
                    'extractor': model_data['extractor'],
                    'scaler': None,
                    'type': 'CustomDecisionTree'
                }
                print("  ✅ Custom tree model loaded")
            except Exception as e:
                print(f"  ❌ Error loading custom tree model: {e}")
        
        print(f"  📊 Loaded {len(self.models)} models")
    
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
    
    def test_model(self, model_name: str, model_data: Dict, X: np.ndarray, y: np.ndarray) -> Dict:
        """Test a single model and return results"""
        print(f"\n🧪 Testing {model_name} ({model_data['type']})...")
        
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
                'model_type': model_data['type'],
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
    
    def compare_models(self):
        """Compare all models and provide recommendations"""
        print(f"\n📊 Quick Model Comparison:")
        print("=" * 60)
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            if results is not None:
                comparison_data.append({
                    'Model': model_name,
                    'Type': results['model_type'],
                    'Accuracy': results['accuracy'],
                    'AI Precision': results['precision'],
                    'AI Recall': results['recall'],
                    'AI F1': results['f1_score'],
                    'Human Precision': results['human_precision'],
                    'Human Recall': results['human_recall'],
                    'Human F1': results['human_f1']
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print(comparison_df.to_string(index=False, float_format='%.4f'))
            
            # Find best model
            best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
            best_balanced = comparison_df.loc[(comparison_df['AI F1'] + comparison_df['Human F1']).idxmax()]
            
            print(f"\n🏆 Best Model by Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
            print(f"⚖️  Best Balanced Model: {best_balanced['Model']} (AI F1: {best_balanced['AI F1']:.4f}, Human F1: {best_balanced['Human F1']:.4f})")
            
            # Quick recommendations
            self._provide_quick_recommendations(comparison_df)
    
    def _provide_quick_recommendations(self, comparison_df: pd.DataFrame):
        """Provide quick improvement recommendations"""
        print(f"\n💡 Quick Recommendations:")
        print("=" * 40)
        
        # Check for class imbalance issues
        for _, row in comparison_df.iterrows():
            if abs(row['AI F1'] - row['Human F1']) > 0.2:
                print(f"⚠️  {row['Model']}: Class imbalance detected (AI F1: {row['AI F1']:.3f}, Human F1: {row['Human F1']:.3f})")
        
        # General recommendations
        print(f"\n🎯 Quick Wins:")
        print(f"  📈 Add more AI tools (currently only V0.dev)")
        print(f"  👥 Add more human-coded websites")
        print(f"  🔄 Try ensemble methods")
        print(f"  ⚖️  Address class imbalance")
    
    def run_quick_test(self):
        """Run quick testing of all models"""
        print("🚀 Starting Quick Model Testing...")
        print("=" * 50)
        
        # Load dataset
        df, labels = self.load_dataset()
        if df is None:
            print("❌ Failed to load dataset")
            return
        
        # Load models
        self.load_models()
        if not self.models:
            print("❌ No models loaded")
            return
        
        # Test each model
        for model_name, model_data in self.models.items():
            results = self.test_model(model_name, model_data, df.values, labels)
            if results:
                self.results[model_name] = results
        
        # Compare models
        self.compare_models()
        
        print(f"\n✅ Quick testing completed in ~30 seconds!")
        print(f"📊 Tested {len(self.results)} models")

def main():
    """Main quick testing function"""
    tester = QuickModelTester()
    tester.run_quick_test()

if __name__ == "__main__":
    main() 