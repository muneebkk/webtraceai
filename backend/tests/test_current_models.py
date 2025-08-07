#!/usr/bin/env python3
"""
Comprehensive Model Testing Script for WebTrace AI
Tests all trained models and provides detailed accuracy analysis
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import cv2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.feature_extract import FeatureExtractor
from custom_tree_model import CustomDecisionTree, CustomTreeTrainer

class ModelTester:
    """Comprehensive model testing and analysis"""
    
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
        file_info = []
        
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
                            file_info.append({'file': filename, 'class': 'ai', 'path': filepath})
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
                            file_info.append({'file': filename, 'class': 'human', 'path': filepath})
                    except Exception as e:
                        print(f"  ❌ Error loading {filename}: {e}")
        
        if not features_list:
            print("❌ No valid images found in dataset!")
            return None, None, None
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        labels = np.array(labels)
        file_df = pd.DataFrame(file_info)
        
        print(f"\n📊 Dataset Summary:")
        print(f"  Total samples: {len(df)}")
        print(f"  AI-generated: {np.sum(labels == 1)}")
        print(f"  Human-coded: {np.sum(labels == 0)}")
        print(f"  Features: {len(self.feature_names)}")
        
        return df, labels, file_df
    
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
        """Test a single model and return detailed results"""
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
            
            # Calculate ROC AUC
            try:
                roc_auc = roc_auc_score(y, y_pred_proba[:, 1])
            except:
                roc_auc = 0.5  # Default for binary classification
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_test, y, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Detailed classification report
            report = classification_report(y, y_pred, target_names=['Human', 'AI'], output_dict=True)
            
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
                'roc_auc': roc_auc,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'human_precision': human_precision,
                'human_recall': human_recall,
                'human_f1': human_f1,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': report
            }
            
            print(f"  ✅ Accuracy: {accuracy:.4f}")
            print(f"  📊 ROC AUC: {roc_auc:.4f}")
            print(f"  🔄 CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
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
    
    def analyze_feature_importance(self, model_name: str, model_data: Dict, X: np.ndarray, y: np.ndarray):
        """Analyze feature importance for the model"""
        print(f"\n🔍 Analyzing feature importance for {model_name}...")
        
        model = model_data['model']
        feature_mask = model_data.get('feature_mask', None)
        
        # Get feature names
        if feature_mask is not None:
            feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if feature_mask[i]]
        else:
            feature_names = self.feature_names
        
        # Get feature importance
        try:
            if hasattr(model, 'feature_importances_'):
                # Random Forest
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Logistic Regression
                importance = np.abs(model.coef_[0])
            elif hasattr(model, 'get_feature_importance'):
                # Custom tree
                importance_dict = model.get_feature_importance()
                importance = [importance_dict.get(name, 0) for name in feature_names]
            else:
                print("  ⚠️  Feature importance not available for this model type")
                return
            
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print(f"\n📊 Top 10 Most Important Features:")
            for i, row in feature_importance_df.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
            
            # Store for later analysis
            self.results[model_name]['feature_importance'] = feature_importance_df
            
        except Exception as e:
            print(f"  ❌ Error analyzing feature importance: {e}")
    
    def compare_models(self):
        """Compare all models and provide recommendations"""
        print(f"\n📊 Model Comparison Summary:")
        print("=" * 80)
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            if results is not None:
                comparison_data.append({
                    'Model': model_name,
                    'Type': results['model_type'],
                    'Accuracy': results['accuracy'],
                    'ROC AUC': results['roc_auc'],
                    'CV Score': results['cv_mean'],
                    'CV Std': results['cv_std'],
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
            
            # Provide improvement recommendations
            self._provide_recommendations(comparison_df)
    
    def _provide_recommendations(self, comparison_df: pd.DataFrame):
        """Provide specific improvement recommendations"""
        print(f"\n💡 Improvement Recommendations:")
        print("=" * 50)
        
        # Check for overfitting
        for _, row in comparison_df.iterrows():
            if row['CV Score'] - row['Accuracy'] > 0.1:
                print(f"⚠️  {row['Model']}: Potential overfitting (CV: {row['CV Score']:.3f}, Test: {row['Accuracy']:.3f})")
        
        # Check for class imbalance issues
        for _, row in comparison_df.iterrows():
            if abs(row['AI F1'] - row['Human F1']) > 0.2:
                print(f"⚠️  {row['Model']}: Class imbalance detected (AI F1: {row['AI F1']:.3f}, Human F1: {row['Human F1']:.3f})")
        
        # General recommendations
        print(f"\n🎯 General Recommendations:")
        
        # Dataset size
        print(f"  📈 Expand dataset to 200+ samples (currently ~40)")
        print(f"  🎨 Add more AI tools (currently only V0.dev)")
        print(f"  👥 Add more diverse human-coded websites")
        
        # Feature engineering
        print(f"\n🔧 Feature Engineering:")
        print(f"  🎯 Focus on visual features (color, layout, texture)")
        print(f"  🚫 Remove highly correlated features")
        print(f"  📊 Use feature selection techniques")
        
        # Model improvements
        print(f"\n🤖 Model Improvements:")
        print(f"  🔄 Try ensemble methods (voting classifiers)")
        print(f"  🎛️  Optimize hyperparameters with grid search")
        print(f"  📏 Use cross-validation for model selection")
        print(f"  ⚖️  Address class imbalance with SMOTE")
    
    def run_comprehensive_test(self):
        """Run comprehensive testing of all models"""
        print("🚀 Starting Comprehensive Model Testing...")
        print("=" * 60)
        
        # Load dataset
        df, labels, file_df = self.load_dataset()
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
                
                # Analyze feature importance
                self.analyze_feature_importance(model_name, model_data, df.values, labels)
        
        # Compare models
        self.compare_models()
        
        # Save detailed results
        self._save_results()
        
        print(f"\n✅ Comprehensive testing completed!")
        print(f"📊 Tested {len(self.results)} models")
    
    def _save_results(self):
        """Save detailed test results"""
        import json
        from datetime import datetime
        
        # Create results summary
        summary = {
            'test_date': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(self.results.get(list(self.results.keys())[0], {}).get('predictions', [])),
                'ai_samples': sum(1 for pred in self.results.get(list(self.results.keys())[0], {}).get('predictions', []) if pred == 1),
                'human_samples': sum(1 for pred in self.results.get(list(self.results.keys())[0], {}).get('predictions', []) if pred == 0)
            },
            'model_results': {}
        }
        
        for model_name, results in self.results.items():
            if results:
                summary['model_results'][model_name] = {
                    'model_type': results['model_type'],
                    'accuracy': results['accuracy'],
                    'roc_auc': results['roc_auc'],
                    'cv_mean': results['cv_mean'],
                    'cv_std': results['cv_std'],
                    'ai_precision': results['precision'],
                    'ai_recall': results['recall'],
                    'ai_f1': results['f1_score'],
                    'human_precision': results['human_precision'],
                    'human_recall': results['human_recall'],
                    'human_f1': results['human_f1']
                }
        
        # Save to file
        with open('model_test_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Detailed results saved to: model_test_results.json")

def main():
    """Main testing function"""
    tester = ModelTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main() 