#!/usr/bin/env python3
"""
Custom Decision Tree Classifier for WebTrace AI
Built from scratch with customizable features
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import cv2
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.feature_extract import FeatureExtractor

class TreeNode:
    """Node in the decision tree"""
    def __init__(self, feature_idx: Optional[int] = None, threshold: Optional[float] = None, 
                 left=None, right=None, is_leaf: bool = False, prediction: Optional[int] = None,
                 samples: int = 0, depth: int = 0):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.samples = samples
        self.depth = depth
        self.feature_name = None  # Will be set during training

class CustomDecisionTree(BaseEstimator, ClassifierMixin):
    """Custom Decision Tree Classifier with advanced features"""
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 5, 
                 min_samples_leaf: int = 2, min_impurity_decrease: float = 0.0,
                 max_features: Optional[str] = 'sqrt', random_state: int = 42,
                 criterion: str = 'gini', prune: bool = True):
        """
        Initialize custom decision tree
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required in a leaf node
            min_impurity_decrease: Minimum impurity decrease to split
            max_features: Number of features to consider for splitting ('sqrt', 'log2', int, or None)
            random_state: Random seed for reproducibility
            criterion: Splitting criterion ('gini' or 'entropy')
            prune: Whether to apply pruning
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state
        self.criterion = criterion
        self.prune = prune
        
        self.root = None
        self.feature_names = None
        self.n_features = 0
        self.n_classes = 0
        self.classes = None
        
        # Set random seed
        np.random.seed(random_state)
        
        # Set required scikit-learn attributes
        self._estimator_type = "classifier"
    
    def get_params(self, deep=True):
        """Get parameters for this estimator (scikit-learn compatibility)"""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_impurity_decrease': self.min_impurity_decrease,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'criterion': self.criterion,
            'prune': self.prune
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator (scikit-learn compatibility)"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity using specified criterion"""
        if len(y) == 0:
            return 0.0
        
        # Count class frequencies
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        if self.criterion == 'gini':
            return 1.0 - np.sum(probabilities ** 2)
        elif self.criterion == 'entropy':
            # Avoid log(0)
            probabilities = probabilities[probabilities > 0]
            return -np.sum(probabilities * np.log2(probabilities))
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """Find the best split for a node"""
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples_split:
            return None, None, 0.0
        
        # Calculate current impurity
        current_impurity = self._calculate_impurity(y)
        
        # Determine number of features to consider
        if self.max_features == 'sqrt':
            n_features_to_consider = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            n_features_to_consider = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            n_features_to_consider = min(self.max_features, n_features)
        else:  # None or 'all'
            n_features_to_consider = n_features
        
        # Randomly select features to consider
        feature_indices = np.random.choice(n_features, n_features_to_consider, replace=False)
        
        best_feature = None
        best_threshold = None
        best_impurity_decrease = 0.0
        
        for feature_idx in feature_indices:
            # Get unique values for this feature
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # Try each unique value as a potential threshold
            for threshold in unique_values:
                # Split the data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # Check minimum samples requirement
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate weighted impurity
                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                impurity_decrease = current_impurity - weighted_impurity
                
                # Check minimum impurity decrease
                if impurity_decrease > best_impurity_decrease and impurity_decrease >= self.min_impurity_decrease:
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_impurity_decrease = impurity_decrease
        
        return best_feature, best_threshold, best_impurity_decrease
    
    def _create_leaf_node(self, y: np.ndarray, depth: int) -> TreeNode:
        """Create a leaf node with majority class prediction"""
        # Find majority class
        unique, counts = np.unique(y, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        
        return TreeNode(
            is_leaf=True,
            prediction=majority_class,
            samples=len(y),
            depth=depth
        )
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        """Recursively build the decision tree"""
        n_samples = len(y)
        
        # Check stopping conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return self._create_leaf_node(y, depth)
        
        # Find best split
        best_feature, best_threshold, impurity_decrease = self._find_best_split(X, y)
        
        # If no good split found, create leaf
        if best_feature is None:
            return self._create_leaf_node(y, depth)
        
        # Split the data
        feature_values = X[:, best_feature]
        left_mask = feature_values <= best_threshold
        right_mask = ~left_mask
        
        # Create node
        node = TreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            samples=n_samples,
            depth=depth
        )
        
        # Recursively build left and right subtrees
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def _prune_tree(self, node: TreeNode, X: np.ndarray, y: np.ndarray) -> TreeNode:
        """Prune the tree using reduced error pruning"""
        if node.is_leaf:
            return node
        
        # Get feature values for this node
        feature_values = X[:, node.feature_idx]
        left_mask = feature_values <= node.threshold
        right_mask = ~left_mask
        
        # Recursively prune subtrees
        node.left = self._prune_tree(node.left, X[left_mask], y[left_mask])
        node.right = self._prune_tree(node.right, X[right_mask], y[right_mask])
        
        # Check if pruning this node would improve performance
        if node.left.is_leaf and node.right.is_leaf:
            # Calculate error if we keep the split
            left_pred = node.left.prediction
            right_pred = node.right.prediction
            
            split_errors = 0
            split_errors += np.sum(y[left_mask] != left_pred)
            split_errors += np.sum(y[right_mask] != right_pred)
            
            # Calculate error if we make this a leaf
            majority_class = self._get_majority_class(y)
            leaf_errors = np.sum(y != majority_class)
            
            # If leaf has fewer errors, prune
            if leaf_errors <= split_errors:
                return self._create_leaf_node(y, node.depth)
        
        return node
    
    def _get_majority_class(self, y: np.ndarray) -> int:
        """Get the majority class from labels"""
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the decision tree to the data"""
        self.n_features = X.shape[1]
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        # Build the tree
        self.root = self._build_tree(X, y)
        
        # Set feature names in tree nodes (use default names if not provided)
        if not hasattr(self, 'feature_names') or self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
        self._set_feature_names(self.root)
        
        # Prune if requested
        if self.prune:
            self.root = self._prune_tree(self.root, X, y)
        
        # Set required scikit-learn attributes
        self._estimator_type = "classifier"
        
        return self
    
    def fit_with_names(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """Fit the decision tree with feature names (for backward compatibility)"""
        self.feature_names = feature_names
        return self.fit(X, y)
    
    def _set_feature_names(self, node: TreeNode):
        """Set feature names in tree nodes for interpretability"""
        if not node.is_leaf:
            if self.feature_names and node.feature_idx < len(self.feature_names):
                node.feature_name = self.feature_names[node.feature_idx]
            self._set_feature_names(node.left)
            self._set_feature_names(node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X"""
        return np.array([self._predict_single(x) for x in X])
    
    def _predict_single(self, x: np.ndarray) -> int:
        """Predict class for a single sample"""
        node = self.root
        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X"""
        predictions = []
        for x in X:
            proba = self._predict_proba_single(x)
            predictions.append(proba)
        return np.array(predictions)
    
    def _predict_proba_single(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for a single sample"""
        node = self.root
        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        # Create probability array
        proba = np.zeros(self.n_classes)
        class_idx = np.where(self.classes == node.prediction)[0][0]
        proba[class_idx] = 1.0
        return proba
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance based on impurity decrease"""
        importance = np.zeros(self.n_features)
        self._calculate_importance(self.root, importance)
        
        # Normalize importance
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        # Create dictionary with feature names
        feature_importance = {}
        for i, imp in enumerate(importance):
            feature_name = self.feature_names[i] if self.feature_names else f"feature_{i}"
            feature_importance[feature_name] = imp
        
        return feature_importance
    
    def _calculate_importance(self, node: TreeNode, importance: np.ndarray):
        """Recursively calculate feature importance"""
        if not node.is_leaf:
            # Add this node's contribution
            importance[node.feature_idx] += node.samples
            
            # Recursively calculate for children
            self._calculate_importance(node.left, importance)
            self._calculate_importance(node.right, importance)
    
    def print_tree(self, node: Optional[TreeNode] = None, indent: str = ""):
        """Print the tree structure for debugging"""
        if node is None:
            node = self.root
        
        if node.is_leaf:
            print(f"{indent}Leaf: {node.prediction} (samples: {node.samples})")
        else:
            feature_name = node.feature_name or f"feature_{node.feature_idx}"
            print(f"{indent}{feature_name} <= {node.threshold:.4f} (samples: {node.samples})")
            print(f"{indent}├── True:")
            self.print_tree(node.left, indent + "│   ")
            print(f"{indent}└── False:")
            self.print_tree(node.right, indent + "    ")

class CustomTreeTrainer:
    """Trainer for the custom decision tree model"""
    
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.feature_names = self.extractor.get_feature_names()
        self.model = None
        
    def load_dataset(self, dataset_path="../dataset"):
        """Load dataset with error handling"""
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
                            file_info.append({'file': filename, 'class': 'ai', 'tool': 'v0'})
                            print(f"  ✅ Loaded: {filename}")
                        else:
                            print(f"  ⚠️  Invalid image: {filename}")
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
                            file_info.append({'file': filename, 'class': 'human', 'tool': 'manual'})
                            print(f"  ✅ Loaded: {filename}")
                        else:
                            print(f"  ⚠️  Invalid image: {filename}")
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
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   max_depth: int = 5, min_samples_split: int = 5,
                   min_samples_leaf: int = 2, criterion: str = 'gini',
                   prune: bool = True) -> CustomDecisionTree:
        """Train the custom decision tree model"""
        print(f"\n🌳 Training Custom Decision Tree...")
        print(f"  Max depth: {max_depth}")
        print(f"  Min samples split: {min_samples_split}")
        print(f"  Min samples leaf: {min_samples_leaf}")
        print(f"  Criterion: {criterion}")
        print(f"  Pruning: {prune}")
        
        # Create and train model
        self.model = CustomDecisionTree(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            prune=prune,
            random_state=42
        )
        
        self.model.fit_with_names(X, y, self.feature_names)
        
        print(f"✅ Custom tree trained successfully!")
        print(f"  Tree depth: {self._get_tree_depth(self.model.root)}")
        print(f"  Total nodes: {self._count_nodes(self.model.root)}")
        print(f"  Leaf nodes: {self._count_leaf_nodes(self.model.root)}")
        
        return self.model
    
    def _get_tree_depth(self, node: TreeNode) -> int:
        """Get the depth of the tree"""
        if node.is_leaf:
            return node.depth
        return max(self._get_tree_depth(node.left), self._get_tree_depth(node.right))
    
    def _count_nodes(self, node: TreeNode) -> int:
        """Count total nodes in the tree"""
        if node.is_leaf:
            return 1
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)
    
    def _count_leaf_nodes(self, node: TreeNode) -> int:
        """Count leaf nodes in the tree"""
        if node.is_leaf:
            return 1
        return self._count_leaf_nodes(node.left) + self._count_leaf_nodes(node.right)
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model performance"""
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        # Make predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        
        print(f"\n📈 Model Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y, y_pred, target_names=['Human', 'AI']))
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(f"  True Negatives (Human): {cm[0, 0]}")
        print(f"  False Positives (Human→AI): {cm[0, 1]}")
        print(f"  False Negatives (AI→Human): {cm[1, 0]}")
        print(f"  True Positives (AI): {cm[1, 1]}")
        
        # Feature importance
        feature_importance = self.model.get_feature_importance()
        print(f"\n🔍 Top 10 Feature Importance:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:10]:
            print(f"  {feature}: {importance:.4f}")
        
        return {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def save_model(self, model_path="custom_tree_model.pkl"):
        """Save the trained model"""
        print(f"\n💾 Saving custom tree model to {model_path}...")
        
        # Instead of saving the entire model object, save the essential parameters
        # This avoids pickle serialization issues with custom classes
        model_data = {
            'model_type': 'CustomDecisionTree',
            'feature_names': self.feature_names,
            'extractor': self.extractor,
            'scaler': None,  # Custom tree doesn't need scaling
            'model_params': {
                'max_depth': self.model.max_depth,
                'min_samples_split': self.model.min_samples_split,
                'min_samples_leaf': self.model.min_samples_leaf,
                'criterion': self.model.criterion,
                'prune': self.model.prune,
                'n_features': self.model.n_features,
                'classes': self.model.classes.tolist(),
                'n_classes': self.model.n_classes
            },
            'tree_structure': self._serialize_tree(self.model.root),
            'training_info': {
                'max_depth': self.model.max_depth,
                'min_samples_split': self.model.min_samples_split,
                'min_samples_leaf': self.model.min_samples_leaf,
                'criterion': self.model.criterion,
                'prune': self.model.prune,
                'tree_depth': self._get_tree_depth(self.model.root),
                'total_nodes': self._count_nodes(self.model.root),
                'leaf_nodes': self._count_leaf_nodes(self.model.root)
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Custom tree model saved successfully!")
        print(f"   Model type: CustomDecisionTree")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Tree depth: {model_data['training_info']['tree_depth']}")
        print(f"   Total nodes: {model_data['training_info']['total_nodes']}")
    
    def _serialize_tree(self, node):
        """Serialize tree structure to avoid pickle issues"""
        if node is None:
            return None
        
        serialized_node = {
            'is_leaf': node.is_leaf,
            'prediction': node.prediction,
            'samples': node.samples,
            'depth': node.depth
        }
        
        if not node.is_leaf:
            serialized_node.update({
                'feature_idx': node.feature_idx,
                'feature_name': node.feature_name,
                'threshold': node.threshold,
                'left': self._serialize_tree(node.left),
                'right': self._serialize_tree(node.right)
            })
        
        return serialized_node
    
    def print_tree_structure(self):
        """Print the tree structure for analysis"""
        print(f"\n🌳 Tree Structure:")
        self.model.print_tree()

def main():
    """Main training function"""
    print("🚀 Starting Custom Decision Tree Training...")
    
    trainer = CustomTreeTrainer()
    
    # Load dataset
    df, labels, file_df = trainer.load_dataset()
    if df is None:
        print("❌ Failed to load dataset")
        return
    
    # Train model with conservative parameters
    model = trainer.train_model(
        df.values, labels,
        max_depth=4,  # Conservative depth
        min_samples_split=8,  # Higher threshold
        min_samples_leaf=4,  # Higher threshold
        criterion='gini',
        prune=True
    )
    
    # Evaluate model
    results = trainer.evaluate_model(df.values, labels)
    
    # Save model
    trainer.save_model()
    
    # Print tree structure (optional, for debugging)
    # trainer.print_tree_structure()
    
    print(f"\n🎉 Custom tree training completed! Final accuracy: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main() 