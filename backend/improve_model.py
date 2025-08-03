#!/usr/bin/env python3
"""
Improved model training script to address overfitting and bias
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.feature_extract import FeatureExtractor

class ImprovedModelTrainer:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.scaler = None  # Will be set only if needed
        self.feature_names = self.extractor.get_feature_names()
        self.best_model = None
        self.best_score = 0
        self.selected_features = None
        
    def load_dataset(self, dataset_path="../dataset"):
        """Load dataset with better error handling and validation"""
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
        
        # Check for class imbalance
        ai_count = np.sum(labels == 1)
        human_count = np.sum(labels == 0)
        
        if ai_count == 0 or human_count == 0:
            print("❌ Dataset must contain both AI and human samples!")
            return None, None, None
        
        imbalance_ratio = min(ai_count, human_count) / max(ai_count, human_count)
        if imbalance_ratio < 0.3:
            print(f"⚠️  Severe class imbalance detected (ratio: {imbalance_ratio:.2f})")
        elif imbalance_ratio < 0.7:
            print(f"⚠️  Moderate class imbalance detected (ratio: {imbalance_ratio:.2f})")
        else:
            print(f"✅ Balanced dataset (ratio: {imbalance_ratio:.2f})")
        
        return df, labels, file_df
    
    def select_features(self, X, y):
        """Select the most important features to reduce overfitting"""
        print("\n🔍 Feature Selection...")
        
        # Calculate feature correlations
        X_df = pd.DataFrame(X, columns=self.feature_names)
        correlation_matrix = X_df.corr().abs()
        
        # Find highly correlated features (correlation > 0.95)
        upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        
        print(f"  Highly correlated features to remove: {len(high_corr_features)}")
        if high_corr_features:
            print(f"    {high_corr_features}")
        
        # Use SelectKBest to select top features based on F-statistic
        # Select top 75% of features to reduce overfitting
        k = max(5, int(len(self.feature_names) * 0.75))
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        self.selected_features = [self.feature_names[i] for i in selected_indices]
        
        print(f"  Selected {len(self.selected_features)} features out of {len(self.feature_names)}")
        print(f"  Selected features: {self.selected_features}")
        
        return X_selected
    
    def analyze_features(self, X, y):
        """Analyze feature importance and correlations"""
        print("\n📊 Feature Analysis...")
        
        # Use the correct feature names (selected features if available)
        feature_names = self.selected_features if self.selected_features else self.feature_names
        
        # Use Random Forest to get feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf.fit(X, y)
        
        # Get feature importance
        importance = rf.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("  Top 10 most important features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title('Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return feature_importance
    
    def handle_class_imbalance(self, X, y):
        """Handle class imbalance using SMOTE"""
        print("\n⚖️  Handling Class Imbalance...")
        
        ai_count = np.sum(y == 1)
        human_count = np.sum(y == 0)
        
        if ai_count == 0 or human_count == 0:
            print("❌ Cannot handle imbalance: missing one class")
            return X, y
        
        imbalance_ratio = min(ai_count, human_count) / max(ai_count, human_count)
        
        if imbalance_ratio < 0.5:
            print(f"  Applying SMOTE to balance classes (ratio: {imbalance_ratio:.2f})")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            print(f"  Before: AI={np.sum(y == 1)}, Human={np.sum(y == 0)}")
            print(f"  After: AI={np.sum(y_resampled == 1)}, Human={np.sum(y_resampled == 0)}")
            
            return X_resampled, y_resampled
        else:
            print(f"  No resampling needed (ratio: {imbalance_ratio:.2f})")
            return X, y
    
    def select_best_model(self, X, y):
        """Select the best model using cross-validation with simplified parameters"""
        print("\n🤖 Model Selection with Cross-Validation...")
        
        # Define models with simplified parameters to reduce overfitting
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,  # Reduced from None to prevent overfitting
                min_samples_split=10,  # Increased to reduce overfitting
                min_samples_leaf=5,  # Added to reduce overfitting
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=50,  # Reduced from 100
                learning_rate=0.1,
                max_depth=3,  # Reduced from 5
                random_state=42
            ),
            'SVM': SVC(random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Simplified parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, 7],
                'min_samples_split': [5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [30, 50],
                'learning_rate': [0.05, 0.1],
                'max_depth': [2, 3]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear']
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l2']  # Removed l1 to avoid solver issues
            }
        }
        
        best_score = 0
        best_model_name = None
        
        for name, model in models.items():
            print(f"\n  Testing {name}...")
            
            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(
                model, 
                param_grids[name], 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            # Scale features ONLY for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                if self.scaler is None:
                    self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
                grid_search.fit(X_scaled, y)
            else:
                # No scaling for tree-based models
                grid_search.fit(X, y)
            
            score = grid_search.best_score_
            print(f"    Best CV score: {score:.4f}")
            print(f"    Best parameters: {grid_search.best_params_}")
            
            if score > best_score:
                best_score = score
                best_model_name = name
                self.best_model = grid_search.best_estimator_
        
        print(f"\n🏆 Best model: {best_model_name} (CV score: {best_score:.4f})")
        self.best_score = best_score
        
        return self.best_model
    
    def train_final_model(self, X, y):
        """Train the final model with the best configuration"""
        print("\n🎯 Training Final Model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features ONLY if needed (SVM or Logistic Regression)
        if isinstance(self.best_model, (SVC, LogisticRegression)):
            if self.scaler is None:
                self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            X_train_final, X_test_final = X_train_scaled, X_test_scaled
        else:
            # No scaling for tree-based models
            X_train_final, X_test_final = X_train, X_test
        
        # Train model
        self.best_model.fit(X_train_final, y_train)
        
        # Evaluate
        y_pred = self.best_model.predict(X_test_final)
        y_pred_proba = self.best_model.predict_proba(X_test_final)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📈 Final Model Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Model type: {type(self.best_model).__name__}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(f"  True Negatives (Human): {cm[0, 0]}")
        print(f"  False Positives (Human→AI): {cm[0, 1]}")
        print(f"  False Negatives (AI→Human): {cm[1, 0]}")
        print(f"  True Positives (AI): {cm[1, 1]}")
        
        # Calculate additional metrics
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n🎯 Detailed Metrics:")
        print(f"  Precision (AI detection): {precision:.4f}")
        print(f"  Recall (AI detection): {recall:.4f}")
        print(f"  F1-Score (AI detection): {f1:.4f}")
        
        return accuracy
    
    def save_model(self, model_path="improved_model.pkl"):
        """Save the trained model with metadata"""
        print(f"\n💾 Saving model to {model_path}...")
        
        model_data = {
            'model': self.best_model,
            'feature_names': self.selected_features if self.selected_features else self.feature_names,
            'extractor': self.extractor,
            'scaler': self.scaler,  # Will be None for tree-based models
            'best_score': self.best_score,
            'model_type': type(self.best_model).__name__,
            'training_info': {
                'total_features': len(self.feature_names),
                'selected_features': len(self.selected_features) if self.selected_features else len(self.feature_names),
                'model_requires_scaling': isinstance(self.best_model, (SVC, LogisticRegression))
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved successfully!")
        print(f"   Model type: {model_data['model_type']}")
        print(f"   Features: {len(model_data['feature_names'])}")
        print(f"   CV score: {self.best_score:.4f}")
        print(f"   Requires scaling: {model_data['training_info']['model_requires_scaling']}")
    
    def generate_recommendations(self, X, y, file_df):
        """Generate recommendations for dataset improvement"""
        print("\n💡 Dataset Improvement Recommendations:")
        
        # Analyze tool distribution
        if 'tool' in file_df.columns:
            tool_counts = file_df['tool'].value_counts()
            print(f"  Tool distribution:")
            for tool, count in tool_counts.items():
                print(f"    {tool}: {count} samples")
        
        # Check for diversity issues
        ai_samples = np.sum(y == 1)
        human_samples = np.sum(y == 0)
        
        if ai_samples < 50:
            print(f"  ⚠️  Low AI sample count ({ai_samples}). Consider collecting more AI-generated websites.")
        
        if human_samples < 50:
            print(f"  ⚠️  Low human sample count ({human_samples}). Consider collecting more human-coded websites.")
        
        # Check for tool bias
        if 'tool' in file_df.columns:
            ai_tools = file_df[file_df['class'] == 'ai']['tool'].value_counts()
            if len(ai_tools) == 1:
                print(f"  ⚠️  AI samples are all from one tool ({ai_tools.index[0]}). Consider diversifying AI tools.")
        
        print(f"  📈 Target dataset size: 100+ samples per class")
        print(f"  🎯 Target accuracy: >85% on external validation")
        
        return {
            'ai_samples': ai_samples,
            'human_samples': human_samples,
            'total_samples': len(y),
            'recommendations': [
                "Collect more diverse AI-generated websites",
                "Include websites from different AI tools",
                "Ensure human samples represent various coding styles",
                "Validate model on external datasets"
            ]
        }

def main():
    """Main training function"""
    print("🚀 Starting Improved Model Training...")
    
    trainer = ImprovedModelTrainer()
    
    # Load dataset
    df, labels, file_df = trainer.load_dataset()
    if df is None:
        print("❌ Failed to load dataset")
        return
    
    # Feature selection
    X_selected = trainer.select_features(df.values, labels)
    
    # Analyze features
    trainer.analyze_features(X_selected, labels)
    
    # Handle class imbalance
    X_balanced, y_balanced = trainer.handle_class_imbalance(X_selected, labels)
    
    # Select best model
    best_model = trainer.select_best_model(X_balanced, y_balanced)
    
    # Train final model
    accuracy = trainer.train_final_model(X_balanced, y_balanced)
    
    # Save model
    trainer.save_model()
    
    # Generate recommendations
    trainer.generate_recommendations(X_balanced, y_balanced, file_df)
    
    print(f"\n🎉 Training completed! Final accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 