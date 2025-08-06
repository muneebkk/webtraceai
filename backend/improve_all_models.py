#!/usr/bin/env python3
"""
Comprehensive model improvement script
Fixes: Custom tree features, improves accuracy, creates ensemble
"""

import os
import pickle
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
from app.feature_extract import FeatureExtractor
from custom_tree_model import CustomTreeTrainer

class ModelImprover:
    def __init__(self):
        self.dataset_path = "../dataset"
        self.extractor = FeatureExtractor()
        self.models = {}
        self.results = {}
        
    def load_dataset_full_features(self):
        """Load dataset with ALL 43 features (not just meaningful ones)"""
        print("📊 Loading dataset with full feature set...")
        
        # Load images
        ai_images = []
        human_images = []
        
        # Load AI images
        ai_path = os.path.join(self.dataset_path, "images/ai")
        for filename in os.listdir(ai_path):
            if filename.endswith('.png'):
                img_path = os.path.join(ai_path, filename)
                ai_images.append(img_path)
        
        # Load human images  
        human_path = os.path.join(self.dataset_path, "images/human")
        for filename in os.listdir(human_path):
            if filename.endswith('.png'):
                img_path = os.path.join(human_path, filename)
                human_images.append(img_path)
        
        print(f"  Found {len(ai_images)} AI images and {len(human_images)} human images")
        
        # Extract ALL features (43 features)
        features_list = []
        labels = []
        file_paths = []
        
        # Process AI images
        print(f"  🔄 Processing {len(ai_images)} AI images...")
        for i, img_path in enumerate(ai_images, 1):
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    # Use extract_image_features for ALL 43 features
                    features = self.extractor.extract_image_features(img)
                    if features is not None:
                        features_list.append(features)
                        labels.append(1)  # AI
                        file_paths.append(img_path)
                        print(f"    ✅ AI image {i}/{len(ai_images)}: {os.path.basename(img_path)}")
                    else:
                        print(f"    ❌ AI image {i}/{len(ai_images)}: {os.path.basename(img_path)} - No features extracted")
                else:
                    print(f"    ❌ AI image {i}/{len(ai_images)}: {os.path.basename(img_path)} - Failed to load")
            except Exception as e:
                print(f"    ❌ AI image {i}/{len(ai_images)}: {os.path.basename(img_path)} - Error: {e}")
        
        # Process human images
        print(f"  🔄 Processing {len(human_images)} human images...")
        for i, img_path in enumerate(human_images, 1):
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    # Use extract_image_features for ALL 43 features
                    features = self.extractor.extract_image_features(img)
                    if features is not None:
                        features_list.append(features)
                        labels.append(0)  # Human
                        file_paths.append(img_path)
                        print(f"    ✅ Human image {i}/{len(human_images)}: {os.path.basename(img_path)}")
                    else:
                        print(f"    ❌ Human image {i}/{len(human_images)}: {os.path.basename(img_path)} - No features extracted")
                else:
                    print(f"    ❌ Human image {i}/{len(human_images)}: {os.path.basename(img_path)} - Failed to load")
            except Exception as e:
                print(f"    ❌ Human image {i}/{len(human_images)}: {os.path.basename(img_path)} - Error: {e}")
        
        # Convert to DataFrame with ALL feature names
        df = pd.DataFrame(features_list, columns=self.extractor.get_feature_names())
        labels = np.array(labels)
        file_df = pd.DataFrame({'file_path': file_paths, 'label': labels})
        
        print(f"  ✅ Successfully extracted features from {len(df)} images")
        print(f"  📈 Dataset shape: {df.shape}")
        print(f"  🎯 Class distribution: {np.bincount(labels)}")
        print(f"  🔍 Features used: {len(self.extractor.get_feature_names())}")
        
        return df, labels, file_df
    
    def train_improved_original_model(self, X_train, X_test, y_train, y_test):
        """Train improved Random Forest with hyperparameter tuning"""
        print("\n🌲 Training Improved Original Model (Random Forest)...")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 8, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Evaluate
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
        
        print(f"  ✅ Training completed")
        print(f"  🔧 Best parameters: {grid_search.best_params_}")
        print(f"  📊 Test Accuracy: {accuracy:.4f}")
        print(f"  📊 Test AUC: {auc:.4f}")
        print(f"  📊 CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save model
        model_data = {
            'model': best_model,
            'accuracy': accuracy,
            'auc_score': auc,
            'cv_scores': cv_scores,
            'feature_names': self.extractor.get_feature_names(),
            'best_params': grid_search.best_params_
        }
        
        with open('model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        self.models['original'] = best_model
        self.results['original'] = {
            'accuracy': accuracy,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        return best_model
    
    def train_high_accuracy_improved_model(self, X_train, X_test, y_train, y_test):
        """Train high-accuracy Logistic Regression with advanced techniques"""
        print("\n📈 Training High-Accuracy Improved Model (Logistic Regression)...")
        
        # Feature selection
        feature_names = self.extractor.get_feature_names()
        
        # Exclude basic properties and font features
        exclude_features = ['width', 'height', 'aspect_ratio', 'total_pixels']
        exclude_features.extend([name for name in feature_names if 'font' in name.lower()])
        
        keep_indices = [i for i, name in enumerate(feature_names) if name not in exclude_features]
        X_train_selected = X_train.iloc[:, keep_indices]
        X_test_selected = X_test.iloc[:, keep_indices]
        selected_feature_names = [feature_names[i] for i in keep_indices]
        
        print(f"  🚫 Excluded {len(exclude_features)} features: {exclude_features}")
        print(f"  ✅ Using {len(selected_feature_names)} selected features")
        
        # Try different scaling methods
        scalers = {
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler()
        }
        
        best_scaler = None
        best_score = 0
        best_model = None
        
        for scaler_name, scaler in scalers.items():
            print(f"  🔄 Testing {scaler_name}...")
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Try different balancing methods
            balancers = {
                'None': None,
                'SMOTE': SMOTE(random_state=42),
                'ADASYN': ADASYN(random_state=42),
                'SMOTETomek': SMOTETomek(random_state=42)
            }
            
            for balancer_name, balancer in balancers.items():
                print(f"    🔄 Testing {balancer_name} balancing...")
                
                if balancer is not None:
                    X_train_balanced, y_train_balanced = balancer.fit_resample(X_train_scaled, y_train)
                else:
                    X_train_balanced, y_train_balanced = X_train_scaled, y_train
                
                # Try different regularization strengths
                for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
                    model = LogisticRegression(
                        C=C,
                        penalty='l2',
                        solver='liblinear',
                        random_state=42,
                        max_iter=1000
                    )
                    
                    model.fit(X_train_balanced, y_train_balanced)
                    cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
                    
                    if cv_score > best_score:
                        best_score = cv_score
                        best_model = model
                        best_scaler = scaler
                        best_balancer_name = balancer_name
                        best_C = C
        
        # Train final model with best parameters
        X_train_scaled = best_scaler.fit_transform(X_train_selected)
        X_test_scaled = best_scaler.transform(X_test_selected)
        
        if best_balancer_name != 'None':
            balancer = balancers[best_balancer_name]
            X_train_balanced, y_train_balanced = balancer.fit_resample(X_train_scaled, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train_scaled, y_train
        
        best_model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate
        y_pred = best_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        print(f"  ✅ Training completed")
        print(f"  🔧 Best parameters:")
        print(f"     Scaler: {type(best_scaler).__name__}")
        print(f"     Balancer: {best_balancer_name}")
        print(f"     C: {best_C}")
        print(f"  📊 Test Accuracy: {accuracy:.4f}")
        print(f"  📊 Test AUC: {auc:.4f}")
        print(f"  📊 CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save model
        model_data = {
            'model': best_model,
            'scaler': best_scaler,
            'feature_mask': keep_indices,
            'feature_names': selected_feature_names,
            'accuracy': accuracy,
            'auc_score': auc,
            'cv_scores': cv_scores,
            'balancing_strategy': best_balancer_name,
            'scaling_method': type(best_scaler).__name__,
            'best_C': best_C
        }
        
        with open('improved_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        self.models['improved'] = best_model
        self.results['improved'] = {
            'accuracy': accuracy,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        return best_model
    
    def train_full_feature_custom_tree(self, X_train, X_test, y_train, y_test):
        """Train Custom Decision Tree with ALL 43 features"""
        print("\n🌳 Training Full-Feature Custom Tree Model...")
        
        # Use CustomTreeTrainer with ALL features
        trainer = CustomTreeTrainer()
        
        # Train with optimized parameters
        model = trainer.train_model(
            X_train, y_train,
            max_depth=8,  # Increased depth
            min_samples_split=3,  # Reduced for more splits
            min_samples_leaf=1,  # Reduced for more leaves
            criterion='gini',
            prune=True
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        print(f"  ✅ Training completed")
        print(f"  📊 Test Accuracy: {accuracy:.4f}")
        print(f"  📊 Test AUC: {auc:.4f}")
        print(f"  📊 CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"  🌳 Tree depth: {trainer._get_tree_depth(model.root)}")
        print(f"  🌳 Total nodes: {trainer._count_nodes(model.root)}")
        
        # Save model with ALL features
        model_data = {
            'model_type': 'CustomDecisionTree',
            'feature_names': self.extractor.get_feature_names(),  # ALL 43 features
            'extractor': self.extractor,
            'scaler': None,
            'model_params': {
                'max_depth': model.max_depth,
                'min_samples_split': model.min_samples_split,
                'min_samples_leaf': model.min_samples_leaf,
                'criterion': model.criterion,
                'prune': model.prune,
                'n_features': model.n_features,
                'classes': model.classes.tolist(),
                'n_classes': model.n_classes
            },
            'tree_structure': trainer._serialize_tree(model.root),
            'training_info': {
                'max_depth': model.max_depth,
                'min_samples_split': model.min_samples_split,
                'min_samples_leaf': model.min_samples_leaf,
                'criterion': model.criterion,
                'prune': model.prune,
                'tree_depth': trainer._get_tree_depth(model.root),
                'total_nodes': trainer._count_nodes(model.root),
                'leaf_nodes': trainer._count_leaf_nodes(model.root)
            },
            'accuracy': accuracy,
            'auc_score': auc,
            'cv_scores': cv_scores
        }
        
        with open('custom_tree_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        self.models['custom_tree'] = model
        self.results['custom_tree'] = {
            'accuracy': accuracy,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        return model
    
    def train_ensemble_model(self, X_train, X_test, y_train, y_test):
        """Train Ensemble Model combining all three improved models"""
        print("\n🎯 Training Ensemble Model...")
        
        # Create ensemble with the three trained models
        ensemble = VotingClassifier(
            estimators=[
                ('rf', self.models['original']),
                ('lr', self.models['improved']),
                ('dt', self.models['custom_tree'])
            ],
            voting='soft',
            weights=[0.4, 0.3, 0.3]  # Give more weight to Random Forest
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, ensemble.predict_proba(X_test)[:, 1])
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
        
        print(f"  ✅ Training completed")
        print(f"  📊 Test Accuracy: {accuracy:.4f}")
        print(f"  📊 Test AUC: {auc:.4f}")
        print(f"  📊 CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save model
        model_data = {
            'model': ensemble,
            'accuracy': accuracy,
            'auc_score': auc,
            'cv_scores': cv_scores,
            'feature_names': self.extractor.get_feature_names(),
            'base_models': ['RandomForest', 'LogisticRegression', 'CustomDecisionTree'],
            'voting_method': 'soft',
            'weights': [0.4, 0.3, 0.3]
        }
        
        with open('ensemble_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        self.models['ensemble'] = ensemble
        self.results['ensemble'] = {
            'accuracy': accuracy,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        return ensemble
    
    def compare_models(self):
        """Compare all models and create visualization"""
        print("\n📊 Comparing Improved Models...")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Test Accuracy': results['accuracy'],
                'Test AUC': results['auc'],
                'CV Accuracy': results['cv_mean'],
                'CV Std': results['cv_std']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\n📈 Model Comparison:")
        print(df_comparison.to_string(index=False, float_format='%.4f'))
        
        # Find best model
        best_model = df_comparison.loc[df_comparison['Test Accuracy'].idxmax()]
        print(f"\n🏆 Best Model: {best_model['Model']} (Accuracy: {best_model['Test Accuracy']:.4f})")
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Test Accuracy
        bars = ax1.bar(df_comparison['Model'], df_comparison['Test Accuracy'])
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Highlight best model
        best_idx = df_comparison['Test Accuracy'].idxmax()
        bars[best_idx].set_color('green')
        
        # Test AUC
        bars = ax2.bar(df_comparison['Model'], df_comparison['Test AUC'])
        ax2.set_title('Test AUC Comparison')
        ax2.set_ylabel('AUC')
        ax2.tick_params(axis='x', rotation=45)
        
        # Highlight best model
        best_idx = df_comparison['Test AUC'].idxmax()
        bars[best_idx].set_color('green')
        
        # CV Accuracy with error bars
        bars = ax3.bar(df_comparison['Model'], df_comparison['CV Accuracy'], 
                yerr=df_comparison['CV Std'], capsize=5)
        ax3.set_title('Cross-Validation Accuracy')
        ax3.set_ylabel('Accuracy')
        ax3.tick_params(axis='x', rotation=45)
        
        # Highlight best model
        best_idx = df_comparison['CV Accuracy'].idxmax()
        bars[best_idx].set_color('green')
        
        # Combined metrics
        x = np.arange(len(df_comparison))
        width = 0.35
        ax4.bar(x - width/2, df_comparison['Test Accuracy'], width, label='Test Accuracy')
        ax4.bar(x + width/2, df_comparison['CV Accuracy'], width, label='CV Accuracy')
        ax4.set_title('Accuracy Comparison')
        ax4.set_ylabel('Accuracy')
        ax4.set_xticks(x)
        ax4.set_xticklabels(df_comparison['Model'], rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('improved_model_comparison.png', dpi=300, bbox_inches='tight')
        print("  📊 Saved improved model comparison plot to 'improved_model_comparison.png'")
        
        return df_comparison
    
    def improve_all_models(self):
        """Main improvement pipeline"""
        print("🚀 Starting Comprehensive Model Improvement...")
        print("⏱️  Expected time: 20-40 minutes")
        
        # Load dataset with ALL features
        df, labels, file_df = self.load_dataset_full_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"  📊 Training set: {X_train.shape[0]} samples")
        print(f"  📊 Test set: {X_test.shape[0]} samples")
        print(f"  🔍 Features: {X_train.shape[1]} (ALL features)")
        
        # Train all improved models
        self.train_improved_original_model(X_train, X_test, y_train, y_test)
        self.train_high_accuracy_improved_model(X_train, X_test, y_train, y_test)
        self.train_full_feature_custom_tree(X_train, X_test, y_train, y_test)
        self.train_ensemble_model(X_train, X_test, y_train, y_test)
        
        # Compare models
        comparison = self.compare_models()
        
        print("\n🎉 Model improvement completed successfully!")
        print("📁 Models saved:")
        print("  - model.pkl (Improved Random Forest)")
        print("  - improved_model.pkl (High-Accuracy Logistic Regression)")
        print("  - custom_tree_model.pkl (Full-Feature Custom Tree)")
        print("  - ensemble_model.pkl (Ensemble)")
        print("  - improved_model_comparison.png (Visualization)")
        
        return comparison

if __name__ == "__main__":
    improver = ModelImprover()
    improver.improve_all_models() 