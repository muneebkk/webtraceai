import os
import pickle
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from app.feature_extract import FeatureExtractor
from custom_tree_model import CustomTreeTrainer

class FastModelRetrainer:
    def __init__(self):
        self.dataset_path = "../dataset"
        self.extractor = FeatureExtractor()
        self.models = {}
        self.results = {}
        
    def load_dataset(self):
        """Load and prepare dataset with meaningful features only"""
        print("📊 Loading dataset...")
        
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
        
        # Extract features
        features_list = []
        labels = []
        file_paths = []
        
        # Process AI images
        print(f"  🔄 Processing {len(ai_images)} AI images...")
        for i, img_path in enumerate(ai_images, 1):
            try:
                # Load image using OpenCV
                img = cv2.imread(img_path)
                if img is not None:
                    features = self.extractor.extract_meaningful_features(img)
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
                # Load image using OpenCV
                img = cv2.imread(img_path)
                if img is not None:
                    features = self.extractor.extract_meaningful_features(img)
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
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list, columns=self.extractor.get_meaningful_feature_names())
        labels = np.array(labels)
        file_df = pd.DataFrame({'file_path': file_paths, 'label': labels})
        
        print(f"  ✅ Successfully extracted features from {len(df)} images")
        print(f"  📈 Dataset shape: {df.shape}")
        print(f"  🎯 Class distribution: {np.bincount(labels)}")
        
        return df, labels, file_df
    
    def train_original_model(self, X_train, X_test, y_train, y_test):
        """Train Random Forest with good defaults"""
        print("\n🌲 Training Original Model (Random Forest)...")
        
        # Use good defaults instead of grid search
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        print(f"  ✅ Training completed")
        print(f"  📊 Test Accuracy: {accuracy:.4f}")
        print(f"  📊 Test AUC: {auc:.4f}")
        print(f"  📊 CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save model
        model_data = {
            'model': model,
            'accuracy': accuracy,
            'auc_score': auc,
            'cv_scores': cv_scores,
            'feature_names': self.extractor.get_meaningful_feature_names()
        }
        
        with open('model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        self.models['original'] = model
        self.results['original'] = {
            'accuracy': accuracy,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        return model
    
    def train_improved_model(self, X_train, X_test, y_train, y_test):
        """Train Logistic Regression with feature selection and scaling"""
        print("\n📈 Training Improved Model (Logistic Regression)...")
        
        # Identify and exclude font-related features
        feature_names = self.extractor.get_meaningful_feature_names()
        font_features = [name for name in feature_names if 'font' in name.lower()]
        non_font_indices = [i for i, name in enumerate(feature_names) if 'font' not in name.lower()]
        
        print(f"  🚫 Excluding {len(font_features)} font features: {font_features}")
        
        # Use only non-font features
        X_train_no_font = X_train.iloc[:, non_font_indices]
        X_test_no_font = X_test.iloc[:, non_font_indices]
        feature_names_no_font = [feature_names[i] for i in non_font_indices]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_no_font)
        X_test_scaled = scaler.transform(X_test_no_font)
        
        # Balance classes
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        # Train with good defaults
        model = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            random_state=42,
            max_iter=1000
        )
        
        model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        print(f"  ✅ Training completed")
        print(f"  📊 Test Accuracy: {accuracy:.4f}")
        print(f"  📊 Test AUC: {auc:.4f}")
        print(f"  📊 CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save model
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_mask': non_font_indices,
            'feature_names': feature_names_no_font,
            'accuracy': accuracy,
            'auc_score': auc,
            'cv_scores': cv_scores,
            'balancing_strategy': 'SMOTE',
            'scaling_method': 'StandardScaler'
        }
        
        with open('improved_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        self.models['improved'] = model
        self.results['improved'] = {
            'accuracy': accuracy,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        return model
    
    def train_custom_tree_model(self, X_train, X_test, y_train, y_test):
        """Train Custom Decision Tree"""
        print("\n🌳 Training Custom Tree Model...")
        
        # Use good defaults
        trainer = CustomTreeTrainer()
        model = trainer.train_model(
            X_train, y_train,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        print(f"  ✅ Training completed")
        print(f"  📊 Test Accuracy: {accuracy:.4f}")
        print(f"  📊 Test AUC: {auc:.4f}")
        print(f"  📊 CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save model
        model_data = {
            'model': model,
            'accuracy': accuracy,
            'auc_score': auc,
            'cv_scores': cv_scores,
            'feature_names': self.extractor.get_meaningful_feature_names()
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
        """Train Ensemble Model combining all three"""
        print("\n🎯 Training Ensemble Model...")
        
        # Create ensemble with the three trained models
        ensemble = VotingClassifier(
            estimators=[
                ('rf', self.models['original']),
                ('lr', self.models['improved']),
                ('dt', self.models['custom_tree'])
            ],
            voting='soft',
            weights=[0.4, 0.3, 0.3]
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, ensemble.predict_proba(X_test)[:, 1])
        
        # Cross-validation
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
            'feature_names': self.extractor.get_meaningful_feature_names()
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
        print("\n📊 Comparing Models...")
        
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
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Test Accuracy
        ax1.bar(df_comparison['Model'], df_comparison['Test Accuracy'])
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Test AUC
        ax2.bar(df_comparison['Model'], df_comparison['Test AUC'])
        ax2.set_title('Test AUC Comparison')
        ax2.set_ylabel('AUC')
        ax2.tick_params(axis='x', rotation=45)
        
        # CV Accuracy with error bars
        ax3.bar(df_comparison['Model'], df_comparison['CV Accuracy'], 
                yerr=df_comparison['CV Std'], capsize=5)
        ax3.set_title('Cross-Validation Accuracy')
        ax3.set_ylabel('Accuracy')
        ax3.tick_params(axis='x', rotation=45)
        
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
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("  📊 Saved model comparison plot to 'model_comparison.png'")
        
        return df_comparison
    
    def retrain_all_models(self):
        """Main training pipeline"""
        print("🚀 Starting Fast Model Training...")
        print("⏱️  Expected time: 15-30 minutes")
        
        # Load dataset
        df, labels, file_df = self.load_dataset()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"  📊 Training set: {X_train.shape[0]} samples")
        print(f"  📊 Test set: {X_test.shape[0]} samples")
        
        # Train all models
        self.train_original_model(X_train, X_test, y_train, y_test)
        self.train_improved_model(X_train, X_test, y_train, y_test)
        self.train_custom_tree_model(X_train, X_test, y_train, y_test)
        self.train_ensemble_model(X_train, X_test, y_train, y_test)
        
        # Compare models
        comparison = self.compare_models()
        
        print("\n🎉 Training completed successfully!")
        print("📁 Models saved:")
        print("  - model.pkl (Random Forest)")
        print("  - improved_model.pkl (Logistic Regression)")
        print("  - custom_tree_model.pkl (Custom Decision Tree)")
        print("  - ensemble_model.pkl (Ensemble)")
        print("  - model_comparison.png (Visualization)")
        
        return comparison

if __name__ == "__main__":
    retrainer = FastModelRetrainer()
    retrainer.retrain_all_models() 