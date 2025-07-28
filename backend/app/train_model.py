# Muneeb: Model training pipeline
# Focus: Train classifiers using scikit-learn (KNN, Decision Tree, etc.)
# Evaluate and save trained models using joblib

import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from .feature_extract import FeatureExtractor
from .model_loader import ModelLoader

class ModelTrainer:
    def __init__(self, dataset_path: str = "../dataset"):
        self.dataset_path = dataset_path
        self.ai_images_path = os.path.join(dataset_path, "images", "ai")
        self.human_images_path = os.path.join(dataset_path, "images", "human")
        self.feature_extractor = FeatureExtractor()
        self.model_loader = ModelLoader()
        self.scaler = StandardScaler()
        
    def load_dataset(self):
        """Load the dataset from CSV and images"""
        labels_df = pd.read_csv(os.path.join(self.dataset_path, "labels.csv"))
        
        features_list = []
        labels_list = []
        
        for _, row in labels_df.iterrows():
            site_id = row['id']
            tool = row['tool']
            
            # Determine image path based on tool
            if tool == "human":
                image_path = os.path.join(self.human_images_path, f"{site_id}.png")
            else:
                image_path = os.path.join(self.ai_images_path, f"{site_id}.png")
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image not found for {site_id}")
                continue
            
            # Extract features from image only
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not load image {image_path}")
                    continue
                features = self.feature_extractor.extract_image_features(image)
                
                features_list.append(features)
                labels_list.append(tool)
                print(f"Processed {site_id} ({tool})")
                
            except Exception as e:
                print(f"Error processing {site_id}: {e}")
                continue
        
        return features_list, labels_list
    
    def prepare_training_data(self, features_list, labels_list):
        """Convert features to training matrix"""
        # Convert features to feature vectors
        feature_vectors = []
        for features in features_list:
            feature_vector = self._features_to_vector(features)
            feature_vectors.append(feature_vector)
        
        return np.array(feature_vectors), np.array(labels_list)
    
    def _features_to_vector(self, features: dict) -> list:
        """Convert features dict to feature vector (image features only)"""
        # Define expected image feature names
        expected_features = [
            # Color features
            'color_mean_h', 'color_mean_s', 'color_mean_v',
            'color_std_h', 'color_std_s', 'color_std_v',
            'color_diversity', 'dominant_colors',
            
            # Layout features
            'edge_density', 'contour_count', 'aspect_ratio', 'white_space_ratio',
            
            # Texture features
            'texture_uniformity', 'gradient_mean', 'gradient_std'
        ]
        
        # Create feature vector with default values
        feature_vector = []
        for feature_name in expected_features:
            if feature_name in features:
                feature_vector.append(float(features[feature_name]))
            else:
                # Use default value for missing features
                feature_vector.append(0.0)
        
        return feature_vector
    
    def train_and_evaluate(self):
        """Train the model and evaluate performance"""
        print("Loading dataset...")
        features_list, labels_list = self.load_dataset()
        
        if len(features_list) == 0:
            print("No valid samples found in dataset!")
            return
        
        print(f"Loaded {len(features_list)} samples")
        
        # Prepare training data
        X, y = self.prepare_training_data(features_list, labels_list)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training model...")
        self.model_loader.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model_loader.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model_loader.model, X_train_scaled, y_train, cv=5)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Cross-validation mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model and scaler
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model_loader.model, "models/webtrace_model.joblib")
        joblib.dump(self.scaler, "models/webtrace_scaler.joblib")
        print(f"\nModel saved to models/webtrace_model.joblib")
        print(f"Scaler saved to models/webtrace_scaler.joblib")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # Feature importance analysis
        self.analyze_feature_importance(X_train_scaled, y_train)
        
        return {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "classification_report": classification_report(y_test, y_pred),
            "n_samples": len(features_list)
        }
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - AI vs Human Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        print("Confusion matrix saved as confusion_matrix.png")
    
    def analyze_feature_importance(self, X_train, y_train):
        """Analyze feature importance"""
        if hasattr(self.model_loader.model, 'feature_importances_'):
            # For tree-based models
            importances = self.model_loader.model.feature_importances_
        elif hasattr(self.model_loader.model, 'named_estimators_'):
            # For ensemble models, use the first estimator
            first_estimator = list(self.model_loader.model.named_estimators_.values())[0]
            if hasattr(first_estimator, 'feature_importances_'):
                importances = first_estimator.feature_importances_
            else:
                return
        else:
            return
        
        # Get feature names (image features only)
        feature_names = [
            'color_mean_h', 'color_mean_s', 'color_mean_v',
            'color_std_h', 'color_std_s', 'color_std_v',
            'color_diversity', 'dominant_colors',
            'edge_density', 'contour_count', 'aspect_ratio', 'white_space_ratio',
            'texture_uniformity', 'gradient_mean', 'gradient_std'
        ]
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top 15 features
        plt.figure(figsize=(10, 8))
        top_features = feature_importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Most Important Image Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        print("Feature importance plot saved as feature_importance.png")
        
        # Save feature importance to CSV
        feature_importance_df.to_csv('feature_importance.csv', index=False)
        print("Feature importance saved to feature_importance.csv")

def main():
    """Main training function"""
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate()
    
    if results:
        print(f"\nTraining completed successfully!")
        print(f"Total samples used: {results['n_samples']}")
        print(f"Model accuracy: {results['accuracy']:.3f}")
        print(f"Cross-validation mean: {results['cv_mean']:.3f}")

if __name__ == "__main__":
    main() 