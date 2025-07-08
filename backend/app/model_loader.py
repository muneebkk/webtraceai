import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import json

class ModelLoader:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "models/webtrace_model.joblib"
        self.model = None
        self.feature_names = None
        self.class_names = [
            "Framer AI", "Wix ADI", "Notion AI", "Durable", 
            "Cursor", "v0 by Vercel", "ChatGPT HTML Generator", "Human"
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained model from disk or create a default one"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"Model loaded from {self.model_path}")
            else:
                print("No trained model found. Creating default model...")
                self.create_default_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.create_default_model()
    
    def create_default_model(self):
        """Create a default model for demonstration purposes"""
        # Create a simple ensemble model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        lr = LogisticRegression(random_state=42, max_iter=1000)
        knn = KNeighborsClassifier(n_neighbors=5)
        dt = DecisionTreeClassifier(random_state=42)
        
        self.model = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('lr', lr),
                ('knn', knn),
                ('dt', dt)
            ],
            voting='soft'
        )
        
        # For now, we'll use dummy data to fit the model
        # In production, this would be replaced with actual training data
        self._fit_with_dummy_data()
    
    def _fit_with_dummy_data(self):
        """Fit model with dummy data for demonstration"""
        # Generate dummy features and labels
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        
        # Create dummy feature data
        X = np.random.randn(n_samples, n_features)
        
        # Create dummy labels (8 classes)
        y = np.random.randint(0, 8, n_samples)
        
        # Fit the model
        self.model.fit(X, y)
        
        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Default model saved to {self.model_path}")
    
    def predict(self, features: Dict) -> Dict:
        """
        Make prediction using the loaded model
        """
        try:
            # Convert features dict to feature vector
            feature_vector = self._features_to_vector(features)
            
            # Make prediction
            prediction_proba = self.model.predict_proba([feature_vector])[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(prediction_proba)
            predicted_class = self.class_names[predicted_class_idx]
            
            # Determine if AI-generated (not Human)
            is_ai_generated = predicted_class != "Human"
            confidence = prediction_proba[predicted_class_idx]
            
            # Create tool probabilities dictionary
            tool_probabilities = {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, prediction_proba)
            }
            
            # Get features used (for debugging/explanation)
            features_used = list(features.keys())
            
            return {
                "is_ai_generated": is_ai_generated,
                "confidence": float(confidence),
                "predicted_tool": predicted_class,
                "tool_probabilities": tool_probabilities,
                "features_used": features_used
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return default prediction
            return {
                "is_ai_generated": False,
                "confidence": 0.5,
                "predicted_tool": "Human",
                "tool_probabilities": {"Human": 1.0},
                "features_used": []
            }
    
    def _features_to_vector(self, features: Dict) -> List[float]:
        """
        Convert features dictionary to feature vector
        """
        # Define expected feature names (should match what FeatureExtractor produces)
        expected_features = [
            # Color features
            'color_mean_h', 'color_mean_s', 'color_mean_v',
            'color_std_h', 'color_std_s', 'color_std_v',
            'color_diversity', 'dominant_colors',
            
            # Layout features
            'edge_density', 'contour_count', 'aspect_ratio', 'white_space_ratio',
            
            # Texture features
            'texture_uniformity', 'gradient_mean', 'gradient_std',
            
            # HTML features
            'div_count', 'span_count', 'p_count', 'img_count', 'link_count',
            'elements_with_class', 'elements_with_id', 'ai_class_patterns',
            'html_length', 'tag_diversity',
            
            # Text features
            'text_length', 'word_count', 'sentence_count', 'avg_word_length',
            'unique_words', 'lexical_diversity', 'ai_text_indicators'
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
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            "model_type": type(self.model).__name__,
            "supported_tools": self.class_names,
            "feature_count": len(self._get_expected_features()),
            "model_path": self.model_path,
            "is_trained": self.model is not None
        }
    
    def _get_expected_features(self) -> List[str]:
        """Get list of expected feature names"""
        return [
            'color_mean_h', 'color_mean_s', 'color_mean_v',
            'color_std_h', 'color_std_s', 'color_std_v',
            'color_diversity', 'dominant_colors',
            'edge_density', 'contour_count', 'aspect_ratio', 'white_space_ratio',
            'texture_uniformity', 'gradient_mean', 'gradient_std',
            'div_count', 'span_count', 'p_count', 'img_count', 'link_count',
            'elements_with_class', 'elements_with_id', 'ai_class_patterns',
            'html_length', 'tag_diversity',
            'text_length', 'word_count', 'sentence_count', 'avg_word_length',
            'unique_words', 'lexical_diversity', 'ai_text_indicators'
        ]
    
    def train_model(self, training_data: pd.DataFrame, labels: List[str]):
        """
        Train the model with new data
        """
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                training_data, labels, test_size=0.2, random_state=42
            )
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save the model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            
            return {
                "accuracy": accuracy,
                "classification_report": classification_report(y_test, y_pred)
            }
            
        except Exception as e:
            print(f"Training error: {e}")
            return {"error": str(e)} 