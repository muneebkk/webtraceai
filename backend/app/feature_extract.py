# Muneeb: Feature extraction from images and HTML
# Focus: Write scripts to extract features from screenshots (OpenCV, NumPy, etc.)
# Store features in usable format (CSV, NumPy array, etc.)

from typing import Dict, List, Optional, Tuple
import io
from PIL import Image
import json

class FeatureExtractor:
    def __init__(self):
        pass
    
    def extract_image_features(self, image_data) -> Dict:
        """
        Extract basic visual features from website screenshot
        Simplified version without ML dependencies
        """
        try:
            # Convert to PIL Image for basic analysis
            if isinstance(image_data, io.BytesIO):
                img = Image.open(image_data)
            else:
                img = image_data
            
            features = {}
            
            # Basic image features
            features.update(self._extract_basic_features(img))
            
            return features
            
        except Exception as e:
            # Return default features if extraction fails
            return {
                'width': 800,
                'height': 600,
                'aspect_ratio': 1.33,
                'mode': 'RGB',
                'format': 'PNG',
                'size_bytes': 0
            }
    
    def _extract_basic_features(self, img: Image.Image) -> Dict:
        """Extract basic image features using PIL"""
        features = {}
        
        # Basic image properties
        features['width'] = img.width
        features['height'] = img.height
        features['aspect_ratio'] = img.width / img.height if img.height > 0 else 1.0
        features['mode'] = img.mode
        features['format'] = img.format or 'Unknown'
        
        # Get image size in bytes (approximate)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        features['size_bytes'] = len(img_byte_arr.getvalue())
        
        return features
    
 