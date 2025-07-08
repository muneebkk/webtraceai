import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from PIL import Image
import io
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv

class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def extract_all_features(
        self, 
        image: Image.Image, 
        html_content: Optional[str] = None,
        visible_text: Optional[str] = None
    ) -> Dict:
        """
        Extract all features from image, HTML, and text
        """
        features = {}
        
        # Image-based features
        features.update(self.extract_image_features(image))
        
        # HTML structure features
        if html_content:
            features.update(self.extract_html_features(html_content))
        
        # Text-based features
        if visible_text:
            features.update(self.extract_text_features(visible_text))
        
        return features
    
    def extract_image_features(self, image: Image.Image) -> Dict:
        """
        Extract visual features from website screenshot
        """
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        features = {}
        
        # Color features
        features.update(self._extract_color_features(img_cv))
        
        # Layout features
        features.update(self._extract_layout_features(img_cv))
        
        # Texture features
        features.update(self._extract_texture_features(img_cv))
        
        return features
    
    def _extract_color_features(self, img: np.ndarray) -> Dict:
        """Extract color-based features"""
        features = {}
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Color statistics
        features['color_mean_h'] = np.mean(hsv[:, :, 0])
        features['color_mean_s'] = np.mean(hsv[:, :, 1])
        features['color_mean_v'] = np.mean(hsv[:, :, 2])
        features['color_std_h'] = np.std(hsv[:, :, 0])
        features['color_std_s'] = np.std(hsv[:, :, 1])
        features['color_std_v'] = np.std(hsv[:, :, 2])
        
        # Color diversity (number of unique colors)
        unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
        features['color_diversity'] = unique_colors
        
        # Dominant colors
        pixels = img.reshape(-1, 3)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        features['dominant_colors'] = len(np.unique(kmeans.labels_))
        
        return features
    
    def _extract_layout_features(self, img: np.ndarray) -> Dict:
        """Extract layout-based features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features['contour_count'] = len(contours)
        
        # Aspect ratio
        height, width = img.shape[:2]
        features['aspect_ratio'] = width / height
        
        # White space analysis
        white_pixels = np.sum(gray > 240)
        total_pixels = gray.shape[0] * gray.shape[1]
        features['white_space_ratio'] = white_pixels / total_pixels
        
        return features
    
    def _extract_texture_features(self, img: np.ndarray) -> Dict:
        """Extract texture-based features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern (simplified)
        lbp = self._local_binary_pattern(gray)
        features['texture_uniformity'] = len(np.unique(lbp))
        
        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        
        return features
    
    def _local_binary_pattern(self, img: np.ndarray) -> np.ndarray:
        """Compute Local Binary Pattern"""
        # Simplified LBP implementation
        patterns = np.zeros_like(img)
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                center = img[i, j]
                pattern = 0
                for k, (di, dj) in enumerate([(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]):
                    if img[i+di, j+dj] >= center:
                        pattern |= (1 << k)
                patterns[i, j] = pattern
        return patterns
    
    def extract_html_features(self, html_content: str) -> Dict:
        """Extract features from HTML structure"""
        features = {}
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Tag counts
        features['div_count'] = len(soup.find_all('div'))
        features['span_count'] = len(soup.find_all('span'))
        features['p_count'] = len(soup.find_all('p'))
        features['img_count'] = len(soup.find_all('img'))
        features['link_count'] = len(soup.find_all('a'))
        
        # Class and ID analysis
        elements_with_class = soup.find_all(attrs={'class': True})
        elements_with_id = soup.find_all(attrs={'id': True})
        
        features['elements_with_class'] = len(elements_with_class)
        features['elements_with_id'] = len(elements_with_id)
        
        # Class name patterns (AI tools often use specific naming conventions)
        class_names = []
        for element in elements_with_class:
            class_names.extend(element.get('class', []))
        
        # Check for common AI tool class patterns
        ai_patterns = ['framer', 'wix', 'notion', 'durable', 'cursor', 'vercel', 'ai-generated']
        features['ai_class_patterns'] = sum(1 for pattern in ai_patterns 
                                          if any(pattern in class_name.lower() for class_name in class_names))
        
        # HTML complexity
        features['html_length'] = len(html_content)
        features['tag_diversity'] = len(set(tag.name for tag in soup.find_all()))
        
        return features
    
    def extract_text_features(self, text: str) -> Dict:
        """Extract features from visible text content"""
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        
        # Average word length
        words = text.split()
        if words:
            features['avg_word_length'] = np.mean([len(word) for word in words])
        else:
            features['avg_word_length'] = 0
        
        # Text complexity
        features['unique_words'] = len(set(words))
        if words:
            features['lexical_diversity'] = features['unique_words'] / len(words)
        else:
            features['lexical_diversity'] = 0
        
        # Check for AI-generated text patterns
        ai_indicators = [
            'welcome to', 'get started', 'learn more', 'contact us',
            'about us', 'our services', 'portfolio', 'testimonials'
        ]
        features['ai_text_indicators'] = sum(1 for indicator in ai_indicators 
                                           if indicator.lower() in text.lower())
        
        return features 