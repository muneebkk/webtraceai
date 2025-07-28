# Muneeb: Feature extraction from images using OpenCV
# Focus: Extract 15 comprehensive features for AI vs Human website classification

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import io
from PIL import Image
import json

class FeatureExtractor:
    def __init__(self):
        pass
    
    def extract_image_features(self, image_data) -> Dict:
        """
        Extract comprehensive visual features from website screenshot
        Returns 15 features for AI vs Human classification
        """
        try:
            # Convert to OpenCV format
            if isinstance(image_data, io.BytesIO):
                # Convert PIL to OpenCV
                pil_img = Image.open(image_data)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            elif isinstance(image_data, Image.Image):
                # Convert PIL to OpenCV
                img = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
            else:
                img = image_data
            
            features = {}
            
            # Extract all feature categories
            features.update(self._extract_basic_features(img))
            features.update(self._extract_color_features(img))
            features.update(self._extract_layout_features(img))
            features.update(self._extract_texture_features(img))
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            # Return default features if extraction fails
            return self._get_default_features()
    
    def _extract_basic_features(self, img: np.ndarray) -> Dict:
        """Extract basic image properties"""
        features = {}
        
        height, width = img.shape[:2]
        features['width'] = width
        features['height'] = height
        features['aspect_ratio'] = width / height if height > 0 else 1.0
        features['total_pixels'] = width * height
        
        return features
    
    def _extract_color_features(self, img: np.ndarray) -> Dict:
        """Extract color-based features (HSV, diversity, dominant colors)"""
        features = {}
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 1. Color diversity (standard deviation of saturation and value)
        features['color_diversity_s'] = np.std(hsv[:, :, 1])  # Saturation std
        features['color_diversity_v'] = np.std(hsv[:, :, 2])  # Value std
        
        # 2. Dominant color analysis
        # Reshape image to 2D array of pixels
        pixels = hsv.reshape(-1, 3)
        
        # Calculate color histogram
        h_bins = 30
        s_bins = 32
        v_bins = 32
        
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [h_bins, s_bins, v_bins], 
                           [0, 180, 0, 256, 0, 256])
        
        # Normalize histogram
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        # 3. Color uniformity (entropy of color distribution)
        hist_flat = hist.flatten()
        hist_flat = hist_flat[hist_flat > 0]  # Remove zero bins
        if len(hist_flat) > 0:
            features['color_uniformity'] = -np.sum(hist_flat * np.log2(hist_flat))
        else:
            features['color_uniformity'] = 0
        
        # 4. Average saturation and brightness
        features['avg_saturation'] = np.mean(hsv[:, :, 1])
        features['avg_brightness'] = np.mean(hsv[:, :, 2])
        
        return features
    
    def _extract_layout_features(self, img: np.ndarray) -> Dict:
        """Extract layout-based features (edges, contours, spatial distribution)"""
        features = {}
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 5. Edge density (Canny edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        features['edge_density'] = edge_pixels / (img.shape[0] * img.shape[1])
        
        # 6. Contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Number of contours
            features['contour_count'] = len(contours)
            
            # Average contour area
            contour_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
            features['avg_contour_area'] = np.mean(contour_areas) if contour_areas else 0
            
            # Contour complexity (perimeter/area ratio)
            complexities = []
            for c in contours:
                area = cv2.contourArea(c)
                if area > 10:
                    perimeter = cv2.arcLength(c, True)
                    complexity = perimeter / area if area > 0 else 0
                    complexities.append(complexity)
            
            features['avg_contour_complexity'] = np.mean(complexities) if complexities else 0
        else:
            features['contour_count'] = 0
            features['avg_contour_area'] = 0
            features['avg_contour_complexity'] = 0
        
        # 7. Spatial distribution (horizontal vs vertical lines)
        # Sobel operators for horizontal and vertical edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate horizontal vs vertical edge ratio
        horizontal_edges = np.sum(np.abs(sobelx))
        vertical_edges = np.sum(np.abs(sobely))
        
        features['horizontal_vertical_ratio'] = horizontal_edges / vertical_edges if vertical_edges > 0 else 1.0
        
        return features
    
    def _extract_texture_features(self, img: np.ndarray) -> Dict:
        """Extract texture-based features (gradients, uniformity, patterns)"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 8. Gradient magnitude (texture intensity)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features['gradient_magnitude'] = np.mean(gradient_magnitude)
        
        # 9. Local Binary Pattern (LBP) for texture analysis
        # Simplified LBP implementation
        lbp = self._compute_lbp(gray)
        features['texture_uniformity'] = np.std(lbp)
        
        # 10. Gray-level co-occurrence matrix features (simplified)
        # Calculate local variance as texture measure
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        features['local_variance'] = np.mean(local_variance)
        
        # 11. Frequency domain analysis (FFT)
        # Apply FFT to detect patterns
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Calculate energy in different frequency bands
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Low frequency energy (center region)
        low_freq = magnitude_spectrum[center_h-20:center_h+20, center_w-20:center_w+20]
        features['low_freq_energy'] = np.mean(low_freq)
        
        # High frequency energy (edges) - use a mask approach
        mask = np.zeros_like(magnitude_spectrum)
        mask[center_h-20:center_h+20, center_w-20:center_w+20] = 1
        high_freq = magnitude_spectrum * (1 - mask)
        features['high_freq_energy'] = np.mean(high_freq)
        
        return features
    
    def _compute_lbp(self, gray_img: np.ndarray) -> np.ndarray:
        """Compute Local Binary Pattern for texture analysis"""
        lbp = np.zeros_like(gray_img)
        
        for i in range(1, gray_img.shape[0] - 1):
            for j in range(1, gray_img.shape[1] - 1):
                center = gray_img[i, j]
                code = 0
                
                # 8-neighbor LBP
                neighbors = [
                    gray_img[i-1, j-1], gray_img[i-1, j], gray_img[i-1, j+1],
                    gray_img[i, j+1], gray_img[i+1, j+1], gray_img[i+1, j],
                    gray_img[i+1, j-1], gray_img[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        return lbp
    
    def _get_default_features(self) -> Dict:
        """Return default features if extraction fails"""
        return {
            'width': 800,
            'height': 600,
            'aspect_ratio': 1.33,
            'total_pixels': 480000,
            'color_diversity_s': 50.0,
            'color_diversity_v': 50.0,
            'color_uniformity': 5.0,
            'avg_saturation': 100.0,
            'avg_brightness': 150.0,
            'edge_density': 0.1,
            'contour_count': 10,
            'avg_contour_area': 1000.0,
            'avg_contour_complexity': 0.5,
            'horizontal_vertical_ratio': 1.0,
            'gradient_magnitude': 50.0,
            'texture_uniformity': 30.0,
            'local_variance': 500.0,
            'low_freq_energy': 10.0,
            'high_freq_energy': 5.0
        }
    
    def get_feature_names(self) -> List[str]:
        """Return list of all feature names"""
        return [
            'width', 'height', 'aspect_ratio', 'total_pixels',
            'color_diversity_s', 'color_diversity_v', 'color_uniformity',
            'avg_saturation', 'avg_brightness', 'edge_density',
            'contour_count', 'avg_contour_area', 'avg_contour_complexity',
            'horizontal_vertical_ratio', 'gradient_magnitude',
            'texture_uniformity', 'local_variance', 'low_freq_energy',
            'high_freq_energy'
        ]
    
 