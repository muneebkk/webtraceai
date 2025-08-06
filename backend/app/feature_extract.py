# Muneeb: Enhanced Feature extraction from images using OpenCV
# Focus: Extract 43 comprehensive features for AI vs Human website classification

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import io
from PIL import Image
import json
from scipy import stats
from skimage import feature, filters, measure
import math

class FeatureExtractor:
    def __init__(self):
        # Basic properties (4) - NOT used for AI vs Human classification
        self.basic_properties = [
            'width', 'height', 'aspect_ratio', 'total_pixels'
        ]
        
        # Meaningful features for AI vs Human classification (39)
        self.meaningful_features = [
            # Color features (12)
            'color_diversity_s', 'color_diversity_v', 'color_uniformity',
            'avg_saturation', 'avg_brightness', 'color_entropy',
            'dominant_colors_count', 'color_contrast', 'color_harmony',
            'color_temperature', 'color_vibrancy', 'color_consistency',
            
            # Layout features (10)
            'edge_density', 'contour_count', 'layout_symmetry',
            'spatial_distribution', 'alignment_score', 'grid_structure',
            'white_space_ratio', 'content_density', 'layout_complexity',
            'visual_hierarchy',
            
            # Texture features (8)
            'texture_uniformity', 'texture_contrast', 'texture_entropy',
            'texture_correlation', 'texture_energy', 'texture_homogeneity',
            'texture_variance', 'texture_smoothness',
            
            # Structural features (9)
            'border_regularity', 'padding_consistency', 'margin_uniformity',
            'element_spacing', 'component_alignment', 'structural_symmetry',
            'design_patterns', 'layout_balance', 'visual_flow'
        ]
        
        # All feature names (for backward compatibility)
        self.feature_names = self.basic_properties + self.meaningful_features
        
        # Legacy feature names for backward compatibility
        self.legacy_feature_names = [
            'width', 'height', 'aspect_ratio', 'total_pixels', 
            'color_diversity_s', 'color_diversity_v', 'color_uniformity', 
            'avg_saturation', 'avg_brightness', 'edge_density', 
            'contour_count', 'avg_contour_area', 'avg_contour_complexity', 
            'horizontal_vertical_ratio', 'gradient_magnitude', 
            'texture_uniformity', 'local_variance', 'low_freq_energy', 
            'high_freq_energy'
        ]
    
    def extract_image_features(self, image_data) -> Dict:
        """
        Extract comprehensive visual features from website screenshot
        Returns 43 features for AI vs Human classification
        """
        try:
            # Convert to OpenCV format
            if isinstance(image_data, io.BytesIO):
                pil_img = Image.open(image_data)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            elif isinstance(image_data, Image.Image):
                img = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
            else:
                img = image_data
            
            # Validate image
            if img is None or img.size == 0:
                print("Warning: Invalid image data, using default features")
                return self._get_default_features()
            
            # Additional validation for image dimensions
            if len(img.shape) < 2 or img.shape[0] <= 0 or img.shape[1] <= 0:
                print("Warning: Invalid image dimensions, using default features")
                return self._get_default_features()
            
            features = {}
            
            # Extract all feature categories with individual error handling
            try:
                features.update(self._extract_basic_features(img))
            except Exception as e:
                print(f"Basic feature extraction error: {e}")
                features.update({
                    'width': 1920.0, 'height': 1080.0, 'aspect_ratio': 16.0/9.0, 'total_pixels': 1920.0 * 1080.0
                })
            
            try:
                features.update(self._extract_color_features(img))
            except Exception as e:
                print(f"Color feature extraction error: {e}")
                features.update({
                    'color_diversity_s': 0.0, 'color_diversity_v': 0.0, 'color_uniformity': 0.0,
                    'avg_saturation': 0.0, 'avg_brightness': 0.0, 'color_entropy': 0.0,
                    'dominant_colors_count': 5.0, 'color_contrast': 0.0, 'color_harmony': 0.0,
                    'color_temperature': 0.0, 'color_vibrancy': 0.0, 'color_consistency': 1.0
                })
            
            try:
                features.update(self._extract_layout_features(img))
            except Exception as e:
                print(f"Layout feature extraction error: {e}")
                features.update({
                    'edge_density': 0.0, 'contour_count': 0.0, 'layout_symmetry': 0.5,
                    'spatial_distribution': 0.0, 'alignment_score': 0.0, 'grid_structure': 0.0,
                    'white_space_ratio': 0.0, 'content_density': 0.0, 'layout_complexity': 0.0,
                    'visual_hierarchy': 0.0
                })
            
            try:
                features.update(self._extract_texture_features(img))
            except Exception as e:
                print(f"Texture feature extraction error: {e}")
                features.update({
                    'texture_uniformity': 1.0, 'texture_contrast': 0.0, 'texture_entropy': 0.0,
                    'texture_correlation': 0.0, 'texture_energy': 0.0, 'texture_homogeneity': 1.0,
                    'texture_variance': 0.0, 'texture_smoothness': 1.0
                })
            
            try:
                features.update(self._extract_structural_features(img))
            except Exception as e:
                print(f"Structural feature extraction error: {e}")
                features.update({
                    'border_regularity': 0.0, 'padding_consistency': 0.0, 'margin_uniformity': 0.0,
                    'element_spacing': 0.0, 'component_alignment': 0.0, 'structural_symmetry': 0.5,
                    'design_patterns': 0.0, 'layout_balance': 0.5, 'visual_flow': 0.0
                })
            
            # Ensure all features are present and are valid numbers
            for feature_name in self.feature_names:
                if feature_name not in features:
                    features[feature_name] = 0.0
                elif not np.isfinite(features[feature_name]):
                    features[feature_name] = 0.0
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return self._get_default_features()
    
    def extract_legacy_features(self, image_data) -> Dict:
        """
        Extract features compatible with old 19-feature models
        Maps new 43 features to legacy 19 features
        """
        try:
            # Get the full 43 features
            full_features = self.extract_image_features(image_data)
            
            # Map to legacy features
            legacy_features = {}
            
            # Direct mappings (features that exist in both)
            direct_mappings = [
                'width', 'height', 'aspect_ratio', 'total_pixels',
                'color_diversity_s', 'color_diversity_v', 'color_uniformity',
                'avg_saturation', 'avg_brightness', 'edge_density',
                'contour_count', 'texture_uniformity'
            ]
            
            for feature in direct_mappings:
                legacy_features[feature] = full_features.get(feature, 0.0)
            
            # Calculate legacy features that need computation
            try:
                if isinstance(image_data, str):
                    gray = cv2.cvtColor(cv2.imread(image_data), cv2.COLOR_BGR2GRAY)
                else:
                    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
                
                if gray is None:
                    raise ValueError("Failed to convert image to grayscale")
                
                # avg_contour_area
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 0]
                    legacy_features['avg_contour_area'] = float(np.mean(areas)) if areas else 0.0
                else:
                    legacy_features['avg_contour_area'] = 0.0
                
                # avg_contour_complexity
                if contours:
                    complexities = [len(c) for c in contours if len(c) > 0]
                    legacy_features['avg_contour_complexity'] = float(np.mean(complexities)) if complexities else 0.0
                else:
                    legacy_features['avg_contour_complexity'] = 0.0
                
                # horizontal_vertical_ratio
                h, w = gray.shape
                legacy_features['horizontal_vertical_ratio'] = float(w / h) if h > 0 else 1.0
                
                # gradient_magnitude
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                legacy_features['gradient_magnitude'] = float(np.mean(gradient_magnitude))
                
                # local_variance
                legacy_features['local_variance'] = float(np.var(gray))
                
                # low_freq_energy and high_freq_energy
                # Simple approximation using blur and edge detection
                blurred = cv2.GaussianBlur(gray, (15, 15), 0)
                edges = cv2.Canny(gray, 50, 150)
                
                legacy_features['low_freq_energy'] = float(np.mean(blurred))
                legacy_features['high_freq_energy'] = float(np.mean(edges))
                
            except Exception as e:
                print(f"Legacy feature computation error: {e}")
                # Set default values for computed features
                legacy_features.update({
                    'avg_contour_area': 0.0, 'avg_contour_complexity': 0.0,
                    'horizontal_vertical_ratio': 1.0, 'gradient_magnitude': 0.0,
                    'local_variance': 0.0, 'low_freq_energy': 0.0, 'high_freq_energy': 0.0
                })
            
            return legacy_features
            
        except Exception as e:
            print(f"Legacy feature extraction error: {e}")
            return {feature: 0.0 for feature in self.legacy_feature_names}
    
    def _extract_basic_features(self, img: np.ndarray) -> Dict:
        """Extract basic image properties"""
        features = {}
        
        height, width = img.shape[:2]
        features['width'] = float(width)
        features['height'] = float(height)
        features['aspect_ratio'] = float(width / height) if height > 0 else 1.0
        features['total_pixels'] = float(width * height)
        
        return features
    
    def _extract_color_features(self, img: np.ndarray) -> Dict:
        """Extract comprehensive color-based features"""
        features = {}
        
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Color diversity
            features['color_diversity_s'] = float(np.std(hsv[:, :, 1]))
            features['color_diversity_v'] = float(np.std(hsv[:, :, 2]))
            
            # 2. Color uniformity (entropy)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [30, 32, 32], [0, 180, 0, 256, 0, 256])
            hist_flat = hist.flatten()
            hist_flat = hist_flat[hist_flat > 0]
            if len(hist_flat) > 0:
                hist_sum = np.sum(hist_flat)
                if hist_sum > 0:
                    hist_normalized = hist_flat / hist_sum
                    features['color_uniformity'] = float(-np.sum(hist_normalized * np.log2(hist_normalized + 1e-10)))
                else:
                    features['color_uniformity'] = 0.0
            else:
                features['color_uniformity'] = 0.0
            
            # 3. Average saturation and brightness
            features['avg_saturation'] = float(np.mean(hsv[:, :, 1]))
            features['avg_brightness'] = float(np.mean(hsv[:, :, 2]))
            
            # 4. Color entropy
            hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_sum = hist_gray.sum()
            if hist_sum > 0:
                hist_gray = hist_gray.flatten() / hist_sum
                features['color_entropy'] = float(-np.sum(hist_gray * np.log2(hist_gray + 1e-10)))
            else:
                features['color_entropy'] = 0.0
            
            # 5. Dominant colors count (with memory optimization and validation)
            try:
                # Sample pixels to reduce memory usage for large images
                h, w = hsv.shape[:2]
                total_pixels = h * w
                
                if total_pixels < 5:
                    # For very small images, use all pixels but reduce clusters
                    pixels = hsv.reshape(-1, 3)
                    n_clusters = min(3, total_pixels)
                elif total_pixels > 1000000:  # If image is larger than 1M pixels
                    # Sample every 4th pixel
                    sample_indices = np.arange(0, total_pixels, 4)
                    pixels = hsv.reshape(-1, 3)[sample_indices]
                    n_clusters = 5
                else:
                    pixels = hsv.reshape(-1, 3)
                    n_clusters = 5
                
                from sklearn.cluster import KMeans
                if len(pixels) >= n_clusters:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    kmeans.fit(pixels)
                    features['dominant_colors_count'] = float(len(np.unique(kmeans.labels_)))
                else:
                    features['dominant_colors_count'] = float(len(pixels))
            except Exception as e:
                print(f"KMeans clustering error: {e}")
                features['dominant_colors_count'] = 5.0
            
            # 6. Color contrast
            features['color_contrast'] = float(np.std(gray))
            
            # 7. Color harmony (based on HSV hue distribution)
            hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            features['color_harmony'] = float(np.std(hue_hist))
            
            # 8. Color temperature (warm vs cool)
            features['color_temperature'] = float(np.mean(hsv[:, :, 0]))
            
            # 9. Color vibrancy
            features['color_vibrancy'] = float(np.mean(hsv[:, :, 1]) * np.mean(hsv[:, :, 2]) / 255.0)
            
            # 10. Color consistency
            sat_std = np.std(hsv[:, :, 1])
            features['color_consistency'] = float(1.0 / (1.0 + sat_std)) if sat_std > 0 else 1.0
            
        except Exception as e:
            print(f"Color feature extraction error: {e}")
            # Set default values
            features.update({
                'color_diversity_s': 0.0, 'color_diversity_v': 0.0, 'color_uniformity': 0.0,
                'avg_saturation': 0.0, 'avg_brightness': 0.0, 'color_entropy': 0.0,
                'dominant_colors_count': 5.0, 'color_contrast': 0.0, 'color_harmony': 0.0,
                'color_temperature': 0.0, 'color_vibrancy': 0.0, 'color_consistency': 1.0
            })
        
        return features
    
    def _extract_layout_features(self, img: np.ndarray) -> Dict:
        """Extract layout-based features"""
        features = {}
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Edge density
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = float(np.sum(edges > 0) / edges.size)
            
            # 2. Contour count
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            features['contour_count'] = float(len(contours))
            
            # 3. Layout symmetry
            height, width = gray.shape
            left_half = gray[:, :width//2]
            right_half = cv2.flip(gray[:, width//2:], 1)
            if left_half.shape == right_half.shape:
                symmetry_score = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
            else:
                symmetry_score = 0.5
            features['layout_symmetry'] = float(symmetry_score)
            
            # 4. Spatial distribution
            y_coords = np.where(edges > 0)[0]
            if len(y_coords) > 0:
                features['spatial_distribution'] = float(np.std(y_coords) / height)
            else:
                features['spatial_distribution'] = 0.0
            
            # 5. Alignment score
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
            if lines is not None:
                features['alignment_score'] = float(len(lines))
            else:
                features['alignment_score'] = 0.0
            
            # 6. Grid structure
            features['grid_structure'] = float(self._detect_grid_pattern(gray))
            
            # 7. White space ratio
            white_pixels = np.sum(gray > 240)
            features['white_space_ratio'] = float(white_pixels / gray.size)
            
            # 8. Content density
            content_pixels = np.sum(gray < 100)
            features['content_density'] = float(content_pixels / gray.size)
            
            # 9. Layout complexity
            features['layout_complexity'] = float(features['edge_density'] * features['contour_count'])
            
            # 10. Visual hierarchy
            features['visual_hierarchy'] = float(self._calculate_visual_hierarchy(gray))
            
        except Exception as e:
            print(f"Layout feature extraction error: {e}")
            features.update({
                'edge_density': 0.0, 'contour_count': 0.0, 'layout_symmetry': 0.5,
                'spatial_distribution': 0.0, 'alignment_score': 0.0, 'grid_structure': 0.0,
                'white_space_ratio': 0.0, 'content_density': 0.0, 'layout_complexity': 0.0,
                'visual_hierarchy': 0.0
            })
        
        return features
    
    def _extract_texture_features(self, img: np.ndarray) -> Dict:
        """Extract texture-based features"""
        features = {}
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Texture uniformity
            gray_std = np.std(gray)
            features['texture_uniformity'] = float(1.0 / (1.0 + gray_std)) if gray_std > 0 else 1.0
            
            # 2. Texture contrast
            features['texture_contrast'] = float(np.max(gray) - np.min(gray))
            
            # 3. Texture entropy
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist = hist.flatten() / hist_sum
                features['texture_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
            else:
                features['texture_entropy'] = 0.0
            
            # 4. Texture correlation
            features['texture_correlation'] = float(self._calculate_texture_correlation(gray))
            
            # 5. Texture energy
            features['texture_energy'] = float(np.sum(gray**2) / gray.size)
            
            # 6. Texture homogeneity
            features['texture_homogeneity'] = float(1.0 / (1.0 + gray_std)) if gray_std > 0 else 1.0
            
            # 7. Texture variance
            features['texture_variance'] = float(np.var(gray))
            
            # 8. Texture smoothness
            gray_var = np.var(gray)
            features['texture_smoothness'] = float(1.0 / (1.0 + gray_var)) if gray_var > 0 else 1.0
            
        except Exception as e:
            print(f"Texture feature extraction error: {e}")
            # Set default values
            features.update({
                'texture_uniformity': 1.0, 'texture_contrast': 0.0, 'texture_entropy': 0.0,
                'texture_correlation': 0.0, 'texture_energy': 0.0, 'texture_homogeneity': 1.0,
                'texture_variance': 0.0, 'texture_smoothness': 1.0
            })
        
        return features
    
    def _extract_structural_features(self, img: np.ndarray) -> Dict:
        """Extract structural design features"""
        features = {}
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Border regularity
            features['border_regularity'] = float(self._detect_border_regularity(gray))
            
            # 2. Padding consistency
            features['padding_consistency'] = float(self._detect_padding_consistency(gray))
            
            # 3. Margin uniformity
            features['margin_uniformity'] = float(self._detect_margin_uniformity(gray))
            
            # 4. Element spacing
            features['element_spacing'] = float(self._calculate_element_spacing(gray))
            
            # 5. Component alignment
            features['component_alignment'] = float(self._detect_component_alignment(gray))
            
            # 6. Structural symmetry
            features['structural_symmetry'] = float(self._calculate_structural_symmetry(gray))
            
            # 7. Design patterns
            features['design_patterns'] = float(self._detect_design_patterns(gray))
            
            # 8. Layout balance
            features['layout_balance'] = float(self._calculate_layout_balance(gray))
            
            # 9. Visual flow
            features['visual_flow'] = float(self._calculate_visual_flow(gray))
            
        except Exception as e:
            print(f"Structural feature extraction error: {e}")
            features.update({
                'border_regularity': 0.0, 'padding_consistency': 0.0, 'margin_uniformity': 0.0,
                'element_spacing': 0.0, 'component_alignment': 0.0, 'structural_symmetry': 0.5,
                'design_patterns': 0.0, 'layout_balance': 0.5, 'visual_flow': 0.0
            })
        
        return features
    
    def _detect_grid_pattern(self, gray: np.ndarray) -> float:
        """Detect grid-like patterns in the image"""
        try:
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
            if lines is not None:
                horizontal_lines = sum(1 for line in lines if abs(line[0][1]) < 0.1)
                vertical_lines = sum(1 for line in lines if abs(line[0][1] - np.pi/2) < 0.1)
                return float(horizontal_lines + vertical_lines)
            return 0.0
        except Exception as e:
            print(f"Grid pattern detection error: {e}")
            return 0.0
    
    def _calculate_visual_hierarchy(self, gray: np.ndarray) -> float:
        """Calculate visual hierarchy score"""
        try:
            # Use gradient magnitude as proxy for visual hierarchy
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            return float(np.mean(gradient_magnitude))
        except Exception as e:
            print(f"Visual hierarchy calculation error: {e}")
            return 0.0
    
    def _calculate_texture_correlation(self, gray: np.ndarray) -> float:
        """Calculate texture correlation"""
        try:
            # Simple correlation between adjacent pixels
            if gray.shape[0] > 1 and gray.shape[1] > 1:
                # Ensure we have enough data for correlation
                if gray.size < 4:
                    return 0.0
                    
                # Calculate correlation in x direction
                gray_x = gray[:-1, :].flatten()
                gray_x_next = gray[1:, :].flatten()
                
                # Calculate correlation in y direction  
                gray_y = gray[:, :-1].flatten()
                gray_y_next = gray[:, 1:].flatten()
                
                # Check if we have enough non-zero variance
                if np.var(gray_x) > 0 and np.var(gray_x_next) > 0:
                    corr_x = np.corrcoef(gray_x, gray_x_next)[0, 1]
                else:
                    corr_x = 0.0
                    
                if np.var(gray_y) > 0 and np.var(gray_y_next) > 0:
                    corr_y = np.corrcoef(gray_y, gray_y_next)[0, 1]
                else:
                    corr_y = 0.0
                
                # Handle NaN values
                if np.isnan(corr_x):
                    corr_x = 0.0
                if np.isnan(corr_y):
                    corr_y = 0.0
                    
                return float((corr_x + corr_y) / 2)
            return 0.0
        except Exception as e:
            print(f"Texture correlation calculation error: {e}")
            return 0.0
    
    def _detect_border_regularity(self, gray: np.ndarray) -> float:
        """Detect regularity of borders"""
        try:
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 0]
                if areas:
                    areas_std = np.std(areas)
                    return float(1.0 / (1.0 + areas_std)) if areas_std > 0 else 1.0
            return 0.0
        except Exception as e:
            print(f"Border regularity detection error: {e}")
            return 0.0
    
    def _detect_padding_consistency(self, gray: np.ndarray) -> float:
        """Detect consistency of padding around elements"""
        try:
            # Analyze edge distribution
            edges = cv2.Canny(gray, 50, 150)
            edge_density_by_region = []
            h, w = gray.shape
            
            # Ensure minimum region size
            if h < 3 or w < 3:
                return 1.0  # Default for very small images
                
            for i in range(3):
                for j in range(3):
                    region = edges[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
                    if region.size > 0:  # Ensure region is not empty
                        edge_density_by_region.append(np.sum(region > 0) / region.size)
                    else:
                        edge_density_by_region.append(0.0)
            
            if len(edge_density_by_region) > 0:
                edge_std = np.std(edge_density_by_region)
                return float(1.0 / (1.0 + edge_std)) if edge_std > 0 else 1.0
            else:
                return 1.0
        except Exception as e:
            print(f"Padding consistency detection error: {e}")
            return 0.0
    
    def _detect_margin_uniformity(self, gray: np.ndarray) -> float:
        """Detect uniformity of margins"""
        try:
            # Analyze border regions
            border_width = min(20, gray.shape[0]//4, gray.shape[1]//4)  # Ensure border width is reasonable
            if border_width <= 0:
                return 0.0
                
            top_border = gray[:border_width, :]
            bottom_border = gray[-border_width:, :]
            left_border = gray[:, :border_width]
            right_border = gray[:, -border_width:]
            
            border_means = [
                np.mean(top_border), np.mean(bottom_border),
                np.mean(left_border), np.mean(right_border)
            ]
            
            border_std = np.std(border_means)
            return float(1.0 / (1.0 + border_std)) if border_std > 0 else 1.0
        except Exception as e:
            print(f"Margin uniformity detection error: {e}")
            return 0.0
    
    def _calculate_element_spacing(self, gray: np.ndarray) -> float:
        """Calculate average spacing between elements"""
        try:
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 1:
                # Calculate distances between contour centroids
                centroids = []
                for contour in contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centroids.append((cx, cy))
                
                if len(centroids) > 1:
                    distances = []
                    for i in range(len(centroids)):
                        for j in range(i+1, len(centroids)):
                            dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                                         (centroids[i][1] - centroids[j][1])**2)
                            distances.append(dist)
                    return float(np.mean(distances)) if distances else 0.0
            return 0.0
        except Exception as e:
            print(f"Element spacing calculation error: {e}")
            return 0.0
    
    def _detect_component_alignment(self, gray: np.ndarray) -> float:
        """Detect alignment of components"""
        try:
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
            if lines is not None:
                angles = [line[0][1] for line in lines]
                # Check for alignment (similar angles)
                aligned_count = 0
                for i, angle1 in enumerate(angles):
                    for j, angle2 in enumerate(angles[i+1:], i+1):
                        if abs(angle1 - angle2) < 0.1:  # Similar angles
                            aligned_count += 1
                return float(aligned_count / max(1, len(angles)))
            return 0.0
        except Exception as e:
            print(f"Component alignment detection error: {e}")
            return 0.0
    
    def _calculate_structural_symmetry(self, gray: np.ndarray) -> float:
        """Calculate structural symmetry"""
        try:
            h, w = gray.shape
            left_half = gray[:, :w//2]
            right_half = cv2.flip(gray[:, w//2:], 1)
            
            if left_half.shape == right_half.shape:
                symmetry = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
            else:
                symmetry = 0.5
            
            return float(symmetry)
        except Exception as e:
            print(f"Structural symmetry calculation error: {e}")
            return 0.5
    
    def _detect_design_patterns(self, gray: np.ndarray) -> float:
        """Detect common design patterns"""
        try:
            # Look for repeated patterns
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 1:
                areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 0]
                if areas:
                    # Check for similar sized elements (potential patterns)
                    similar_count = 0
                    for i, area1 in enumerate(areas):
                        for j, area2 in enumerate(areas[i+1:], i+1):
                            max_area = max(area1, area2)
                            if max_area > 0 and abs(area1 - area2) / max_area < 0.2:  # 20% similarity
                                similar_count += 1
                    return float(similar_count / max(1, len(areas)))
            return 0.0
        except Exception as e:
            print(f"Design patterns detection error: {e}")
            return 0.0
    
    def _calculate_layout_balance(self, gray: np.ndarray) -> float:
        """Calculate layout balance"""
        try:
            h, w = gray.shape
            if h < 2:
                return 0.5
                
            top_half = gray[:h//2, :]
            bottom_half = gray[h//2:, :]
            
            top_weight = np.mean(top_half)
            bottom_weight = np.mean(bottom_half)
            
            balance = 1.0 - abs(top_weight - bottom_weight) / 255.0
            return float(balance)
        except Exception as e:
            print(f"Layout balance calculation error: {e}")
            return 0.5
    
    def _calculate_visual_flow(self, gray: np.ndarray) -> float:
        """Calculate visual flow direction"""
        try:
            # Use gradient direction as proxy for visual flow
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate dominant flow direction
            flow_direction = np.arctan2(grad_y, grad_x)
            flow_hist, _ = np.histogram(flow_direction.flatten(), bins=8, range=(-np.pi, np.pi))
            
            # Measure flow consistency
            flow_sum = np.sum(flow_hist)
            if flow_sum > 0:
                flow_consistency = np.max(flow_hist) / flow_sum
            else:
                flow_consistency = 0.0
            return float(flow_consistency)
        except Exception as e:
            print(f"Visual flow calculation error: {e}")
            return 0.0
    
    def _get_default_features(self) -> Dict:
        """Return default features if extraction fails"""
        return {feature_name: 0.0 for feature_name in self.feature_names}
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return self.feature_names.copy()
    
    def get_meaningful_feature_names(self) -> List[str]:
        """Get list of meaningful features (excluding basic properties) for AI vs Human classification"""
        return self.meaningful_features.copy()
    
    def get_basic_property_names(self) -> List[str]:
        """Get list of basic properties (width, height, etc.) that should NOT be used for classification"""
        return self.basic_properties.copy()
    
    def extract_meaningful_features(self, image_data) -> Dict:
        """
        Extract only meaningful features for AI vs Human classification
        Excludes basic properties like width, height, aspect_ratio, total_pixels
        """
        all_features = self.extract_image_features(image_data)
        meaningful_features = {k: v for k, v in all_features.items() if k in self.meaningful_features}
        return meaningful_features
    
    def map_to_legacy_features(self, features: Dict) -> Dict:
        """
        Map 43 features to the 19 features expected by existing models
        Excludes basic properties like width, height, aspect_ratio, total_pixels
        """
        # Old model expected these 19 features:
        legacy_feature_names = [
            'width', 'height', 'aspect_ratio', 'total_pixels',  # Basic properties (will be excluded)
            'color_diversity_s', 'color_diversity_v', 'color_uniformity',
            'avg_saturation', 'avg_brightness', 'edge_density', 'contour_count',
            'avg_contour_area', 'avg_contour_complexity', 'horizontal_vertical_ratio',
            'gradient_magnitude', 'texture_uniformity', 'local_variance',
            'low_freq_energy', 'high_freq_energy'
        ]
        
        # Create mapping from new features to legacy features
        # Focus on meaningful features, exclude basic properties
        legacy_features = {}
        
        # Map basic properties (set to reasonable defaults since they don't correlate with AI generation)
        legacy_features['width'] = 1920.0  # Default width
        legacy_features['height'] = 1080.0  # Default height  
        legacy_features['aspect_ratio'] = 16.0/9.0  # Default aspect ratio
        legacy_features['total_pixels'] = 1920.0 * 1080.0  # Default total pixels
        
        # Map color features (direct mapping where available)
        legacy_features['color_diversity_s'] = features.get('color_diversity_s', 0.0)
        legacy_features['color_diversity_v'] = features.get('color_diversity_v', 0.0)
        legacy_features['color_uniformity'] = features.get('color_uniformity', 0.0)
        legacy_features['avg_saturation'] = features.get('avg_saturation', 0.0)
        legacy_features['avg_brightness'] = features.get('avg_brightness', 0.0)
        
        # Map layout features to legacy features
        legacy_features['edge_density'] = features.get('edge_density', 0.0)
        legacy_features['contour_count'] = features.get('contour_count', 0.0)
        
        # Map structural features to legacy contour features
        # Use structural complexity as proxy for contour complexity
        layout_complexity = features.get('layout_complexity', 0.0)
        visual_hierarchy = features.get('visual_hierarchy', 0.0)
        legacy_features['avg_contour_area'] = (layout_complexity + visual_hierarchy) / 2.0
        legacy_features['avg_contour_complexity'] = layout_complexity
        
        # Map alignment and spacing to horizontal/vertical ratio
        alignment_score = features.get('alignment_score', 0.0)
        component_alignment = features.get('component_alignment', 0.0)
        legacy_features['horizontal_vertical_ratio'] = (alignment_score + component_alignment) / 2.0
        
        # Map texture features
        legacy_features['texture_uniformity'] = features.get('texture_uniformity', 0.0)
        
        # Map structural features to variance measures
        structural_symmetry = features.get('structural_symmetry', 0.0)
        layout_balance = features.get('layout_balance', 0.0)
        legacy_features['local_variance'] = (structural_symmetry + layout_balance) / 2.0
        
        # Map visual flow and patterns to frequency features
        visual_flow = features.get('visual_flow', 0.0)
        design_patterns = features.get('design_patterns', 0.0)
        legacy_features['low_freq_energy'] = visual_flow
        legacy_features['high_freq_energy'] = design_patterns
        
        # Map gradient magnitude from edge density and texture contrast
        edge_density = features.get('edge_density', 0.0)
        texture_contrast = features.get('texture_contrast', 0.0)
        legacy_features['gradient_magnitude'] = (edge_density + texture_contrast) / 2.0
        
        return legacy_features
    
 