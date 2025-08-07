#!/usr/bin/env python3
"""
Dataset Validation Script
Checks for potential issues before training on large dataset
"""

import os
import cv2
import numpy as np
from app.feature_extract import FeatureExtractor
import time

class DatasetValidator:
    def __init__(self):
        self.dataset_path = "../dataset"
        self.extractor = FeatureExtractor()
        self.issues = []
        self.stats = {
            'total_images': 0,
            'valid_images': 0,
            'failed_images': 0,
            'ai_images': 0,
            'human_images': 0,
            'feature_extraction_time': 0
        }
        
    def validate_image(self, img_path, category):
        """Validate a single image"""
        try:
            # Check if file exists
            if not os.path.exists(img_path):
                self.issues.append(f"File not found: {img_path}")
                return False
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                self.issues.append(f"Failed to load image: {img_path}")
                return False
            
            # Check image size
            if img.size == 0:
                self.issues.append(f"Empty image: {img_path}")
                return False
            
            # Check image dimensions
            height, width = img.shape[:2]
            if width < 100 or height < 100:
                self.issues.append(f"Image too small ({width}x{height}): {img_path}")
                return False
            
            # Test feature extraction (with timeout)
            start_time = time.time()
            try:
                features = self.extractor.extract_image_features(img)
                extraction_time = time.time() - start_time
                
                if features is None:
                    self.issues.append(f"Feature extraction failed: {img_path}")
                    return False
                
                # Check if we got the expected number of features
                expected_features = len(self.extractor.get_feature_names())
                actual_features = len(features)
                
                if actual_features != expected_features:
                    self.issues.append(f"Feature count mismatch ({actual_features}/{expected_features}): {img_path}")
                    return False
                
                # Check for NaN or infinite values
                feature_values = list(features.values())
                if any(np.isnan(val) for val in feature_values) or any(np.isinf(val) for val in feature_values):
                    self.issues.append(f"Invalid feature values (NaN/Inf): {img_path}")
                    return False
                
                self.stats['feature_extraction_time'] += extraction_time
                return True
                
            except Exception as e:
                self.issues.append(f"Feature extraction error in {img_path}: {str(e)}")
                return False
                
        except Exception as e:
            self.issues.append(f"General error with {img_path}: {str(e)}")
            return False
    
    def validate_category(self, category_path, category_name):
        """Validate all images in a category"""
        print(f"🔍 Validating {category_name} images...")
        
        if not os.path.exists(category_path):
            print(f"  ❌ Directory not found: {category_path}")
            return
        
        image_files = [f for f in os.listdir(category_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"  📁 Found {len(image_files)} images")
        
        valid_count = 0
        for i, filename in enumerate(image_files, 1):
            img_path = os.path.join(category_path, filename)
            
            if self.validate_image(img_path, category_name):
                valid_count += 1
                if i % 10 == 0:  # Progress indicator
                    print(f"    ✅ Processed {i}/{len(image_files)} images")
            else:
                self.stats['failed_images'] += 1
            
            self.stats['total_images'] += 1
        
        if category_name == 'ai':
            self.stats['ai_images'] = valid_count
        else:
            self.stats['human_images'] = valid_count
            
        self.stats['valid_images'] += valid_count
        
        print(f"  ✅ {valid_count}/{len(image_files)} {category_name} images valid")
    
    def run_validation(self):
        """Run full dataset validation"""
        print("🚀 Starting Dataset Validation...")
        print("=" * 50)
        
        # Validate AI images
        ai_path = os.path.join(self.dataset_path, "images", "ai")
        self.validate_category(ai_path, "ai")
        
        print()
        
        # Validate human images
        human_path = os.path.join(self.dataset_path, "images", "human")
        self.validate_category(human_path, "human")
        
        print()
        print("=" * 50)
        print("📊 Validation Results:")
        print(f"  Total images: {self.stats['total_images']}")
        print(f"  Valid images: {self.stats['valid_images']}")
        print(f"  Failed images: {self.stats['failed_images']}")
        print(f"  AI images: {self.stats['ai_images']}")
        print(f"  Human images: {self.stats['human_images']}")
        print(f"  Average extraction time: {self.stats['feature_extraction_time']/max(1, self.stats['valid_images']):.2f}s per image")
        
        if self.issues:
            print(f"\n❌ Issues Found ({len(self.issues)}):")
            for i, issue in enumerate(self.issues[:10], 1):  # Show first 10 issues
                print(f"  {i}. {issue}")
            if len(self.issues) > 10:
                print(f"  ... and {len(self.issues) - 10} more issues")
        else:
            print("\n✅ No issues found!")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if self.stats['valid_images'] < 50:
            print("  ⚠️  Dataset is quite small (< 50 images)")
        elif self.stats['valid_images'] < 100:
            print("  📈 Dataset size is moderate (50-100 images)")
        else:
            print("  🎉 Dataset size is good (> 100 images)")
        
        if self.stats['ai_images'] == 0 or self.stats['human_images'] == 0:
            print("  ❌ Missing one of the classes (AI or Human)")
        elif abs(self.stats['ai_images'] - self.stats['human_images']) > 10:
            print("  ⚖️  Class imbalance detected")
        else:
            print("  ✅ Classes are well balanced")
        
        if self.stats['feature_extraction_time']/max(1, self.stats['valid_images']) > 2:
            print("  ⏱️  Feature extraction is slow (> 2s per image)")
        else:
            print("  ⚡ Feature extraction speed is good")
        
        # Final recommendation
        if self.stats['failed_images'] == 0 and self.stats['valid_images'] > 20:
            print(f"\n🎉 Dataset is ready for training!")
            print(f"   Expected training time: {self.stats['valid_images'] * 0.5:.0f}-{self.stats['valid_images'] * 1.5:.0f} minutes")
        else:
            print(f"\n⚠️  Dataset has issues that should be fixed before training")
        
        return self.stats['failed_images'] == 0 and self.stats['valid_images'] > 20

def main():
    validator = DatasetValidator()
    is_ready = validator.run_validation()
    
    if is_ready:
        print(f"\n✅ Ready to run: python improve_all_models.py")
    else:
        print(f"\n❌ Fix issues before running training script")

if __name__ == "__main__":
    main() 