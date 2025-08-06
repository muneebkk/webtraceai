#!/usr/bin/env python3
"""
Demo script for WebTrace AI Model Visualization Features
This script demonstrates the new visualization capabilities
"""

import requests
import json
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section"""
    print(f"\n📋 {title}")
    print("-" * 40)

def demo_visualization_features():
    """Demonstrate the new visualization features"""
    
    print_header("WebTrace AI - Model Visualization Demo")
    
    print("🎯 This demo showcases the new model visualization features")
    print("   that show exactly how each AI model makes predictions.")
    
    print_section("Testing Visualization Endpoint")
    
    # Test all three models
    models = [
        {
            "id": "original",
            "name": "Original Random Forest",
            "description": "Basic ensemble model with all features"
        },
        {
            "id": "improved", 
            "name": "Improved Logistic Regression",
            "description": "Optimized model with feature selection"
        },
        {
            "id": "custom_tree",
            "name": "Custom Decision Tree", 
            "description": "Interpretable tree-based model"
        }
    ]
    
    results = {}
    
    # Test with a sample image
    test_image_path = "dataset/images/human/human_001.png"
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        print("Please ensure the test image exists before running this demo.")
        return
    
    for model in models:
        print(f"\n🧪 Testing {model['name']}...")
        
        try:
            # Prepare request data with image file
            with open(test_image_path, 'rb') as f:
                files = {'screenshot': f}
                data = {'model': model['id']}
            
            # Make request to visualization endpoint
            response = requests.post(
                "http://localhost:8000/api/predict-visualization",
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                results[model['id']] = result
                
                print(f"✅ Success!")
                print(f"   Prediction: {result['prediction']}")
                print(f"   Confidence: {result['confidence']:.1%}")
                print(f"   Model Type: {result['model_type']}")
                print(f"   Features Analyzed: {len(result['feature_values'])}")
                print(f"   Decision Steps: {len(result['decision_path'])}")
                
                # Show top 3 features
                top_features = sorted(result['feature_importance'].items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
                print(f"   Top Features: {[f[0] for f in top_features]}")
                
            else:
                print(f"❌ Failed with status {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print_section("Visualization Features Overview")
    
    print("🎨 The visualization system provides:")
    print("   1. 📊 Feature Importance Charts")
    print("   2. 🛤️  Decision Path Analysis") 
    print("   3. 📈 Model Comparison Views")
    print("   4. 📝 Human-Readable Explanations")
    print("   5. 🔍 Interactive Modal Interface")
    
    print_section("How to Use the Visualization")
    
    print("🚀 To use the visualization features:")
    print("   1. Start the backend: cd backend && python -m uvicorn main:app --reload")
    print("   2. Start the frontend: cd frontend && npm run dev")
    print("   3. Open http://localhost:5173 in your browser")
    print("   4. Upload a screenshot")
    print("   5. Select your preferred model")
    print("   6. Click 'Analyze Website'")
    print("   7. Click 'Visualize' to see detailed breakdown")
    
    print_section("Model-Specific Insights")
    
    for model_id, result in results.items():
        model_info = next(m for m in models if m['id'] == model_id)
        print(f"\n🔍 {model_info['name']}:")
        print(f"   • {model_info['description']}")
        print(f"   • Prediction: {result['prediction']}")
        print(f"   • Confidence: {result['confidence']:.1%}")
        
        # Show explanation
        explanation = result['explanation']
        if len(explanation) > 100:
            explanation = explanation[:100] + "..."
        print(f"   • Explanation: {explanation}")
    
    print_section("Educational Benefits")
    
    print("📚 The visualization system helps users understand:")
    print("   • How different AI algorithms work")
    print("   • Which features are most important")
    print("   • Why specific predictions are made")
    print("   • How to interpret AI model outputs")
    print("   • The trade-offs between different models")
    
    print_section("Technical Implementation")
    
    print("⚙️  Key technical components:")
    print("   • Backend: New /api/predict-visualization endpoint")
    print("   • Frontend: Interactive React components")
    print("   • Data: Feature importance, decision paths, explanations")
    print("   • UI: Tabbed modal with responsive design")
    print("   • Integration: Seamless workflow integration")
    
    print_section("Future Enhancements")
    
    print("🔮 Planned improvements:")
    print("   • Interactive decision tree visualization")
    print("   • Real-time feature analysis")
    print("   • Export capabilities for reports")
    print("   • Comparative analysis across models")
    print("   • Advanced chart types and animations")
    
    print_header("Demo Complete!")
    
    print("🎉 The visualization demo is complete!")
    print("   Try uploading your own website screenshots to see")
    print("   how the different models analyze and predict.")
    print("\n   For more information, see docs/MODEL_VISUALIZATION_GUIDE.md")

if __name__ == "__main__":
    try:
        demo_visualization_features()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("   Make sure the backend is running on http://localhost:8000") 