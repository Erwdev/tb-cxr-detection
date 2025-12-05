import cv2
import numpy as np
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import functions (sekarang src/ sudah di path)
from src.tes import (
    preprocess_image,
    segment_lungs,
    apply_morphology,
    extract_lbp_features
)
import joblib

def test_single_image(image_path: str):
    """Test complete pipeline dengan single image"""
    
    print("="*60)
    print("TB DETECTION PIPELINE TEST")
    print("="*60)
    
    # STEP 1: Preprocessing
    print("\n[1/5] Preprocessing...")
    try:
        preprocessed = preprocess_image(image_path)
        print(f"✓ Shape: {preprocessed.shape}")
        print(f"✓ Range: [{preprocessed.min()}, {preprocessed.max()}]")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # STEP 2: Segmentation
    print("\n[2/5] Segmentation...")
    try:
        segments = segment_lungs(preprocessed)
        print(f"✓ Lung mask: {segments['lung'].shape}")
        print(f"✓ Nodule mask: {segments['nodule'].shape}")
        print(f"✓ Cavity mask: {segments['cavity'].shape}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # STEP 3: Morphology
    print("\n[3/5] Morphology...")
    try:
        lung_morph = apply_morphology(segments['lung'], kernel_size=5)
        print(f"✓ Operations: {list(lung_morph.keys())}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # STEP 4: Feature Extraction
    print("\n[4/5] Feature Extraction...")
    try:
        features = extract_lbp_features(preprocessed, mask=segments['lung'])
        print(f"✓ Features extracted:")
        for k, v in features.items():
            if k != 'lbp_hist':
                print(f"  - {k}: {v}")
            else:
                print(f"  - {k}: {len(v)} bins")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # STEP 5: Prediction
    print("\n[5/5] Prediction...")
    model_path = "E:\\SEMESTER 5\\PCD\\FINAL PROJECT\\tb-cxr-detection\\models\\tb_model_raw.pkl"
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print("  → Train model first using src/classification/train.py")
        return
    
    try:
        model = joblib.load(model_path)
        print(f"✓ Model loaded: {type(model)}")
        
        # Convert features to array
        lbp_hist = features['lbp_hist']
        X = np.array([
            features['edge_sum'],
            features['num_lines'],
            features['glcm_contrast'],
            features['glcm_homogeneity'],
            *lbp_hist
        ]).reshape(1, -1)
        
        print(f"✓ Feature vector shape: {X.shape}")
        
        # Predict
        prediction = model.predict(X)[0]
        label = "Normal" if prediction == 0 else "Tuberculosis"
        
        try:
            proba = model.predict_proba(X)[0]
            confidence = max(proba)
            print(f"\n{'='*60}")
            print(f"PREDICTION: {label}")
            print(f"Confidence: {confidence*100:.2f}%")
            print(f"Normal: {proba[0]*100:.2f}% | TB: {proba[1]*100:.2f}%")
            print(f"{'='*60}")
        except:
            print(f"\n{'='*60}")
            print(f"PREDICTION: {label}")
            print(f"{'='*60}")
            
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Path ke gambar test
    test_image = "E:\\SEMESTER 5\\PCD\\FINAL PROJECT\\tb-cxr-detection\\notebooks\\revy-segmentation\\Normal-1.png"  # GANTI INI
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    if not os.path.exists(test_image):
        print(f"Error: Image not found - {test_image}")
        print(f"Usage: python test_pipeline.py <image_path>")
        sys.exit(1)
    
    test_single_image(test_image)