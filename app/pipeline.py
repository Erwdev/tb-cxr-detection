"""
Backend Pipeline untuk Streamlit App
Handles image analysis dengan proper state management
"""
import sys
import os
import numpy as np
import joblib
import tempfile
from typing import Dict, Optional
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.tes import (
    preprocess_image,
    segment_lungs,
    apply_morphology,
    extract_lbp_features
)

# ============================================================================
# MODEL LOADING (Cached - loads only once)
# ============================================================================
_model_cache = None

def load_model(model_path: str = None) -> object:
    """
    Load trained model with caching
    This will be wrapped with @st.cache_resource in Streamlit
    """
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    if model_path is None:
        model_path = os.path.join(project_root, "models", "tb_model_raw.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    _model_cache = joblib.load(model_path)
    return _model_cache


# ============================================================================
# FEATURE PROCESSING
# ============================================================================
def features_to_array(features: Dict) -> np.ndarray:
    """
    Convert features dict to numpy array (14 features)
    Order MUST match training data!
    """
    lbp_hist = features['lbp_hist']
    
    feature_array = np.array([
        features['edge_sum'],
        features['num_lines'],
        features['glcm_contrast'],
        features['glcm_homogeneity'],
        *lbp_hist  # 9 LBP histogram bins
    ]).reshape(1, -1)
    
    return feature_array


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================
def analyze_image(
    image_path: str,
    model: object,
    include_morphology: bool = True,
    progress_callback: Optional[callable] = None
) -> Dict:
    """
    Complete analysis pipeline untuk single image
    
    Args:
        image_path: Path to input image
        model: Trained classifier model
        include_morphology: Whether to compute morphology operations
        progress_callback: Function to call for progress updates (for Streamlit)
    
    Returns:
        Dict containing all results and prediction
    """
    
    results = {}
    
    # STEP 1: Preprocessing (20%)
    if progress_callback:
        progress_callback(0.2, "Preprocessing image...")
    
    preprocessed = preprocess_image(image_path)
    results['preprocessed'] = preprocessed
    
    # STEP 2: Segmentation (40%)
    if progress_callback:
        progress_callback(0.4, "Segmenting lung regions...")
    
    segments = segment_lungs(preprocessed)
    results['segments'] = segments
    
    # STEP 3: Morphology (60%) - Optional
    if include_morphology:
        if progress_callback:
            progress_callback(0.6, "Applying morphological operations...")
        
        results['morphology'] = {
            'lung': apply_morphology(segments['lung'], kernel_size=5),
            'nodule': apply_morphology(segments['nodule'], kernel_size=3),
            'cavity': apply_morphology(segments['cavity'], kernel_size=3)
        }
    
    # STEP 4: Feature Extraction (80%)
    if progress_callback:
        progress_callback(0.8, "Extracting features...")
    
    # IMPORTANT: Only use LUNG mask for features (matching training)
    features = extract_lbp_features(preprocessed, mask=segments['lung'])
    results['features'] = features
    
    # STEP 5: Prediction (100%)
    if progress_callback:
        progress_callback(0.9, "Making prediction...")
    
    X = features_to_array(features)
    prediction_label = model.predict(X)[0]
    
    # Get probabilities if available
    try:
        proba = model.predict_proba(X)[0]
        probabilities = {
            "Normal": float(proba[0]),
            "Tuberculosis": float(proba[1])
        }
        confidence = float(max(proba))
    except AttributeError:
        # Model doesn't have predict_proba
        probabilities = {
            "Normal": 1.0 if prediction_label == 0 else 0.0,
            "Tuberculosis": 0.0 if prediction_label == 0 else 1.0
        }
        confidence = 1.0
    
    results['prediction'] = {
        'label': "Normal" if prediction_label == 0 else "Tuberculosis",
        'label_code': int(prediction_label),
        'confidence': confidence,
        'probabilities': probabilities
    }
    
    if progress_callback:
        progress_callback(1.0, "Analysis complete!")
    
    return results


# ============================================================================
# FILE HANDLING
# ============================================================================
def save_uploaded_file(uploaded_file) -> str:
    """
    Save Streamlit uploaded file to temp directory
    Returns path to saved file
    """
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return temp_path


def cleanup_temp_file(file_path: str):
    """Remove temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Warning: Could not delete temp file {file_path}: {e}")


# ============================================================================
# BATCH PROCESSING (Optional - for multiple images)
# ============================================================================
def analyze_batch(
    image_paths: list,
    model: object,
    progress_callback: Optional[callable] = None
) -> list:
    """
    Analyze multiple images
    Returns list of results
    """
    results = []
    total = len(image_paths)
    
    for idx, img_path in enumerate(image_paths):
        if progress_callback:
            progress = (idx + 1) / total
            progress_callback(progress, f"Processing {idx+1}/{total}...")
        
        try:
            result = analyze_image(img_path, model, include_morphology=False)
            result['image_path'] = img_path
            result['status'] = 'success'
        except Exception as e:
            result = {
                'image_path': img_path,
                'status': 'error',
                'error': str(e)
            }
        
        results.append(result)
    
    return results


# ============================================================================
# VALIDATION
# ============================================================================
def validate_image(image_path: str) -> tuple[bool, str]:
    """
    Validate if image can be processed
    Returns (is_valid, error_message)
    """
    import cv2
    
    if not os.path.exists(image_path):
        return False, "File not found"
    
    # Try to read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return False, "Invalid image format"
    
    # Check dimensions
    if img.shape[0] < 100 or img.shape[1] < 100:
        return False, "Image too small (minimum 100x100)"
    
    if img.shape[0] > 4096 or img.shape[1] > 4096:
        return False, "Image too large (maximum 4096x4096)"
    
    return True, ""