import os
import glob
import pandas as pd
import numpy as np
import cv2
from src.preprocessing.preprocess import preprocess_image
from src.segmentation.core import segment
from src.morphology.morphology_fix import clean_mask, fill_holes
from src.feature_extraction.extract import extract_lbp_features

def process_single_image(image_path: str):
    
    # 1. Preprocess
    img = preprocess_image(image_path)
    
    # 2. Segment
    masks = segment(img)
    lung_mask = masks['lung']
    
    # 3. Clean Mask
    cleaned = clean_mask(lung_mask)
    final_mask = fill_holes(cleaned)
    
    # 4. Extract Features
    features = extract_lbp_features(img, mask=final_mask)
    return features

def build_dataset(raw_data_path: str, output_csv: str):
    
    data = []
    classes = ["Normal", "Tuberculosis"]
    
    print(" Starting Pipeline Processing...")
    
    for label in classes:
        folder_path = os.path.join(raw_data_path, label)
        image_files = glob.glob(os.path.join(folder_path, "*.*"))
        
        print(f"Processing Class: {label} ({len(image_files)} images)")
        
        for i, img_path in enumerate(image_files):
            try:
                # Extract features
                feats = process_single_image(img_path)
                
                # Add label
                feats['label'] = 0 if label == "Normal" else 1
                feats['filename'] = os.path.basename(img_path)
                data.append(feats)
                
                if i % 100 == 0:
                    print(f"  - Processed {i} images...")
                    
            except Exception as e:
                print(f"  [Error] {img_path}: {e}")
                continue

    # Save to CSV
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f" Dataset saved to: {output_csv}")

if __name__ == "__main__":
    # Update this path to your actual raw data location
    RAW_PATH = "data/raw" 
    OUT_CSV = "data/processed/features/dataset.csv"
    
    if os.path.exists(RAW_PATH):
        build_dataset(RAW_PATH, OUT_CSV)
    else:
        print(f" Error: Path '{RAW_PATH}' not found. Please create it or update the path.")