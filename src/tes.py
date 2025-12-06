"""
TB Detection Pipeline - Complete System (Corrected Logic)
Preprocessing → Segmentation → Morphology (Sequential) → Feature Extraction
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import json
import os

# ============================================================================
# STEP 1: PREPROCESSING
# ============================================================================
def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocessing gambar dengan Gaussian Blur dan CLAHE"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Gambar tidak dapat dibaca: {image_path}")
    
    # Gaussian blur untuk mengurangi noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # CLAHE untuk meningkatkan kontras
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    return img


# ============================================================================
# STEP 2: SEGMENTATION (K-Means)
# ============================================================================
def segment_lungs(img: np.ndarray) -> dict:
    """Segmentasi paru-paru, nodule, dan cavity menggunakan K-Means"""
    pixels = img.reshape((-1, 1)).astype(np.float32)
    
    # K-Means clustering dengan 3 cluster
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Urutkan centers untuk mendapatkan lung region (intensitas menengah)
    sorted_centers = np.argsort(centers.flatten())
    lung_idx = sorted_centers[1]
    
    labels_img = labels.reshape(img.shape)
    raw_lung_mask = (labels_img == lung_idx).astype(np.uint8) * 255
    
    # Lung ROI
    lung_roi = cv2.bitwise_and(img, img, mask=raw_lung_mask)
    
    # Nodule mask (adaptive threshold)
    raw_nodule_mask = cv2.adaptiveThreshold(
        lung_roi, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=21, 
        C=-5
    )
    raw_nodule_mask = cv2.bitwise_and(raw_nodule_mask, raw_nodule_mask, mask=raw_lung_mask)
    
    # Cavity mask
    cavity_s_img = lung_roi.copy()
    cavity_s_img[raw_lung_mask == 0] = 128
    
    raw_cavity_mask = cv2.adaptiveThreshold(
        cavity_s_img, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=21, 
        C=7
    )
    raw_cavity_mask = cv2.bitwise_and(raw_cavity_mask, raw_cavity_mask, mask=raw_lung_mask)
    
    return {
        "lung": raw_lung_mask,
        "nodule": raw_nodule_mask,
        "cavity": raw_cavity_mask,
        "lung_roi": lung_roi
    }


# ============================================================================
# STEP 3: MORPHOLOGICAL PROCESSING (FIXED: SEQUENTIAL)
# ============================================================================
def otsu_threshold(img):
    """Otsu thresholding manual"""
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    total = img.size
    sum_total = np.dot(np.arange(256), hist)
    
    sumB = 0
    wB = 0
    max_var = 0
    threshold = 0
    
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    
    return threshold, (img > threshold).astype(np.uint8)

def fill_holes(mask: np.ndarray) -> np.ndarray:
    """[NEW] Isi lubang terbesar (paru-paru)"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    if cnts:
        # Find largest contour by area
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(filled_mask, [c], -1, 255, -1)
    
    return filled_mask

def apply_morphology(mask: np.ndarray, kernel_size: int = 5, do_fill: bool = True) -> dict:
    """
    [FIXED] Aplikasikan operasi morfologi secara SEKUENSIAL:
    Otsu -> Opening -> Closing -> Hole Filling
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 0. Otsu thresholding
    threshold_val, mask_bin = otsu_threshold(mask)
    mask_bin_255 = (mask_bin * 255).astype(np.uint8)
    
    # 1. Opening: Hapus noise kecil
    opened = cv2.morphologyEx(mask_bin_255, cv2.MORPH_OPEN, kernel)
    
    # 2. Closing: Tutup lubang kecil (input = opened)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # 3. Hole Filling (Optional, usually for lungs)
    final_result = closed
    if do_fill:
        final_result = fill_holes(closed)
    
    # Visualization helpers (optional)
    eroded = cv2.erode(mask_bin_255, kernel, iterations=1)
    dilated = cv2.dilate(mask_bin_255, kernel, iterations=1)
    
    return {
        "original": mask,
        "otsu_binary": mask_bin_255,
        "eroded": eroded,
        "dilated": dilated,
        "opened": opened,
        "closed": closed,
        "final": final_result, # Use this one!
        "threshold_value": threshold_val
    }


# ============================================================================
# STEP 4: FEATURE EXTRACTION (LBP + HOG)
# ============================================================================
def extract_lbp_features(img: np.ndarray, mask: np.ndarray = None) -> dict:
    """Ekstraksi fitur tekstur menggunakan LBP, GLCM, dan edge detection"""
    
    # Masking gambar ke ROI jika ada mask
    if mask is not None:
        img = cv2.bitwise_and(img, img, mask=mask)
    
    features = {}
    
    # 1. Edge Detection (Canny)
    edges = cv2.Canny(img, 100, 200)
    features['edge_sum'] = float(np.sum(edges))
    
    # 2. Hough Transform untuk deteksi garis
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
    features['num_lines'] = 0 if lines is None else len(lines)
    
    # 3. GLCM Features (kontras dan homogenitas)
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, 
                        symmetric=True, normed=True)
    features['glcm_contrast'] = float(graycoprops(glcm, 'contrast')[0, 0])
    features['glcm_homogeneity'] = float(graycoprops(glcm, 'homogeneity')[0, 0])
    
    # 4. Local Binary Pattern (LBP)
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    features['lbp_hist'] = hist.tolist()
    
    return features


# ============================================================================
# PIPELINE LENGKAP
# ============================================================================
def process_single_image(image_path: str, output_dir: str = None, visualize: bool = True):
    """Pipeline lengkap untuk satu gambar"""
    
    print(f"\n{'='*70}")
    print(f"Processing: {os.path.basename(image_path)}")
    print(f"{'='*70}")
    
    # STEP 1: Preprocessing
    print("→ Step 1: Preprocessing...")
    preprocessed = preprocess_image(image_path)
    
    # STEP 2: Segmentation
    print("→ Step 2: Segmentation...")
    segments = segment_lungs(preprocessed)
    
    # STEP 3: Morphological Processing
    print("→ Step 3: Morphological Processing...")
    
    # LUNG: Use Fill Holes (True)
    lung_morph = apply_morphology(segments['lung'], kernel_size=5, do_fill=True)
    
    # NODULE/CAVITY: No Fill Holes (False) - prevents selecting only 1 nodule
    nodule_morph = apply_morphology(segments['nodule'], kernel_size=3, do_fill=False)
    cavity_morph = apply_morphology(segments['cavity'], kernel_size=3, do_fill=False)
    
    # STEP 4: Feature Extraction
    print("→ Step 4: Feature Extraction...")
    
    # [FIXED] Use the 'final' cleaned mask, not the raw segment
    lung_features = extract_lbp_features(preprocessed, lung_morph['final'])
    nodule_features = extract_lbp_features(preprocessed, nodule_morph['final'])
    cavity_features = extract_lbp_features(preprocessed, cavity_morph['final'])
    
    # Hasil akhir
    results = {
        "image_path": image_path,
        "preprocessed_shape": preprocessed.shape,
        "segmentation": {
            "lung_area": int(np.sum(segments['lung'] > 0)),
            "nodule_area": int(np.sum(segments['nodule'] > 0)),
            "cavity_area": int(np.sum(segments['cavity'] > 0))
        },
        "morphology": {
            "lung_threshold": lung_morph['threshold_value'],
            "nodule_threshold": nodule_morph['threshold_value'],
            "cavity_threshold": cavity_morph['threshold_value']
        },
        "features": {
            "lung": lung_features,
            "nodule": nodule_features,
            "cavity": cavity_features
        }
    }
    
    # Visualisasi (Updated to show 'final')
    if visualize:
        visualize_pipeline(
            preprocessed, 
            segments, 
            lung_morph, 
            nodule_morph, 
            cavity_morph
        )
    
    # Simpan hasil
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Simpan JSON features
        json_path = os.path.join(output_dir, f"{basename}_features.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Simpan gambar hasil
        cv2.imwrite(os.path.join(output_dir, f"{basename}_preprocessed.png"), preprocessed)
        cv2.imwrite(os.path.join(output_dir, f"{basename}_lung_mask.png"), lung_morph['final']) # Save the CLEAN one
        cv2.imwrite(os.path.join(output_dir, f"{basename}_nodule_mask.png"), nodule_morph['final'])
        cv2.imwrite(os.path.join(output_dir, f"{basename}_cavity_mask.png"), cavity_morph['final'])
        
        print(f"✓ Results saved to: {output_dir}")
    
    return results


def visualize_pipeline(preprocessed, segments, lung_morph, nodule_morph, cavity_morph):
    """Visualisasi hasil pipeline"""
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('TB Detection Pipeline - Complete Results', fontsize=16, fontweight='bold')
    
    # Row 1: Original & Segmentation
    axes[0, 0].imshow(preprocessed, cmap='gray')
    axes[0, 0].set_title('1. Preprocessed')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(segments['lung'], cmap='gray')
    axes[0, 1].set_title('2. Lung Mask (Raw)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(segments['nodule'], cmap='gray')
    axes[0, 2].set_title('3. Nodule Mask (Raw)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(segments['cavity'], cmap='gray')
    axes[0, 3].set_title('4. Cavity Mask (Raw)')
    axes[0, 3].axis('off')
    
    # Row 2: Lung Morphology
    axes[1, 0].imshow(lung_morph['otsu_binary'], cmap='gray')
    axes[1, 0].set_title(f"Lung Otsu")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(lung_morph['opened'], cmap='gray')
    axes[1, 1].set_title('Lung Opening')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(lung_morph['closed'], cmap='gray')
    axes[1, 2].set_title('Lung Closing')
    axes[1, 2].axis('off')
    
    # [UPDATED] Show Final Clean Mask instead of just Closing
    axes[1, 3].imshow(lung_morph['final'], cmap='gray')
    axes[1, 3].set_title('Lung FINAL (Filled)')
    axes[1, 3].axis('off')
    
    # Row 3: Nodule Morphology
    axes[2, 0].imshow(nodule_morph['otsu_binary'], cmap='gray')
    axes[2, 0].set_title(f"Nodule Otsu")
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(nodule_morph['opened'], cmap='gray')
    axes[2, 1].set_title('Nodule Opening')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(nodule_morph['closed'], cmap='gray')
    axes[2, 2].set_title('Nodule Closing')
    axes[2, 2].axis('off')
    
    axes[2, 3].imshow(nodule_morph['final'], cmap='gray')
    axes[2, 3].set_title('Nodule FINAL')
    axes[2, 3].axis('off')
    
    # Row 4: Cavity Morphology
    axes[3, 0].imshow(cavity_morph['otsu_binary'], cmap='gray')
    axes[3, 0].set_title(f"Cavity Otsu")
    axes[3, 0].axis('off')
    
    axes[3, 1].imshow(cavity_morph['opened'], cmap='gray')
    axes[3, 1].set_title('Cavity Opening')
    axes[3, 1].axis('off')
    
    axes[3, 2].imshow(cavity_morph['closed'], cmap='gray')
    axes[3, 2].set_title('Cavity Closing')
    axes[3, 2].axis('off')
    
    axes[3, 3].imshow(cavity_morph['final'], cmap='gray')
    axes[3, 3].set_title('Cavity FINAL')
    axes[3, 3].axis('off')
    
    plt.tight_layout()
    plt.show()


def process_batch(input_folder: str, output_folder: str, class_name: str = None):
    """Process batch images dari folder"""
    
    print("\n" + "="*70)
    print("TB DETECTION BATCH PROCESSING")
    print("="*70)
    
    # Cari semua gambar
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
        image_paths.extend(glob(os.path.join(input_folder, ext)))
    
    print(f"Found {len(image_paths)} images")
    
    if class_name:
        output_folder = os.path.join(output_folder, class_name)
    
    all_results = []
    
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing...")
        try:
            result = process_single_image(
                img_path, 
                output_dir=output_folder, 
                visualize=False
            )
            all_results.append(result)
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    # Simpan summary
    summary_path = os.path.join(output_folder, "batch_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✓ Batch processing complete!")
    print(f"✓ Processed: {len(all_results)}/{len(image_paths)} images")
    print(f"✓ Summary saved: {summary_path}")
    print("="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    # ===== MODE 1: Process Single Image =====
    print("\nMODE 1: Single Image Processing")
    single_image_path = r"D:\UGM\Pengantar Citra Digital\Tbc\TB_Chest_Radiography_Database\Tuberculosis\Tuberculosis-5"
    
    # Check current directory if path doesn't exist
    if not os.path.exists(single_image_path):
        print(f"⚠ Note: Path '{single_image_path}' not found.")
        print("  Running on sample data/raw/Normal/Normal-1.png if available...")
        single_image_path = "data/raw/Normal/Normal-1.png"

    if os.path.exists(single_image_path):
        result = process_single_image(
            single_image_path, 
            output_dir="output/single",
            visualize=True
        )
    else:
        print(f"⚠ Image not found: {single_image_path}")
    
    
    # ===== MODE 2: Process Batch (Normal & TB) =====
    print("\n\nMODE 2: Batch Processing")
    
    input_root = r"D:\UGM\Pengantar Citra Digital\Tbc\TB_Chest_Radiography_Database"
    output_root = r"D:\UGM\Pengantar Citra Digital\Tbc\Proccessed"
    
    classes = ["Normal", "Tuberculosis"]
    
    if os.path.exists(input_root):
        for cls in classes:
            input_folder = os.path.join(input_root, cls)
            
            if os.path.exists(input_folder):
                process_batch(
                    input_folder=input_folder,
                    output_folder=output_root,
                    class_name=cls
                )
            else:
                print(f"⚠ Folder not found: {input_folder}")
    else:
        print(f"Skipping batch processing (Folder not found: {input_root})")
    
    print("\n✓ All processing complete!")