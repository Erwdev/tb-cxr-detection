import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def extract_lbp_features(img: np.ndarray, mask: np.ndarray = None) -> dict:
    

    # masking gambar ke ROI jika belum berupa mask 
    if mask is not None:
        img = cv2.bitwise_and(img, img, mask=mask)
        
    features = {}
    
    # deteksi edge menggunakan Canny dan juga menghitung kekasaran dengan edge sum 
    edges = cv2.Canny(img, 100, 200)
    features['edge_sum'] = np.sum(edges)
    
    
    # menggunakan hough transform untuk deteksi garis 
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
    features['num_lines'] = 0 if lines is None else len(lines)
    
    # gray level co-occurrence matrix (GLCM) features
    # menghitung kontras dan homogenitas ke dalam sebuah matrix 
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features['glcm_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    features['glcm_homogenity'] = graycoprops(glcm, 'homogeneity')[0,0]
    
    # menangkap fitur LBP , tekstur lokal yang muncul lalu dihitung histogramnya dari tiap pixel 
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    features['lbp_hist'] = hist.tolist()

    return features 