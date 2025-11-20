import cv2
import numpy as np
from sklearn.cluster import KMeans

def segment(img: np.ndarray) -> dict:

    pixels = img.reshape((-1, 1)).astype(np.float32)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    sorted_centers = np.argsort(centers.flatten())
    
    lung_idx = sorted_centers[1]
    
    labels_img = labels.reshape(img.shape)
    raw_lung_mask = (labels_img == lung_idx).astype(np.uint8) * 255

    lung_roi = cv2.bitwise_and(img, img, mask=raw_lung_mask)

    raw_nodule_mask = cv2.adaptiveThreshold(
        lung_roi, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=21, 
        C=-5
    )

    raw_nodule_mask = cv2.bitwise_and(raw_nodule_mask, raw_nodule_mask, mask=raw_lung_mask)

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
        "cavity": raw_cavity_mask
    }