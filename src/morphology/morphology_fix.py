import cv2
import numpy as np

def clean_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Remove small white noise (background spots)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Close small black holes (inside the lungs)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def fill_holes(mask: np.ndarray) -> np.ndarray:
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    # Fill the largest contour (assumed to be the lung/object)
    if cnts:
        # Find largest contour by area
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(filled_mask, [c], -1, 255, -1)
    
    return filled_mask