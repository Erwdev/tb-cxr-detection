import cv2 
import numpy as np

def preprocess_image(image_path:str) -> np.ndarray:
    # baca gambar grayscale 
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None: 
        raise ValueError(f"Gambar tidak dapat dibaca: {image_path}")
    
    # gaussian blur untuk mengurangi noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # CLAHE untuk meningkatkan kontras
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # output berupa array (HxW)
    return img