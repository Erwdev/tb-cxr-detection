import cv2 
import numpy as np

def preprocess_image(image_path:str) -> np.ndarray:
    # baca gambar grayscale 
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)