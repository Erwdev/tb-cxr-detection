import os
from glob import glob

import cv2
import numpy as np


def preprocess_image(image_path: str) -> np.ndarray:
    # baca gambar grayscale 
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Gambar tidak dapat dibaca: {image_path}")
    
    # gaussian blur untuk mengurangi noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # CLAHE untuk meningkatkan kontras
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # output berupa array (H x W)
    return img


def preprocess_folder(input_root: str, output_root: str):
    classes = ["Normal", "Tuberculosis"]

    for cls in classes:
        in_dir = os.path.join(input_root, cls)
        out_dir = os.path.join(output_root, cls)

        os.makedirs(out_dir, exist_ok=True)

        # ambil semua file gambar di folder input
        image_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            image_paths.extend(glob(os.path.join(in_dir, ext)))

        print(f"Memproses kelas: {cls}, jumlah gambar: {len(image_paths)}")

        # loop proses & simpan gambar
        for img_path in image_paths:
            try:
                img = preprocess_image(img_path)
            except ValueError as e:
                print(e)
                continue

            # simpan gambar hasil preprocess 
            filename = os.path.basename(img_path)
            save_path = os.path.join(out_dir, filename)
            cv2.imwrite(save_path, img)

    print("Selesai preprocessing semua kelas!")


# if __name__ == "__main__":
#     # base_dir = folder root project (tb-cxr-detection)
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#     # path input & output relatif dari base_dir
#     input_root = os.path.join("D:/Bismillah Kuliah/Semester 3/PCD/Final Project/tb-cxr-detection/data/raw/TB_Chest_Radiography_Database")
#     output_root = os.path.join("D:/Bismillah Kuliah/Semester 3/PCD/Final Project/tb-cxr-detection/data/preprocessed")

#     preprocess_folder(input_root, output_root)
