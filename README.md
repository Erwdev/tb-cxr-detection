
# TB-CXR Detection  
**Deteksi Tuberkulosis dari Citra Chest X-Ray Menggunakan Computer Vision**

---

## Tim  
| Nama | NPM | Email | Tugas Utama |
|------|-----|-------|-------------|
| **Azhar Maulana** | 24/533487/PA/22582 | azharmaulana533487@mail.ugm.ac.id | **Preprocessing** |
| **Revy Satya Gunawan** | 24/538296/PA/22835 | revysatyagunawan538296@mail.ugm.ac.id | **Segmentation** |
| **Raditya Nathaniel Nugroho** | 24/543188/PA/23069 | radityanathanielnugroho2005@mail.ugm.ac.id | **Morphological Processing** |
| **Benedictus Erwin Widianto** | 23/520176/PA/22350 | benedictuserwinwidianto@mail.ugm.ac.id | **Feature Extraction + Project Lead** |

---

## Technical Implementation (Pipeline)

| Tahap | Teknik | Input | Output |
|-------|--------|-------|--------|
| **1. Preprocessing** | Grayscale + CLAHE + Gaussian Blur | `image_path: str` | `img: np.ndarray` (H×W) |
| **2. Segmentation** | ASM, OTSU, Canny, K-Means, Region Growing | `img` | `masks: dict` (lung, nodule, cavity, ...) |
| **3. Morphology** | Opening, Closing, Hole Filling | `mask` | `clean_mask: np.ndarray` |
| **4. Feature Extraction** | Shape (edge, line, corner) + Texture (GLCM, LBP) | `img + clean_mask` | `features: dict` → `.pkl` |
| **5. Classification** | SLDT-MSA (Stacked Loopy Decision Tree + Moth Search Algorithm) | `feature_vector` | `label: "Normal"/"TB"` |

---

## Dataset
- **Sumber**: [Kaggle TB Chest X-ray](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)  
- **Total**: 4.200 citra (3.500 Normal, 700 TB)  
- **Struktur**:
  ```
  data/raw/
    ├── Normal/          → 3.500 citra
    └── Tuberculosis/    → 700 citra
  ```

---

## Struktur Proyek
```
tb-cxr-detection/
├── config.yaml
├── requirements.txt
├── data/
│   ├── raw/              (git ignore)
│   ├── processed/
│   │   ├── images/       (.npy)
│   │   ├── masks/        (.png)
│   │   └── features/     (.pkl)
│   └── mock/             (untuk testing)
├── src/
│   ├── preprocessing/
│   ├── segmentation/
│   ├── morphology/
│   ├── feature_extraction/
│   └── pipeline/
├── app/main.py           (FastAPI)
├── tests/
└── notebooks/
```

---

## Setup (SEMUA ANGGOTA WAJIB)

```bash
# 1. Clone repo
git clone https://github.com/benedictuserwinwidianto/tb-cxr-detection.git
cd tb-cxr-detection

# 2. Buat environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test config
python -c "import yaml; print('config OK')"
```

---

## Workflow (PR + Review)

```bash
git checkout -b feature/nama-modul-nama
# Kerjakan → commit → push
git push origin feature/preprocessing-azhar
```

> **Buat PR → tag @benedictuserwinwidianto + 1 teammate untuk review**  
> **Merge ke `main` setelah 1 approve**

---

## FastAPI Demo
```bash
uvicorn app.main:app --reload
# → http://localhost:8000/docs

