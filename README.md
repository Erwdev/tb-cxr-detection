# TB-CXR Detection  
**Deteksi Tuberkulosis dari Citra Chest X-Ray Menggunakan Computer Vision & Machine Learning**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tb-detection-pcd-k3.streamlit.app/)

---

## 🚀 Live Demo
👉 **[Try the App Here!](https://tb-detection-pcd-k3.streamlit.app/)** 

Upload X-Ray image → Get instant TB detection results with visualization!

---

## 👥 Tim Pengembang
| Nama | NPM | Email | Tugas Utama |
|------|-----|-------|-------------|
| **Azhar Maulana** | 24/533487/PA/22582 | azharmaulana533487@mail.ugm.ac.id | **Preprocessing** |
| **Revy Satya Gunawan** | 24/538296/PA/22835 | revysatyagunawan538296@mail.ugm.ac.id | **Segmentation** |
| **Raditya Nathaniel Nugroho** | 24/543188/PA/23069 | radityanathanielnugroho2005@mail.ugm.ac.id | **Morphological Processing** |
| **Benedictus Erwin Widianto** | 23/520176/PA/22350 | benedictuserwinwidianto@mail.ugm.ac.id | **Feature Extraction + Project Lead** |

---

## 🎯 Features

- ✅ **Automated Lung Segmentation** - K-Means clustering untuk isolasi region paru-paru
- ✅ **Advanced Preprocessing** - CLAHE + Gaussian Blur untuk enhancement
- ✅ **Multi-Region Detection** - Deteksi lung, nodule, dan cavity
- ✅ **Feature Extraction** - LBP, GLCM, Edge, dan Hough Line features
- ✅ **ML Classification** - SLDT-MSA (Stacking + Moth Search Algorithm)
- ✅ **Interactive Visualization** - Real-time visualization dengan Streamlit
- ✅ **Morphological Analysis** - Complete morphology operations analysis

---

## 🔬 Technical Implementation (Pipeline)

| Tahap | Teknik | Input | Output |
|-------|--------|-------|--------|
| **1. Preprocessing** | Grayscale → Gaussian Blur → CLAHE | `image_path: str` | `preprocessed: np.ndarray (H×W)` |
| **2. Segmentation** | K-Means (3 clusters) + Adaptive Threshold | `preprocessed` | `masks: dict` (lung, nodule, cavity) |
| **3. Morphology** | Otsu + Erosion/Dilation/Opening/Closing | `mask` | `morphology_results: dict` |
| **4. Feature Extraction** | Edge (Canny) + Lines (Hough) + GLCM + LBP | `img + lung_mask` | `features: dict` (14 features) |
| **5. Classification** | SLDT-MSA (Stacked Decision Tree + Moth Search) | `feature_vector` | `prediction: "Normal"/"Tuberculosis"` |

### Feature Set (14 Features)
- **Shape Features (3)**: Edge Sum, Number of Lines, Corner Count
- **Texture Features (2)**: GLCM Contrast, GLCM Homogeneity
- **LBP Features (9)**: Local Binary Pattern Histogram (9 bins)

---

## 📊 Dataset
- **Sumber**: [Kaggle TB Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)  
- **Total**: 4.200 citra (3.500 Normal, 700 TB)  
- **Split**: 80% Training, 20% Testing (stratified)
- **Format**: PNG/JPG grayscale images
- **Struktur**:
  ```
  data/raw/TB_Chest_Radiography_Database/
    ├── Normal/          → 3.500 citra
    └── Tuberculosis/    → 700 citra
  ```

---

## 📁 Struktur Proyek
```
tb-cxr-detection/
├── README.md
├── requirements.txt         # Dependencies untuk production
├── requirements-local.txt   # Dependencies untuk development
├── packages.txt            # System dependencies (deployment)
├── config.yaml             # Configuration file
├── .streamlit/
│   └── config.toml         # Streamlit app configuration
├── app/
│   ├── main.py            # 🎨 Streamlit UI
│   ├── pipeline.py        # Backend analysis pipeline
│   └── utils/
│       └── visualizer.py  # Visualization functions
├── src/
│   ├── __init__.py
│   ├── tes.py             # Complete pipeline (preprocessing → features)
│   ├── preprocessing/
│   ├── segmentation/
│   ├── morphology/
│   └── classification/
│       ├── train.py       # Model training (SLDT-MSA)
│       └── test_a.py      # Model evaluation
├── models/
│   └── tb_model_raw.pkl   # Trained classifier (Git LFS)
├── data/
│   ├── raw/               # Original dataset (gitignored)
│   ├── processed/
│   │   ├── features/
│   │   │   └── dataset.csv  # Extracted features
│   │   ├── images/
│   │   └── masks/
│   └── mock/              # Sample images for testing
├── notebooks/             # Jupyter notebooks (development)
└── tests/                 # Unit tests
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10 or 3.11
- Git
- (Optional) Git LFS for model file

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/benedictuserwinwidianto/tb-cxr-detection.git
cd tb-cxr-detection

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import streamlit; import cv2; import sklearn; print('✓ All dependencies installed!')"
```

---

## 🚀 Running the Application

### Local Development
```bash
# From project root directory
streamlit run app/main.py

# App will open at http://localhost:8501
```

## 📊 Model Performance

**Classifier**: SLDT-MSA (Stacking Loopy Decision Tree + Moth Search Algorithm)

### Model Architecture
- **Feature Selection**: Moth Search Algorithm (MSA)
- **Base Learners**: Decision Tree + Random Forest
- **Meta Learner**: Decision Tree
- **Optimization**: Grid Search (class weight + max depth)

---

## 💻 Usage Examples

### Using Streamlit App
1. Upload chest X-Ray image (PNG/JPG)
2. Click "🔬 Analyze Image"
3. View results:
   - Prediction (Normal/TB) with confidence
   - Segmentation masks
   - Morphological operations
   - Extracted features


### Training Custom Model
```bash
# Generate features from dataset
python src/tes.py

# Train classifier
python src/classification/train.py

# Test model
python src/classification/test_a.py
```

---

## 🔧 Development Workflow

### Branch Strategy
```bash
# Create feature branch
git checkout -b feature/module-name-yourname

# Example
git checkout -b feature/segmentation-revy

# Work → Commit → Push
git add .
git commit -m "Add segmentation module"
git push origin feature/segmentation-revy
```

### Pull Request Process
1. Create PR dari feature branch ke `main`
2. Tag: `@benedictuserwinwidianto` + 1 teammate untuk review
3. Merge setelah mendapat **1 approval**
4. Delete feature branch setelah merge

---

## 📚 Documentation

### Pipeline Modules

#### 1. Preprocessing (`src/tes.py`)
```python
preprocess_image(image_path: str) -> np.ndarray
```
- Gaussian Blur (3×3)
- CLAHE (clipLimit=2.0, tileGridSize=8×8)

#### 2. Segmentation (`src/tes.py`)
```python
segment_lungs(img: np.ndarray) -> dict
```
- K-Means clustering (k=3)
- Adaptive threshold untuk nodule/cavity

#### 3. Morphology (`src/tes.py`)
```python
apply_morphology(mask: np.ndarray, kernel_size: int) -> dict
```
- Otsu thresholding
- Erosion, Dilation, Opening, Closing

#### 4. Feature Extraction (`src/tes.py`)
```python
extract_lbp_features(img: np.ndarray, mask: np.ndarray) -> dict
```
- Edge detection (Canny)
- Line detection (Hough)
- GLCM features
- LBP histogram

---



## 🙏 Acknowledgments
- Dataset: [Tawsifur Rahman et al.](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
- Streamlit Community
- UGM - Pengantar Citra Digital Course

---

## 📧 Contact
**Project Lead**: Benedictus Erwin Widianto  
📧 benedictuserwinwidianto@mail.ugm.ac.id  
🔗 [GitHub Issues](https://github.com/benedictuserwinwidianto/tb-cxr-detection/issues)

