"""
Streamlit App - TB Detection System
Handles async-like processing with progress indicators
"""
import streamlit as st
import os
import sys
import time

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pipeline import (
    load_model,
    analyze_image,
    save_uploaded_file,
    cleanup_temp_file,
    validate_image
)
from utils.visualizer import (
    plot_prediction_gauge,
    plot_segmentation_masks,
    plot_morphology_operations,
    plot_features_table
)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="TB Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - MONOCHROME BLACK & WHITE
# ============================================================================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #ffffff;
    }
    
    /* Hero section */
    .hero-title {
        font-size: 3.5em;
        font-weight: 800;
        color: #000000;
        text-align: center;
        margin: 40px 0 20px 0;
        letter-spacing: -1px;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.3em;
        color: #666666;
        text-align: center;
        margin-bottom: 50px;
        font-weight: 300;
    }
    
    /* Pipeline steps */
    .pipeline-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin: 40px 0;
    }
    
    .pipeline-step {
        flex: 1;
        text-align: center;
        padding: 30px 20px;
        background: #f8f9fa;
        border: 2px solid #000000;
        border-radius: 8px;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .pipeline-step:hover {
        background: #000000;
        color: #ffffff;
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    .pipeline-step:hover .step-icon {
        color: #ffffff;
    }
    
    .pipeline-step:hover .step-number {
        background: #ffffff;
        color: #000000;
    }
    
    .step-number {
        display: inline-block;
        width: 40px;
        height: 40px;
        line-height: 40px;
        border-radius: 50%;
        background: #000000;
        color: #ffffff;
        font-weight: bold;
        margin-bottom: 15px;
    }
    
    .step-icon {
        font-size: 2em;
        margin-bottom: 10px;
        color: #000000;
    }
    
    .step-title {
        font-size: 1.1em;
        font-weight: 700;
        margin: 10px 0;
        color: inherit;
    }
    
    .step-description {
        font-size: 0.9em;
        color: #666666;
        margin-top: 5px;
    }
    
    .pipeline-step:hover .step-description {
        color: #cccccc;
    }
    
    /* Arrow between steps */
    .pipeline-arrow {
        position: absolute;
        right: -30px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 2em;
        color: #cccccc;
    }
    
    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 30px;
        margin: 40px 0;
    }
    
    .feature-card {
        padding: 40px 30px;
        background: #ffffff;
        border: 2px solid #000000;
        border-radius: 8px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        background: #000000;
        color: #ffffff;
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .feature-card:hover .feature-icon {
        color: #ffffff;
    }
    
    .feature-icon {
        font-size: 3em;
        margin-bottom: 20px;
        color: #000000;
    }
    
    .feature-title {
        font-size: 1.3em;
        font-weight: 700;
        margin-bottom: 15px;
        color: inherit;
    }
    
    .feature-description {
        font-size: 1em;
        color: #666666;
        line-height: 1.6;
    }
    
    .feature-card:hover .feature-description {
        color: #cccccc;
    }
    
    /* Stats section */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin: 40px 0;
        padding: 40px;
        background: #f8f9fa;
        border: 2px solid #000000;
        border-radius: 8px;
    }
    
    .stat-item {
        text-align: center;
        padding: 20px;
        border-right: 1px solid #dddddd;
    }
    
    .stat-item:last-child {
        border-right: none;
    }
    
    .stat-number {
        font-size: 2.5em;
        font-weight: 800;
        color: #000000;
        margin-bottom: 10px;
    }
    
    .stat-label {
        font-size: 0.9em;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 2em;
        font-weight: 700;
        color: #000000;
        text-align: center;
        margin: 60px 0 30px 0;
        position: relative;
        padding-bottom: 15px;
    }
    
    .section-header:after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 3px;
        background: #000000;
    }
    
    /* CTA section */
    .cta-container {
        text-align: center;
        padding: 50px;
        background: #000000;
        color: #ffffff;
        border-radius: 8px;
        margin: 40px 0;
    }
    
    .cta-title {
        font-size: 2em;
        font-weight: 700;
        margin-bottom: 20px;
    }
    
    .cta-description {
        font-size: 1.2em;
        color: #cccccc;
        margin-bottom: 30px;
    }
    
    /* Technique badges */
    .technique-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 15px;
        margin: 30px 0;
    }
    
    .technique-badge {
        padding: 15px;
        background: #ffffff;
        border: 2px solid #000000;
        border-radius: 50px;
        text-align: center;
        font-weight: 600;
        font-size: 0.9em;
        transition: all 0.3s ease;
    }
    
    .technique-badge:hover {
        background: #000000;
        color: #ffffff;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = None

# ============================================================================
# LOAD MODEL (Cached)
# ============================================================================
@st.cache_resource
def get_model():
    """Load model once and cache it"""
    with st.spinner("Loading AI model..."):
        return load_model()

# ============================================================================
# PROGRESS CALLBACK
# ============================================================================
def create_progress_callback(progress_bar, status_text):
    """Create callback function for progress updates"""
    def callback(progress: float, message: str):
        progress_bar.progress(progress)
        status_text.text(message)
    return callback

# ============================================================================
# LANDING PAGE COMPONENTS
# ============================================================================
def render_landing_page():
    """Render landing page with monochrome black & white design"""
    
    # Hero section
    st.markdown("""
    <div class="hero-title">
        🫁 TB Chest X-Ray Detection
    </div>
    <div class="hero-subtitle">
        AI-Powered Tuberculosis Detection from Medical Imaging
    </div>
    """, unsafe_allow_html=True)
    
    # Stats section
    st.markdown("""
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-number">14</div>
            <div class="stat-label">Features</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">5</div>
            <div class="stat-label">Pipeline Steps</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">10+</div>
            <div class="stat-label">Algorithms</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">2</div>
            <div class="stat-label">Classes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Section: Processing Pipeline
    st.markdown('<div class="section-header">Processing Pipeline</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="pipeline-step">
            <div class="step-number">1</div>
            <div class="step-icon">📥</div>
            <div class="step-title">Preprocessing</div>
            <div class="step-description">CLAHE + Gaussian Blur</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pipeline-step">
            <div class="step-number">2</div>
            <div class="step-icon">🎯</div>
            <div class="step-title">Segmentation</div>
            <div class="step-description">K-Means Clustering</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="pipeline-step">
            <div class="step-number">3</div>
            <div class="step-icon">⚙️</div>
            <div class="step-title">Morphology</div>
            <div class="step-description">Erosion + Dilation</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="pipeline-step">
            <div class="step-number">4</div>
            <div class="step-icon">📊</div>
            <div class="step-title">Features</div>
            <div class="step-description">LBP + GLCM + Hough</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="pipeline-step">
            <div class="step-number">5</div>
            <div class="step-icon">🤖</div>
            <div class="step-title">Classification</div>
            <div class="step-description">SLDT + MSA</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Section: Techniques Used
    st.markdown('<div class="section-header">Techniques & Algorithms</div>', unsafe_allow_html=True)
    
    techniques = [
        "K-Means", "CLAHE", "Gaussian Blur", "LBP", "GLCM",
        "Hough Transform", "SLDT", "MSA", "Morphology", "Edge Detection"
    ]
    
    cols = st.columns(5)
    for idx, tech in enumerate(techniques):
        with cols[idx % 5]:
            st.markdown(f"""
            <div class="technique-badge">
                {tech}
            </div>
            """, unsafe_allow_html=True)
    
    # Section: Key Features
    st.markdown('<div class="section-header">Key Features</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Accurate Detection</div>
            <div class="feature-description">
                Machine learning powered classification with optimized 
                feature selection using Moth Search Algorithm
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Fast Processing</div>
            <div class="feature-description">
                Automated pipeline processes chest X-Ray images 
                in seconds with real-time progress tracking
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Detailed Analysis</div>
            <div class="feature-description">
                Complete visualization of segmentation masks, 
                morphological operations, and extracted features
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("""
    <div class="cta-container">
        <div class="cta-title">Ready to Analyze?</div>
        <div class="cta-description">
            Upload a chest X-Ray image from the sidebar to get started with TB detection
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical details
    with st.expander("🔬 Technical Details & Pipeline Breakdown"):
        st.markdown("""
        ### Complete Processing Pipeline
        
        #### 1. **Preprocessing** 📥
        - **Grayscale Conversion**: Convert RGB to grayscale
        - **Gaussian Blur**: 3×3 kernel for noise reduction
        - **CLAHE**: Contrast Limited Adaptive Histogram Equalization
          - Clip Limit: 2.0
          - Tile Grid Size: 8×8
        
        #### 2. **Segmentation** 🎯
        - **K-Means Clustering**: Segment lung regions
          - Number of clusters: 3
          - Random state: 42
        - **Adaptive Thresholding**: Detect nodules and cavities
          - Block size: 21
          - Method: Gaussian
        
        #### 3. **Morphological Operations** ⚙️
        - **Otsu Thresholding**: Automatic threshold calculation
        - **Erosion**: Remove noise (5×5 kernel)
        - **Dilation**: Fill gaps (5×5 kernel)
        - **Opening**: Remove small objects
        - **Closing**: Fill small holes
        
        #### 4. **Feature Extraction** 📊
        - **Edge Features**
          - Canny edge detection (threshold: 100, 200)
          - Total edge sum
        
        - **Line Features**
          - Hough line transform
          - Line count and distribution
        
        - **Texture Features (GLCM)**
          - Contrast: Measure of local variations
          - Homogeneity: Measure of uniformity
          - Distance: [1], Angle: [0]
        
        - **Pattern Features (LBP)**
          - Local Binary Pattern histogram
          - Points: 8, Radius: 1
          - Method: uniform
          - Output: 9 bins
        
        #### 5. **Classification** 🤖
        - **Feature Selection**: Moth Search Algorithm (MSA)
          - Population: 8 moths
          - Iterations: 10
          - Metric: F1-Score (macro)
        
        - **Classifier**: Stacking Loopy Decision Tree (SLDT)
          - Base Learners:
            - Decision Tree (entropy, balanced)
            - Random Forest (10 trees)
          - Meta Learner: Decision Tree
          - Hyperparameter Tuning: Grid Search
        
        ### Model Architecture
        ```
        Input (14 features)
              ↓
        Feature Selection (MSA)
              ↓
        Selected Features
              ↓
        ┌─────────────────────┐
        │  Base Learners      │
        │  - Decision Tree    │
        │  - Random Forest    │
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │  Meta Learner       │
        │  - Decision Tree    │
        └─────────────────────┘
              ↓
        Output: Normal / TB
        ```
        
        ### Feature Set (14 Features)
        1. Edge Sum (Canny)
        2. Number of Lines (Hough)
        3. GLCM Contrast
        4. GLCM Homogeneity
        5-13. LBP Histogram (9 bins)
        14. Corner Count
        """)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Title
    st.title("🫁 TB Chest X-Ray Detection System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📤 Upload X-Ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose an X-Ray image",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a chest X-Ray image for TB detection"
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Options
            st.markdown("---")
            st.subheader("⚙️ Options")
            include_morphology = st.checkbox(
                "Include morphology analysis",
                value=True,
                help="Compute morphological operations (slower but more detailed)"
            )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info(
            "This system uses machine learning to detect tuberculosis "
            "from chest X-ray images. Upload an image and click Analyze."
        )
        
        st.markdown("---")
        st.markdown("### 👥 Tim Pengembang")
        st.markdown("""
        - **Azhar Maulana**  
          *Preprocessing*
        
        - **Revy Satya Gunawan**  
          *Segmentation*
        
        - **Raditya Nathaniel Nugroho**  
          *Morphological Processing*
        
        - **Benedictus Erwin Widianto**  
          *Feature Extraction & Lead*
        """)
    
    # Main Area - Show landing page if no image uploaded
    if uploaded_file is None:
        render_landing_page()
        return
    
    # Analyze Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button(
            "🔬 Analyze Image",
            type="primary",
            use_container_width=True
        )
    
    # Handle Analysis
    if analyze_button:
        # Save uploaded file
        temp_path = save_uploaded_file(uploaded_file)
        st.session_state.temp_file_path = temp_path
        
        # Validate image
        is_valid, error_msg = validate_image(temp_path)
        if not is_valid:
            st.error(f"❌ Invalid image: {error_msg}")
            cleanup_temp_file(temp_path)
            return
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Load model
            model = get_model()
            
            # Analyze with progress callback
            callback = create_progress_callback(progress_bar, status_text)
            results = analyze_image(
                temp_path,
                model,
                include_morphology=include_morphology,
                progress_callback=callback
            )
            
            # Store results in session state
            st.session_state.analysis_results = results
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Success message
            st.success("✅ Analysis complete!")
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
            return
    
    # Display Results
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        
        # Prediction Card
        st.markdown("## 📊 Prediction Result")
        
        pred = results['prediction']
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Metric card
            label = pred['label']
            confidence = pred['confidence']
            
            # Color based on prediction
            if label == "Normal":
                st.success(f"### {label}")
            else:
                st.error(f"### {label}")
            
            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )
        
        with col2:
            # Probability bars
            st.markdown("#### Probabilities")
            prob_normal = pred['probabilities']['Normal']
            prob_tb = pred['probabilities']['Tuberculosis']
            
            st.progress(prob_normal, text=f"Normal: {prob_normal*100:.1f}%")
            st.progress(prob_tb, text=f"TB: {prob_tb*100:.1f}%")
        
        st.markdown("---")
        
        # Detailed Results Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔬 Segmentation",
            "⚙️ Morphology",
            "📈 Features",
            "🖼️ Original"
        ])
        
        with tab1:
            st.markdown("### Lung Segmentation Results")
            plot_segmentation_masks(results['segments'])
        
        with tab2:
            if 'morphology' in results:
                st.markdown("### Morphological Operations")
                plot_morphology_operations(results['morphology'])
            else:
                st.info("Morphology analysis was skipped")
        
        with tab3:
            st.markdown("### Extracted Features")
            plot_features_table(results['features'])
        
        with tab4:
            st.markdown("### Original & Preprocessed")
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Original", use_column_width=True)
            with col2:
                st.image(
                    results['preprocessed'],
                    caption="Preprocessed (CLAHE)",
                    use_column_width=True,
                    clamp=True
                )
        
        # Download Results
        st.markdown("---")
        st.markdown("### 💾 Download Results")
        
        # TODO: Implement download functionality
        st.info("Download feature coming soon!")

if __name__ == "__main__":
    main()