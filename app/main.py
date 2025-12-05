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
# CUSTOM CSS FOR MARQUEE
# ============================================================================
st.markdown("""
<style>
    .marquee-container {
        width: 100%;
        overflow: hidden;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px 0;
        border-radius: 10px;
        margin: 20px 0;
    }
    
    .marquee {
        display: flex;
        animation: scroll 30s linear infinite;
        white-space: nowrap;
    }
    
    .marquee-item {
        display: inline-flex;
        align-items: center;
        margin: 0 30px;
        padding: 15px 25px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50px;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .marquee-item:hover {
        transform: scale(1.1);
        background: rgba(255, 255, 255, 0.2);
    }
    
    .marquee-icon {
        font-size: 24px;
        margin-right: 10px;
    }
    
    .marquee-text {
        font-size: 16px;
        font-weight: 600;
        color: white;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    @keyframes scroll {
        0% {
            transform: translateX(0);
        }
        100% {
            transform: translateX(-50%);
        }
    }
    
    .pipeline-step {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        margin: 10px;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .pipeline-step:hover {
        transform: translateY(-5px);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
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
    """Render landing page with animated pipeline"""
    
    # Hero section
    st.markdown("""
    <div style='text-align: center; padding: 40px 0;'>
        <h1 style='font-size: 3em; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🫁 TB Chest X-Ray Detection
        </h1>
        <p style='font-size: 1.2em; color: #666; margin-top: 10px;'>
            AI-Powered Tuberculosis Detection from Medical Imaging
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Marquee of techniques
    techniques = [
        ("🔄", "K-Means Clustering"),
        ("✨", "CLAHE Enhancement"),
        ("🌀", "Gaussian Blur"),
        ("🔍", "LBP Features"),
        ("📊", "GLCM Texture"),
        ("📏", "Hough Transform"),
        ("🌳", "SLDT Classifier"),
        ("🦋", "MSA Optimization"),
        ("⚙️", "Morphology Ops"),
        ("🎯", "Edge Detection"),
    ]
    
    # Double the techniques for seamless loop
    techniques_doubled = techniques + techniques
    
    marquee_html = """
    <div class="marquee-container">
        <div class="marquee">
    """
    
    for icon, name in techniques_doubled:
        marquee_html += f"""
            <div class="marquee-item">
                <span class="marquee-icon">{icon}</span>
                <span class="marquee-text">{name}</span>
            </div>
        """
    
    marquee_html += """
        </div>
    </div>
    """
    
    st.markdown(marquee_html, unsafe_allow_html=True)
    
    # Pipeline explanation
    st.markdown("### 🔄 Complete Processing Pipeline")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="pipeline-step">
            📥<br>Preprocessing<br>
            <small>CLAHE + Gaussian</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pipeline-step">
            🎯<br>Segmentation<br>
            <small>K-Means (k=3)</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="pipeline-step">
            ⚙️<br>Morphology<br>
            <small>Erosion + Dilation</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="pipeline-step">
            📊<br>Features<br>
            <small>LBP + GLCM + Hough</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="pipeline-step">
            🤖<br>Classification<br>
            <small>SLDT + MSA</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("---")
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Accurate Detection</h3>
            <p>Machine learning powered classification with optimized feature selection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Fast Processing</h3>
            <p>Automated pipeline processes images in seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Detailed Analysis</h3>
            <p>Complete visualization of segmentation, features, and predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Upload prompt
    st.markdown("---")
    st.info("👈 **Get Started**: Upload a chest X-Ray image from the sidebar to begin analysis!")
    
    # Technical details
    with st.expander("🔬 Technical Details"):
        st.markdown("""
        #### Processing Pipeline:
        
        1. **Preprocessing** 🔄
           - Grayscale conversion
           - Gaussian Blur (3×3 kernel) - noise reduction
           - CLAHE (Contrast Limited Adaptive Histogram Equalization) - contrast enhancement
        
        2. **Segmentation** 🎯
           - K-Means clustering (k=3) for lung region isolation
           - Adaptive thresholding for nodule detection
           - Adaptive thresholding for cavity detection
        
        3. **Morphological Operations** ⚙️
           - Otsu thresholding
           - Erosion & Dilation
           - Opening & Closing operations
        
        4. **Feature Extraction** 📊
           - **Edge Features**: Canny edge detection
           - **Line Features**: Hough line transform
           - **Texture Features**: GLCM (contrast, homogeneity)
           - **Pattern Features**: LBP histogram (9 bins)
        
        5. **Classification** 🤖
           - **SLDT**: Stacking Loopy Decision Tree
           - **MSA**: Moth Search Algorithm for feature selection
           - Output: Normal vs Tuberculosis with confidence score
        
        #### Model Performance:
        - **Feature Selection**: Moth Search Algorithm
        - **Base Classifiers**: Decision Tree + Random Forest
        - **Meta Classifier**: Decision Tree
        - **Optimization**: Grid Search (class weights + max depth)
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