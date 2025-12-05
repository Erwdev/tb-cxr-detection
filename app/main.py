"""
Streamlit App - TB Detection System
Handles async-like processing with progress indicators
"""
import streamlit as st
import os
import sys

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
    
    # Main Area
    if uploaded_file is None:
        st.info("👈 Please upload an X-Ray image to begin analysis")
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