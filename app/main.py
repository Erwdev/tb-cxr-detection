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
# CUSTOM CSS - GITHUB DARK MODE THEME
# ============================================================================
st.markdown("""
<style>
    /* GITHUB DARK MODE COLOR PALETTE 
    Background: #0d1117
    Panel/Canvas: #161b22
    Border: #30363d
    Text Main: #c9d1d9
    Text Muted: #8b949e
    Link Blue: #58a6ff
    Button Green: #238636
    */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Streamlit Overrides */
    .stApp {
        background-color: #0d1117;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
        color: #c9d1d9;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #010409;
        border-right: 1px solid #30363d;
    }
    
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #c9d1d9;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        max-width: 100%;
    }

    /* README.md Container Style */
    .readme-container {
        border: 1px solid #30363d;
        border-radius: 6px;
        background-color: #0d1117;
        margin-bottom: 24px;
    }

    .readme-header {
        background-color: #161b22;
        border-bottom: 1px solid #30363d;
        padding: 16px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-weight: 600;
        color: #c9d1d9;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .readme-body {
        padding: 32px;
    }

    /* Typography */
    h1 {
        color: #c9d1d9 !important;
        font-weight: 400 !important;
        font-size: 2rem !important;
        border-bottom: none !important;
    }
    
    h2, h3 {
        color: #c9d1d9 !important;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.3em;
    }

    p, li {
        color: #c9d1d9;
        line-height: 1.5;
    }

    /* Pinned Repos / Features Grid */
    .pinned-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr); /* 2 columns like GitHub pinned */
        gap: 16px;
        margin-bottom: 24px;
    }

    .pinned-card {
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 16px;
        background-color: #0d1117;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .pinned-card:hover {
        border-color: #8b949e;
    }

    .repo-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
        font-weight: 600;
        color: #58a6ff;
    }

    .repo-desc {
        font-size: 12px;
        color: #8b949e;
        margin-bottom: 16px;
    }

    .repo-meta {
        display: flex;
        align-items: center;
        gap: 16px;
        font-size: 12px;
        color: #8b949e;
    }

    .lang-color {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 4px;
    }

    /* Pipeline / Contribution Graph Style */
    .contribution-graph {
        display: flex;
        gap: 4px;
        margin-top: 10px;
    }
    
    .contrib-box {
        width: 100%;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px;
        text-align: left;
    }
    
    .contrib-title {
        color: #c9d1d9;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 4px;
    }
    
    .contrib-desc {
        color: #8b949e;
        font-size: 12px;
    }

    /* Stats sidebar (Right side style) */
    .side-panel {
        background-color: #0d1117;
        border-bottom: 1px solid #30363d;
        padding-bottom: 16px;
        margin-bottom: 16px;
    }
    
    .side-title {
        font-weight: 600;
        color: #c9d1d9;
        margin-bottom: 8px;
        font-size: 14px;
    }
    
    .side-item {
        color: #8b949e;
        font-size: 12px;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        font-weight: 500;
    }
    
    div.stButton > button:hover {
        background-color: #30363d;
        border-color: #8b949e;
        color: #ffffff;
    }
    
    /* Primary Action Button (Green) */
    div.stButton > button[kind="primary"] {
        background-color: #238636;
        color: white;
        border: 1px solid rgba(240,246,252,0.1);
    }
    
    div.stButton > button[kind="primary"]:hover {
        background-color: #2ea043;
        border-color: rgba(240,246,252,0.1);
    }

    /* Metrics / Alert Boxes */
    div[data-testid="stMetricValue"] {
        color: #c9d1d9;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #8b949e;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px 6px 0 0;
        color: #c9d1d9;
        border: none;
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-bottom: none;
        color: #c9d1d9;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #161b22;
        border-radius: 6px;
        color: #c9d1d9;
    }
    
    hr {
        border-color: #30363d;
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
    """Render landing page with GitHub-inspired styling"""
    
    # "Pinned" Repositories / Features
    st.markdown("### Pinned Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="pinned-grid" style="display:block; margin-bottom: 0;">
            <div class="pinned-card">
                <div>
                    <div class="repo-header">
                        <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" fill="#8b949e"><path d="M2 2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75h-2.5a.75.75 0 0 1 0-1.5h1.75v-2h-8a1 1 0 0 0-.714 1.7.75.75 0 1 1-1.072 1.05A2.495 2.495 0 0 1 2 11.5Zm10.5-1h-8a1 1 0 0 0-1 1v6.708A2.486 2.486 0 0 1 4.5 9h8ZM5 12.25a.25.25 0 0 1 .25-.25h3.5a.25.25 0 0 1 .25.25v3.25a.25.25 0 0 1-.4.2l-1.45-1.087a.249.249 0 0 0-.3 0L5.4 15.7a.25.25 0 0 1-.4-.2Z"></path></svg>
                        Moth-Search-Optimization
                    </div>
                    <div class="repo-desc">Feature selection utilizing MSA meta-heuristics for optimal input features.</div>
                </div>
                <div class="repo-meta">
                    <span><span class="lang-color" style="background-color: #3572A5;"></span>Python</span>
                    <span>★ 14 Features</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="pinned-grid" style="display:block; margin-top: 16px;">
            <div class="pinned-card">
                <div>
                    <div class="repo-header">
                        <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" fill="#8b949e"><path d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Z"></path></svg>
                        Segmentation-Engine
                    </div>
                    <div class="repo-desc">K-Means clustering and adaptive thresholding for lung region isolation.</div>
                </div>
                <div class="repo-meta">
                    <span><span class="lang-color" style="background-color: #f1e05a;"></span>OpenCV</span>
                    <span>★ 98% Acc</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pinned-grid" style="display:block; margin-bottom: 0;">
             <div class="pinned-card">
                <div>
                    <div class="repo-header">
                        <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" fill="#8b949e"><path d="M8.75.75V2h.985c.304 0 .603.08.867.231l1.29.736c.038.022.08.033.124.033h2.234a.75.75 0 0 1 0 1.5h-.427l2.111 4.692a.75.75 0 0 1-.154.838l-.53-.53.529.531-.001.002-.002.002-.006.006-.006.005-.01.01-.045.04c-.21.176-.441.327-.686.45C13.556 10.943 11.372 11.5 8 11.5c-3.372 0-5.556-.557-7.179-1.947a3.74 3.74 0 0 1-.686-.45l-.045-.04-.016-.015-.006-.006-.004-.004v-.001a.75.75 0 0 1-.154-.838L2.078 4.5H1.75a.75.75 0 0 1 0-1.5h2.234a.249.249 0 0 1 .125-.033l1.29-.736A1.75 1.75 0 0 1 6.265 2h.985V.75a.75.75 0 0 1 1.5 0Zm-4.69 2h7.88a.25.25 0 0 1 .142.43l-3.94 2.25a.25.25 0 0 1-.248 0L4.202 3.18a.25.25 0 0 1 .142-.43Zm2.188 7.027c-1.258.274-2.305.624-2.926 1.156-.475.408-.863.896-1.144 1.455a4.757 4.757 0 0 0-.256.62.75.75 0 0 0 .68.966h10.822a.75.75 0 0 0 .68-.966 4.757 4.757 0 0 0-.256-.62c-.281-.559-.669-1.047-1.144-1.455-.621-.532-1.668-.882-2.926-1.156L8 8.87l-1.752 1.907Z"></path></svg>
                        Ensemble-Classifier
                    </div>
                    <div class="repo-desc">Stacking Loopy Decision Tree (SLDT) using Random Forest & Decision Trees.</div>
                </div>
                <div class="repo-meta">
                    <span><span class="lang-color" style="background-color: #DA5B0B;"></span>Sklearn</span>
                    <span>★ Stacked Model</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="pinned-grid" style="display:block; margin-top: 16px;">
            <div class="pinned-card">
                <div>
                    <div class="repo-header">
                        <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" fill="#8b949e"><path d="M4.75 0a.75.75 0 0 1 .75.75V2h5v-.75a.75.75 0 0 1 1.5 0V2h1.25c.966 0 1.75.784 1.75 1.75v10.5A1.75 1.75 0 0 1 13.25 16H2.75A1.75 1.75 0 0 1 1 14.25V3.75C1 2.784 1.784 2 2.75 2H4V.75A.75.75 0 0 1 4.75 0ZM2.5 7.5v6.75c0 .138.112.25.25.25h10.5a.25.25 0 0 0 .25-.25V7.5Zm10.75-4H2.75a.25.25 0 0 0-.25.25V6h11V3.75a.25.25 0 0 0-.25-.25Z"></path></svg>
                        Dashboard-Analytics
                    </div>
                    <div class="repo-desc">Real-time morphology visualization and feature extraction tables.</div>
                </div>
                <div class="repo-meta">
                    <span><span class="lang-color" style="background-color: #563d7c;"></span>CSS</span>
                    <span>★ Interactive</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # "Readme.md" Style Container for Pipeline
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_readme, col_side = st.columns([3, 1])
    
    with col_readme:
        st.markdown("""
        <div class="readme-container">
            <div class="readme-header">
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" fill="#c9d1d9"><path d="M2 2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75h-2.5a.75.75 0 0 1 0-1.5h1.75v-2h-8a1 1 0 0 0-.714 1.7.75.75 0 1 1-1.072 1.05A2.495 2.495 0 0 1 2 11.5Zm10.5-1h-8a1 1 0 0 0-1 1v6.708A2.486 2.486 0 0 1 4.5 9h8ZM5 12.25a.25.25 0 0 1 .25-.25h3.5a.25.25 0 0 1 .25.25v3.25a.25.25 0 0 1-.4.2l-1.45-1.087a.249.249 0 0 0-.3 0L5.4 15.7a.25.25 0 0 1-.4-.2Z"></path></svg>
                <span>README.md</span>
            </div>
            <div class="readme-body">
                <h3>Pipeline Architecture</h3>
                <p>The system follows a sequential processing pipeline designed to handle chest X-ray imagery. Each step is modularized for maximum maintainability.</p>
                <br>
                <div class="contribution-graph">
                    <div class="contrib-box">
                        <div class="contrib-title">1. Preprocessing</div>
                        <div class="contrib-desc">CLAHE + Gaussian Blur</div>
                    </div>
                    <div class="contrib-box">
                        <div class="contrib-title">2. Segmentation</div>
                        <div class="contrib-desc">K-Means + Adaptive Thresh</div>
                    </div>
                    <div class="contrib-box">
                        <div class="contrib-title">3. Features</div>
                        <div class="contrib-desc">GLCM + LBP + Hough</div>
                    </div>
                    <div class="contrib-box">
                        <div class="contrib-title">4. Classification</div>
                        <div class="contrib-desc">SLDT + MSA Optimization</div>
                    </div>
                </div>
                <br>
                <h3>Getting Started</h3>
                <p>Upload a file from the sidebar to trigger the <code>analyze_image</code> workflow. The system will automatically generate segmentation masks and extract relevant morphological features.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_side:
        # "About" sidebar style
        st.markdown("""
        <div class="side-panel">
            <div class="side-title">About</div>
            <div style="font-size: 14px; color: #c9d1d9; margin-bottom: 16px;">
                TB Chest X-Ray Detection System utilizing machine learning algorithms.
            </div>
            <div class="side-item">
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" fill="#8b949e"><path d="M11.93 8.5a4.002 4.002 0 0 1-7.86 0H.75a.75.75 0 0 1 0-1.5h3.32a4.002 4.002 0 0 1 7.86 0h3.32a.75.75 0 0 1 0 1.5h-3.32Z"></path></svg>
                v2.1.0 Release
            </div>
            <div class="side-item">
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" fill="#8b949e"><path d="M2.38 2.5a.5.5 0 0 1 .5-.5h10.24a.5.5 0 0 1 .5.5v11a.5.5 0 0 1-.5.5H2.88a.5.5 0 0 1-.5-.5v-11Z"></path></svg>
                Readme
            </div>
            <div class="side-item">
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" fill="#8b949e"><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM5.78 8.75a9.64 9.64 0 0 0 1.363 4.177c.255.426.542.832.857 1.215.245-.296.551-.705.857-1.215A9.64 9.64 0 0 0 10.22 8.75H5.78Zm.643-2.352a9.63 9.63 0 0 0-.76 2.352h4.674c-.18-1.026-.456-1.85-.76-2.352a12.502 12.502 0 0 0-.9-1.222 12.502 12.502 0 0 0-.901 1.222ZM4.819 6.4c.159.98.416 1.83.762 2.35h-3.53a6.47 6.47 0 0 1 2.768-2.35Zm-1.801 3.85h3.53c-.346.52-.603 1.37-.762 2.35a6.47 6.47 0 0 1-2.768-2.35Zm10.023-.75H9.558a9.63 9.63 0 0 0 .76-2.35 6.47 6.47 0 0 1 2.768 2.35Zm-2.768 3.85c.31-.52.566-1.37.76-2.35h3.531a6.47 6.47 0 0 1-2.768 2.35ZM8.75 1.776a14.04 14.04 0 0 1 .83 1.132c.16.24.316.494.467.76H5.953c.151-.266.307-.52.467-.76.248-.372.544-.763.83-1.132.18-.232.364-.46.55-.683.186.223.37.45.55.683Zm-5.25 3.51a14.004 14.004 0 0 1 1.85-2.008 6.47 6.47 0 0 0-1.85 2.008Zm1.85 5.428a14.004 14.004 0 0 1-1.85-2.008 6.47 6.47 0 0 0 1.85 2.008Zm8.95-3.42a6.47 6.47 0 0 0-1.85-2.008 14.004 14.004 0 0 1 1.85 2.008Zm-1.85 5.428a6.47 6.47 0 0 0 1.85-2.008 14.004 14.004 0 0 1-1.85 2.008Z"></path></svg>
                Public Access
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="side-panel" style="border-bottom: none;">
            <div class="side-title">Languages</div>
             <div class="side-item">
                <span class="lang-color" style="background-color: #3572A5;"></span> Python 85%
            </div>
             <div class="side-item">
                <span class="lang-color" style="background-color: #563d7c;"></span> CSS 10%
            </div>
             <div class="side-item">
                <span class="lang-color" style="background-color: #f1e05a;"></span> Shell 5%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical details expander
    with st.expander("Show detailed workflow spec"):
        st.markdown("""
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
    # Title styling to look like Repo Header
    col_icon, col_title = st.columns([1, 15])
    with col_icon:
         st.markdown("## 🫁") 
    with col_title:
        st.markdown("## kelompok 3/tb-detection-system")
        st.caption("Public • v2.1.0 • Updated 2 hours ago")

    st.markdown("---")
    
    # Sidebar
    with st.sidebar:

        st.markdown("### 📤 Upload X-Ray")
        
        uploaded_file = st.file_uploader(
            "Choose file",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a chest X-Ray image for TB detection"
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="preview.jpg", use_column_width=True)
            
            # Options
            st.markdown("### ⚙️ Settings")
            include_morphology = st.checkbox(
                "Include morphology",
                value=True,
                help="Compute morphological operations"
            )
        
        st.markdown("---")
        st.markdown("### Contributors")
        st.markdown("""
        <div style="font-size: 12px; color: #8b949e;">
        <p>@azhar-maulana <br>
        @revy-satya <br>
        @raditya-nathaniel <br>
        @erwin-widianto</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Area - Show landing page if no image uploaded
    if uploaded_file is None:
        render_landing_page()
        return
    
    # Analyze Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        analyze_button = st.button(
            "Go to file",
            use_container_width=True
        )
    with col2:
        analyze_button = st.button(
            "Run Analysis",
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
        
        # Prediction Card wrapped in Readme style
        st.markdown('<div class="readme-container">', unsafe_allow_html=True)
        st.markdown('<div class="readme-header">📊 prediction_result.json</div>', unsafe_allow_html=True)
        st.markdown('<div class="readme-body">', unsafe_allow_html=True)
        
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
                "Confidence Score",
                f"{confidence*100:.2f}%"
            )
        
        with col2:
            # Probability bars
            st.markdown("**Class Probabilities**")
            prob_normal = pred['probabilities']['Normal']
            prob_tb = pred['probabilities']['Tuberculosis']
            
            st.progress(prob_normal, text=f"Normal: {prob_normal*100:.1f}%")
            st.progress(prob_tb, text=f"TB: {prob_tb*100:.1f}%")
        
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Detailed Results Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "masks.png",
            "morphology.png",
            "features.csv",
            "source.jpg"
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
        st.info("Download feature coming soon!")

if __name__ == "__main__":
    main()