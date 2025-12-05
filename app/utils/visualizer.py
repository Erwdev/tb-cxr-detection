"""
Visualization Utilities untuk Streamlit App
Plotting functions untuk semua hasil analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import cv2
from typing import Dict
import plotly.graph_objects as go
import pandas as pd

# ============================================================================
# PREDICTION VISUALIZATION
# ============================================================================
def plot_prediction_gauge(prediction: Dict):
    """
    Plot gauge chart untuk prediction confidence
    
    Args:
        prediction: Dict dengan keys: label, confidence, probabilities
    """
    confidence = prediction['confidence']
    label = prediction['label']
    
    # Determine color based on prediction
    color = "green" if label == "Normal" else "red"
    
    # Create gauge chart dengan Plotly
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Prediction: {label}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_probability_bars(probabilities: Dict):
    """
    Plot horizontal bar chart untuk probabilities
    
    Args:
        probabilities: Dict dengan keys: Normal, Tuberculosis
    """
    labels = list(probabilities.keys())
    values = [probabilities[k] * 100 for k in labels]
    colors = ['green', 'red']
    
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Probability (%)",
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SEGMENTATION VISUALIZATION
# ============================================================================
def plot_segmentation_masks(segments: Dict):
    """
    Plot segmentation masks (lung, nodule, cavity)
    
    Args:
        segments: Dict dengan keys: lung, nodule, cavity, lung_roi
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Lung Mask")
        st.image(
            segments['lung'],
            caption="Lung Region (K-Means)",
            use_column_width=True,
            clamp=True
        )
    
    with col2:
        st.markdown("#### Nodule Mask")
        st.image(
            segments['nodule'],
            caption="Potential Nodules",
            use_column_width=True,
            clamp=True
        )
    
    with col3:
        st.markdown("#### Cavity Mask")
        st.image(
            segments['cavity'],
            caption="Potential Cavities",
            use_column_width=True,
            clamp=True
        )
    
    # Overlay visualization
    st.markdown("---")
    st.markdown("#### Combined Overlay")
    
    # Create RGB overlay
    lung_roi = segments['lung_roi']
    overlay = cv2.cvtColor(lung_roi, cv2.COLOR_GRAY2RGB)
    
    # Add colored masks
    overlay[segments['nodule'] > 0] = [255, 0, 0]  # Red for nodules
    overlay[segments['cavity'] > 0] = [0, 0, 255]  # Blue for cavities
    
    st.image(
        overlay,
        caption="Overlay: Red=Nodules, Blue=Cavities",
        use_column_width=True,
        clamp=True
    )


# ============================================================================
# MORPHOLOGY VISUALIZATION
# ============================================================================
def plot_morphology_operations(morphology: Dict):
    """
    Plot morphological operations results
    
    Args:
        morphology: Dict dengan keys: lung, nodule, cavity
                   Each contains: original, otsu_binary, eroded, dilated, opened, closed
    """
    # Tabs untuk setiap mask type
    tab1, tab2, tab3 = st.tabs(["Lung", "Nodule", "Cavity"])
    
    for tab, (name, morph_dict) in zip([tab1, tab2, tab3], morphology.items()):
        with tab:
            st.markdown(f"#### {name.capitalize()} Morphology")
            
            # Create 2x3 grid
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(
                    morph_dict['original'],
                    caption="Original",
                    use_column_width=True,
                    clamp=True
                )
                st.image(
                    morph_dict['eroded'],
                    caption="Eroded",
                    use_column_width=True,
                    clamp=True
                )
            
            with col2:
                st.image(
                    morph_dict['otsu_binary'],
                    caption=f"Otsu (threshold={morph_dict['threshold_value']:.0f})",
                    use_column_width=True,
                    clamp=True
                )
                st.image(
                    morph_dict['dilated'],
                    caption="Dilated",
                    use_column_width=True,
                    clamp=True
                )
            
            with col3:
                st.image(
                    morph_dict['opened'],
                    caption="Opened",
                    use_column_width=True,
                    clamp=True
                )
                st.image(
                    morph_dict['closed'],
                    caption="Closed",
                    use_column_width=True,
                    clamp=True
                )


# ============================================================================
# FEATURE VISUALIZATION
# ============================================================================
def plot_features_table(features: Dict):
    """
    Display extracted features in a table
    
    Args:
        features: Dict dengan keys: edge_sum, num_lines, glcm_*, lbp_hist
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Shape Features")
        
        shape_data = {
            "Feature": ["Edge Sum", "Number of Lines"],
            "Value": [
                f"{features['edge_sum']:.2f}",
                f"{features['num_lines']}"
            ]
        }
        st.dataframe(
            pd.DataFrame(shape_data),
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown("#### Texture Features (GLCM)")
        
        texture_data = {
            "Feature": ["Contrast", "Homogeneity"],
            "Value": [
                f"{features['glcm_contrast']:.4f}",
                f"{features['glcm_homogeneity']:.4f}"
            ]
        }
        st.dataframe(
            pd.DataFrame(texture_data),
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### LBP Histogram (9 bins)")
        
        # Plot LBP histogram
        lbp_hist = features['lbp_hist']
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(range(len(lbp_hist))),
                y=lbp_hist,
                marker=dict(
                    color=lbp_hist,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f"{v:.3f}" for v in lbp_hist],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Local Binary Pattern Distribution",
            xaxis_title="Bin",
            yaxis_title="Frequency",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# COMPARISON VISUALIZATION
# ============================================================================
def plot_side_by_side(img1, img2, title1="Image 1", title2="Image 2"):
    """
    Plot two images side by side
    
    Args:
        img1, img2: numpy arrays
        title1, title2: titles untuk setiap image
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img1, caption=title1, use_column_width=True, clamp=True)
    
    with col2:
        st.image(img2, caption=title2, use_column_width=True, clamp=True)


def plot_preprocessing_comparison(original_path: str, preprocessed: np.ndarray):
    """
    Compare original and preprocessed image
    
    Args:
        original_path: Path to original image
        preprocessed: Preprocessed numpy array
    """
    import cv2
    
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    
    st.markdown("### Preprocessing Comparison")
    plot_side_by_side(
        original,
        preprocessed,
        "Original Image",
        "Preprocessed (Gaussian Blur + CLAHE)"
    )
    
    # Show histograms
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Histogram")
        fig_orig = plot_histogram(original)
        st.pyplot(fig_orig)
    
    with col2:
        st.markdown("#### Preprocessed Histogram")
        fig_prep = plot_histogram(preprocessed)
        st.pyplot(fig_prep)


def plot_histogram(img: np.ndarray):
    """
    Plot grayscale histogram
    
    Args:
        img: Grayscale image numpy array
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    
    ax.hist(img.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.set_title('Intensity Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# EDGE & LINE VISUALIZATION
# ============================================================================
def plot_edge_detection(img: np.ndarray, mask: np.ndarray = None):
    """
    Visualize edge detection results
    
    Args:
        img: Preprocessed image
        mask: Optional mask to apply before edge detection
    """
    import cv2
    
    # Apply mask if provided
    if mask is not None:
        img = cv2.bitwise_and(img, img, mask=mask)
    
    # Detect edges
    edges = cv2.Canny(img, 100, 200)
    
    # Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
    
    # Create visualization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(img, caption="Input", use_column_width=True, clamp=True)
    
    with col2:
        st.image(edges, caption="Canny Edges", use_column_width=True, clamp=True)
    
    with col3:
        # Draw lines on image
        line_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        if lines is not None:
            for line in lines[:50]:  # Limit to 50 lines
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        st.image(
            line_img,
            caption=f"Hough Lines ({0 if lines is None else len(lines)})",
            use_column_width=True,
            clamp=True
        )


# ============================================================================
# SUMMARY METRICS
# ============================================================================
def plot_summary_metrics(results: Dict):
    """
    Display summary metrics dalam metric cards
    
    Args:
        results: Complete analysis results
    """
    features = results['features']
    prediction = results['prediction']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Edge Density",
            f"{features['edge_sum']:.0f}",
            help="Sum of detected edges (Canny)"
        )
    
    with col2:
        st.metric(
            "Line Features",
            f"{features['num_lines']}",
            help="Number of lines detected (Hough Transform)"
        )
    
    with col3:
        st.metric(
            "GLCM Contrast",
            f"{features['glcm_contrast']:.3f}",
            help="Texture contrast measure"
        )
    
    with col4:
        st.metric(
            "GLCM Homogeneity",
            f"{features['glcm_homogeneity']:.3f}",
            help="Texture homogeneity measure"
        )


# ============================================================================
# EXPORT VISUALIZATION
# ============================================================================
def create_report_image(results: Dict) -> np.ndarray:
    """
    Create a single report image with all key visualizations
    (For download/export)
    
    Args:
        results: Complete analysis results
    
    Returns:
        Combined visualization as numpy array
    """
    # TODO: Implement report generation
    # This would combine all visualizations into one image
    pass