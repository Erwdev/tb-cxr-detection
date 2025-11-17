FEATURE_CONFIG = {
    "canny": {"threshold1": 100, "threshold2": 200},
    "hough": {"rho": 1, "theta": np.pi/180, "threshold": 150},
    "lbp": {"P": 8, "R": 1, "method": "uniform"},
    "glcm": {"distances": [1], "angles": [0], "levels": 256, "symmetric": True, "normed": True}
}
import numpy as np 
