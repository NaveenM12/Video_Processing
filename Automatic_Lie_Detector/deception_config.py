#!/usr/local/bin/python3
"""
Configuration file for the deception detection system.
This module stores default parameters that can be loaded when running the script.

This configuration specifically supports the PBM/EVM-exclusive detector which
ONLY uses Phase-Based Magnification for motion detection and Eulerian Video 
Magnification for heart rate analysis.
"""

import os

# Base directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Default output directory
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_videos")

# Default input directory
DEFAULT_INPUT_DIR = os.path.join(ROOT_DIR, "trimmed_mp4")

# Default detection parameters
DEFAULT_WINDOW_SIZE = 30      # Window size for movement detection (in frames)
DEFAULT_THRESHOLD_FACTOR = 1.5  # Threshold factor for significant movement
DEFAULT_BIN_SIZE = 15        # Size of bins for aggregating frames

# Maximum window parameters
MAX_WINDOW_SPAN = 90        # Maximum size of the detection window (in frames)
DEFAULT_SPAN_IN_BINS = 6     # Number of bins to use around the most significant bin

# Heart rate analysis parameters
HEART_RATE_BIN_SIZE = 60    # Size of bins for heart rate data (larger for smoother HR curves)
HEART_RATE_BOOST = 0.2      # Factor to boost detection score when heart rate confirms micro-expression

# Face detection parameters for heart rate
FACE_ROI_SCALE = 0.65       # Scale factor for face ROI

# PBM parameters (Phase-Based Magnification for micro-expression detection)
# Matching parameters from Face_Motion_Magnification
PBM_PARAMS = {
    'phase_mag': 15.0,       # Magnification factor (matches Face_Motion_Magnification)
    'f_lo': 0.25,            # Low cutoff frequency (matches Face_Motion_Magnification)
    'f_hi': 0.3,             # High cutoff frequency (matches Face_Motion_Magnification)
    'sigma': 1.0,            # Sigma for Gaussian smoothing (matches Face_Motion_Magnification)
    'attenuate': True        # Attenuate frequencies outside band (matches Face_Motion_Magnification)
}

# EVM parameters (Eulerian Video Magnification for heart rate detection)
# Matching parameters from config.py COLOR_MAG_PARAMS with reduced alpha
EVM_PARAMS = {
    'alpha': 30.0,            # Reduced amplification factor for more subtle magnification
    'level': 5,               # Pyramid level
    'f_lo': 50/60,            # Low cutoff frequency (50 BPM)
    'f_hi': 180/60,           # High cutoff frequency (180 BPM)
    'chromAttenuation': 1.0   # Add chroma attenuation to reduce color artifacts
}

# Feature weights for detection (PBM/EVM exclusive)
FEATURE_WEIGHTS = {
    'phase_change': 3.0,      # Strong weight for PBM micro-expression features
    'heart_rate': 0.3,        # Heart rate weight now 1/10th of phase change weight
    'cross_correlation': 0.0  # No weight for correlation - not using any other technique
}

# Peak detection parameters - applies uniformly to all frames
PEAK_DETECTION = {
    'threshold_multiplier': 1.2,  # Lowered threshold to detect more peaks (more sensitive)
    'cluster_importance': 3.0,    # Importance factor for clustered peaks (higher = more emphasis)
    'min_peaks_required': 2,      # Minimum peaks to consider significant (lower to catch subtle expressions)
    'max_gap_frames': 30,         # Maximum frames between peaks to be considered a single cluster
}

def get_config():
    """Return a dictionary with all configuration parameters"""
    return {
        "window_size": DEFAULT_WINDOW_SIZE,
        "threshold_factor": DEFAULT_THRESHOLD_FACTOR,
        "bin_size": DEFAULT_BIN_SIZE,
        "max_window_span": MAX_WINDOW_SPAN,
        "span_in_bins": DEFAULT_SPAN_IN_BINS,
        "face_roi_scale": FACE_ROI_SCALE,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "input_dir": DEFAULT_INPUT_DIR,
        "heart_rate_bin_size": HEART_RATE_BIN_SIZE,
        "heart_rate_boost": HEART_RATE_BOOST,
        "pbm_params": PBM_PARAMS,
        "evm_params": EVM_PARAMS,
        "feature_weights": FEATURE_WEIGHTS,
        "peak_detection": PEAK_DETECTION
    } 