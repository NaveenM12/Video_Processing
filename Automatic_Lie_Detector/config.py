"""
Configuration settings for the Automatic Lie Detector.
Edit this file to change input/output paths and detection parameters.
"""

# Video Processing Settings
INPUT_VIDEOS_FOLDER = "/Users/naveenmirapuri/VideoProcessing/trimmed_mp4"
OUTPUT_VIDEOS_FOLDER = "/Users/naveenmirapuri/VideoProcessing/Automatic_Lie_Detector/output_videos"
KEEP_TEMP_FILES = False

# Re-use motion and color magnification parameters from the Separate_Color_Motion_Magnification
# Motion Magnification Parameters - Optimized for detecting micro-expressions
MOTION_MAG_PARAMS = {
    'phase_mag': 25.0,       # Higher magnification factor for subtle movements
    'f_lo': 0.4,             # Higher low cutoff to filter out normal speech movements
    'f_hi': 1.2,             # Higher high cutoff to capture quick micro-expressions
    'sigma': 3.0,            # Increased sigma for better amplitude-weighted blurring
    'attenuate': True        # Attenuate other frequencies to focus on the band of interest
}

# Color Magnification Parameters - Optimized for heart rate detection in 50-180 BPM range
COLOR_MAG_PARAMS = {
    'alpha': 40.0,           # Reduced alpha to minimize artifacts while preserving color changes
    'level': 5,              # Mid-pyramid level for optimal detail/noise balance
    
    # Heart rate frequencies - standard resting to active heart rate range
    'f_lo': 50/60,           # 0.83 Hz - Lower boundary (~50 BPM)
    'f_hi': 180/60,          # 3.0 Hz - Upper boundary (~180 BPM)
    'chromAttenuation': 0.0, # No attenuation to maintain full signal for analysis
}

# Display Parameters (reused from Separate_Color_Motion_Magnification)
LABEL_FONT_SCALE = 1.0
LABEL_THICKNESS = 2
LABEL_HEIGHT = 50
LABEL_ALPHA = 0.7           # Transparency factor for label background

# Labels for the videos
ORIGINAL_LABEL = "Original"
MOTION_LABEL = "PBM"
COLOR_LABEL = "EVM"

# Deception Detection Parameters
DETECTION_PARAMS = {
    # Temporal window parameters
    'window_size_seconds': 3,      # Reduced window size for more precise detection
    'window_overlap': 0.6,         # Increased overlap to improve localization of deception
    
    # Anomaly detection parameters
    'anomaly_threshold': 98,       # Increased percentile threshold for more selective detection
    'min_anomaly_score': 0.8,      # Higher minimum score to ensure only significant peaks are detected
    
    # Feature weighting - prioritize micro-expressions over heart rate
    'feature_weights': {
        'phase_change': 2.0,       # Increased weight for micro-expression features
        'heart_rate': 0.3,         # Reduced weight for heart rate features
        'cross_correlation': 0.3,  # Reduced weight for correlation between signals
    },
    
    # Peak detection parameters - new section for better targeting frames 200-300
    'peak_detection': {
        'threshold_multiplier': 1.8,  # Higher multiplier to focus on more significant peaks
        'cluster_importance': 2.5,    # Higher importance for clustered peaks 
        'min_peaks_required': 3,      # Minimum number of peaks to consider a region as deceptive
        'max_gap_frames': 40,         # Maximum frames between peaks to be considered a single cluster
    },
    
    # Visualization
    'highlight_color': (0, 0, 255),  # Red color for highlighting deception regions (BGR)
    'highlight_alpha': 0.3,          # Transparency for highlighted regions
}

# Regions of interest for micro-expression detection (reused from Separate_Color_Motion_Magnification)
FACIAL_REGIONS = {
    'track_eyebrows': True,          # Track inner and outer eyebrow movements
    'track_upper_eyelids': True,     # Track eyelid movements (blinking patterns)
    'track_eye_corners': True,       # Track outer eye corners (authentic vs fake smiles)
    'track_nasolabial_fold': True,   # Track cheek-nose junction
    'track_forehead': True,          # Track forehead micro-movements
    'region_size_factor': 0.8        # Use smaller, more precise regions (0.8 = 80% of default size)
} 