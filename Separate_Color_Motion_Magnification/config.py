"""
Configuration settings for the Side-by-Side Video Magnification.
Edit this file to change input/output paths and magnification parameters.
"""

# Input/Output Settings
INPUT_VIDEO_PATH = "/Users/naveenmirapuri/VideoProcessing/trimmed_mp4/1.mp4"
# Alternative test path for a longer video (uncomment to use)
# INPUT_VIDEO_PATH = "/Users/naveenmirapuri/VideoProcessing/test_videos/face_longer.mp4"
OUTPUT_VIDEO_PATH = "output_videos/side_by_side_output.mp4"
KEEP_TEMP_FILES = False

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

# Display Parameters
LABEL_FONT_SCALE = 1.0
LABEL_THICKNESS = 2
LABEL_HEIGHT = 50
LABEL_ALPHA = 0.7           # Transparency factor for label background

# Labels for the videos
ORIGINAL_LABEL = "Original"
MOTION_LABEL = "PBM"
COLOR_LABEL = "EVM"

# Facial Region Detection Parameters
# Regions of interest for micro-expression detection during speech
FACIAL_REGIONS = {
    'track_eyebrows': True,          # Track inner and outer eyebrow movements
    'track_upper_eyelids': True,     # Track eyelid movements (blinking patterns)
    'track_eye_corners': True,       # Track outer eye corners (authentic vs fake smiles)
    'track_nasolabial_fold': True,   # Track cheek-nose junction
    'track_forehead': True,          # Track forehead micro-movements
    'region_size_factor': 0.8        # Use smaller, more precise regions (0.8 = 80% of default size)
} 