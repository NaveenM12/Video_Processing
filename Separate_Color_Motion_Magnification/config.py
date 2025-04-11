"""
Configuration settings for the Side-by-Side Video Magnification.
Edit this file to change input/output paths and magnification parameters.
"""

# Input/Output Settings
INPUT_VIDEO_PATH = "/Users/naveenmirapuri/VideoProcessing/test_videos/face.mp4"
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

# Color Magnification Parameters - Optimized for heart rate detection based on article
COLOR_MAG_PARAMS = {
    'alpha': 50.0,           # Standard magnification factor as per the article
    'level': 4,              # Pyramid level 4-6 recommended in the article
    
    # Heart rate frequencies - focusing on 50-60 BPM as mentioned in the article
    'f_lo': 50/60,           # 0.83 Hz - Lower boundary (~50 BPM)
    'f_hi': 60/60,           # 1.0 Hz - Upper boundary (~60 BPM)
}

# Display Parameters
LABEL_FONT_SCALE = 1.0
LABEL_THICKNESS = 2
LABEL_HEIGHT = 50
LABEL_ALPHA = 0.7           # Transparency factor for label background

# Labels for the videos
ORIGINAL_LABEL = "Original"
MOTION_LABEL = "Motion Magnified"
COLOR_LABEL = "Color Magnified"

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