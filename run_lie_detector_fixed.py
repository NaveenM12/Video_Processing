#!/usr/local/bin/python3
"""
Fixed run script for the Automatic Lie Detector.

This script sets the PYTHONPATH environment variable to include all necessary directories
before importing and running the detector.
"""

import os
import sys
import argparse

# Set up the environment variables and paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add all relevant directories to Python path
sys.path.insert(0, ROOT_DIR)  # VideoProcessing root
sys.path.insert(0, os.path.join(ROOT_DIR, "Automatic_Lie_Detector"))
sys.path.insert(0, os.path.join(ROOT_DIR, "Separate_Color_Motion_Magnification"))
sys.path.insert(0, os.path.join(ROOT_DIR, "Face_Motion_Magnification"))
sys.path.insert(0, os.path.join(ROOT_DIR, "Face_Color_Magnification"))

# Import configuration from Separate_Color_Motion_Magnification directly
from Separate_Color_Motion_Magnification.config import (
    INPUT_VIDEO_PATH as SCMM_INPUT_PATH,
    OUTPUT_VIDEO_PATH as SCMM_OUTPUT_PATH,
    MOTION_MAG_PARAMS,
    COLOR_MAG_PARAMS,
    LABEL_FONT_SCALE,
    LABEL_THICKNESS,
    LABEL_HEIGHT,
    LABEL_ALPHA,
    ORIGINAL_LABEL,
    MOTION_LABEL,
    COLOR_LABEL,
    FACIAL_REGIONS
)

# Define parameters for the Automatic Lie Detector
INPUT_VIDEOS_FOLDER = "/Users/naveenmirapuri/VideoProcessing/trimmed_mp4"
OUTPUT_VIDEOS_FOLDER = "/Users/naveenmirapuri/VideoProcessing/Automatic_Lie_Detector/output_videos"

# Define detection parameters - use these directly with the AutomaticLieDetector
DETECTION_PARAMS = {
    # Temporal window parameters
    'window_size_seconds': 3,      # Size of sliding window in seconds (changed from 5 to 3)
    'window_overlap': 0.5,         # Overlap between consecutive windows (0.5 = 50% overlap)
    
    # Anomaly detection parameters
    'anomaly_threshold': 95,       # Percentile threshold for anomaly detection (95th percentile)
    'min_anomaly_score': 0.7,      # Minimum anomaly score to consider (0-1 scale)
    
    # Feature weighting
    'feature_weights': {
        'phase_change': 1.0,       # Weight for micro-expression features
        'heart_rate': 0.7,         # Weight for heart rate features
        'cross_correlation': 0.5,  # Weight for correlation between signals
    },
    
    # Visualization
    'highlight_color': (0, 0, 255),  # Red color for highlighting deception regions (BGR)
    'highlight_alpha': 0.3,          # Transparency for highlighted regions
}

# Replace the config module directly
class ConfigModule:
    pass

config_mod = ConfigModule()
config_mod.DETECTION_PARAMS = DETECTION_PARAMS
config_mod.INPUT_VIDEOS_FOLDER = INPUT_VIDEOS_FOLDER
config_mod.OUTPUT_VIDEOS_FOLDER = OUTPUT_VIDEOS_FOLDER
config_mod.MOTION_MAG_PARAMS = MOTION_MAG_PARAMS
config_mod.COLOR_MAG_PARAMS = COLOR_MAG_PARAMS
config_mod.LABEL_FONT_SCALE = LABEL_FONT_SCALE
config_mod.LABEL_THICKNESS = LABEL_THICKNESS
config_mod.LABEL_HEIGHT = LABEL_HEIGHT
config_mod.LABEL_ALPHA = LABEL_ALPHA
config_mod.ORIGINAL_LABEL = ORIGINAL_LABEL
config_mod.MOTION_LABEL = MOTION_LABEL
config_mod.COLOR_LABEL = COLOR_LABEL
config_mod.FACIAL_REGIONS = FACIAL_REGIONS

# Replace the config module
sys.modules['config'] = config_mod

# Make upper_face_detector and cheek_detector available globally
sys.modules['upper_face_detector'] = __import__('Separate_Color_Motion_Magnification.upper_face_detector', fromlist=['UpperFaceDetector'])
sys.modules['cheek_detector'] = __import__('Separate_Color_Motion_Magnification.cheek_detector', fromlist=['CheekDetector'])

# Import after path setup - now the import will use our custom module above
from Automatic_Lie_Detector.automatic_lie_detector import AutomaticLieDetector

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Automatic Lie Detector - Video Processing')
    
    parser.add_argument(
        '--input', '-i', type=str, default=INPUT_VIDEOS_FOLDER,
        help=f'Path to input videos folder (default: {INPUT_VIDEOS_FOLDER})'
    )
    
    parser.add_argument(
        '--output', '-o', type=str, default=OUTPUT_VIDEOS_FOLDER,
        help=f'Path to output videos folder (default: {OUTPUT_VIDEOS_FOLDER})'
    )
    
    parser.add_argument(
        '--single', '-s', type=str, default=None,
        help='Process a single video file instead of the entire folder'
    )
    
    return parser.parse_args()

def main():
    """Run the automatic lie detector."""
    args = parse_arguments()
    
    print("===== Starting Automatic Lie Detector with Fixed Path Settings =====")
    print(f"Current directory: {os.getcwd()}")
    print(f"DETECTION_PARAMS in config: {hasattr(sys.modules['config'], 'DETECTION_PARAMS')}")
    
    # Create detector instance - directly pass the parameters
    detector = AutomaticLieDetector(
        motion_params=MOTION_MAG_PARAMS,
        color_params=COLOR_MAG_PARAMS
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    if args.single:
        # Process a single video
        if not os.path.exists(args.single):
            print(f"Error: Input file '{args.single}' not found")
            return
        
        print(f"Processing single video: {args.single}")
        basename = os.path.basename(args.single)
        file_name = os.path.splitext(basename)[0]
        output_path = os.path.join(args.output, f"deception_detection_{file_name}.mp4")
        
        detector.process_video(args.single, output_path)
    else:
        # Process all videos in the folder
        print(f"Processing all videos in: {args.input}")
        print(f"Output folder: {args.output}")
        
        detector.process_folder(args.input, args.output)
    
    print("===== Automatic Lie Detector Complete =====")

if __name__ == "__main__":
    main() 