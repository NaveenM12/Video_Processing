#!/usr/local/bin/python3
"""
Run script for the Automatic Lie Detector from the VideoProcessing directory level.

This script processes videos through the Automatic Lie Detector to identify
potential regions of deception.
"""

import os
import sys
import argparse
import importlib.util

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the config module from Separate_Color_Motion_Magnification first
scmm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Separate_Color_Motion_Magnification')
config_path = os.path.join(scmm_dir, 'config.py')

# Load the Separate_Color_Motion_Magnification config module
spec = importlib.util.spec_from_file_location('config', config_path)
config_module = importlib.util.module_from_spec(spec)
sys.modules['config'] = config_module  # Make it available as 'config'
spec.loader.exec_module(config_module)

# Now import from Automatic_Lie_Detector
from Automatic_Lie_Detector.automatic_lie_detector import AutomaticLieDetector
from Automatic_Lie_Detector.config import (
    INPUT_VIDEOS_FOLDER, 
    OUTPUT_VIDEOS_FOLDER, 
    MOTION_MAG_PARAMS, 
    COLOR_MAG_PARAMS
)

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
    
    print("===== Starting Automatic Lie Detector =====")
    print(f"Current directory: {os.getcwd()}")
    
    # Create detector instance
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