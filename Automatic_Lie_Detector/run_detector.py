#!/usr/local/bin/python3
"""
Run script for the Automatic Lie Detector.

This script processes videos through the Automatic Lie Detector to identify
potential regions of deception.
"""

import os
import sys
import argparse

# Add parent directory to Python path to ensure imports work correctly
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Now import our modules
from automatic_lie_detector import AutomaticLieDetector
from config import INPUT_VIDEOS_FOLDER, OUTPUT_VIDEOS_FOLDER, MOTION_MAG_PARAMS, COLOR_MAG_PARAMS

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
    print(f"Python path: {sys.path}")
    
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