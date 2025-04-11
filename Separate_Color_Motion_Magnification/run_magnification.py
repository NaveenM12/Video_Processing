#!/usr/local/bin/python3
import argparse
import os
from side_by_side_magnification import SideBySideMagnification
from config import *

def main():
    """Run the side-by-side magnification using configuration from config.py"""
    print("Running Side-by-Side Video Magnification with the following settings:")
    print(f"Input video: {INPUT_VIDEO_PATH}")
    print(f"Output video: {OUTPUT_VIDEO_PATH}")
    print(f"Motion magnification parameters: {MOTION_MAG_PARAMS}")
    print(f"Color magnification parameters: {COLOR_MAG_PARAMS}")
    
    # Check if input file exists
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"Error: Input file '{INPUT_VIDEO_PATH}' not found")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    
    # Process the video using settings from config.py
    processor = SideBySideMagnification(
        motion_params=MOTION_MAG_PARAMS,
        color_params=COLOR_MAG_PARAMS
    )
    processor.process_video(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)
    
    print(f"Video processing complete. Output saved to '{OUTPUT_VIDEO_PATH}'")

if __name__ == "__main__":
    main() 