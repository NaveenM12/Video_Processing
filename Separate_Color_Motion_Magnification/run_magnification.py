#!/usr/local/bin/python3
import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from side_by_side_magnification import SideBySideMagnification
from config import *

# Print the version information
print("Running Side-by-Side Video Magnification with the following settings:")
print(f"Input video: {INPUT_VIDEO_PATH}")
print(f"Output video: {OUTPUT_VIDEO_PATH}")
print(f"Motion magnification parameters: {MOTION_MAG_PARAMS}")
print(f"Color magnification parameters: {COLOR_MAG_PARAMS}")

# Create processor with parameters from config
processor = SideBySideMagnification(
    motion_params=MOTION_MAG_PARAMS,
    color_params=COLOR_MAG_PARAMS
)

# Process video
try:
    processor.process_video(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)
    print(f"Video processing complete. Output saved to '{OUTPUT_VIDEO_PATH}'")
except Exception as e:
    print(f"ERROR processing video: {str(e)}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    pass 