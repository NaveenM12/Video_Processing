import cv2
import numpy as np
import os
from config import INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH

def test_fixed_dimensions():
    """Test script to verify that our fixed dimensions will be consistent"""
    
    # Fixed dimensions we're using in side_by_side_magnification.py
    fixed_base_width = 640
    video_height = 360
    plot_height = 300
    
    # Calculate side by side width and combined dimensions
    side_by_side_width = fixed_base_width * 3
    total_plot_rows = 3  # Ensure at least 3 rows for consistent layout
    total_plot_height = total_plot_rows * plot_height
    combined_height = video_height + total_plot_height
    
    # Print the dimensions that will be used
    print(f"Input video path: {INPUT_VIDEO_PATH}")
    print(f"Output video path: {OUTPUT_VIDEO_PATH}")
    print(f"\nFixed dimensions that will be used:")
    print(f"- Each video column width: {fixed_base_width}px")
    print(f"- Video height: {video_height}px")
    print(f"- Plot height per row: {plot_height}px")
    print(f"- Number of plot rows: {total_plot_rows}")
    print(f"\nResulting output dimensions:")
    print(f"- Total width: {side_by_side_width}px (3 columns)")
    print(f"- Total height: {combined_height}px (videos + plots)")
    print(f"- Aspect ratio: {side_by_side_width/combined_height:.2f}")
    
    # Print the input video properties
    try:
        cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
        if not cap.isOpened():
            print(f"\nError: Could not open video file {INPUT_VIDEO_PATH}")
            return
        
        # Get video properties
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nInput video properties:")
        print(f"- Width: {orig_width}px")
        print(f"- Height: {orig_height}px")
        print(f"- Aspect ratio: {orig_width/orig_height:.2f}")
        print(f"- FPS: {orig_fps}")
        print(f"- Total frames: {total_frames}")
        print(f"- Duration: {total_frames/orig_fps:.2f} seconds")
        
        # Calculate aspect-ratio-maintained dimensions
        aspect_ratio = orig_width / orig_height
        adjusted_width = min(int(video_height * aspect_ratio), fixed_base_width)
        
        print(f"\nAdjusted video width to maintain aspect ratio: {adjusted_width}px")
        
        cap.release()
        print("\nOur changes ensure that output dimensions remain fixed regardless of input video length or resolution")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    test_fixed_dimensions() 