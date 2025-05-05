#!/usr/local/bin/python3
"""
Simplified Improved Deception Detection Script

This script provides a simplified interface to the improved deception detection
algorithm, matching the original workflow while still using the advanced 
peak detection and visualization improvements.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from collections import deque
import time
from PIL import Image
import scipy.signal as signal

# Import the complete functions from the main improved detection script
from run_improved_detection import (
    compute_frame_difference,
    find_significant_movement_region,
    compute_smoothed_movement_data,
    calculate_heart_rate,
    ImprovedDetector,
    get_basename
)

def parse_arguments():
    """Parse command line arguments similar to the original script."""
    parser = argparse.ArgumentParser(description='Simplified Improved Deception Detection')
    
    parser.add_argument(
        '--single',
        help='Path to single video file for processing'
    )
    
    parser.add_argument(
        '--output', '-o', type=str, default='./output_videos',
        help='Directory for output video (default: ./output_videos)'
    )
    
    parser.add_argument(
        '--window', '-w', type=int, default=30,
        help='Window size for movement detection (default: 30 frames)'
    )
    
    parser.add_argument(
        '--threshold', '-t', type=float, default=1.5,
        help='Threshold factor for significant movement (default: 1.5)'
    )
    
    # Add focus on frames parameter to match previous behavior
    parser.add_argument(
        '--focus-frames', '-f', type=str, default='200-300',
        help='Frame range to focus on for deception detection (default: 200-300)'
    )
    
    return parser.parse_args()

def process_single_video(video_path, output_dir, window_size=30, threshold_factor=1.5, focus_frames=None):
    """
    Process a single video using the improved detection algorithm
    
    Args:
        video_path: Path to input video file
        output_dir: Directory for output
        window_size: Size of window for movement detection
        threshold_factor: Factor for significance threshold
        focus_frames: Optional frame range to focus on (format: "start-end")
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n===== Processing single video: {video_path} =====")
    
    # Check if input file exists
    if not os.path.isfile(video_path):
        print(f"Error: Input file '{video_path}' not found")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    file_name = get_basename(video_path)
    output_path = os.path.join(output_dir, f"improved_detection_{file_name}.mp4")
    
    # Parse focus frames if provided
    start_frame = None
    end_frame = None
    if focus_frames:
        try:
            parts = focus_frames.split('-')
            if len(parts) == 2:
                start_frame = int(parts[0])
                end_frame = int(parts[1])
                print(f"Will focus on frame range: {start_frame}-{end_frame}")
        except ValueError:
            print(f"Warning: Invalid focus frame format: {focus_frames}. Using auto-detection.")
    
    # Create detector and process the video
    detector = ImprovedDetector()
    
    # If focus frames are provided, we'll need to modify how detection works
    if start_frame is not None and end_frame is not None:
        # Temporarily store the original function to restore later
        original_find_region = find_significant_movement_region
        
        # Create a modified version that returns the specified frame range
        def modified_find_region(*args, **kwargs):
            print(f"Using specified frame range: {start_frame}-{end_frame}")
            return start_frame, end_frame
        
        # Override the detection function
        globals()['find_significant_movement_region'] = modified_find_region
    
    # Process the video
    success = detector.process_video(
        video_path, 
        output_path,
        window_size,
        threshold_factor
    )
    
    # Restore original function if we overrode it
    if start_frame is not None and end_frame is not None:
        globals()['find_significant_movement_region'] = original_find_region
    
    if success:
        print(f"===== Processing Complete for {file_name} =====")
        print(f"Output saved to: {output_path}")
        return True
    else:
        print(f"===== Processing Failed for {file_name} =====")
        return False

def main():
    """Main function to run the simplified improved detector."""
    args = parse_arguments()
    
    print("===== Starting Simplified Improved Deception Detector =====")
    print(f"Current directory: {os.getcwd()}")
    
    # Process single video if specified
    if args.single:
        success = process_single_video(
            args.single,
            args.output,
            args.window,
            args.threshold,
            args.focus_frames
        )
        return 0 if success else 1
    else:
        print("Error: No input video specified. Use --single to specify a video.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 