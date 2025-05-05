#!/usr/bin/env python3
"""
Script to run the modified Deception Detector focusing on a single deception region
where there is a significant cluster of micro-expression peaks.
"""

import cv2
import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import detector modules
from Automatic_Lie_Detector.deception_detector import DeceptionDetector
from Separate_Color_Motion_Magnification.fixed_side_by_side_magnification import SideBySideMagnification
import config

def get_basename(file_path):
    """Get base filename without extension"""
    return os.path.splitext(os.path.basename(file_path))[0]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run focused deception detection on a video file")
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("--output", "-o", default="./output_videos", 
                      help="Directory for output video (default: ./output_videos)")
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Generate output filename
    file_name = get_basename(args.input)
    output_path = os.path.join(args.output, f"focused_deception_{file_name}.mp4")
    
    print(f"Processing video: {args.input}")
    print(f"Output will be saved to: {output_path}")
    
    # Initialize the side-by-side magnification processor
    processor = SideBySideMagnification(
        motion_params=config.MOTION_MAG_PARAMS,
        color_params=config.COLOR_MAG_PARAMS
    )
    
    # Process the video to get magnified results and extracted features
    phase_changes, bpm_data, fps, total_frames = processor.process_video(
        input_path=args.input,
        output_path=None  # Don't save intermediate output
    )
    
    # Initialize the deception detector with our parameters
    detector = DeceptionDetector(config.DETECTION_PARAMS)
    
    # Fit the detector to the data
    detector.fit(phase_changes, bpm_data, fps, total_frames)
    
    # Create video writer for final output
    cap = cv2.VideoCapture(args.input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process and write each frame with deception highlighting
    frame_idx = 0
    
    print("Creating output video with deception highlighting...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get deception score for current frame
        score, is_deception = detector.get_frame_deception_score(frame_idx)
        
        # Highlight the frame if it's in the deception region
        if is_deception:
            # Add red tint to frame to indicate deception
            red_overlay = np.zeros_like(frame)
            red_overlay[:,:,2] = 255  # Set red channel to maximum
            # Blend the original frame with the red overlay
            highlighted_frame = cv2.addWeighted(frame, 0.7, red_overlay, 0.3, 0)
            
            # Add text indicator
            cv2.putText(
                highlighted_frame, 
                f"DECEPTION DETECTED - Score: {score:.2f}", 
                (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 0, 255), 
                2, 
                cv2.LINE_AA
            )
            
            # Write the highlighted frame
            out.write(highlighted_frame)
        else:
            # Write original frame
            out.write(frame)
        
        frame_idx += 1
        
        # Print progress
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    # Release resources
    cap.release()
    out.release()
    
    # Print results
    if detector.deception_windows:
        window = detector.deception_windows[0]
        start_frame = window['start_frame']
        end_frame = window['end_frame']
        start_time = window['time_info']['start_time']
        end_time = window['time_info']['end_time']
        
        print(f"\nFocused Deception Detection Results:")
        print(f"Detected single deception region:")
        print(f"  Frames: {start_frame}-{end_frame}")
        print(f"  Time: {start_time:.2f}s-{end_time:.2f}s")
        print(f"  Peak count: {window.get('detailed_peaks', {}).get('count', 0)}")
        print(f"  Peak density: {window.get('peak_density', 0):.2f}")
        print(f"  Anomaly score: {window.get('score_normalized', 0):.2f}")
    else:
        print("No deception regions detected.")
    
    print(f"\nOutput video saved to {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 