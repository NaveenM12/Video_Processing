#!/usr/local/bin/python3
import os
import glob
from side_by_side_magnification import SideBySideMagnification
from config import MOTION_MAG_PARAMS, COLOR_MAG_PARAMS, INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH

def process_all_videos():
    """
    Simple batch processor for video magnification.
    Processes all videos in trimmed_mp4 folder without modifying any parameters.
    """
    # Get workspace directory
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Set up paths
    input_dir = os.path.join(workspace_dir, "trimmed_mp4")
    output_dir = os.path.join(workspace_dir, "2T_1L_outputs")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all mp4 files
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    
    print(f"Found {len(input_files)} videos to process")
    
    # Create a single processor instance
    processor = SideBySideMagnification(
        motion_params=MOTION_MAG_PARAMS,
        color_params=COLOR_MAG_PARAMS
    )
    
    # Process each video
    for i, input_file in enumerate(input_files, 1):
        # Get just the filename
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        
        print(f"\n[{i}/{len(input_files)}] Processing {filename}")
        
        # Process video with the original method
        try:
            processor.process_video(input_file, output_file)
            print(f"Completed processing {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print("All videos processed!")

if __name__ == "__main__":
    process_all_videos() 