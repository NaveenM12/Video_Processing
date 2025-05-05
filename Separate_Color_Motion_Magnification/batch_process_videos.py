#!/usr/local/bin/python3
import os
import glob
from side_by_side_magnification import SideBySideMagnification
from config import MOTION_MAG_PARAMS, COLOR_MAG_PARAMS

def batch_process_videos():
    """Process all videos in the trimmed_mp4 folder."""
    # Input directory containing MP4 videos
    input_dir = "/Users/naveenmirapuri/VideoProcessing/trimmed_mp4"
    
    # Output directory for processed videos
    output_dir = "output_videos"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all MP4 files in the input directory
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    
    # Initialize the processor once with the parameters from config
    processor = SideBySideMagnification(
        motion_params=MOTION_MAG_PARAMS,
        color_params=COLOR_MAG_PARAMS
    )
    
    # Process each video
    for input_file in input_files:
        # Extract the base filename without extension
        basename = os.path.basename(input_file)
        file_number = os.path.splitext(basename)[0]  # Should be a number like "1", "2", etc.
        
        # Create output filename with _x suffix
        output_file = os.path.join(output_dir, f"side_by_side_output_{file_number}.mp4")
        
        print(f"\n\n===== Processing {basename} =====")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        
        # Process the video
        processor.process_video(input_file, output_file)
        
        print(f"===== Completed {basename} =====\n\n")
    
    print(f"All {len(input_files)} videos have been processed.")

if __name__ == "__main__":
    batch_process_videos() 