#!/bin/bash

# Run Improved Detector Script
# This script runs the improved deception detection with the updated layout
# using parameters from the configuration file

# Check if an input video was provided
if [ $# -lt 1 ]; then
    echo "Usage: ./run_improved_detector.sh [path_to_video]"
    echo "Example: ./run_improved_detector.sh /Users/naveenmirapuri/VideoProcessing/trimmed_mp4/1.mp4"
    echo "Note: detection parameters are now loaded from deception_config.py"
    exit 1
fi

# Run the improved detector with the provided video path
# Parameters are now loaded from deception_config.py
cd /Users/naveenmirapuri/VideoProcessing && /usr/local/bin/python3 run_improved_detection.py "$1" --output output_videos/

echo ""
echo "Video processing complete. Please check the output_videos directory for results." 