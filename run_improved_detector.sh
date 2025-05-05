#!/bin/bash

# Run Improved Detector Script
# This script runs the improved deception detection with the updated layout
# and bins-based aggregation approach

# Check if an input video was provided
if [ $# -lt 1 ]; then
    echo "Usage: ./run_improved_detector.sh [path_to_video] [bin_size]"
    echo "Example: ./run_improved_detector.sh /Users/naveenmirapuri/VideoProcessing/trimmed_mp4/1.mp4 15"
    echo "Note: bin_size is optional (default is 15 frames)"
    exit 1
fi

# Set default bin size
BIN_SIZE=15

# If a second parameter is provided, use it as the bin size
if [ $# -ge 2 ]; then
    BIN_SIZE=$2
fi

# Run the improved detector with the provided video path and bin size
cd /Users/naveenmirapuri/VideoProcessing && /usr/local/bin/python3 run_improved_detection.py "$1" --output output_videos/ --bin-size $BIN_SIZE

echo ""
echo "Video processing complete. Please check the output_videos directory for results." 