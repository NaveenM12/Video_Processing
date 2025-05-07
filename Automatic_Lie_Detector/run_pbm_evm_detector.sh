#!/bin/bash

# Script to run the PBM/EVM-EXCLUSIVE detector on a video
# This detector ONLY uses PBM for motion analysis and EVM for heart rate analysis

# Verify that an argument was provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 [path_to_video]"
    echo "Example: $0 ../trimmed_mp4/1.mp4"
    exit 1
fi

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Video path is the first argument
VIDEO_PATH="$1"

# Run the PBM/EVM detector
echo "Running PBM/EVM-EXCLUSIVE deception detector on video: $VIDEO_PATH"
python3 pbm_evm_detection.py --single "$VIDEO_PATH"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Processing complete. Check the output_videos directory for results."
    echo "The analysis uses ONLY:"
    echo "- Phase-Based Magnification (PBM) for micro-expression detection"
    echo "- Eulerian Video Magnification (EVM) for heart rate analysis"
else
    echo "Error processing video. Check the console output for details."
fi 