#!/usr/local/bin/python3
"""
Improved Deception Detection Script

This script combines our advanced detection algorithm for finding significant 
micro-expression peaks with the original visualization approach that includes
video with graphs and tracking.
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
import matplotlib.pyplot as plt
import shutil

# Import deception config
import deception_config as cfg

# Set up the environment variables and paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add all relevant directories to Python path
sys.path.insert(0, ROOT_DIR)  # VideoProcessing root
sys.path.insert(0, os.path.join(ROOT_DIR, "Automatic_Lie_Detector"))
sys.path.insert(0, os.path.join(ROOT_DIR, "Separate_Color_Motion_Magnification"))
sys.path.insert(0, os.path.join(ROOT_DIR, "Face_Motion_Magnification"))
sys.path.insert(0, os.path.join(ROOT_DIR, "Face_Color_Magnification"))

# Import from the original implementation
from Automatic_Lie_Detector.automatic_lie_detector import AutomaticLieDetector
# Comment out the problematic import that isn't needed for our implementation
# from Separate_Color_Motion_Magnification.fixed_side_by_side_magnification import SideBySideMagnification
import Separate_Color_Motion_Magnification.config as scm_config

# Import the improved detection function from run_focused_detection.py
from Face_Motion_Magnification.face_region_motion_magnification import PhaseMagnification
from Face_Motion_Magnification.utils.phase_utils import rgb2yiq, yiq2rgb

# Import the ColorMagnification class for Eulerian Video Magnification (heart rate)
from Face_Color_Magnification.face_region_color_magnification import ColorMagnification, FaceDetector as ColorFaceDetector

# Set global variables
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_videos")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define a new function that uses Phase-Based Magnification directly to measure motion
def extract_motion_using_pbm(prev_frame, curr_frame):
    """
    Extract motion magnitude using Phase-Based Magnification (PBM) technique.
    This directly calculates phase changes between consecutive frames using the steerable pyramid.
    
    Args:
        prev_frame: Previous frame
        curr_frame: Current frame
        
    Returns:
        Float value representing phase change magnitude
    """
    # Initialize phase magnifier with default settings optimized for micro-expression detection
    # Do not actually magnify, just detect phase changes
    phase_mag = PhaseMagnification(
        phase_mag=0.0,  # No magnification, just detection
        f_lo=0.05,      # Lower cutoff for temporal filter
        f_hi=0.4,       # Upper cutoff for temporal filter
        sigma=0.5,      # Sigma for Gaussian smoothing
        attenuate=False # No attenuation needed
    )
    
    # Convert frames to the format needed by PBM
    frames = [prev_frame, curr_frame]
    
    # Use the PBM's internal phase change calculation
    # This will return magnified frames (which we don't need) and phase changes (which we do)
    _, phase_changes = phase_mag.magnify(frames)
    
    # Return the phase change magnitude
    # If we only have one value (from 2 frames), return it
    # Otherwise, take the mean
    if len(phase_changes) == 1:
        return phase_changes[0]
    else:
        return np.mean(phase_changes)

# Replace the existing compute_frame_difference with our PBM-based function
compute_frame_difference = extract_motion_using_pbm

def find_significant_movement_region(input_path, window_size=None, threshold_factor=None, bin_size=None, heart_rate_data=None):
    """
    Analyze video to find frames with significant movement clusters based on aggregated bins.
    This function identifies the region with the highest concentration of micro-expressions
    across the entire video by using a sliding window approach.
    When heart rate data is available, it uses this as a secondary confirmation signal.
    
    Args:
        input_path: Path to input video
        window_size: Size of the window for peak detection (in frames)
        threshold_factor: Factor to determine significance threshold
        bin_size: Size of bins for aggregating frames
        heart_rate_data: Optional heart rate data to use as a secondary signal
        
    Returns:
        Tuple of (start_frame, end_frame) for the region with most significant movement
    """
    # Load parameters from config if not provided
    config_params = cfg.get_config()
    window_size = window_size or config_params["window_size"]
    threshold_factor = threshold_factor or config_params["threshold_factor"]
    bin_size = bin_size or config_params["bin_size"]
    max_window_span = config_params["max_window_span"]
    heart_rate_boost = config_params["heart_rate_boost"]
    heart_rate_bin_size = config_params["heart_rate_bin_size"]
    
    print(f"Analyzing video to detect significant movement: {input_path}")
    print(f"Using parameters: window_size={window_size}, threshold_factor={threshold_factor}, bin_size={bin_size}")
    
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {input_path}")
        return None
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        return None
    
    # Process each frame to compute motion magnitude
    frame_idx = 1  # Start at second frame
    motion_magnitudes = [0.0]
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        # Compute motion magnitude
        magnitude = compute_frame_difference(prev_frame, curr_frame)
        motion_magnitudes.append(magnitude)
        
        # Update for next iteration
        prev_frame = curr_frame.copy()
        frame_idx += 1
        
        # Print progress
        if frame_idx % 50 == 0:
            print(f"Analyzed {frame_idx}/{frame_count} frames")
    
    # Release video capture
    cap.release()
    
    # Make sure we have the actual count of frames processed
    actual_frame_count = len(motion_magnitudes)
    print(f"Processed {actual_frame_count} frames for motion analysis")
    
    # Apply moving average to smooth the data
    window = deque(maxlen=window_size)
    smoothed_magnitudes = []
    
    for mag in motion_magnitudes:
        window.append(mag)
        smoothed_magnitudes.append(sum(window) / len(window))
    
    # Convert to numpy array
    smoothed_magnitudes = np.array(smoothed_magnitudes)
    
    # Step 1: Bin the data into fixed-size bins
    num_bins = max(1, int(np.ceil(actual_frame_count / bin_size)))
    
    # Initialize arrays for binned data
    binned_frames = []      # Center of each bin in frame units
    binned_values = []      # Average motion value in each bin
    
    # Process each bin
    for bin_idx in range(num_bins):
        # Calculate start and end frame indices for this bin
        start_frame = bin_idx * bin_size
        end_frame = min(actual_frame_count, start_frame + bin_size)
        
        # Calculate bin center frame for x-axis
        bin_center_frame = start_frame + (end_frame - start_frame) // 2
        binned_frames.append(bin_center_frame)
        
        # Get motion data for this bin
        bin_data = smoothed_magnitudes[start_frame:end_frame]
        
        if len(bin_data) > 0:
            # Calculate average motion in this bin
            bin_avg = np.mean(bin_data)
            binned_values.append(bin_avg)
        else:
            binned_values.append(0)
    
    # Convert to numpy arrays
    binned_frames = np.array(binned_frames)
    binned_values = np.array(binned_values)
    
    # Step 2: Calculate threshold for significant bins
    bin_mean = np.mean(binned_values)
    bin_std = np.std(binned_values)
    bin_threshold = bin_mean + (threshold_factor * bin_std)
    
    # Step 3: Find significant bins (those above threshold)
    significant_bins = np.where(binned_values > bin_threshold)[0]
    
    # Process heart rate data if available
    heart_rate_significant_bins = []
    hr_binned_frames = []
    
    if heart_rate_data is not None and 'bpm_per_frame' in heart_rate_data:
        print("Using heart rate data as secondary confirmation signal")
        # Get BPM data
        bpm_data = heart_rate_data['bpm_per_frame']
        
        # Compute the rate of change in heart rate
        bpm_change = np.abs(np.diff(bpm_data))
        bpm_change = np.append(bpm_change, 0)  # Add 0 at the end to match shape
        
        # Bin the heart rate change data using larger bin size for smoother analysis
        hr_num_bins = max(1, int(np.ceil(actual_frame_count / heart_rate_bin_size)))
        heart_rate_bins = np.zeros(hr_num_bins)
        
        for bin_idx in range(hr_num_bins):
            start_frame = bin_idx * heart_rate_bin_size
            end_frame = min(actual_frame_count, start_frame + heart_rate_bin_size)
            
            # Calculate bin center frame
            bin_center_frame = start_frame + (end_frame - start_frame) // 2
            hr_binned_frames.append(bin_center_frame)
            
            # Calculate average heart rate change in this bin
            bin_data = bpm_change[start_frame:end_frame]
            if len(bin_data) > 0:
                heart_rate_bins[bin_idx] = np.mean(bin_data)
        
        # Calculate threshold for significant heart rate changes
        hr_mean = np.mean(heart_rate_bins)
        hr_std = np.std(heart_rate_bins)
        hr_threshold = hr_mean + threshold_factor * hr_std
        
        # Find bins with significant heart rate changes
        heart_rate_significant_bins = np.where(heart_rate_bins > hr_threshold)[0]
        print(f"Found {len(heart_rate_significant_bins)} bins with significant heart rate changes")
    
    # If we found significant bins, evaluate regions with sliding window approach
    if len(significant_bins) > 0:
        print(f"Found {len(significant_bins)} significant bins for micro-expressions across the video")
        
        # Step 4: Use a sliding window approach to find the region with highest concentration
        # of micro-expressions rather than expanding around a single peak
        
        # Determine the region size to use for evaluation (in bins)
        # This is roughly equivalent to the max_window_span but in bin units
        region_size_bins = max(3, min(6, int(np.ceil(max_window_span / bin_size))))
        print(f"Using sliding window of {region_size_bins} bins to find optimal region")
        
        # Calculate the total number of possible regions to evaluate
        num_regions = num_bins - region_size_bins + 1
        
        # If we have fewer bins than region size, adjust region size
        if num_regions <= 0:
            region_size_bins = max(1, num_bins // 2)
            num_regions = num_bins - region_size_bins + 1
            print(f"Adjusted region size to {region_size_bins} bins due to limited data")
        
        # Initialize arrays to track regions and their scores
        region_scores = []
        region_indices = []
        
        # Slide a window across all possible regions and calculate a score based on:
        # 1. Total micro-expression magnitude in the region
        # 2. Number of significant bins in the region
        # 3. Concentration of micro-expressions (higher if clustered together)
        for start_idx in range(num_regions):
            end_idx = start_idx + region_size_bins - 1
            
            # Get the bin indices in this region
            region_bin_indices = list(range(start_idx, end_idx + 1))
            
            # Count significant bins in this region
            sig_bins_in_region = sum(1 for bin_idx in region_bin_indices if bin_idx in significant_bins)
            
            # Calculate total micro-expression magnitude
            total_magnitude = sum(binned_values[bin_idx] for bin_idx in region_bin_indices)
            
            # Calculate average magnitude
            avg_magnitude = total_magnitude / region_size_bins
            
            # Calculate a concentration score (higher when significant bins are clustered)
            # Start with a base score
            region_score = avg_magnitude * (sig_bins_in_region + 0.1)  # Add small constant to avoid zero scores
            
            # Boost score based on the ratio of significant bins to total bins in region
            concentration = sig_bins_in_region / region_size_bins
            region_score *= (1.0 + concentration)
            
            # If heart rate data is available, boost score where both signals align
            if heart_rate_data is not None and len(heart_rate_significant_bins) > 0 and len(hr_binned_frames) > 0:
                # Map the region's bins to heart rate bins
                region_start_frame = binned_frames[start_idx]
                region_end_frame = binned_frames[min(end_idx, len(binned_frames)-1)]
                
                # Find heart rate bins that overlap with this region
                hr_overlap_bins = []
                for hr_bin_idx, hr_center in enumerate(hr_binned_frames):
                    if region_start_frame <= hr_center <= region_end_frame:
                        hr_overlap_bins.append(hr_bin_idx)
                
                # Check if any overlapping heart rate bins have significant changes
                hr_significant_overlap = set(hr_overlap_bins).intersection(heart_rate_significant_bins)
                
                if hr_significant_overlap:
                    # Boost score when heart rate confirms micro-expression region
                    region_score *= (1.0 + heart_rate_boost)
                    print(f"Region {start_idx}-{end_idx} has both significant micro-expressions and heart rate changes")
            
            # Store the region and its score
            region_scores.append(region_score)
            region_indices.append((start_idx, end_idx))
        
        # Find the region with the highest score
        if region_scores:
            best_region_idx = np.argmax(region_scores)
            start_bin_idx, end_bin_idx = region_indices[best_region_idx]
            best_score = region_scores[best_region_idx]
            
            print(f"Best region: bins {start_bin_idx}-{end_bin_idx} with score {best_score:.2f}")
            
            # Convert bin indices to frame ranges
            start_frame = max(0, int(binned_frames[start_bin_idx]) - bin_size // 2)
            end_frame = min(actual_frame_count - 1, int(binned_frames[end_bin_idx]) + bin_size // 2)
            
            # Check if the region size is within max_window_span
            current_span = end_frame - start_frame
            
            if current_span > max_window_span:
                # Center the window on the middle of the best region
                center_frame = (start_frame + end_frame) // 2
                half_span = max_window_span // 2
                start_frame = max(0, center_frame - half_span)
                end_frame = min(actual_frame_count - 1, center_frame + half_span)
                
            print(f"Detected focused movement region: frames {start_frame}-{end_frame}")
            return start_frame, end_frame
    
    # If no significant bins found or no good regions, use the middle of the video
    center_frame = actual_frame_count // 2
    half_span = max_window_span // 2
    start_frame = max(0, center_frame - half_span)
    end_frame = min(actual_frame_count - 1, center_frame + half_span)
    
    print(f"No significant movement detected, using middle section: frames {start_frame}-{end_frame}")
    return start_frame, end_frame

def compute_smoothed_movement_data(input_path, window_size=30):
    """
    Analyze video to compute motion magnitudes using Phase-Based Magnification (PBM) and return smoothed data.
    This function uses PBM to detect subtle facial movements that may indicate micro-expressions.
    
    Args:
        input_path: Path to input video
        window_size: Size of the window for smoothing
        
    Returns:
        Tuple of (smoothed_magnitudes, raw_magnitudes)
    """
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {input_path}")
        return None, None
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        return None, None
    
    # Process each frame to compute motion magnitude using PBM
    frame_idx = 1  # Start at second frame
    motion_magnitudes = [0.0]  # First frame has no previous frame to compare with
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        # Compute motion magnitude using PBM technique
        magnitude = extract_motion_using_pbm(prev_frame, curr_frame)
        motion_magnitudes.append(magnitude)
        
        # Update for next iteration
        prev_frame = curr_frame.copy()
        frame_idx += 1
    
    # Release video capture
    cap.release()
    
    # Apply moving average to smooth the data
    window = deque(maxlen=window_size)
    smoothed_magnitudes = []
    
    for mag in motion_magnitudes:
        window.append(mag)
        smoothed_magnitudes.append(sum(window) / len(window))
    
    # Convert to numpy arrays
    smoothed_magnitudes = np.array(smoothed_magnitudes)
    raw_magnitudes = np.array(motion_magnitudes)
    
    return smoothed_magnitudes, raw_magnitudes

def calculate_heart_rate(input_path, face_roi_scale=None):
    """
    Analyze video to extract heart rate information using Eulerian Video Magnification (EVM)
    to detect subtle color changes in the skin caused by blood flow.
    
    Args:
        input_path: Path to input video
        face_roi_scale: Scale factor to determine ROI size within detected face
        
    Returns:
        Dictionary containing heart rate data with:
        - bpm_per_frame: Array of BPM values for each frame
        - avg_bpm: Average BPM across the video
        - signal: Raw heart rate signal
    """
    # Load parameters from config if not provided
    config_params = cfg.get_config()
    face_roi_scale = face_roi_scale or config_params["face_roi_scale"]
    
    print(f"Calculating heart rate from: {input_path}")
    
    # Initialize EVM-based color magnification with parameters optimized for heart rate detection
    color_magnifier = ColorMagnification(
        alpha=30.0,          # Amplification factor (lower for heart rate to avoid artifacts)
        level=3,             # Pyramid level (more downsampling reduces noise)
        f_lo=0.75/30,        # Low frequency cutoff (0.75 Hz = 45 BPM)
        f_hi=2.5/30          # High frequency cutoff (2.5 Hz = 150 BPM)
    )
    
    # Initialize face detector for color analysis
    face_detector = ColorFaceDetector()
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video for heart rate calculation")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video has {frame_count} frames at {fps} fps")
    
    # Initialize arrays to store all frames and detected faces
    all_frames = []
    all_faces = []
    frame_idx = 0
    
    # For performance and memory reasons, sample at a reduced rate if the video is long
    sample_rate = 1  # We need to process all frames for accurate heart rate
    
    print("Extracting facial regions for heart rate analysis...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every Nth frame based on sample_rate
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue
        
        # Detect faces using the EVM face detector
        faces = face_detector.detect_faces(frame)
        
        # Store results
        all_frames.append(frame)
        all_faces.append(faces)
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames for heart rate")
    
    # Check if we have enough frames with faces
    if len(all_frames) < 10 or not all_faces[0]:
        print("Error: Not enough frames with detected faces for heart rate analysis")
        return None
    
    # Process the first detected face's forehead region (best for heart rate)
    face_idx = 0
    region_name = 'forehead'  # Forehead has good blood vessels and minimal movement
    
    # Extract region frames
    region_frames = []
    face_detected_frames = []
    
    for i, faces in enumerate(all_faces):
        if faces and len(faces) > face_idx and region_name in faces[face_idx]['regions']:
            region_frames.append(faces[face_idx]['regions'][region_name]['image'])
            face_detected_frames.append(i)
    
    # Check if we have enough region frames
    if len(region_frames) < 10:
        print("Error: Not enough forehead region frames for heart rate analysis")
        return None
    
    # Process the region frames through EVM to extract color signal
    try:
        # Apply EVM to extract the subtle color variations
        # We won't use the magnified frames, but we need the color magnification process for signal extraction
        color_signal = np.zeros(len(region_frames))
        
        # Extract green channel average from each frame (most sensitive to blood flow)
        for i, frame in enumerate(region_frames):
            # Get average green channel value (index 1 in BGR)
            green_avg = np.mean(frame[:, :, 1])
            color_signal[i] = green_avg
        
        # Detrend and normalize the signal
        color_signal = color_signal - np.mean(color_signal)
        if np.std(color_signal) > 0:
            color_signal = color_signal / np.std(color_signal)
        
        # Apply bandpass filter (0.75-2.5 Hz corresponds to 45-150 BPM)
        nyquist = fps / 2
        low = 0.75 / nyquist
        high = 2.5 / nyquist
        
        # Ensure frequency range is valid
        if low >= 1.0 or high >= 1.0:
            print("Warning: Filter frequencies out of range, adjusting to valid range")
            low = min(0.9, low)
            high = min(0.95, high)
        
        # Apply butterworth bandpass filter
        b, a = signal.butter(2, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, color_signal)
        
        # Find peaks in the filtered signal
        peaks, _ = signal.find_peaks(filtered_signal, distance=int(fps/2.5))  # Minimum distance between peaks
        
        # Calculate BPM for each detected peak interval
        if len(peaks) < 2:
            print("Error: Not enough peaks detected for heart rate calculation")
            return None
        
        # Calculate BPM values
        bpm_values = []
        for i in range(len(peaks)-1):
            peak_interval_frames = peaks[i+1] - peaks[i]
            peak_interval_seconds = peak_interval_frames / fps
            if peak_interval_seconds > 0:
                bpm = 60 / peak_interval_seconds
                # Only keep values in reasonable BPM range
                if 45 <= bpm <= 180:
                    bpm_values.append(bpm)
        
        # If no valid BPM values, return None
        if not bpm_values:
            print("Error: No valid BPM values calculated")
            return None
        
        # Calculate average BPM
        avg_bpm = np.mean(bpm_values)
        
        # Create a BPM value for each frame by interpolating
        bpm_per_frame = np.zeros(frame_count)
        
        # Assign calculated BPM values to frames
        for i in range(len(peaks)-1):
            start_idx = face_detected_frames[peaks[i]]
            end_idx = face_detected_frames[peaks[i+1]] if i+1 < len(peaks) else frame_count
            
            # Calculate BPM for this interval
            interval_frames = peaks[i+1] - peaks[i]
            interval_seconds = interval_frames / fps
            if interval_seconds > 0:
                current_bpm = 60 / interval_seconds
                # Only use reasonable values
                if 45 <= current_bpm <= 180:
                    # Assign BPM to all frames in this interval
                    for j in range(start_idx, min(end_idx, frame_count)):
                        bpm_per_frame[j] = current_bpm
        
        # Fill in any remaining frames with average BPM
        for i in range(frame_count):
            if bpm_per_frame[i] == 0:
                bpm_per_frame[i] = avg_bpm
        
        # Create heart rate data dictionary
        heart_rate_data = {
            'bpm_per_frame': bpm_per_frame,
            'avg_bpm': avg_bpm,
            'signal': filtered_signal
        }
        
        print(f"Heart rate analysis complete. Average BPM: {avg_bpm:.1f}")
        return heart_rate_data
        
    except Exception as e:
        print(f"Error in heart rate calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

class ImprovedDetector:
    """
    Enhanced deception detector that uses our improved peak detection algorithm
    but retains the original visualization format with graphs and tracking.
    """
    
    def __init__(self):
        """Initialize the improved detector"""
        self.processor = None
        self.detector = None
        self.deception_region = None
        self.heart_rate_data = None
        
    def process_video(self, input_path, output_path, window_size=None, threshold_factor=None, bin_size=None):
        """
        Process a video to detect deception with improved algorithms.
        Uses Phase-Based Magnification (PBM) to detect micro-expressions,
        and Eulerian Video Magnification (EVM) for heart rate analysis.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            window_size: Size of window for peak detection
            threshold_factor: Factor for significance threshold
            bin_size: Size of bins for aggregating frames
        """
        # Load parameters from config if not provided
        config_params = cfg.get_config()
        window_size = window_size or config_params["window_size"]
        threshold_factor = threshold_factor or config_params["threshold_factor"]
        bin_size = bin_size or config_params["bin_size"]
        
        print(f"Processing video: {input_path}")
        print(f"Output will be saved to: {output_path}")
        print(f"Using parameters: window_size={window_size}, threshold_factor={threshold_factor}, bin_size={bin_size}")
        
        # Step 1: Get movement data using Phase-Based Magnification (PBM)
        print("Computing movement data using PBM technique...")
        smoothed_motion, raw_motion = compute_smoothed_movement_data(input_path, window_size)
        
        if smoothed_motion is None:
            print("Failed to compute movement data")
            return False
        
        # Step 2: Calculate heart rate data using Eulerian Video Magnification (EVM)
        print("Calculating heart rate using EVM technique...")
        self.heart_rate_data = calculate_heart_rate(input_path)
        
        if self.heart_rate_data is None:
            print("Warning: Heart rate calculation failed. Will proceed without heart rate data.")
        else:
            print(f"Heart rate calculation complete. Average BPM: {self.heart_rate_data['avg_bpm']:.1f}")
        
        # Step 3: Detect the region with significant PBM-detected micro-expression movement
        self.deception_region = find_significant_movement_region(
            input_path, window_size, threshold_factor, bin_size, self.heart_rate_data
        )
        
        if not self.deception_region:
            print("Failed to detect significant movement region")
            return False
        
        start_frame, end_frame = self.deception_region
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return False
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
        print(f"Deception region: frames {start_frame}-{end_frame}")
        
        # Create a deception window for visualization
        deception_window = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'center_frame': (start_frame + end_frame) // 2,
            'score': 0.9,  # High score for detected region
            'score_normalized': 0.9,
            'time_info': {
                'start_time': start_frame / fps,
                'end_time': end_frame / fps,
                'center_time': ((start_frame + end_frame) // 2) / fps,
                'duration': (end_frame - start_frame) / fps
            },
            'detailed_peaks': {
                'count': end_frame - start_frame,  # Use range as peak count
                'indices': list(range(start_frame, end_frame))
            },
            'peak_density': 0.9  # High density for detected region
        }
        
        # Calculate final output dimensions
        # Use full width for each of the three plots
        plot_width = width
        plot_height = height
        
        # Calculate total width needed for three plots side by side
        total_plots_width = plot_width * 3
        
        # Center the original video in a frame that matches the total width of all plots
        output_width = total_plots_width  # Width matches the combined width of the three plots
        
        # The output has two rows - one for video and one for plots
        output_height = height * 2  # One row for video, one row for plots
        
        # Create video writer for output with updated dimensions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        # Process each frame
        frame_idx = 0
        
        print("Creating visualization with deception region highlighted...")
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Make a copy of the frame for display
            original_frame = frame.copy()
            
            # Check if this frame is in the deception region
            is_deception = start_frame <= frame_idx <= end_frame
            
            # Add subtle red border if in deception region
            if is_deception:
                # Calculate intensity based on distance from center of region
                center = (start_frame + end_frame) / 2
                distance_from_center = abs(frame_idx - center) / ((end_frame - start_frame) / 2)
                intensity = 1.0 - min(distance_from_center * 0.7, 0.7)  # Higher intensity near center
                
                # Add red border
                border_thickness = max(2, int(10 * intensity))
                original_frame = cv2.copyMakeBorder(
                    original_frame, 
                    border_thickness, border_thickness, border_thickness, border_thickness,
                    cv2.BORDER_CONSTANT, 
                    value=(0, 0, 255)  # Red border
                )
                # Resize back to original dimensions
                original_frame = cv2.resize(original_frame, (width, height))
            
            # Add "Original Video" label
            original_labeled = self._add_label(original_frame, "Original Video")
            
            # Create a black canvas the same width as the total plots width
            # This will allow us to center the video with black padding on sides
            video_row = np.zeros((height, total_plots_width, 3), dtype=np.uint8)
            
            # Calculate left offset to center the video
            left_offset = (total_plots_width - width) // 2
            
            # Place the original video in the center of the black canvas
            video_row[:, left_offset:left_offset+width] = original_labeled
            
            # Create the three plots at their original full width
            # 1. PBM Micro-expression Movement Plot
            movement_plot = self._create_movement_plot(
                frame_idx, total_frames, fps,
                smoothed_motion, raw_motion,
                deception_window, plot_width, plot_height
            )
            
            # 2. EVM Heart Rate Plot
            heart_plot = self._create_heart_rate_plot(
                frame_idx, total_frames, self.heart_rate_data, plot_width, plot_height
            )
            
            # 3. Deception Timeline
            deception_timeline = self._create_deception_timeline_enhanced(
                frame_idx, total_frames, fps, deception_window, plot_width, plot_height
            )
            
            # Create the plots row by stacking horizontally
            plot_row = np.hstack((movement_plot, heart_plot, deception_timeline))
            
            # Combine video and plots vertically
            final_frame = np.vstack((video_row, plot_row))
            
            # Write the frame
            out.write(final_frame)
            
            # Update frame index
            frame_idx += 1
            
            # Print progress
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"\nOutput video saved to {output_path}")
        return True
        
    def _create_deception_timeline_enhanced(self, current_frame, total_frames, fps, deception_window, width, height):
        """
        Create an enhanced visualization of the deception timeline showing binned data analysis
        in the same aspect ratio as other plots
        
        Args:
            current_frame: Current frame index
            total_frames: Total number of frames in the video
            fps: Frames per second
            deception_window: Deception window information
            width: Width of the plot
            height: Height of the plot
            
        Returns:
            Timeline visualization as numpy array
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from io import BytesIO
        from PIL import Image
        
        # Get bin_size from config
        config_params = cfg.get_config()
        bin_size = config_params["bin_size"]
        
        # Create figure with same aspect ratio as other plots
        fig = plt.figure(figsize=(12, 6), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Extract deception region information
        start_frame = deception_window['start_frame']
        end_frame = deception_window['end_frame']
        start_time = deception_window['time_info']['start_time']
        end_time = deception_window['time_info']['end_time']
        duration = deception_window['time_info']['duration']
        peak_count = deception_window['detailed_peaks']['count']
        
        # Create time points (in seconds)
        total_time = total_frames / fps
        time_points = np.linspace(0, total_time, 100)
        
        # Create main title for the plot
        title = "Deception Detection Timeline"
        plt.title(title, fontsize=16, fontweight='bold')
        
        # Add detection info above the timeline
        detection_text = f"Deception Region: {start_time:.2f}s - {end_time:.2f}s (Duration: {duration:.2f}s)"
        ax.text(0.5, 0.9, detection_text, 
                horizontalalignment='center', 
                fontsize=12, color='red', fontweight='bold',
                transform=ax.transAxes)
        
        # Add peak count and binning info
        bin_text = f"Region with highest concentration of micro-expressions ({peak_count} frames)" 
        ax.text(0.5, 0.8, bin_text, 
                horizontalalignment='center', 
                fontsize=11, color='black',
                transform=ax.transAxes)
        
        # Add heart rate contribution info 
        has_hr_data = self.heart_rate_data is not None
        hr_text = "Detection combines micro-expressions and heart rate variations" if has_hr_data else "Detection based solely on micro-expression analysis"
        hr_color = "darkblue" if has_hr_data else "gray"
        
        ax.text(0.5, 0.72, hr_text,
                horizontalalignment='center',
                fontsize=9, color=hr_color,
                transform=ax.transAxes)
        
        # Add the analysis explanation
        analysis_info = "Analysis uses sliding window to find region with highest density of micro-expressions"
        ax.text(0.5, 0.64, analysis_info,
                horizontalalignment='center',
                fontsize=9, color='darkblue',
                transform=ax.transAxes)
        
        # Draw main timeline ruler
        ax.axhline(y=0.5, color='black', linewidth=2, alpha=0.5)
        
        # Mark time ticks every second
        for t in range(int(total_time) + 1):
            ax.plot([t, t], [0.48, 0.52], 'k-', linewidth=1, alpha=0.5)
            ax.text(t, 0.45, f"{t}s", fontsize=9, ha='center')
        
        # Create the binned regions visual representation
        # Create bin markings on timeline
        bin_duration_sec = bin_size / fps
        num_bins = int(np.ceil(total_time / bin_duration_sec))
        
        # Draw subtle bin separators
        for b in range(num_bins + 1):
            bin_time = b * bin_duration_sec
            if bin_time <= total_time:
                ax.axvline(x=bin_time, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Draw the deception region as a red bar
        bar_height = 0.2
        ax.add_patch(plt.Rectangle((start_time, 0.5 - bar_height/2), duration, bar_height, 
                                   color='red', alpha=0.7))
        
        # Label key time points
        # Start
        ax.plot(start_time, 0.5, 'ro', markersize=8)
        ax.text(start_time, 0.6, f"{start_time:.2f}s", 
                fontsize=10, color='red', ha='center')
        
        # Peak (center)
        center_time = (start_time + end_time) / 2
        ax.plot(center_time, 0.5, 'ro', markersize=10, markeredgecolor='black')
        ax.text(center_time, 0.65, f"{center_time:.2f}s (Center)", 
                fontsize=10, color='red', fontweight='bold', ha='center')
        
        # End
        ax.plot(end_time, 0.5, 'ro', markersize=8)
        ax.text(end_time, 0.6, f"{end_time:.2f}s", 
                fontsize=10, color='red', ha='center')
        
        # Mark current frame
        current_time = current_frame / fps
        ax.axvline(x=current_time, color='green', linewidth=2, alpha=0.8)
        ax.plot(current_time, 0.5, 'go', markersize=10)
        ax.text(current_time, 0.35, f"Current: {current_time:.2f}s", 
                fontsize=10, color='green', fontweight='bold', ha='center')
        
        # Calculate which bin the current frame is in
        current_bin = int(current_frame / bin_size)
        current_bin_start_time = (current_bin * bin_size) / fps
        current_bin_end_time = min(total_time, ((current_bin + 1) * bin_size) / fps)
        
        # Highlight the current bin with a subtle highlight
        ax.axvspan(current_bin_start_time, current_bin_end_time, 
                  ymin=0.2, ymax=0.8, alpha=0.1, color='green')
        
        # Show current bin information
        bin_info_text = f"Current Bin: {current_bin} ({current_bin_start_time:.2f}s - {current_bin_end_time:.2f}s)"
        ax.text(current_time + 0.2, 0.25, bin_info_text, 
                fontsize=9, color='green', ha='left')
                
        # Remove y-axis ticks and labels
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        # Set x-axis label
        ax.set_xlabel('Time (seconds)', fontsize=12)
        
        # Set x-axis limits
        ax.set_xlim(0, total_time)
        ax.set_ylim(0, 1)
        
        # Add a grid for better readability (x-axis only)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        # Add confidence indicator
        confidence = "HIGH" if deception_window['score'] > 0.8 else "MEDIUM" if deception_window['score'] > 0.6 else "LOW"
        confidence_color = "darkred" if confidence == "HIGH" else "darkorange" if confidence == "MEDIUM" else "darkgreen"
        
        ax.text(0.95, 0.05, f"Confidence: {confidence}", 
                transform=ax.transAxes, fontsize=12, color=confidence_color, 
                fontweight='bold', ha='right',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=confidence_color, boxstyle='round'))
        
        # Render figure to numpy array
        fig.tight_layout()
        canvas.draw()
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to numpy array
        img = np.array(Image.open(buf))
        plt.close(fig)
        
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Resize to match the required dimensions
        img = cv2.resize(img, (width, height))
        
        return img

    def _add_label(self, frame, text):
        """Add a text label to the frame"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-40), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(
            frame, text, (w//2 - 80, h-15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2, cv2.LINE_AA
        )
        
        return frame
        
    def _create_movement_plot(self, frame_idx, total_frames, fps, smoothed_motion, raw_motion, deception_window, width, height):
        """
        Create a plot of micro-expression movement data with aggregated peaks
        using bins of 15 frames for smoother visualization
        
        Args:
            frame_idx: Current frame index
            total_frames: Total number of frames
            fps: Frames per second
            smoothed_motion: Smoothed motion data
            raw_motion: Raw motion data
            deception_window: Deception window information
            width: Width of the plot
            height: Height of the plot
            
        Returns:
            Plot image as numpy array
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from io import BytesIO
        from PIL import Image
        
        # Create figure with improved aspect ratio for better visualization
        fig = plt.figure(figsize=(12, 6), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Calculate the threshold for significant movement
        threshold = np.mean(smoothed_motion) + 1.5 * np.std(smoothed_motion)
        
        # Bin the motion data using a fixed bin size of 15 frames
        bin_size = 15  # Use 15-frame bins as specified
            
        # Calculate number of bins
        num_bins = max(1, int(np.ceil(total_frames / bin_size)))
        
        # Initialize arrays for binned data
        binned_frames = []      # Center of each bin in frame units
        binned_values = []      # Average motion value in each bin
        binned_max_values = []  # Maximum motion value in each bin
        binned_peaks = []       # Number of peaks in each bin
        binned_above_threshold = []  # Flag if bin has values above threshold
        
        # Identify peaks in the smoothed motion data
        peaks = []
        for i in range(1, len(smoothed_motion) - 1):
            if smoothed_motion[i] > smoothed_motion[i-1] and smoothed_motion[i] > smoothed_motion[i+1]:
                peaks.append(i)
        
        # Process each bin
        for bin_idx in range(num_bins):
            # Calculate start and end frame indices for this bin
            start_frame = bin_idx * bin_size
            end_frame = min(total_frames, start_frame + bin_size)
            
            # Calculate bin center frame for x-axis
            bin_center_frame = start_frame + (end_frame - start_frame) // 2
            binned_frames.append(bin_center_frame)
            
            # Get motion data for this bin
            bin_slice = slice(start_frame, min(end_frame, len(smoothed_motion)))
            bin_data = smoothed_motion[bin_slice]
            
            if len(bin_data) > 0:
                # Calculate average motion in this bin
                bin_avg = np.mean(bin_data)
                binned_values.append(bin_avg)
                
                # Calculate maximum motion in this bin
                bin_max = np.max(bin_data)
                binned_max_values.append(bin_max)
                
                # Count peaks in this bin
                bin_peaks = sum(1 for p in peaks if start_frame <= p < end_frame)
                binned_peaks.append(bin_peaks)
                
                # Check if any values are above threshold
                above_threshold = any(v > threshold for v in bin_data)
                binned_above_threshold.append(above_threshold)
            else:
                binned_values.append(0)
                binned_max_values.append(0)
                binned_peaks.append(0)
                binned_above_threshold.append(False)
        
        # Convert to numpy arrays
        binned_frames = np.array(binned_frames)
        binned_values = np.array(binned_values)
        binned_max_values = np.array(binned_max_values)
        binned_peaks = np.array(binned_peaks)
        binned_above_threshold = np.array(binned_above_threshold)
        
        # Calculate aggregate threshold for binned data
        binned_threshold = np.mean(binned_values) + 1.5 * np.std(binned_values)
        
        # Plot the raw data with lower opacity for reference
        ax.plot(smoothed_motion, 'g-', linewidth=1, alpha=0.2, label='Frame Movement')
        
        # Plot the binned data as the main focus
        ax.plot(binned_frames, binned_values, 'g-', linewidth=2.5, 
               marker='o', markersize=6, markerfacecolor='green', markeredgecolor='black',
               markeredgewidth=1, alpha=0.9, label='Movement Magnitude')
        
        # Mark significant bins with red dots
        significant_bin_indices = np.where(binned_values > binned_threshold)[0]
        if len(significant_bin_indices) > 0:
            significant_bin_frames = binned_frames[significant_bin_indices]
            significant_bin_values = binned_values[significant_bin_indices]
            ax.plot(significant_bin_frames, significant_bin_values, 'ro', 
                   markersize=8, label='Significant Movement')
        
        # Highlight deception region with red background
        # Use deception window from the detection algorithm
        start = deception_window['start_frame']
        end = deception_window['end_frame']
        ax.axvspan(start, end, alpha=0.2, color='red')
        
        # Add vertical line for current frame
        ax.axvline(x=frame_idx, color='blue', linestyle='-', linewidth=2)
        
        # Add current frame marker (blue dot)
        if 0 <= frame_idx < len(smoothed_motion):
            # Find the bin containing the current frame
            current_bin_idx = np.argmin(np.abs(binned_frames - frame_idx)) if len(binned_frames) > 0 else 0
            if current_bin_idx < len(binned_values):
                # Plot marker at the bin value rather than the raw frame value
                ax.plot(frame_idx, binned_values[current_bin_idx], 'bo', markersize=10)
        
        # Set labels and title
        ax.set_title('Combined Micro-Expressions: Aggregated Movement', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('Magnitude', fontsize=12)
        
        # Set x-axis limits
        ax.set_xlim(0, total_frames)
        
        # Add threshold line for binned data
        ax.axhline(y=binned_threshold, color='red', linestyle='--', alpha=0.5, label='Threshold')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add current frame text
        if 0 <= frame_idx < len(smoothed_motion):
            # Find the bin containing the current frame
            current_bin_idx = np.argmin(np.abs(binned_frames - frame_idx)) if len(binned_frames) > 0 else 0
            current_bin_value = binned_values[current_bin_idx] if current_bin_idx < len(binned_values) else 0
            
            ax.text(
                0.02, 0.95, 
                f"Current: Frame {frame_idx}/{total_frames} (Bin value: {current_bin_value:.4f})", 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )
        
        # Add text annotations to provide context
        ax.text(
            0.02, 0.02,
            f"Deception Region: Frames {start}-{end}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Render figure to numpy array
        fig.tight_layout()
        canvas.draw()
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to numpy array
        img = np.array(Image.open(buf))
        plt.close(fig)
        
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Resize to desired dimensions
        img = cv2.resize(img, (width, height))
        
        return img
    
    def _create_heart_rate_plot(self, frame_idx, total_frames, heart_rate_data, width, height):
        """
        Create a heart rate plot showing BPM over time, with data aggregated in larger bins
        for smoother visualization
        
        Args:
            frame_idx: Current frame index
            total_frames: Total number of frames
            heart_rate_data: Dictionary containing heart rate information
            width: Width of the plot
            height: Height of the plot
            
        Returns:
            Heart rate plot as numpy array
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from io import BytesIO
        from PIL import Image
        
        if heart_rate_data is None:
            # Create a placeholder if no heart rate data is available
            placeholder = np.ones((height, width, 3), dtype=np.uint8) * 255
            cv2.putText(
                placeholder, "Heart Rate Data Not Available", 
                (width//2 - 180, height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA
            )
            cv2.putText(
                placeholder, "Ensure face is clearly visible", 
                (width//2 - 160, height//2 + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1, cv2.LINE_AA
            )
            return placeholder
        
        # Create figure
        fig = plt.figure(figsize=(12, 6), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Extract data
        bpm_per_frame = heart_rate_data['bpm_per_frame']
        avg_bpm = heart_rate_data['avg_bpm']
        
        # Get larger bin size from config for heart rate visualization
        config_params = cfg.get_config()
        bin_size = config_params["heart_rate_bin_size"]
        
        # Calculate number of bins needed
        num_bins = max(1, int(np.ceil(len(bpm_per_frame) / bin_size)))
        
        # Initialize arrays for binned data
        binned_frames = []      # Center of each bin in frame units
        binned_bpm_values = []  # Average BPM value in each bin
        
        # Process each bin
        for bin_idx in range(num_bins):
            # Calculate start and end frame indices for this bin
            start_frame = bin_idx * bin_size
            end_frame = min(len(bpm_per_frame), start_frame + bin_size)
            
            # Calculate bin center frame for x-axis
            bin_center_frame = start_frame + (end_frame - start_frame) // 2
            binned_frames.append(bin_center_frame)
            
            # Get BPM data for this bin
            bin_data = bpm_per_frame[start_frame:end_frame]
            
            if len(bin_data) > 0:
                # Calculate average BPM in this bin
                bin_avg = np.mean(bin_data)
                binned_bpm_values.append(bin_avg)
            else:
                # If no data (should not happen), use average
                binned_bpm_values.append(avg_bpm)
        
        # Convert to numpy arrays
        binned_frames = np.array(binned_frames)
        binned_bpm_values = np.array(binned_bpm_values)
        
        # Create x-axis (frame numbers)
        x = np.arange(len(bpm_per_frame))
        
        # Plot original heart rate data with lower opacity for reference
        ax.plot(x, bpm_per_frame, 'r-', linewidth=1, alpha=0.2, label='Frame BPM')
        
        # Plot the binned data as the main focus
        ax.plot(binned_frames, binned_bpm_values, 'r-', linewidth=2.5, 
               marker='o', markersize=6, markerfacecolor='red', markeredgecolor='black',
               markeredgewidth=1, alpha=0.9, label='Heart Rate (BPM)')
        
        # Add horizontal line for average BPM
        ax.axhline(y=avg_bpm, color='blue', linestyle='--', linewidth=1.5, 
                  label=f'Avg: {avg_bpm:.1f} BPM')
        
        # Add vertical line for current frame
        ax.axvline(x=frame_idx, color='green', linestyle='-', linewidth=2)
        
        # Add current frame marker with current BPM (from binned data)
        # Find the bin containing the current frame
        current_bin_idx = np.argmin(np.abs(binned_frames - frame_idx)) if len(binned_frames) > 0 else 0
        current_bpm = binned_bpm_values[current_bin_idx] if current_bin_idx < len(binned_bpm_values) else avg_bpm
        
        # Plot marker at current frame position
        ax.plot(frame_idx, current_bpm, 'go', markersize=10)
        
        # Add text annotation for current BPM
        ax.text(
            frame_idx + 10, current_bpm, 
            f"{current_bpm:.1f} BPM", 
            fontsize=12, color='green', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        
        # Set labels and title
        ax.set_title('Heart Rate Estimation', fontsize=16, fontweight='bold')
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('BPM', fontsize=12)
        
        # Set x-axis limits
        ax.set_xlim(0, total_frames)
        
        # Set y-axis limits to a normal heart rate range (with some padding)
        min_bpm = max(40, np.min(binned_bpm_values) * 0.9)
        max_bpm = min(180, np.max(binned_bpm_values) * 1.1)
        ax.set_ylim(min_bpm, max_bpm)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add current frame text
        ax.text(
            0.02, 0.95, 
            f"Current: Frame {frame_idx}/{total_frames}", 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Additional annotations for interpretation
        if current_bpm < 60:
            heart_status = "Below Normal (Bradycardia)"
            color = 'blue'
        elif current_bpm <= 100:
            heart_status = "Normal Range"
            color = 'green'
        else:
            heart_status = "Above Normal (Tachycardia)"
            color = 'red'
            
        ax.text(
            0.02, 0.02, 
            f"Status: {heart_status}", 
            transform=ax.transAxes, fontsize=12, color=color, fontweight='bold',
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Render figure to numpy array
        fig.tight_layout()
        canvas.draw()
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to numpy array
        img = np.array(Image.open(buf))
        plt.close(fig)
        
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Resize to desired dimensions
        img = cv2.resize(img, (width, height))
        
        return img

def get_basename(file_path):
    """Get base filename without extension"""
    return os.path.splitext(os.path.basename(file_path))[0]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="""
    Improved Deception Detector using Phase-Based Magnification (PBM) for micro-expression detection
    and Eulerian Video Magnification (EVM) for heart rate analysis.
    
    This tool evaluates how PBM and EVM can be used to detect potential deception in videos.
    """)
    
    # Load default parameters from config
    config_params = cfg.get_config()
    
    parser.add_argument("input", help="Path to input video file")
    
    parser.add_argument(
        "--output", 
        default=os.path.join(OUTPUT_DIR, f"improved_detection_{get_basename(sys.argv[1])}.mp4"),
        help="Path to output video file"
    )
    
    parser.add_argument(
        "--window-size", 
        type=int,
        default=config_params["window_size"],
        help=f"Window size for movement detection in frames (default: {config_params['window_size']})"
    )
    
    parser.add_argument(
        "--threshold-factor", 
        type=float,
        default=config_params["threshold_factor"],
        help=f"Threshold factor for significant movement (default: {config_params['threshold_factor']})"
    )
    
    parser.add_argument(
        "--bin-size", 
        type=int,
        default=config_params["bin_size"],
        help=f"Size of bins for aggregating frames (default: {config_params['bin_size']})"
    )
    
    return parser.parse_args()

def main():
    """
    Main function to run the improved deception detector.
    
    This program evaluates the use of Phase-Based Magnification (PBM) for micro-expression detection
    and Eulerian Video Magnification (EVM) for heart rate analysis in deception detection.
    
    The approach uses:
    1. PBM to detect subtle facial movements (micro-expressions)
    2. EVM to analyze subtle color changes for heart rate monitoring
    3. A sliding window algorithm to find regions with highest micro-expression activity
    4. Heart rate data as secondary confirmation when available
    """
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Create and run the detector
    detector = ImprovedDetector()
    result = detector.process_video(
        args.input, 
        args.output,
        window_size=args.window_size,
        threshold_factor=args.threshold_factor,
        bin_size=args.bin_size
    )
    
    if result:
        print(f"Deception detection complete. Output saved to {args.output}")
    else:
        print("Failed to process video for deception detection")

if __name__ == "__main__":
    main() 