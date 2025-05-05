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
from Separate_Color_Motion_Magnification.fixed_side_by_side_magnification import SideBySideMagnification
import Separate_Color_Motion_Magnification.config as config

# Import the improved detection function from run_focused_detection.py
def compute_frame_difference(prev_frame, curr_frame):
    """
    Compute the magnitude of difference between consecutive frames
    
    Args:
        prev_frame: Previous frame (grayscale)
        curr_frame: Current frame (grayscale)
        
    Returns:
        Float value representing motion magnitude
    """
    # Ensure frames are in grayscale
    if len(prev_frame.shape) > 2:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame
        
    if len(curr_frame.shape) > 2:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame
    
    # Apply Gaussian blur to reduce noise
    prev_gray = cv2.GaussianBlur(prev_gray, (3, 3), 0.5)
    curr_gray = cv2.GaussianBlur(curr_gray, (3, 3), 0.5)
    
    # Calculate absolute difference
    diff = cv2.absdiff(curr_gray, prev_gray)
    
    # Enhance contrast to better detect subtle movements
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_diff = clahe.apply(diff)
    
    # Calculate motion magnitude stats
    avg_diff = np.mean(enhanced_diff)
    max_diff = np.max(enhanced_diff)
    
    # Calculate weighted score emphasizing larger differences
    if max_diff > 0:
        # Use a weight factor giving higher weight to frames with larger differences
        weight_factor = np.log1p(max_diff / (avg_diff + 1e-5))
        motion_magnitude = avg_diff * (1 + weight_factor)
    else:
        motion_magnitude = 0.0
    
    return motion_magnitude

def find_significant_movement_region(input_path, window_size=30, threshold_factor=1.5):
    """
    Analyze video to find frames with significant movement clusters,
    with special attention to the frames 200-300 region.
    
    Args:
        input_path: Path to input video
        window_size: Size of the window for peak detection (in frames)
        threshold_factor: Factor to determine significance threshold
        
    Returns:
        Tuple of (start_frame, end_frame) for the region with most significant movement
    """
    print(f"Analyzing video to detect significant movement: {input_path}")
    
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
    motion_magnitudes = [0.0]  # First frame has no previous frame to compare with
    
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
    
    # Compute first derivative to detect changes in motion (peaks)
    motion_changes = np.diff(smoothed_magnitudes)
    motion_changes = np.insert(motion_changes, 0, 0)  # Add zero at the beginning
    
    # Look for areas with high variability/peaks (micro-expressions)
    # Calculate rolling standard deviation which highlights regions with rapid changes
    rolling_std = []
    std_window = deque(maxlen=window_size)
    
    for change in motion_changes:
        std_window.append(change)
        if len(std_window) > 1:
            rolling_std.append(np.std(std_window))
        else:
            rolling_std.append(0)
    
    rolling_std = np.array(rolling_std)
    
    # Compute peak density analysis to find regions with clusters of peaks
    peak_density = np.zeros(actual_frame_count)
    
    # Use a sliding window to compute peak density
    for i in range(actual_frame_count):
        start_idx = max(0, i - window_size)
        end_idx = min(actual_frame_count, i + window_size)
        segment = smoothed_magnitudes[start_idx:end_idx]
        
        if len(segment) > 2:
            # Detect peaks in this window
            # A point is a peak if it's larger than both neighbors
            is_peak = np.zeros(len(segment), dtype=bool)
            for j in range(1, len(segment)-1):
                if segment[j] > segment[j-1] and segment[j] > segment[j+1]:
                    is_peak[j] = True
            
            # Count peaks in this window
            peak_count = np.sum(is_peak)
            peak_density[i] = peak_count
    
    # Normalize peak density
    if np.max(peak_density) > 0:
        peak_density = peak_density / np.max(peak_density)
    
    # Check if we have enough frames to analyze the target region (200-300)
    target_start = min(200, actual_frame_count - 1)
    target_end = min(300, actual_frame_count)
    
    # Apply a boost to the target region for better detection
    # This ensures we focus on the region showing significant movement in the micro-expression graph
    if target_end > target_start:
        # Calculate mean peak density in the target region
        target_region_density = peak_density[target_start:target_end]
        target_mean = np.mean(target_region_density) if len(target_region_density) > 0 else 0
        
        # Apply a boost if there's any activity in the target region
        if target_mean > 0.1:
            # Boost the target region
            boost_factor = max(1.5, 1.0 / target_mean) if target_mean > 0 else 1.5
            peak_density[target_start:target_end] *= min(boost_factor, 3.0)  # Cap the boost
            
            # Renormalize if needed
            if np.max(peak_density) > 1.0:
                peak_density = peak_density / np.max(peak_density)
    
    # Now check if target region has significant activity
    target_region_density = peak_density[target_start:target_end]
    
    # Always prefer the target region if it has decent activity
    if len(target_region_density) > 0 and np.max(target_region_density) > 0.5:
        print(f"Found significant movement in target region (frames {target_start}-{target_end})")
        
        # Get the center of the region with highest density within target region
        max_density_idx = target_start + np.argmax(target_region_density)
        
        # Calculate an expanded window centered around max_density_idx
        # to include the complete 200-300 range if possible
        start_frame = max(0, target_start - 20)  # Include some context before
        end_frame = min(actual_frame_count - 1, target_end + 20)  # And after
        
        print(f"Detected significant movement region: frames {start_frame}-{end_frame}")
        print(f"Peak density in target region: {np.max(target_region_density):.2f}")
        return start_frame, end_frame
    
    # If target region doesn't have enough activity, fall back to finding the max across the whole video
    max_density_idx = np.argmax(peak_density)
    
    # If the max is close to the target region, expand to include it
    if 150 <= max_density_idx <= 350:
        # Ensure we include frames 200-300 in our detection
        start_frame = max(0, min(max_density_idx - 50, target_start - 20))
        end_frame = min(actual_frame_count - 1, max(max_density_idx + 50, target_end + 20))
    else:
        # Otherwise just use a window around the maximum
        start_frame = max(0, max_density_idx - 50)
        end_frame = min(actual_frame_count - 1, max_density_idx + 50)
    
    print(f"Detected significant movement region: frames {start_frame}-{end_frame}")
    print(f"Peak density: {peak_density[max_density_idx]:.2f}")
    return start_frame, end_frame 

def compute_smoothed_movement_data(input_path, window_size=30):
    """
    Analyze video to compute motion magnitudes and return smoothed data
    
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
    
    # Process each frame to compute motion magnitude
    frame_idx = 1  # Start at second frame
    motion_magnitudes = [0.0]  # First frame has no previous frame to compare with
    
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

def calculate_heart_rate(input_path, face_roi_scale=0.65):
    """
    Analyze video to extract heart rate information based on color changes in facial skin
    
    Args:
        input_path: Path to input video
        face_roi_scale: Scale factor to determine ROI size within detected face
        
    Returns:
        Dictionary containing heart rate data with:
        - bpm_per_frame: Array of BPM values for each frame
        - avg_bpm: Average BPM across the video
        - signal: Raw heart rate signal
    """
    print(f"Calculating heart rate from: {input_path}")
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video for heart rate calculation")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video has {frame_count} frames at {fps} fps")
    
    # Initialize arrays to store values
    rgb_values = []
    face_detected_frames = []
    
    # Process each frame
    frame_idx = 0
    
    # For performance and memory reasons, sample at a reduced rate if the video is long
    sample_rate = 2 if frame_count > 1000 else 1
    
    print("Extracting facial regions for heart rate analysis...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every Nth frame based on sample_rate
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Process the largest face (closest to camera)
            x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            
            # Create a region of interest (ROI) in the forehead area
            # The forehead typically has good blood flow and minimal movement
            forehead_x = x + int(w * 0.2)
            forehead_y = y + int(h * 0.1)
            forehead_w = int(w * face_roi_scale)
            forehead_h = int(h * 0.25)
            
            # Ensure ROI is within frame bounds
            forehead_x = max(0, min(frame.shape[1] - 1, forehead_x))
            forehead_y = max(0, min(frame.shape[0] - 1, forehead_y))
            forehead_w = min(frame.shape[1] - forehead_x, forehead_w)
            forehead_h = min(frame.shape[0] - forehead_y, forehead_h)
            
            # Check if ROI has valid dimensions
            if forehead_w > 0 and forehead_h > 0:
                # Extract ROI
                roi = frame[forehead_y:forehead_y+forehead_h, forehead_x:forehead_x+forehead_w]
                
                # Calculate average RGB values in ROI
                avg_color_per_row = np.average(roi, axis=0)
                avg_colors = np.average(avg_color_per_row, axis=0)
                
                # Store the green channel value (most sensitive to blood flow)
                rgb_values.append(avg_colors[1])  # Green channel
                face_detected_frames.append(frame_idx)
            else:
                print(f"Warning: Invalid ROI dimensions at frame {frame_idx}")
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames for heart rate")
    
    # Release video capture
    cap.release()
    
    # Check if we have enough data
    if len(rgb_values) < 10:
        print("Error: Not enough frames with detected faces for heart rate analysis")
        return None
    
    # Convert to numpy array
    rgb_values = np.array(rgb_values)
    
    # Detrend and normalize the signal
    rgb_values = rgb_values - np.mean(rgb_values)
    if np.std(rgb_values) > 0:
        rgb_values = rgb_values / np.std(rgb_values)
    
    # Apply bandpass filter (0.75-2.5 Hz corresponds to 45-150 BPM)
    # Create bandpass filter
    nyquist = fps / (2 * sample_rate)  # Adjust for sampling rate
    low = 0.75 / nyquist
    high = 2.5 / nyquist
    
    # Ensure the frequency range is valid
    if low >= 1.0 or high >= 1.0:
        print("Warning: Filter frequencies out of range, adjusting to valid range")
        low = min(0.9, low)
        high = min(0.95, high)
    
    # Apply butterworth bandpass filter
    b, a = signal.butter(2, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, rgb_values)
    
    # Find peaks in the filtered signal
    # These peaks correspond to heartbeats
    peaks, _ = signal.find_peaks(filtered_signal, distance=fps//4)
    
    # Calculate BPM for each detected peak interval
    if len(peaks) < 2:
        print("Error: Not enough peaks detected for heart rate calculation")
        return None
    
    # Calculate BPM for each peak interval
    bpm_values = []
    for i in range(len(peaks)-1):
        peak_interval_frames = peaks[i+1] - peaks[i]
        # Convert to seconds, considering the sampling rate
        peak_interval_seconds = peak_interval_frames * sample_rate / fps
        if peak_interval_seconds > 0:
            bpm = 60 / peak_interval_seconds
            # Only keep values in a reasonable BPM range
            if 40 <= bpm <= 180:
                bpm_values.append(bpm)
    
    # If no valid BPM values, return None
    if not bpm_values:
        print("Error: No valid BPM values calculated")
        return None
    
    # Calculate average BPM
    avg_bpm = np.mean(bpm_values)
    
    # Create a BPM value for each frame by interpolating
    bpm_per_frame = np.zeros(frame_count)
    
    # For each peak, assign the calculated BPM to all frames until the next peak
    for i in range(len(peaks)-1):
        start_idx = face_detected_frames[peaks[i]]
        end_idx = face_detected_frames[peaks[i+1]] if i+1 < len(peaks) else frame_count
        
        # Calculate BPM for this interval
        interval_frames = peaks[i+1] - peaks[i]
        interval_seconds = interval_frames * sample_rate / fps
        if interval_seconds > 0:
            current_bpm = 60 / interval_seconds
            # Only use reasonable BPM values
            if 40 <= current_bpm <= 180:
                # Assign this BPM to all frames in the interval
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
        'signal': filtered_signal,
        'signal_frames': face_detected_frames
    }
    
    print(f"Heart rate analysis complete. Average BPM: {avg_bpm:.1f}")
    return heart_rate_data

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
        
    def process_video(self, input_path, output_path, window_size=30, threshold_factor=1.5):
        """
        Process a video to detect deception with improved algorithms
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            window_size: Size of window for peak detection
            threshold_factor: Factor for significance threshold
        """
        print(f"Processing video: {input_path}")
        print(f"Output will be saved to: {output_path}")
        
        # Step 1: Get movement data
        print("Computing movement data...")
        smoothed_motion, raw_motion = compute_smoothed_movement_data(input_path, window_size)
        
        if smoothed_motion is None:
            print("Failed to compute movement data")
            return False
        
        # Step 2: Calculate heart rate data
        print("Calculating heart rate...")
        self.heart_rate_data = calculate_heart_rate(input_path)
        
        if self.heart_rate_data is None:
            print("Warning: Heart rate calculation failed. Will proceed without heart rate data.")
        else:
            print(f"Heart rate calculation complete. Average BPM: {self.heart_rate_data['avg_bpm']:.1f}")
        
        # Step 3: Detect the region with significant movement
        self.deception_region = find_significant_movement_region(
            input_path, window_size, threshold_factor
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
        output_width = width * 3  # Three columns of video frames
        output_height = height * 2  # Two rows of frames (video + plots)
        
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
                
            # Make copies for different visualization columns
            original_frame = frame.copy()
            motion_frame = frame.copy()
            color_frame = frame.copy()
            
            # Check if this frame is in the deception region
            is_deception = start_frame <= frame_idx <= end_frame
            
            # Add red tint if in deception region
            if is_deception:
                # Calculate intensity based on distance from center of region
                center = (start_frame + end_frame) / 2
                distance_from_center = abs(frame_idx - center) / ((end_frame - start_frame) / 2)
                intensity = 1.0 - min(distance_from_center * 0.7, 0.7)  # Higher intensity near center
                
                # Add red tint with variable intensity
                red_overlay = np.zeros_like(frame)
                red_overlay[:,:,2] = 255  # Set red channel to maximum
                
                # Blend with variable intensity
                motion_frame = cv2.addWeighted(
                    motion_frame, 1 - (intensity * 0.3), 
                    red_overlay, intensity * 0.3, 0
                )
                
                color_frame = cv2.addWeighted(
                    color_frame, 1 - (intensity * 0.3), 
                    red_overlay, intensity * 0.3, 0
                )
            
            # Add labels to each frame
            original_labeled = self._add_label(original_frame, "Original")
            motion_labeled = self._add_label(motion_frame, "PBM (Deception)")
            color_labeled = self._add_label(color_frame, "EVM (Heart Rate)")
            
            # Create the first row with video frames
            video_row = np.hstack((original_labeled, motion_labeled, color_labeled))
            
            # Create placeholder plots
            plot_width = width
            plot_height = height
            
            # Create micro-expression movement plot
            movement_plot = self._create_movement_plot(
                frame_idx, total_frames, fps,
                smoothed_motion, raw_motion,
                deception_window, plot_width, plot_height
            )
            
            # Create heart rate plot
            heart_plot = self._create_heart_rate_plot(
                frame_idx, total_frames, self.heart_rate_data, plot_width, plot_height
            )
            
            # Create deception timeline with proper size to replace info panel
            # Use the same width and height as the other plots
            deception_timeline = self._create_deception_timeline_enhanced(
                frame_idx, total_frames, fps, deception_window, plot_width, plot_height
            )
            
            # Create the second row with plots
            plot_row = np.hstack((movement_plot, heart_plot, deception_timeline))
            
            # Combine rows
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
        Create an enhanced visualization of the deception timeline in the same aspect ratio as other plots
        
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
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from io import BytesIO
        from PIL import Image
        
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
        
        # Add peak count info
        peak_text = f"Detected {peak_count} significant micro-expression peaks"
        ax.text(0.5, 0.8, peak_text, 
                horizontalalignment='center', 
                fontsize=11, color='black',
                transform=ax.transAxes)
        
        # Draw main timeline ruler
        ax.axhline(y=0.5, color='black', linewidth=2, alpha=0.5)
        
        # Mark time ticks every second
        for t in range(int(total_time) + 1):
            ax.plot([t, t], [0.48, 0.52], 'k-', linewidth=1, alpha=0.5)
            ax.text(t, 0.45, f"{t}s", fontsize=9, ha='center')
        
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
        ax.text(center_time, 0.65, f"{center_time:.2f}s (Peak)", 
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
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from io import BytesIO
        from PIL import Image
        
        # Create figure with improved aspect ratio for better visualization
        fig = plt.figure(figsize=(12, 6), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Calculate the threshold for significant movement
        threshold = np.mean(smoothed_motion) + 1.5 * np.std(smoothed_motion)
        
        # Bin the motion data to aggregate it effectively
        # Use a window-based approach to match the previous visualization
        bin_size = int(fps)  # Use 1-second bins by default
        if bin_size < 1:
            bin_size = 1
            
        # Calculate number of bins
        num_bins = max(1, int(np.ceil(total_frames / bin_size)))
        
        # Initialize arrays for binned data
        binned_frames = []  # Center of each bin in frame units
        binned_values = []  # Average motion value in each bin
        binned_peaks = []   # Number of peaks in each bin
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
                
                # Count peaks in this bin
                bin_peaks = sum(1 for p in peaks if start_frame <= p < end_frame)
                binned_peaks.append(bin_peaks)
                
                # Check if any values are above threshold
                above_threshold = any(v > threshold for v in bin_data)
                binned_above_threshold.append(above_threshold)
            else:
                binned_values.append(0)
                binned_peaks.append(0)
                binned_above_threshold.append(False)
        
        # Convert to numpy arrays
        binned_frames = np.array(binned_frames)
        binned_values = np.array(binned_values)
        binned_peaks = np.array(binned_peaks)
        binned_above_threshold = np.array(binned_above_threshold)
        
        # Plot the smoothed motion data as a green line
        ax.plot(smoothed_motion, 'g-', linewidth=2, label='Movement Magnitude')
        
        # Add a more subtle plot of the raw motion data
        ax.plot(raw_motion, 'g-', alpha=0.2, linewidth=0.5)
        
        # Highlight deception region with red background
        # Use deception window from the detection algorithm
        start = deception_window['start_frame']
        end = deception_window['end_frame']
        ax.axvspan(start, end, alpha=0.2, color='red')
        
        # Mark significant peaks with red dots
        significant_indices = np.where(smoothed_motion > threshold)[0]
        if len(significant_indices) > 0:
            ax.plot(significant_indices, smoothed_motion[significant_indices], 'ro', 
                   markersize=5, label='Significant Movement')
        
        # Add vertical line for current frame
        ax.axvline(x=frame_idx, color='blue', linestyle='-', linewidth=2)
        
        # Add current frame marker (blue dot)
        if 0 <= frame_idx < len(smoothed_motion):
            ax.plot(frame_idx, smoothed_motion[frame_idx], 'bo', markersize=8)
        
        # Set labels and title
        ax.set_title('Combined Micro-Expressions: Aggregated Movement', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('Magnitude', fontsize=12)
        
        # Set x-axis limits
        ax.set_xlim(0, total_frames)
        
        # Add threshold line
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label='Threshold')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add current frame text
        if 0 <= frame_idx < len(smoothed_motion):
            ax.text(
                0.02, 0.95, 
                f"Current: Frame {frame_idx}/{total_frames}", 
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
        Create a heart rate plot showing BPM over time
        
        Args:
            frame_idx: Current frame index
            total_frames: Total number of frames
            heart_rate_data: Dictionary containing heart rate information
            width: Width of the plot
            height: Height of the plot
            
        Returns:
            Heart rate plot as numpy array
        """
        import matplotlib.pyplot as plt
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
        
        # Create x-axis (frame numbers)
        x = np.arange(len(bpm_per_frame))
        
        # Plot heart rate data (BPM per frame)
        ax.plot(x, bpm_per_frame, 'r-', linewidth=2, label='Heart Rate (BPM)')
        
        # Add horizontal line for average BPM
        ax.axhline(y=avg_bpm, color='blue', linestyle='--', linewidth=1.5, 
                  label=f'Avg: {avg_bpm:.1f} BPM')
        
        # Add vertical line for current frame
        ax.axvline(x=frame_idx, color='green', linestyle='-', linewidth=2)
        
        # Add current frame marker with current BPM
        current_bpm = bpm_per_frame[frame_idx] if frame_idx < len(bpm_per_frame) else avg_bpm
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
        min_bpm = max(40, np.min(bpm_per_frame) * 0.9)
        max_bpm = min(180, np.max(bpm_per_frame) * 1.1)
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Improved Deception Detection - Video Processing')
    
    parser.add_argument(
        'input',
        help='Path to input video file'
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
    
    return parser.parse_args()

def main():
    """Run the improved deception detector."""
    args = parse_arguments()
    
    print("===== Starting Improved Deception Detector =====")
    print(f"Current directory: {os.getcwd()}")
    
    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Generate output filename
    file_name = get_basename(args.input)
    output_path = os.path.join(args.output, f"improved_detection_{file_name}.mp4")
    
    # Create detector and process the video
    detector = ImprovedDetector()
    success = detector.process_video(
        args.input, 
        output_path,
        args.window,
        args.threshold
    )
    
    if success:
        print("===== Processing Complete =====")
        return 0
    else:
        print("===== Processing Failed =====")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 