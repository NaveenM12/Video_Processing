#!/usr/local/bin/python3
"""
PBM/EVM Deception Detection

This module EXCLUSIVELY implements:
1. Phase-Based Magnification (PBM) for micro-expression detection
2. Eulerian Video Magnification (EVM) for heart rate analysis

No other techniques are used for motion analysis or physiological measurements.
This is a focused study on evaluating PBM and EVM in deception detection.
"""

import os
import sys
import cv2
import numpy as np
import time
import argparse
from typing import Dict, List, Tuple
from scipy import signal

# Add parent directory to path for imports when run directly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Make upper_face_detector and cheek_detector available globally
# We need to do this before importing from Automatic_Lie_Detector to avoid import errors
sys.modules['upper_face_detector'] = __import__('Separate_Color_Motion_Magnification.upper_face_detector', fromlist=['UpperFaceDetector'])
sys.modules['cheek_detector'] = __import__('Separate_Color_Motion_Magnification.cheek_detector', fromlist=['CheekDetector'])

# Import from original detector to extend its functionality
from Automatic_Lie_Detector.automatic_lie_detector import AutomaticLieDetector
from Automatic_Lie_Detector.deception_detector import DeceptionDetector

# Import direct implementations of PBM and EVM
from Face_Motion_Magnification.face_region_motion_magnification import PhaseMagnification, FacialPhaseMagnification
from Face_Color_Magnification.face_region_color_magnification import ColorMagnification

# Import CheekDetector for heart rate region detection
from Separate_Color_Motion_Magnification.cheek_detector import CheekDetector

# Import configuration from Separate_Color_Motion_Magnification for labels and other settings
from Separate_Color_Motion_Magnification.config import (
    LABEL_FONT_SCALE,
    LABEL_THICKNESS,
    LABEL_HEIGHT,
    LABEL_ALPHA
)

# Import our consolidated configuration
from Automatic_Lie_Detector.deception_config import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    PBM_PARAMS,
    EVM_PARAMS,
    FEATURE_WEIGHTS,
    HEART_RATE_BIN_SIZE,
    get_config
)

# Explicit labels for PBM and EVM technologies
PBM_LABEL = "PBM (Micro-Expression Detection)"
EVM_LABEL = "EVM (Heart Rate Analysis)"
ORIGINAL_LABEL = "Original Video"

# Define detection parameters with ONLY PBM for motion and EVM for heart rate
DETECTION_PARAMS = {
    # Temporal window parameters
    'window_size_seconds': 3,      # Size of sliding window in seconds
    'window_overlap': 0.5,         # Overlap between consecutive windows
    
    # Anomaly detection parameters
    'anomaly_threshold': 95,       # Percentile threshold for anomaly detection
    'min_anomaly_score': 0.7,      # Minimum anomaly score to consider
    
    # Heart rate parameters
    'heart_rate_bin_size': HEART_RATE_BIN_SIZE,  # Size of bins for heart rate analysis (frames)
    
    # Feature weighting - ONLY use phase change for motion (PBM) and heart rate for physiological (EVM)
    'feature_weights': FEATURE_WEIGHTS,
    
    # Visualization
    'highlight_color': (0, 0, 255),  # Red color for highlighting deception regions (BGR)
    'highlight_alpha': 0.3,          # Transparency for highlighted regions
}

class PBMEVMDetector(AutomaticLieDetector):
    """
    A specialized detector that exclusively uses:
    1. Phase-Based Magnification (PBM) for micro-expression detection
    2. Eulerian Video Magnification (EVM) for heart rate analysis
    
    This detector does not use any other motion detection or physiological
    measurement techniques. It is intended for evaluating the effectiveness
    of PBM and EVM specifically in deception detection.
    """
    
    def __init__(self, motion_params=None, color_params=None):
        """
        Initialize the detector with direct PBM and EVM implementations.
        
        Args:
            motion_params: Parameters for PBM
            color_params: Parameters for EVM
        """
        # Use params from config if not provided
        motion_params = motion_params or PBM_PARAMS
        color_params = color_params or EVM_PARAMS
        
        # Call parent initialization
        super().__init__(motion_params, color_params)
        
        # Store the heart rate bin size parameter for use in BPM calculations
        self.params = get_config()
        
        # Store fps and video dimensions (will be set during processing)
        self.fps = 30
        self.total_frames = 0
        self.video_dimensions = (0, 0)
        
        # Override detector with specialized detection parameters
        self.detector = DeceptionDetector(DETECTION_PARAMS)
        
        # Initialize CheekDetector for heart rate detection
        self.cheek_detector = CheekDetector()
        
        # Override the processor with direct implementations
        # Initialize PBM directly for micro-expression detection
        self.pbm_processor = FacialPhaseMagnification()
        if motion_params:
            self.pbm_processor.phase_magnifier.phase_mag = motion_params.get('phase_mag', 15.0)
            self.pbm_processor.phase_magnifier.f_lo = motion_params.get('f_lo', 0.25)
            self.pbm_processor.phase_magnifier.f_hi = motion_params.get('f_hi', 0.3)
            self.pbm_processor.phase_magnifier.sigma = motion_params.get('sigma', 1.0)
            self.pbm_processor.phase_magnifier.attenuate = motion_params.get('attenuate', True)
        
        # Initialize EVM directly for heart rate analysis
        self.evm_processor = ColorMagnification()
        if color_params:
            # Use exact parameters from config without hardcoding
            self.evm_processor.alpha = color_params.get('alpha', 24.0) 
            self.evm_processor.level = color_params.get('level', 3)
            self.evm_processor.f_lo = color_params.get('f_lo', 0.83)
            self.evm_processor.f_hi = color_params.get('f_hi', 1.0)
            if 'chromAttenuation' in color_params:
                self.evm_processor.chromAttenuation = color_params.get('chromAttenuation', 0.0)
            else:
                # Add default chromAttenuation to reduce color artifacts
                self.evm_processor.chromAttenuation = 1.0
    
    def create_deception_info_panel(self, frame_idx: int, plot_width: int, plot_height: int) -> np.ndarray:
        """
        Override the info panel method to return an empty panel, as we're not using it.
        This is kept for compatibility with the parent class.
        """
        # Return empty panel - not used in this implementation
        empty_panel = np.ones((10, 10, 3), dtype=np.uint8) * 255
        return empty_panel
        
    def process_video(self, input_path: str, output_path: str) -> None:
        """
        Process a video to detect deception using ONLY PBM and EVM.
        
        This overrides the parent implementation to ensure only PBM and EVM are used.
        
        Args:
            input_path: Path to the input video
            output_path: Path to save the output video
        """
        print(f"\n\n===== Processing {os.path.basename(input_path)} with PBM/EVM ONLY =====")
        start_time = time.time()
        
        # Make EVM parameters available to SideBySideMagnification via global variables
        # This allows the calculate_color_changes method to use them without parameter changes
        import builtins
        builtins.EVM_LOW_FREQ = self.evm_processor.f_lo  # Low frequency for bandpass filter
        builtins.EVM_HIGH_FREQ = self.evm_processor.f_hi  # High frequency for bandpass filter
        builtins.EVM_ALPHA = self.evm_processor.alpha  # Amplification factor
        builtins.EVM_LEVEL = self.evm_processor.level  # Pyramid level
        if hasattr(self.evm_processor, 'chromAttenuation'):
            builtins.EVM_CHROM_ATTENUATION = self.evm_processor.chromAttenuation
        else:
            builtins.EVM_CHROM_ATTENUATION = 0.0
            
        print(f"Set EVM parameters globally: f_lo={builtins.EVM_LOW_FREQ:.4f}, f_hi={builtins.EVM_HIGH_FREQ:.4f}, alpha={builtins.EVM_ALPHA}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.fps = fps
        self.total_frames = total_frames
        self.video_dimensions = (width, height)
        
        print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Read all frames
        print("Reading frames...")
        all_frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            all_frames.append(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Read {frame_count}/{total_frames} frames")
        
        # Close input video
        cap.release()
        
        # Initialize result arrays
        magnified_motion_frames = []
        magnified_color_frames = []
        phase_changes = {}
        heart_rate_data = None
        
        # Step 1: Detect faces and extract regions only once per frame (efficiency improvement)
        print("Detecting faces and extracting regions...")
        
        # Detect faces and regions for all frames
        face_detector = self.pbm_processor.face_detector
        cheek_detector = CheekDetector()
        
        all_face_results = []
        all_cheek_results = []
        
        for i, frame in enumerate(all_frames):
            if i % 30 == 0:
                print(f"Detecting faces in frame {i}/{total_frames}")
            
            # Detect face regions for PBM
            face_results = face_detector.detect_faces(frame)
            all_face_results.append(face_results)
            
            # Detect cheek regions for EVM
            detected, cheek_regions = cheek_detector.detect_cheeks(frame)
            all_cheek_results.append((detected, cheek_regions))
        
        # Step 2: Apply PBM for micro-expression detection
        print("Step 1: Applying PBM for micro-expression detection...")
        
        # Initialize output frames
        magnified_motion_frames = [frame.copy() for frame in all_frames]
        
        # Track key regions for motion magnification
        motion_regions = ['left_eye', 'right_eye', 'nose_tip']
        
        # Combined phase changes array for overall micro-expression analysis
        combined_phase_changes = np.zeros(total_frames)
        valid_regions_count = 0
        
        # Process the eye/nose regions with PBM for motion magnification
        if all_face_results[0] and len(all_face_results[0]) > 0 and 'regions' in all_face_results[0][0]:
            for region_name in motion_regions:
                print(f"Processing {region_name} with PBM magnification...")
                
                # Collect region frames for this region
                region_frames = []
                valid_frames = []
                
                for frame_idx, face_results in enumerate(all_face_results):
                    if face_results and len(face_results) > 0 and 'regions' in face_results[0]:
                        face_data = face_results[0]
                        if region_name in face_data['regions']:
                            region_frames.append(face_data['regions'][region_name]['image'])
                            valid_frames.append(frame_idx)
                
                if len(region_frames) > 0:
                    print(f"Collected {len(region_frames)} frames for {region_name}")
                    
                    # Apply PBM magnification to this region
                    magnified_frames, phase_changes_data = self.pbm_processor.phase_magnifier.magnify(region_frames)
                    
                    # Store phase changes for this region
                    region_key = f"face1_{region_name}"
                    phase_changes[region_key] = phase_changes_data
                    
                    # Add to combined phase changes
                    padded_phase_data = np.zeros(total_frames)
                    for i, frame_idx in enumerate(valid_frames):
                        if i < len(phase_changes_data) and frame_idx < total_frames:
                            padded_phase_data[frame_idx] = phase_changes_data[i]
                    
                    combined_phase_changes += padded_phase_data
                    valid_regions_count += 1
                    
                    # Replace regions in magnified motion frames
                    for i, (frame_idx, magnified) in enumerate(zip(valid_frames, magnified_frames)):
                        if i < len(magnified_frames) and frame_idx < len(all_frames):
                            face_data = all_face_results[frame_idx][0]
                            if region_name in face_data['regions']:
                                bounds = face_data['regions'][region_name]['bounds']
                                magnified_motion_frames[frame_idx][bounds[1]:bounds[3], bounds[0]:bounds[2]] = magnified
        
        # Average the combined phase changes and store it
        if valid_regions_count > 0:
            combined_phase_changes /= valid_regions_count
            phase_changes['combined'] = combined_phase_changes
        
        # Step 3: Apply EVM only to cheek regions for heart rate detection
        print("Step 2: Applying EVM for heart rate detection...")
        print(f"EVM Parameters: alpha={self.evm_processor.alpha}, level={self.evm_processor.level}, "
              f"f_lo={self.evm_processor.f_lo:.4f}, f_hi={self.evm_processor.f_hi:.4f}, "
              f"chromAttenuation={getattr(self.evm_processor, 'chromAttenuation', 0.0):.2f}")
        
        # Initialize color magnified frames as copies of original frames
        magnified_color_frames = [frame.copy() for frame in all_frames]
        
        # Extract cheek regions for heart rate detection
        left_cheek_frames = []
        right_cheek_frames = []
        left_cheek_bounds = []
        right_cheek_bounds = []
        
        for frame_idx, (detected, cheek_regions) in enumerate(all_cheek_results):
            if detected and cheek_regions:
                regions = cheek_regions[0]['regions']
                
                if 'left_cheek' in regions:
                    left_cheek_frames.append(regions['left_cheek']['image'])
                    left_cheek_bounds.append(regions['left_cheek']['bounds'])
                else:
                    left_cheek_frames.append(None)
                    left_cheek_bounds.append(None)
                
                if 'right_cheek' in regions:
                    right_cheek_frames.append(regions['right_cheek']['image'])
                    right_cheek_bounds.append(regions['right_cheek']['bounds'])
                else:
                    right_cheek_frames.append(None)
                    right_cheek_bounds.append(None)
        
        # Apply EVM to left cheek regions
        if len([f for f in left_cheek_frames if f is not None]) > 0:
            valid_left_frames = [f for f in left_cheek_frames if f is not None]
            left_magnified = self.evm_processor.magnify(valid_left_frames)
            
            # Replace in the original frames
            left_idx = 0
            for frame_idx, cheek_frame in enumerate(left_cheek_frames):
                if cheek_frame is not None and left_idx < len(left_magnified):
                    bounds = left_cheek_bounds[frame_idx]
                    
                    # Create a subtle blend instead of direct replacement
                    original_region = magnified_color_frames[frame_idx][bounds[1]:bounds[3], bounds[0]:bounds[2]].copy()
                    magnified_region = left_magnified[left_idx].copy()
                    
                    # Ensure regions are the same size
                    if magnified_region.shape[:2] != original_region.shape[:2]:
                        magnified_region = cv2.resize(magnified_region, 
                                                     (original_region.shape[1], original_region.shape[0]))
                    
                    # Make blend factor proportional to EVM alpha but keep it subtle
                    max_blend = 0.25  # Maximum blend percentage allowed
                    blend_factor = min(max_blend, self.evm_processor.alpha / 100.0)  # Scale based on alpha
                    print(f"Using visual blend factor of {blend_factor:.2f} based on alpha={self.evm_processor.alpha}")
                    blended_region = cv2.addWeighted(
                        magnified_region, blend_factor,
                        original_region, 1.0 - blend_factor,
                        0
                    )
                    
                    # Replace the region with the blended result
                    magnified_color_frames[frame_idx][bounds[1]:bounds[3], bounds[0]:bounds[2]] = blended_region
                    
                    left_idx += 1
        
        # Apply EVM to right cheek regions
        if len([f for f in right_cheek_frames if f is not None]) > 0:
            valid_right_frames = [f for f in right_cheek_frames if f is not None]
            right_magnified = self.evm_processor.magnify(valid_right_frames)
            
            # Replace in the original frames
            right_idx = 0
            for frame_idx, cheek_frame in enumerate(right_cheek_frames):
                if cheek_frame is not None and right_idx < len(right_magnified):
                    bounds = right_cheek_bounds[frame_idx]
                    
                    # Create a subtle blend instead of direct replacement
                    original_region = magnified_color_frames[frame_idx][bounds[1]:bounds[3], bounds[0]:bounds[2]].copy()
                    magnified_region = right_magnified[right_idx].copy()
                    
                    # Ensure regions are the same size
                    if magnified_region.shape[:2] != original_region.shape[:2]:
                        magnified_region = cv2.resize(magnified_region, 
                                                     (original_region.shape[1], original_region.shape[0]))
                    
                    # Make blend factor proportional to EVM alpha but keep it subtle
                    max_blend = 0.25  # Maximum blend percentage allowed
                    blend_factor = min(max_blend, self.evm_processor.alpha / 100.0)  # Scale based on alpha
                    print(f"Using visual blend factor of {blend_factor:.2f} based on alpha={self.evm_processor.alpha}")
                    blended_region = cv2.addWeighted(
                        magnified_region, blend_factor,
                        original_region, 1.0 - blend_factor,
                        0
                    )
                    
                    # Replace the region with the blended result
                    magnified_color_frames[frame_idx][bounds[1]:bounds[3], bounds[0]:bounds[2]] = blended_region
                    
                    right_idx += 1
        
        # Calculate heart rate using EVM
        try:
            # Calculate color changes for each cheek using EVM
            # Note: This is the ONLY source for heart rate/BPM calculation in the system
            # The entire heart rate detection is based on Eulerian Video Magnification (EVM)
            color_signals = {}
            
            # Process left cheek
            valid_left_frames = [f for f in left_cheek_frames if f is not None]
            if valid_left_frames:
                color_signals['left_cheek'] = self.calculate_color_changes(valid_left_frames)
            
            # Process right cheek
            valid_right_frames = [f for f in right_cheek_frames if f is not None]
            if valid_right_frames:
                color_signals['right_cheek'] = self.calculate_color_changes(valid_right_frames)
            
            # Calculate heart rate data
            if color_signals:
                heart_rate_bin_size = self.params['heart_rate_bin_size']
                
                # Log EVM parameters being used for heart rate calculation
                print("----- EVM Parameters for Heart Rate Detection -----")
                print(f"Alpha (magnification): {self.evm_processor.alpha:.1f}")
                print(f"Level (pyramid level): {self.evm_processor.level}")
                print(f"Frequency range: {self.evm_processor.f_lo*60:.1f}-{self.evm_processor.f_hi*60:.1f} BPM")
                print(f"ChromAttenuation: {getattr(self.evm_processor, 'chromAttenuation', 0.0):.2f}")
                print(f"Heart rate bin size: {heart_rate_bin_size} frames")
                print("---------------------------------------------------")
                
                # Calculate heart rate data using our EVM-based method
                print("Calculating heart rate from EVM color signals...")
                heart_rate_data = self.calculate_bpm(color_signals, fps=fps)
                
                if heart_rate_data is not None:
                    # Now manually apply the binning to the avg_bpm data
                    inst_bpm, avg_bpm, signal = heart_rate_data
                    
                    # Create binned/flattened BPM array
                    flattened_bpm = np.zeros_like(avg_bpm)
                    num_bins = max(1, int(np.ceil(len(avg_bpm) / heart_rate_bin_size)))
                    
                    # Calculate average BPM for each bin and apply it to all frames in that bin
                    for bin_idx in range(num_bins):
                        bin_start = bin_idx * heart_rate_bin_size
                        bin_end = min(len(avg_bpm), bin_start + heart_rate_bin_size)
                        
                        if bin_end > bin_start:
                            # Calculate bin average (only for frames with valid data)
                            bin_values = avg_bpm[bin_start:bin_end]
                            valid_values = bin_values[bin_values > 0]
                            
                            if len(valid_values) > 0:
                                bin_avg = np.mean(valid_values)
                                # Apply the same average to all frames in this bin
                                flattened_bpm[bin_start:bin_end] = bin_avg
                    
                    # Print statistics about the flattened heart rate
                    valid_avg = avg_bpm[avg_bpm > 0]
                    valid_flattened = flattened_bpm[flattened_bpm > 0]
                    if len(valid_avg) > 0 and len(valid_flattened) > 0:
                        print(f"Original BPM range: {np.min(valid_avg):.2f} - {np.max(valid_avg):.2f} BPM")
                        print(f"Flattened BPM range: {np.min(valid_flattened):.2f} - {np.max(valid_flattened):.2f} BPM")
                    
                    # Replace the original avg_bpm with our flattened version
                    # and create a new heart_rate_data tuple
                    heart_rate_data = (inst_bpm, flattened_bpm, signal)
                    
                print(f"Extracted heart rate data with {len(heart_rate_data[1])} points")
        except Exception as e:
            print(f"Warning: Failed to extract heart rate data: {str(e)}")
            heart_rate_data = None
        
        # Step 4: Fit the deception detector using PBM and EVM data
        print("Step 3: Fitting deception detection model with PBM/EVM data...")
        # Use ONLY phase changes for feature extraction, no other motion features
        self.detector.fit(phase_changes, heart_rate_data, fps, total_frames)
        
        # Step 5: Create the output visualization with horizontal layout
        print("Step 4: Creating annotated output video...")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Calculate dimensions for the output video with horizontal layout for graphs
        # Make sure all dimensions are even numbers (required by video codecs)
        output_width = width * 3  # Three videos side by side
        output_width = output_width if output_width % 2 == 0 else output_width - 1
        
        # Make plots the same size as videos for better visibility and consistency
        plot_height = height  # Make plots as tall as videos
        plot_height = plot_height if plot_height % 2 == 0 else plot_height - 1
        
        plot_width = width  # Make plots as wide as videos
        plot_width = plot_width if plot_width % 2 == 0 else plot_width - 1
        
        # Calculate total output height - 2 rows: videos and plots
        output_height = height + plot_height
        output_height = output_height if output_height % 2 == 0 else output_height - 1
        
        print(f"Output video dimensions: {output_width}x{output_height}")
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        # Process each frame to create final output
        for frame_idx in range(len(all_frames)):
            if frame_idx % 30 == 0:
                print(f"Processing output frame {frame_idx}/{total_frames}")
            
            # Get frames
            original_frame = all_frames[frame_idx]
            motion_frame = magnified_motion_frames[frame_idx]
            color_frame = magnified_color_frames[frame_idx]
            
            # Use combined phase data for visualization
            combined_phase_data = phase_changes.get('combined', None)
            
            if combined_phase_data is not None and len(combined_phase_data) > frame_idx:
                phase_data = combined_phase_data
                phase_title = "PBM: Micro-Expression Detection: Aggregated Movement"
            else:
                # Create empty phase data if none available
                phase_data = np.zeros(total_frames)
                phase_title = "No Micro-Expression Data"
            
            # Create micro-expression movement graph
            phase_plot = self.processor.create_single_plot(
                phase_data, frame_idx, plot_width, plot_height,
                plot_type="diff", title=phase_title, total_frames=total_frames
            )
            
            # Highlight deception regions on the phase plot
            phase_plot = self.detector.highlight_deception_regions(phase_plot, frame_idx)
            
            # Create heart rate plot
            if heart_rate_data is not None:
                heart_plot = self.processor.create_heart_rate_plot(
                    heart_rate_data, frame_idx, plot_width, plot_height
                )
                heart_plot = self.detector.highlight_deception_regions(heart_plot, frame_idx)
                heart_title = f"Heart Rate Estimation (Bin Size: {self.params['heart_rate_bin_size']} frames)"
                # Add title to heart rate plot if not already present - make larger and more visible
                cv2.putText(heart_plot, heart_title, (plot_width//10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            else:
                # Create empty heart rate plot if data is not available
                heart_plot = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
                cv2.putText(heart_plot, "No Heart Rate Data Available", 
                           (plot_width//10, plot_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            
            # Create deception timeline plot
            timeline_plot = self.detector.create_deception_timeline_plot(
                frame_idx, plot_width, plot_height
            )
            timeline_title = "Deception Detection Timeline"
            # Add title to timeline plot if not already present - make larger and more visible
            cv2.putText(timeline_plot, timeline_title, (plot_width//10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            
            # Stitch everything together with horizontal layout for plots
            output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            
            # Add videos on top row
            video_row = np.hstack([
                cv2.resize(original_frame, (width, height)),
                cv2.resize(motion_frame, (width, height)),
                cv2.resize(color_frame, (width, height))
            ])
            output_frame[:height, :] = video_row
            
            # Add plots in bottom row - same size as videos
            plots_row = np.hstack([phase_plot, heart_plot, timeline_plot])
            output_frame[height:, :] = plots_row
            
            # Add labels for each video
            label_bg_height = 40  # Increased height for better visibility
            label_y = height - label_bg_height
            
            # Original video label
            cv2.rectangle(output_frame, (0, label_y), (width, height), (0, 0, 0), -1)
            cv2.putText(output_frame, ORIGINAL_LABEL, (width//2 - 90, height - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # PBM video label
            cv2.rectangle(output_frame, (width, label_y), (width*2, height), (0, 0, 0), -1)
            cv2.putText(output_frame, PBM_LABEL, (width + width//2 - 130, height - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # EVM video label
            cv2.rectangle(output_frame, (width*2, label_y), (width*3, height), (0, 0, 0), -1)
            cv2.putText(output_frame, EVM_LABEL, (width*2 + width//2 - 100, height - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Write frame to output
            out.write(output_frame)
        
        # Release output video
        out.release()
        
        # Print completion message
        elapsed_time = time.time() - start_time
        print(f"Processed {os.path.basename(input_path)} with PBM/EVM in {elapsed_time:.1f} seconds")
        print(f"Output saved to {output_path}")
        print(f"===== Completed processing with PBM/EVM ONLY =====\n\n")

    def calculate_color_changes(self, frames):
        """Calculate color changes across frames specifically using Eulerian Video Magnification
        
        This method extracts color change intensity directly using EVM magnification process
        rather than using any green channel analysis or other approaches. This ensures the
        color change intensity and heartrate calculations are purely based on EVM.
        
        Args:
            frames: List of region frames
            
        Returns:
            Array of color changes across frames from EVM processing
        """
        try:
            if not frames or len(frames) < 2:
                return np.zeros(len(frames) if frames else 0)
                
            # Get parameters from our initialized EVM processor
            print(f"Using EVM parameters: f_lo={self.evm_processor.f_lo:.4f}, f_hi={self.evm_processor.f_hi:.4f}, alpha={self.evm_processor.alpha}, level={self.evm_processor.level}")
            
            # Convert all frames to YIQ color space (used by EVM)
            yiq_frames = np.array([self.evm_processor.bgr2yiq(frame) for frame in frames])
            
            # Get dimensions for the pyramid
            height, width = frames[0].shape[:2]
            pyr_height = height
            pyr_width = width
            for _ in range(self.evm_processor.level):
                pyr_height = (pyr_height + 1) // 2
                pyr_width = (pyr_width + 1) // 2
            
            # Focus on Y channel (luminance) - same as in the EVM magnify method
            channel = 0  # Y channel in YIQ
            channel_frames = yiq_frames[:, :, :, channel]
            
            # Build pyramid for each frame using EVM's own method
            pyramid_frames = np.zeros((len(frames), pyr_height, pyr_width))
            
            for i, frame in enumerate(channel_frames):
                # Create 2D frame
                frame_2d = frame.reshape(height, width)
                pyramid = self.evm_processor.build_gaussian_pyramid(frame_2d)
                # Get the last level (most downsampled)
                pyramid_frames[i] = cv2.resize(pyramid[-1], (pyr_width, pyr_height))
            
            # Apply temporal filter to pyramid level using EVM's own method
            filtered = self.evm_processor.temporal_bandpass_filter(pyramid_frames)
            
            # To obtain a 1D signal, compute the mean of the filtered signal at each frame
            # This represents the overall color change intensity from the EVM process
            signal_data = np.array([np.mean(filtered[i]) for i in range(len(frames))])
            
            # Normalize signal
            signal_data = signal_data - np.mean(signal_data)
            if np.std(signal_data) > 0:
                signal_data = signal_data / np.std(signal_data)
            
            return signal_data
            
        except Exception as e:
            print(f"Error in calculate_color_changes: {str(e)}")
            # Return zeros with the same length as input
            return np.zeros(len(frames) if frames else 0)

    def calculate_bpm(self, color_signals, fps=30):
        """Calculate heart rate (BPM) from color signals using EVM-based approach
        
        Extracts heart rate (beats per minute) from the color signals generated
        by the EVM process. This ensures the heart rate calculation is purely based
        on the Eulerian Video Magnification approach.
        
        Args:
            color_signals: Dictionary of color signals from different regions
            fps: Frames per second of the video
            
        Returns:
            Tuple of:
                - Array of instantaneous BPM values per frame
                - Array of average BPM values per window (smoother)
                - Array of combined color signal
        """
        try:
            if not color_signals:
                print("No color signals provided to calculate BPM")
                return None
                
            # Find the first non-empty signal to get the length
            signal_length = 0
            for signal_data in color_signals.values():
                if len(signal_data) > 0:
                    signal_length = len(signal_data)
                    break
                    
            if signal_length == 0:
                print("All color signals are empty")
                return None
            
            # Combine signals from all regions (typically left and right cheeks)
            combined_signal = np.zeros(signal_length)
            count = 0
            
            for signal_data in color_signals.values():
                if len(signal_data) == signal_length:
                    combined_signal += signal_data
                    count += 1
                    
            if count == 0:
                print("No valid color signals to combine")
                return None
                
            # Normalize combined signal
            combined_signal = combined_signal - np.mean(combined_signal)
            if np.std(combined_signal) > 0:
                combined_signal = combined_signal / np.std(combined_signal)
            
            # Get heart rate range directly from EVM parameters (Hz to BPM)
            f_lo_bpm = self.evm_processor.f_lo * 60
            f_hi_bpm = self.evm_processor.f_hi * 60
            print(f"Using EVM heart rate frequency range: {f_lo_bpm:.1f}-{f_hi_bpm:.1f} BPM")
            
            # Initialize BPM arrays
            inst_bpm = np.zeros(signal_length)
            
            # Detect peaks in the signal using consistent parameters with EVM
            # Minimum distance based on lowest expected heart rate frequency
            min_heart_rate_hz = self.evm_processor.f_lo
            peak_distance = int(fps / min_heart_rate_hz / 2)  # Frames between peaks
            
            # Keep within reasonable bounds
            peak_distance = max(3, min(peak_distance, signal_length // 4))
            print(f"Peak detection distance: {peak_distance} frames (based on EVM f_lo={self.evm_processor.f_lo:.4f}Hz)")
            
            # Find peaks in the combined signal
            peaks, _ = signal.find_peaks(combined_signal, distance=peak_distance)
            
            # If we have enough peaks, calculate BPM from peak intervals
            if len(peaks) >= 2:
                print(f"Found {len(peaks)} peaks in the heart rate signal")
                
                # Calculate instantaneous heart rate between consecutive peaks
                for i in range(len(peaks) - 1):
                    # Time between peaks in seconds
                    time_between_peaks = (peaks[i+1] - peaks[i]) / fps
                    
                    # Convert to BPM
                    if time_between_peaks > 0:
                        curr_bpm = 60 / time_between_peaks
                        
                        # Only use values within EVM frequency range
                        if f_lo_bpm <= curr_bpm <= f_hi_bpm:
                            # Apply to frames between these peaks
                            start_idx = peaks[i]
                            end_idx = peaks[i+1]
                            inst_bpm[start_idx:end_idx] = curr_bpm
                
                # Fill in any remaining frames at the end
                if peaks[-1] < signal_length and len(peaks) >= 2:
                    # Use last valid interval
                    last_interval = (peaks[-1] - peaks[-2]) / fps
                    if last_interval > 0:
                        last_bpm = 60 / last_interval
                        if f_lo_bpm <= last_bpm <= f_hi_bpm:
                            inst_bpm[peaks[-1]:] = last_bpm
            else:
                # Not enough peaks - use frequency domain analysis (consistent with EVM)
                print(f"Not enough peaks detected ({len(peaks)}). Using frequency domain analysis.")
                
                # Perform FFT - consistent with frequency-domain approach of EVM
                fft_values = np.abs(np.fft.rfft(combined_signal))
                frequencies = np.fft.rfftfreq(len(combined_signal), d=1.0/fps)
                
                # Limit to EVM frequency range
                valid_range = (frequencies >= self.evm_processor.f_lo) & (frequencies <= self.evm_processor.f_hi)
                if np.any(valid_range):
                    valid_freqs = frequencies[valid_range]
                    valid_fft = fft_values[valid_range]
                    
                    if len(valid_fft) > 0:
                        # Find dominant frequency in the EVM band
                        peak_idx = np.argmax(valid_fft)
                        peak_freq = valid_freqs[peak_idx]
                        
                        # Convert to BPM
                        default_bpm = peak_freq * 60
                        print(f"Dominant frequency: {peak_freq:.4f}Hz ({default_bpm:.1f} BPM)")
                        
                        # Apply to all frames
                        inst_bpm[:] = default_bpm
                    else:
                        # Use middle of the EVM range if no clear peak
                        default_bpm = (f_lo_bpm + f_hi_bpm) / 2
                        inst_bpm[:] = default_bpm
                else:
                    # Use middle of the EVM range if no valid frequencies
                    default_bpm = (f_lo_bpm + f_hi_bpm) / 2
                    inst_bpm[:] = default_bpm
            
            # Calculate average BPM with sliding window
            # Window size based on EVM parameters for consistency
            avg_bpm = np.zeros_like(inst_bpm)
            window_len = int(fps * 3 / self.evm_processor.f_lo)  # 3 cycles of lowest frequency
            window_len = min(window_len, signal_length // 2)
            window_len = max(window_len, int(fps * 2))  # At least 2 seconds
            
            print(f"Using heart rate averaging window of {window_len} frames ({window_len/fps:.1f} seconds)")
            
            # Apply sliding window average
            for i in range(signal_length):
                start_idx = max(0, i - window_len//2)
                end_idx = min(signal_length, i + window_len//2)
                values = inst_bpm[start_idx:end_idx]
                values = values[values > 0]  # Only use non-zero values
                if len(values) > 0:
                    avg_bpm[i] = np.mean(values)
                else:
                    # Use midpoint of valid range if no values
                    avg_bpm[i] = (f_lo_bpm + f_hi_bpm) / 2
            
            # Apply final smoothing for display
            if signal_length > 5:
                # Determine window size (must be odd)
                sg_window = min(21, signal_length // 2)
                if sg_window % 2 == 0:
                    sg_window += 1
                
                if sg_window >= 5:  # Need at least 5 points for quadratic fit
                    avg_bpm = signal.savgol_filter(avg_bpm, sg_window, 2)
            
            print(f"Heart rate analysis complete. BPM range: {np.min(avg_bpm[avg_bpm > 0]):.1f}-{np.max(avg_bpm):.1f}")
            return inst_bpm, avg_bpm, combined_signal
            
        except Exception as e:
            print(f"Error in calculate_bpm: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="""
    PBM/EVM-EXCLUSIVE Deception Detector
    
    This tool EXCLUSIVELY uses:
    - Phase-Based Magnification (PBM) for micro-expression detection
    - Eulerian Video Magnification (EVM) for heart rate analysis
    
    No other motion detection or physiological measurement techniques are used.
    This is a focused study on evaluating these specific technologies.
    """)
    
    parser.add_argument(
        '--input', '-i', type=str, default=DEFAULT_INPUT_DIR,
        help=f'Path to input videos folder (default: {DEFAULT_INPUT_DIR})'
    )
    
    parser.add_argument(
        '--output', '-o', type=str, default=DEFAULT_OUTPUT_DIR,
        help=f'Path to output videos folder (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--single', '-s', type=str, default=None,
        help='Process a single video file instead of the entire folder'
    )
    
    return parser.parse_args()

def run_detector():
    """
    Run the lie detector using ONLY PBM for micro-expression detection
    and EVM for heart rate analysis.
    """
    args = parse_arguments()
    
    print("\n" + "="*80)
    print(" PBM/EVM-EXCLUSIVE Deception Detector ".center(80, "="))
    print("="*80)
    print("This detector EXCLUSIVELY utilizes:")
    print("1. Phase-Based Magnification (PBM) for micro-expression detection")
    print("2. Eulerian Video Magnification (EVM) for heart rate analysis")
    print("No other motion detection or physiological measurement techniques are used.")
    print("="*80 + "\n")
    
    print(f"Current working directory: {os.getcwd()}")
    
    # Create our specialized detector instance that uses only PBM and EVM
    detector = PBMEVMDetector(
        motion_params=PBM_PARAMS,
        color_params=EVM_PARAMS
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    if args.single:
        # Process a single video
        if not os.path.exists(args.single):
            print(f"Error: Input file '{args.single}' not found")
            return
        
        print(f"Processing single video: {args.single}")
        basename = os.path.basename(args.single)
        file_name = os.path.splitext(basename)[0]
        output_path = os.path.join(args.output, f"pbm_evm_only_{file_name}.mp4")
        
        detector.process_video(args.single, output_path)
    else:
        # Process all videos in the folder
        print(f"Processing all videos in: {args.input}")
        print(f"Output folder: {args.output}")
        
        detector.process_folder(args.input, args.output)
    
    print("\n" + "="*80)
    print(" PBM/EVM-EXCLUSIVE Deception Detector Complete ".center(80, "="))
    print("="*80 + "\n")

if __name__ == "__main__":
    run_detector() 