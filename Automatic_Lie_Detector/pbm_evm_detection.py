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
        
        # Override detector with specialized detection parameters
        self.detector = DeceptionDetector(DETECTION_PARAMS)
        
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
            self.evm_processor.alpha = color_params.get('alpha', 50)
            self.evm_processor.level = color_params.get('level', 3)
            self.evm_processor.f_lo = color_params.get('f_lo', 0.83)
            self.evm_processor.f_hi = color_params.get('f_hi', 1.0)
    
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
                    magnified_color_frames[frame_idx][bounds[1]:bounds[3], bounds[0]:bounds[2]] = left_magnified[left_idx]
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
                    magnified_color_frames[frame_idx][bounds[1]:bounds[3], bounds[0]:bounds[2]] = right_magnified[right_idx]
                    right_idx += 1
        
        # Calculate heart rate using EVM
        try:
            # Calculate color changes for each cheek using EVM
            color_signals = {}
            
            # Process left cheek
            valid_left_frames = [f for f in left_cheek_frames if f is not None]
            if valid_left_frames:
                color_signals['left_cheek'] = self.processor.calculate_color_changes(valid_left_frames)
            
            # Process right cheek
            valid_right_frames = [f for f in right_cheek_frames if f is not None]
            if valid_right_frames:
                color_signals['right_cheek'] = self.processor.calculate_color_changes(valid_right_frames)
            
            # Calculate heart rate data - but don't pass the heart_rate_bin_size parameter yet
            # The original error shows we're actually using SideBySideMagnification's method
            # which doesn't accept this parameter
            if color_signals:
                heart_rate_bin_size = self.params['heart_rate_bin_size']
                print(f"Using heart rate bin size of {heart_rate_bin_size} frames for BPM calculation")
                
                # Get regular heart rate data first
                heart_rate_data = self.processor.calculate_bpm(color_signals, fps=fps)
                
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
    
    print("===== Starting PBM/EVM-EXCLUSIVE Deception Detector =====")
    print(f"Current directory: {os.getcwd()}")
    print("Using ONLY Phase-Based Magnification (PBM) for micro-expression detection")
    print("Using ONLY Eulerian Video Magnification (EVM) for heart rate analysis")
    
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
    
    print("===== PBM/EVM-EXCLUSIVE Deception Detector Complete =====")

if __name__ == "__main__":
    run_detector() 