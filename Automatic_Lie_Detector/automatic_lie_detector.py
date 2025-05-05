"""
Automatic Lie Detector main module.

This module processes videos through micro-expression and heart rate magnification,
then applies an unsupervised anomaly detection algorithm to identify potential
regions of deception in the video.
"""

import os
import sys
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import glob
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from io import BytesIO
from PIL import Image

# Import from Separate_Color_Motion_Magnification
from Separate_Color_Motion_Magnification.side_by_side_magnification import SideBySideMagnification
from Separate_Color_Motion_Magnification.upper_face_detector import UpperFaceDetector
from Separate_Color_Motion_Magnification.cheek_detector import CheekDetector

# Import from Face_Motion_Magnification
from Face_Motion_Magnification.face_region_motion_magnification import FacialPhaseMagnification
# Import from Face_Color_Magnification
from Face_Color_Magnification.face_region_color_magnification import ColorMagnification

# Import from config - this will be the custom ConfigModule from run_lie_detector_fixed.py
import config
from .deception_detector import DeceptionDetector


# Define detection parameters - this is needed because config.DETECTION_PARAMS might not be available
DETECTION_PARAMS = {
    # Temporal window parameters
    'window_size_seconds': 3,      # Size of sliding window in seconds
    'window_overlap': 0.5,         # Overlap between consecutive windows (0.5 = 50% overlap)
    
    # Anomaly detection parameters
    'anomaly_threshold': 90,       # Percentile threshold for anomaly detection (lowered from 95)
    'min_anomaly_score': 0.6,      # Minimum anomaly score (lowered from 0.7)
    
    # Feature weighting
    'feature_weights': {
        'phase_change': 8.0,       # Extreme weight for micro-expression features (PBM motion)
        'heart_rate': 0.1,         # Very low weight for heart rate features (EVM color)
        'cross_correlation': 0.1,  # Minimal weight for correlation between signals
    },
    
    # Peak detection parameters (new)
    'peak_detection': {
        'threshold_multiplier': 1.25,  # Lower threshold to detect more peaks (standard deviations above mean)
        'cluster_importance': 2.0,     # Multiplier for cluster score importance
    },
    
    # Visualization
    'highlight_color': (0, 0, 255),  # Red color for highlighting deception regions (BGR)
    'highlight_alpha': 0.3,          # Transparency for highlighted regions
}


class AutomaticLieDetector:
    """
    Processes videos to detect potential regions of deception by analyzing
    micro-expressions and physiological patterns like heart rate.
    """
    
    def __init__(self, motion_params=None, color_params=None):
        """
        Initialize the lie detector with magnification parameters.
        
        Args:
            motion_params: Optional parameters for motion magnification
            color_params: Optional parameters for color magnification
        """
        # Use provided parameters or default from config
        self.motion_params = motion_params or config.MOTION_MAG_PARAMS
        self.color_params = color_params or config.COLOR_MAG_PARAMS
        
        # Initialize the side-by-side magnification processor
        # This is reused from the existing implementation
        self.processor = SideBySideMagnification(
            motion_params=self.motion_params,
            color_params=self.color_params
        )
        
        # Initialize the deception detector - directly pass the detection params
        self.detector = DeceptionDetector(DETECTION_PARAMS)
        
        # Store video metadata
        self.fps = 30  # Default, will be updated with actual fps
        self.total_frames = 0
        self.video_dimensions = (0, 0)  # (width, height)
    
    def is_valid_array(self, arr, min_length=0):
        """
        Check if an array is valid and has at least min_length elements.
        
        Args:
            arr: Array to check
            min_length: Minimum length required
            
        Returns:
            True if the array is valid and has at least min_length elements
        """
        return arr is not None and isinstance(arr, (list, np.ndarray)) and len(arr) > min_length
    
    def add_label(self, frame: np.ndarray, text: str) -> np.ndarray:
        """Add a text label to the frame - reused from SideBySideMagnification"""
        frame_with_label = frame.copy()
        h, w = frame_with_label.shape[:2]
        
        # Create a semi-transparent overlay for the label background
        overlay = frame_with_label.copy()
        cv2.rectangle(overlay, (0, h-config.LABEL_HEIGHT), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, config.LABEL_ALPHA, frame_with_label, 1-config.LABEL_ALPHA, 0, frame_with_label)
        
        # Add text - centering calculation improved to prevent cut-off
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, config.LABEL_FONT_SCALE, config.LABEL_THICKNESS)[0]
        text_x = max(20, int(w/2 - text_size[0]/2))  # Ensure minimum margin of 20px
        
        cv2.putText(
            frame_with_label, 
            text, 
            (text_x, h-15), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            config.LABEL_FONT_SCALE, 
            (255, 255, 255), 
            config.LABEL_THICKNESS, 
            cv2.LINE_AA
        )
        
        return frame_with_label
    
    def create_deception_info_panel(self, frame_idx: int, plot_width: int, plot_height: int) -> np.ndarray:
        """
        Create an information panel displaying deception detection results.
        
        Args:
            frame_idx: Current frame index
            plot_width: Width of the plot
            plot_height: Height of the plot
            
        Returns:
            Information panel as numpy array
        """
        # Create base panel with white background
        panel = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
        
        # Get deception score and peak count for current frame
        score, is_deception = self.detector.get_frame_deception_score(frame_idx)
        
        # Calculate current time in seconds
        current_time = frame_idx / self.fps
        
        # Find the deception window containing the current frame, if any
        current_window = None
        for window in self.detector.deception_windows:
            if window['start_frame'] <= frame_idx <= window['end_frame']:
                current_window = window
                break
        
        # Get peak details if we have them
        peak_count = 0
        if current_window and 'detailed_peaks' in current_window:
            peak_count = current_window['detailed_peaks']['count']
        
        # Create header with deception indicator
        if is_deception:
            # Red banner for potential deception
            cv2.rectangle(panel, (0, 0), (plot_width, 40), (0, 0, 255), -1)
            
            if peak_count > 0:
                text = f"POTENTIAL DECEPTION: {peak_count} MICRO-EXPRESSION PEAKS"
            else:
                text = "POTENTIAL DECEPTION DETECTED"
                
            color = (255, 255, 255)  # White text on red
        else:
            # Green banner for no deception
            cv2.rectangle(panel, (0, 0), (plot_width, 40), (0, 128, 0), -1)
            text = "No Deception Indicators"
            color = (255, 255, 255)  # White text on green
        
        # Add header text
        cv2.putText(
            panel, text, (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA
        )
        
        # Add score indicator
        score_text = f"Anomaly Score: {score:.2f}"
        cv2.putText(
            panel, score_text, (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA
        )
        
        # Add time information - prominently display seconds instead of frames
        time_text = f"Time: {current_time:.2f}s (Frame: {frame_idx}/{self.total_frames})"
        cv2.putText(
            panel, time_text, (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA
        )
        
        # Add deception window time range if in a deception window
        if current_window and 'time_info' in current_window:
            time_info = current_window['time_info']
            window_text = f"Deception Window: {time_info['start_time']:.2f}s - {time_info['end_time']:.2f}s (Duration: {time_info['duration']:.2f}s)"
            
            # Use red text for deception window info
            cv2.putText(
                panel, window_text, (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 2, cv2.LINE_AA
            )
        
        # Add deception timeline plot (majority of the panel)
        timeline_plot = self.detector.create_deception_timeline_plot(
            frame_idx, plot_width, plot_height - 150
        )
        
        # Place the timeline plot in the panel
        panel[150:, :] = timeline_plot
        
        return panel
    
    def stitch_frames_with_plots(self, original: np.ndarray, motion: np.ndarray, 
                                color: np.ndarray, phase_plot: np.ndarray, 
                                heart_plot: np.ndarray, info_panel: np.ndarray) -> np.ndarray:
        """
        Stitch together the video frames and plots to create the final output frame.
        
        Args:
            original: Original video frame
            motion: Motion-magnified frame
            color: Color-magnified frame
            phase_plot: Phase change plot
            heart_plot: Heart rate plot
            info_panel: Deception information panel
            
        Returns:
            Combined frame as numpy array
        """
        # Add labels to video frames
        original_labeled = self.add_label(original.copy(), config.ORIGINAL_LABEL)
        motion_labeled = self.add_label(motion.copy(), config.MOTION_LABEL)
        color_labeled = self.add_label(color.copy(), config.COLOR_LABEL)
        
        # Ensure all frames have the same dimensions
        h_orig, w_orig = original_labeled.shape[:2]
        motion_labeled = cv2.resize(motion_labeled, (w_orig, h_orig))
        color_labeled = cv2.resize(color_labeled, (w_orig, h_orig))
        
        # Stitch the three videos horizontally
        videos_row = np.hstack((original_labeled, motion_labeled, color_labeled))
        h_videos, w_videos = videos_row.shape[:2]
        
        # Calculate correct plot height for 16:9 aspect ratio (ensuring even numbers)
        plot_width = w_videos // 2  # Half the width of the video row
        plot_width = plot_width if plot_width % 2 == 0 else plot_width - 1  # Ensure even
        plot_height = (plot_width * 9) // 16  # 16:9 aspect ratio
        plot_height = plot_height if plot_height % 2 == 0 else plot_height - 1  # Ensure even
        
        # Resize plots to the correct dimensions
        phase_plot_resized = cv2.resize(phase_plot, (plot_width, plot_height))
        heart_plot_resized = cv2.resize(heart_plot, (plot_width, plot_height))
        
        # Stack plots horizontally
        plots_row = np.hstack((phase_plot_resized, heart_plot_resized))
        
        # Calculate info panel height (ensuring even numbers)
        info_height = (plot_height * 7) // 10  # 70% of plot height
        info_height = info_height if info_height % 2 == 0 else info_height - 1  # Ensure even
        
        # Resize info panel
        info_panel_resized = cv2.resize(info_panel, (w_videos, info_height))
        
        # Stack plots and info panel vertically
        bottom_section = np.vstack((plots_row, info_panel_resized))
        
        # Stack everything vertically
        result = np.vstack((videos_row, bottom_section))
        
        # Final check to ensure dimensions are even
        h_result, w_result = result.shape[:2]
        if h_result % 2 != 0 or w_result % 2 != 0:
            # If either dimension is odd, crop to make even
            new_h = h_result if h_result % 2 == 0 else h_result - 1
            new_w = w_result if w_result % 2 == 0 else w_result - 1
            result = result[:new_h, :new_w]
        
        return result
    
    def process_video(self, input_path: str, output_path: str) -> None:
        """
        Process a video to detect deception.
        
        Args:
            input_path: Path to the input video
            output_path: Path to save the output video
        """
        print(f"\n\n===== Processing {os.path.basename(input_path)} for deception detection =====")
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
        
        # Create temporary output paths for processing
        temp_dir = os.path.join(os.path.dirname(output_path), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Use the SideBySideMagnification class to process the video and extract features
        print("Step 1: Applying magnification and extracting features...")
        
        # Create a processor instance
        processor = SideBySideMagnification(
            motion_params=self.motion_params,
            color_params=self.color_params
        )
        
        # Read frames and apply magnification
        all_frames = []
        magnified_motion_frames = []
        magnified_color_frames = []
        phase_changes = {}
        heart_rate_data = None
        
        # Read all frames
        print("Reading frames...")
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
        
        # Process with the side-by-side magnification
        # This will apply both the motion and color magnification
        print("Applying motion and color magnification...")
        
        # First detect faces in the initial frame
        face_data = processor.detect_micro_expression_regions(all_frames[0])
        
        # Track key regions for motion magnification
        motion_regions = ['left_eye', 'right_eye', 'nose_tip']
        
        # Combined phase changes array for overall micro-expression analysis
        combined_phase_changes = None
        valid_regions_count = 0
        
        # Only process if we have detected regions
        if 'regions' in face_data and len(face_data['regions']) > 0:
            # Process each key region separately for motion magnification
            for region_name in motion_regions:
                # Standardize region key for storage
                region_key = f"face1_{region_name}"
                print(f"Processing {region_key} with PBM magnification...")
                
                # Collect region frames
                region_frames = []
                valid_frames = []
                
                for frame_idx, frame in enumerate(all_frames):
                    # Detect regions for this frame
                    current_face_data = processor.detect_micro_expression_regions(frame)
                    if 'regions' in current_face_data:
                        # Find matching region in current frame
                        for curr_name, curr_info in current_face_data['regions'].items():
                            if ((region_name == 'left_eye' and 'left' in curr_name and 'eye' in curr_name) or
                                (region_name == 'right_eye' and 'right' in curr_name and 'eye' in curr_name) or
                                (region_name == 'nose_tip' and 'nose' in curr_name)):
                                region_frames.append(curr_info['image'])
                                valid_frames.append(frame_idx)
                                break
                
                if len(region_frames) > 0:
                    print(f"Collected {len(region_frames)} frames for {region_key}")
                    
                    # Apply PBM magnification to get phase changes
                    magnified_frames, phase_changes_data = processor.motion_processor.phase_magnifier.magnify(region_frames)
                    
                    # Store phase changes for this region
                    phase_changes[region_key] = phase_changes_data
                    print(f"Extracted phase changes for {region_key}: {len(phase_changes_data)} values")
                    
                    # Add to combined phase changes
                    if combined_phase_changes is None:
                        combined_phase_changes = np.zeros(total_frames)
                    
                    # Ensure the phase data covers the full video length
                    padded_phase_data = np.zeros(total_frames)
                    for i, frame_idx in enumerate(valid_frames):
                        if i < len(phase_changes_data) and frame_idx < total_frames:
                            padded_phase_data[frame_idx] = phase_changes_data[i]
                    
                    # Add to combined data
                    combined_phase_changes += padded_phase_data
                    valid_regions_count += 1
                    
                    # Replace regions in magnified motion frames
                    for i, (frame_idx, magnified) in enumerate(zip(valid_frames, magnified_frames)):
                        if frame_idx < len(all_frames):
                            # Initialize magnified motion frame if needed
                            if frame_idx >= len(magnified_motion_frames):
                                while len(magnified_motion_frames) <= frame_idx:
                                    magnified_motion_frames.append(all_frames[len(magnified_motion_frames)].copy())
                            
                            # Get current face data for this frame
                            current_face_data = processor.detect_micro_expression_regions(all_frames[frame_idx])
                            if 'regions' in current_face_data:
                                # Find matching region to replace
                                for curr_name, curr_info in current_face_data['regions'].items():
                                    if ((region_name == 'left_eye' and 'left' in curr_name and 'eye' in curr_name) or
                                        (region_name == 'right_eye' and 'right' in curr_name and 'eye' in curr_name) or
                                        (region_name == 'nose_tip' and 'nose' in curr_name)):
                                        # Get bounds
                                        bounds = curr_info['bounds']
                                        # Replace region
                                        magnified_motion_frames[frame_idx][bounds[1]:bounds[3], 
                                                                         bounds[0]:bounds[2]] = magnified
                                        break
            
            # Average the combined phase changes
            if combined_phase_changes is not None and valid_regions_count > 0:
                combined_phase_changes /= valid_regions_count
                # Store the combined data
                phase_changes['combined'] = combined_phase_changes
        
        # Process color magnification
        print("Applying color magnification for heart rate detection...")
        magnified_color_frames = processor.process_color_magnification(all_frames, fps=fps)
        
        # Calculate heart rate data from color changes in facial regions
        # First detect cheeks in the initial frame for heart rate detection
        try:
            cheek_detector = CheekDetector()
            detected, cheek_regions = cheek_detector.detect_cheeks(all_frames[0])
            
            if detected and cheek_regions:
                # Extract cheek regions from all frames
                left_cheek_frames = []
                right_cheek_frames = []
                
                for frame in all_frames:
                    detected, regions = cheek_detector.detect_cheeks(frame)
                    if detected and regions:
                        if 'left_cheek' in regions[0]['regions']:
                            left_cheek_frames.append(regions[0]['regions']['left_cheek']['image'])
                        if 'right_cheek' in regions[0]['regions']:
                            right_cheek_frames.append(regions[0]['regions']['right_cheek']['image'])
                
                # Calculate color changes for each cheek
                color_signals = {}
                if left_cheek_frames:
                    color_signals['left_cheek'] = processor.calculate_color_changes(left_cheek_frames)
                if right_cheek_frames:
                    color_signals['right_cheek'] = processor.calculate_color_changes(right_cheek_frames)
                
                # Calculate heart rate data
                if color_signals:
                    heart_rate_data = processor.calculate_bpm(color_signals, fps=fps)
                    print(f"Extracted heart rate data with {len(heart_rate_data[1])} points")
        except Exception as e:
            print(f"Warning: Failed to extract heart rate data: {str(e)}")
            heart_rate_data = None
        
        # Fill in any missing frames
        while len(magnified_motion_frames) < len(all_frames):
            magnified_motion_frames.append(all_frames[len(magnified_motion_frames)].copy())
        
        while len(magnified_color_frames) < len(all_frames):
            magnified_color_frames.append(all_frames[len(magnified_color_frames)].copy())
        
        # Step 2: Fit the deception detector
        print("Step 2: Fitting deception detection model...")
        self.detector.fit(phase_changes, heart_rate_data, fps, total_frames)
        
        # Step 3: Create output video with annotations
        print("Step 3: Creating annotated output video...")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Calculate dimensions for the output video with better proportions
        # Make sure all dimensions are even numbers (required by video codecs)
        output_width = width * 3  # Three videos side by side
        # Ensure output_width is even
        output_width = output_width if output_width % 2 == 0 else output_width - 1
        
        # Calculate plot dimensions using 16:9 aspect ratio
        plot_width = output_width // 2  # Two plots side by side
        plot_height = (plot_width * 9) // 16  # 16:9 aspect ratio
        
        # Ensure plot_height is even
        plot_height = plot_height if plot_height % 2 == 0 else plot_height - 1
        
        # Calculate info panel height (70% of plot height)
        info_height = (plot_height * 7) // 10
        info_height = info_height if info_height % 2 == 0 else info_height - 1
        
        # Calculate total output height
        output_height = height + plot_height + info_height
        # Ensure output_height is even
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
                phase_title = "Combined Micro-Expressions"
            else:
                # Create empty phase data if none available
                phase_data = np.zeros(total_frames)
                phase_title = "No Micro-Expression Data"
            
            # Create phase plot using the dimensions calculated earlier
            phase_plot = processor.create_single_plot(
                phase_data, frame_idx, plot_width, plot_height,
                plot_type="diff", title=phase_title, total_frames=total_frames
            )
            
            # Highlight deception regions on the phase plot
            phase_plot = self.detector.highlight_deception_regions(phase_plot, frame_idx)
            
            # Create heart rate plot
            if heart_rate_data is not None:
                heart_plot = processor.create_heart_rate_plot(
                    heart_rate_data, frame_idx, plot_width, plot_height
                )
            else:
                # Create empty heart rate plot if data is not available
                heart_plot = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
                cv2.putText(heart_plot, "No Heart Rate Data Available", 
                          (plot_width//10, plot_height//2), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Highlight deception regions on the heart rate plot
            heart_plot = self.detector.highlight_deception_regions(heart_plot, frame_idx)
            
            # Create deception info panel
            info_panel = self.create_deception_info_panel(
                frame_idx, plot_width*2, info_height
            )
            
            # Stitch everything together with corrected proportions
            output_frame = self.stitch_frames_with_plots(
                original_frame, motion_frame, color_frame,
                phase_plot, heart_plot, info_panel
            )
            
            # Write frame to output
            out.write(output_frame)
        
        # Release output video
        out.release()
        
        # Clean up temporary files if any
        
        # Print completion message
        elapsed_time = time.time() - start_time
        print(f"Processed {os.path.basename(input_path)} in {elapsed_time:.1f} seconds")
        print(f"Output saved to {output_path}")
        print(f"===== Completed processing {os.path.basename(input_path)} =====\n\n")
    
    def process_folder(self, input_folder=None, output_folder=None):
        """
        Process all videos in a folder for deception detection.
        
        Args:
            input_folder: Path to the folder containing input videos
            output_folder: Path to save the output videos
        """
        # Use provided paths or default from config
        input_folder = input_folder or config.INPUT_VIDEOS_FOLDER
        output_folder = output_folder or config.OUTPUT_VIDEOS_FOLDER
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all MP4 files in the input directory
        input_files = sorted(glob.glob(os.path.join(input_folder, "*.mp4")))
        
        if not input_files:
            print(f"No MP4 files found in {input_folder}")
            return
        
        print(f"Found {len(input_files)} videos to process")
        
        # Process each video
        for i, input_file in enumerate(input_files):
            # Extract the base filename without extension
            basename = os.path.basename(input_file)
            file_name = os.path.splitext(basename)[0]
            
            # Create output filename
            output_file = os.path.join(output_folder, f"deception_detection_{file_name}.mp4")
            
            print(f"\nProcessing video {i+1}/{len(input_files)}: {basename}")
            self.process_video(input_file, output_file)
        
        print(f"\nAll {len(input_files)} videos have been processed for deception detection.")


def main():
    """Run the automatic lie detector on a folder of videos."""
    print("===== Starting Automatic Lie Detector =====")
    print(f"Input folder: {config.INPUT_VIDEOS_FOLDER}")
    print(f"Output folder: {config.OUTPUT_VIDEOS_FOLDER}")
    
    # Create and run the detector
    detector = AutomaticLieDetector(
        motion_params=config.MOTION_MAG_PARAMS,
        color_params=config.COLOR_MAG_PARAMS
    )
    
    # Process all videos in the input folder
    detector.process_folder()
    
    print("===== Automatic Lie Detector Complete =====")


if __name__ == "__main__":
    main() 