import cv2
import numpy as np
import os
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from io import BytesIO
from PIL import Image
from scipy import signal
import matplotlib

# Set up matplotlib for non-interactive use
matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 50  # Increase from 10 to 50
matplotlib.rcParams['figure.figsize'] = [10, 6]
matplotlib.rcParams['figure.dpi'] = 100

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Face_Motion_Magnification.face_region_motion_magnification import FacialPhaseMagnification
# Import ColorMagnification directly from Face_Color_Magnification
from Face_Color_Magnification.face_region_color_magnification import ColorMagnification
import mediapipe as mp
from copy import deepcopy
from config import *
from upper_face_detector import UpperFaceDetector

class SideBySideMagnification:
    def __init__(self, motion_params=None, color_params=None):
        """Initialize both motion and color magnification processors"""
        # Use provided parameters or default from config
        motion_params = motion_params or MOTION_MAG_PARAMS
        color_params = color_params or COLOR_MAG_PARAMS
        
        # Initialize motion processor with parameters
        self.motion_processor = FacialPhaseMagnification()
        # Set motion parameters
        self.motion_processor.phase_mag = motion_params['phase_mag']
        self.motion_processor.f_lo = motion_params['f_lo']
        self.motion_processor.f_hi = motion_params['f_hi']
        self.motion_processor.sigma = motion_params['sigma']
        self.motion_processor.attenuate = motion_params['attenuate']
        
        # Initialize ColorMagnification directly from Face_Color_Magnification
        self.color_processor = ColorMagnification(
            alpha=color_params['alpha'],
            level=color_params['level'],
            f_lo=color_params['f_lo'],
            f_hi=color_params['f_hi']
        )
        
        # Initialize upper face detector for heart rate detection
        self.upper_face_detector = UpperFaceDetector()
        
        # Initialize face detection for micro-expression tracking
        # Use MediaPipe Face Mesh for more accurate landmark tracking
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize standard face detection as backup
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        
        # Landmark indices for micro-expression regions
        # Based on MediaPipe Face Mesh landmarks:
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
        
        # Landmark indices for key regions
        self.EYEBROW_LANDMARKS = {
            'left_inner': [336, 296, 334, 293, 276],  # Left inner eyebrow
            'left_outer': [297, 338, 337, 284, 251],  # Left outer eyebrow
            'right_inner': [66, 105, 63, 70, 53],     # Right inner eyebrow
            'right_outer': [107, 55, 65, 52, 46]      # Right outer eyebrow
        }
        
        self.EYELID_LANDMARKS = {
            'left_upper': [159, 145, 144, 163, 160],  # Left upper eyelid
            'right_upper': [386, 374, 373, 390, 388]  # Right upper eyelid
        }
        
        self.EYE_CORNER_LANDMARKS = {
            'left_outer': [33, 133, 173, 157, 158],   # Left outer eye corner
            'right_outer': [362, 263, 398, 384, 385]  # Right outer eye corner
        }
        
        # Add explicit nose landmark indices
        self.NOSE_LANDMARKS = {
            'nose_tip': [1, 2, 3, 4, 5, 6],           # Nose tip
            'nose_bridge': [168, 195, 197, 4, 19, 94], # Nose bridge
            'nose_bottom': [2, 326, 328, 330, 331]    # Bottom of nose
        }
        
        self.NASOLABIAL_LANDMARKS = {
            'left': [129, 209, 226, 207, 208],        # Left nasolabial fold area
            'right': [358, 429, 442, 427, 428]        # Right nasolabial fold area
        }
        
        self.FOREHEAD_LANDMARKS = {
            'center': [10, 151, 8, 9, 107],          # Center forehead area
            'left': [109, 67, 104, 54, 68],           # Left forehead area
            'right': [338, 297, 332, 284, 298]        # Right forehead area
        }
        
        # Define region dimensions with scaling factor
        self.region_size_factor = FACIAL_REGIONS.get('region_size_factor', 1.0)
        
        # Define base region dimensions (width, height) in pixels
        self.BASE_REGION_DIMENSIONS = {
            'eyebrow': (100, 50),      # Eyebrow region
            'eyelid': (80, 40),        # Eyelid region
            'eye_corner': (60, 60),    # Eye corner region
            'nasolabial': (80, 80),    # Nasolabial fold region
            'forehead': (120, 60)      # Forehead region
        }
        
        # Apply region size factor
        self.REGION_DIMENSIONS = {
            region: (int(dim[0] * self.region_size_factor), 
                    int(dim[1] * self.region_size_factor))
            for region, dim in self.BASE_REGION_DIMENSIONS.items()
        }
    
    def add_label(self, frame: np.ndarray, text: str) -> np.ndarray:
        """Add a text label to the frame"""
        frame_with_label = frame.copy()
        h, w = frame_with_label.shape[:2]
        
        # Create a semi-transparent overlay for the label background
        overlay = frame_with_label.copy()
        cv2.rectangle(overlay, (0, h-LABEL_HEIGHT), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, LABEL_ALPHA, frame_with_label, 1-LABEL_ALPHA, 0, frame_with_label)
        
        # Add text - centering calculation improved to prevent cut-off
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS)[0]
        text_x = max(20, int(w/2 - text_size[0]/2))  # Ensure minimum margin of 20px
        
        cv2.putText(
            frame_with_label, 
            text, 
            (text_x, h-15), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            LABEL_FONT_SCALE, 
            (255, 255, 255), 
            LABEL_THICKNESS, 
            cv2.LINE_AA
        )
        
        return frame_with_label
    
    def stitch_frames(self, original: np.ndarray, motion: np.ndarray, color: np.ndarray) -> np.ndarray:
        """Stitch three frames side by side with proper aspect ratio and labels"""
        # Make deep copies to avoid modifying the originals
        original_copy = original.copy()
        motion_copy = motion.copy()
        color_copy = color.copy()
        
        # Get dimensions
        h, w = original.shape[:2]
        
        # Add labels
        original_labeled = self.add_label(original_copy, ORIGINAL_LABEL)
        motion_labeled = self.add_label(motion_copy, MOTION_LABEL)
        color_labeled = self.add_label(color_copy, COLOR_LABEL)
        
        # Stitch frames horizontally
        stitched = np.hstack((original_labeled, motion_labeled, color_labeled))
        
        return stitched
    
    def get_region_center(self, landmarks: List[Tuple[float, float]], indices: List[int]) -> Tuple[int, int]:
        """Calculate the center point of a region based on landmark points"""
        x_coords = [landmarks[idx][0] for idx in indices]
        y_coords = [landmarks[idx][1] for idx in indices]
        
        center_x = int(sum(x_coords) / len(x_coords))
        center_y = int(sum(y_coords) / len(y_coords))
        
        return center_x, center_y
    
    def extract_rectangle_region(self, frame: np.ndarray, 
                               center: Tuple[int, int], 
                               dimensions: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extract a rectangular region around a center point with specified dimensions"""
        height, width = frame.shape[:2]
        rect_width, rect_height = dimensions
        
        # Calculate region bounds
        half_width = rect_width // 2
        half_height = rect_height // 2
        
        x_min = max(0, center[0] - half_width)
        x_max = min(width, center[0] + half_width)
        y_min = max(0, center[1] - half_height)
        y_max = min(height, center[1] + half_height)
        
        # Extract region
        region = frame[y_min:y_max, x_min:x_max]
        
        return region, (x_min, y_min, x_max, y_max)
    
    def create_single_plot(self, phase_data, frame_idx, plot_width, plot_height, plot_type="raw", title=None, total_frames=None, x_min=None, x_max=None):
        """Creates a single square plot of either raw phase data or frame-to-frame changes
        
        Args:
            phase_data: Array of phase change values
            frame_idx: Current frame index
            plot_width: Width of the plot in pixels
            plot_height: Height of the plot in pixels
            plot_type: "raw" or "diff" for raw phase changes or frame-to-frame differences
            title: Title to display on the plot
            total_frames: Total number of frames in the video
            x_min: Optional global minimum x-axis value for consistent scaling (in frames)
            x_max: Optional global maximum x-axis value for consistent scaling (in frames)
            
        Returns:
            Square plot image as numpy array
        """
        try:
            # Make sure phase_data is a valid numpy array of floats
            phase_data = np.array(phase_data, dtype=np.float64)
            if len(phase_data) == 0:
                # Handle empty data
                blank = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
                cv2.putText(blank, f"No data for {title}", 
                          (20, plot_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return blank
            
            # Create a figure with a wider aspect ratio for better visibility of the full timeline
            # Increase width to ensure entire timeline is visible
            fig = plt.figure(figsize=(12, 5), dpi=100, facecolor='white')
            # Add padding around the plot for better appearance
            ax = fig.add_subplot(1, 1, 1)
            plt.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.15)
            
            # Get array length
            max_len = len(phase_data)
            
            # If total_frames is not provided, use phase data length
            if total_frames is None or total_frames <= 0:
                total_frames = max_len
            
            # Scale frame_idx to match the phase data length
            # This ensures the green dot moves correctly across the entire graph
            # regardless of video length
            if max_len > 1:
                # Use linear mapping for the scaled frame index
                scaled_frame_idx = min(int((frame_idx / (total_frames - 1)) * (max_len - 1)), max_len - 1) if total_frames > 1 else 0
            else:
                scaled_frame_idx = 0
            
            # Prepare data for plotting
            if plot_type == "diff" and len(phase_data) > 1:
                # Calculate derivatives for frame-to-frame changes
                y_original = np.abs(np.diff(phase_data))
                y_original = np.insert(y_original, 0, 0)  # Add zero at the beginning
                color = 'green'
                title_prefix = "Phase Change Rate"
            else:
                # Raw phase data
                y_original = phase_data
                color = 'blue'
                title_prefix = "Phase Magnitude"
            
            # Time binning to aggregate movements over intervals based on detected fps
            fps = 30  # Default FPS
            if total_frames is not None and 'self' in locals() and hasattr(self, 'fps'):
                fps = self.fps  # Use detected fps if available
            
            # Calculate bin size in frames to represent 0.5 second of data (reduced from 1.0 second)
            bin_size_frames = int(fps * 0.5)  # 0.5-second bins in frames
            
            # Ensure we have at least one bin even for short videos
            if bin_size_frames < 1:
                bin_size_frames = 1
            
            # Set the x-axis limits early to ensure we're using the full range
            if x_min is not None and x_max is not None:
                full_x_min = x_min
                full_x_max = x_max
            else:
                full_x_min = 0
                full_x_max = total_frames
            
            # Calculate number of bins to cover the ENTIRE frame range
            # This ensures the bins extend across the full video duration
            num_bins = max(1, int(np.ceil((full_x_max - full_x_min) / bin_size_frames)))
            
            # Initialize arrays for aggregated data
            binned_frames = []
            binned_values = []
            binned_max_values = []
            
            # Group data into bins by frame numbers, covering the FULL video range
            for bin_idx in range(num_bins):
                # Calculate start and end frame indices for this bin
                start_frame = full_x_min + (bin_idx * bin_size_frames)
                end_frame = min(full_x_max, start_frame + bin_size_frames)
                
                # Map these frame indices to phase_data indices
                if max_len > 0 and total_frames > 0:
                    # Scale from video frame range to phase_data index range
                    start_idx = min(max_len - 1, int((start_frame / total_frames) * max_len))
                    end_idx = min(max_len, int((end_frame / total_frames) * max_len))
                else:
                    start_idx = 0
                    end_idx = 0
                
                # Calculate bin center frame for x-axis (in video frame units)
                bin_center_frame = start_frame + (end_frame - start_frame) // 2
                binned_frames.append(bin_center_frame)
                
                # Get data for this bin if we have valid indices
                if start_idx < max_len and start_idx < end_idx:
                    bin_data = y_original[start_idx:end_idx]
                    
                    # For movement intensity (diff plot), use different aggregation strategies
                    if plot_type == "diff":
                        # Use 90th percentile instead of mean to focus on larger movements
                        if len(bin_data) > 0:
                            # Calculate percentile value that emphasizes larger movements
                            bin_value = np.percentile(bin_data, 90) if len(bin_data) >= 10 else np.max(bin_data)
                            # Also track maximum for highlighting significant movements
                            bin_max = np.max(bin_data)
                        else:
                            bin_value = 0
                            bin_max = 0
                    else:
                        # For raw data, use mean with some weighting toward extremes
                        if len(bin_data) > 0:
                            # For raw plots, calculate weighted mean that emphasizes extremes
                            weights = np.abs(bin_data - np.mean(bin_data)) + 0.1  # Add small constant to avoid zeros
                            bin_value = np.average(bin_data, weights=weights)
                            bin_max = np.max(bin_data)
                        else:
                            bin_value = 0
                            bin_max = 0
                else:
                    # No data for this bin, use the last known value or 0
                    bin_value = binned_values[-1] if binned_values else 0
                    bin_max = binned_max_values[-1] if binned_max_values else 0
                
                binned_values.append(bin_value)
                binned_max_values.append(bin_max)
            
            # Convert to numpy arrays
            binned_frames = np.array(binned_frames)
            binned_values = np.array(binned_values)
            binned_max_values = np.array(binned_max_values)
            
            # Find the bin containing the current frame for position marker
            if frame_idx >= 0 and frame_idx <= full_x_max:
                # Find closest bin to current frame
                current_bin_idx = np.argmin(np.abs(binned_frames - frame_idx))
            else:
                current_bin_idx = 0
            
            # Determine threshold for "significant" movements (focusing on upper portion)
            # Calculate the threshold as a percentage of the way from min to max
            if len(binned_values) > 0 and np.max(binned_values) > 0:
                # Focus on upper portion by cutting off the bottom part of movements
                # Determine the focus threshold as a percentage of the max value
                if plot_type == "diff":
                    # For diff plots, focus on significant movements by setting a higher threshold
                    focus_threshold = np.percentile(binned_values, 25)  # Bottom 25% considered baseline noise
                else:
                    # For raw plots, use a lower threshold to show more data
                    focus_threshold = np.percentile(binned_values, 10)  # Bottom 10% cut off
                
                # Ensure the threshold isn't too high in case of very skewed data
                if focus_threshold > np.max(binned_values) * 0.5:
                    focus_threshold = np.max(binned_values) * 0.25
            else:
                focus_threshold = 0
            
            # Plot the aggregated data as a line graph instead of bar plot
            # Main line connecting all points
            ax.plot(binned_frames, binned_values, '-', color=color, linewidth=2.5, 
                   marker='o', markersize=6, markerfacecolor=color, markeredgecolor='black',
                   markeredgewidth=1, alpha=0.8, zorder=5)
            
            # Add a current position indicator
            if current_bin_idx >= 0 and current_bin_idx < len(binned_frames):
                # Mark the current frame bin with a vertical line
                ax.axvline(x=binned_frames[current_bin_idx], color='gray', linestyle='--', alpha=0.7)
                
                # Add a bright green dot on the current point
                ax.plot(binned_frames[current_bin_idx], binned_values[current_bin_idx], 'o', 
                       color='lime', markersize=12, markeredgecolor='black', markeredgewidth=2, zorder=10)
            
            # Highlight significant movement periods (those with higher values)
            # Find bins with values above the 75th percentile
            if len(binned_values) > 3:  # Need enough data for percentile calculation
                significant_threshold = np.percentile(binned_values, 75)
                significant_bins = np.where(binned_values >= significant_threshold)[0]
                
                # Highlight these bins with a different color
                if len(significant_bins) > 0:
                    # Plot significant points in red with larger markers
                    significant_frames = binned_frames[significant_bins]
                    significant_values = binned_values[significant_bins]
                    ax.plot(significant_frames, significant_values, 'o', 
                           color='tomato', markersize=10, markeredgecolor='darkred', 
                           markeredgewidth=1.5, alpha=0.9, zorder=7)
                    
                    # Connect significant points with line segments
                    # Only if they are adjacent bins
                    for i in range(len(significant_bins)-1):
                        if significant_bins[i+1] - significant_bins[i] == 1:
                            ax.plot([binned_frames[significant_bins[i]], binned_frames[significant_bins[i+1]]],
                                  [binned_values[significant_bins[i]], binned_values[significant_bins[i+1]]],
                                  '-', color='tomato', linewidth=3.5, alpha=0.7, zorder=6)
            
            # Set title with increased font size
            if title:
                ax.set_title(f"{title}: {title_prefix}", fontsize=18, fontweight='bold')
            else:
                ax.set_title(title_prefix, fontsize=18, fontweight='bold')
                
            # Set labels and grid with increased font size
            ax.set_xlabel("Frame Number", fontsize=12)
            ax.set_ylabel("Magnitude", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Increase tick font size
            ax.tick_params(axis='both', which='major', labelsize=11)
            
            # ALWAYS set the x-axis to span the full frame range
            ax.set_xlim(full_x_min, full_x_max)
            
            # Set tick positions at intervals appropriate for frame numbers
            x_range = full_x_max - full_x_min
            if x_range > 1000:
                tick_spacing = 200
            elif x_range > 500:
                tick_spacing = 100
            elif x_range > 200:
                tick_spacing = 50
            else:
                tick_spacing = 20
            
            # Set x-axis ticks at regular intervals
            tick_positions = np.arange(
                int(full_x_min / tick_spacing) * tick_spacing,  # Round to nearest tick_spacing
                full_x_max + tick_spacing,
                tick_spacing
            )
            ax.set_xticks(tick_positions)
            
            # Set y-axis limits - focus on upper portion by cutting off the bottom
            if len(binned_values) > 0:
                # Calculate more focused y-axis limits based on average to maximum range
                avg_value = np.mean(binned_values)
                max_value = np.max(binned_values)
                min_value = np.min(binned_values)
                padding = 0.05  # Add small padding above max value
                
                # For frame-to-frame plots, show more of the data range vertically
                if plot_type == "diff":
                    # Show more of the range - start from below the minimum value
                    # This ensures more of the vertical range is visible
                    y_min = min_value * 0.9  # 10% below minimum value
                    
                    # Set upper limit to max value plus more padding for diff plots
                    y_max = max_value + (max_value - min_value) * 0.2  # 20% additional range above max
                else:
                    # For raw plots, still focus on the average to max range
                    # Set lower limit to average value, but no less than the focus threshold
                    y_min = max(focus_threshold, avg_value) * 0.95  # Slightly below avg for context
                    
                    # Set upper limit to max value plus padding
                    y_max = max_value + padding
                
                # Ensure there's always some range to display
                if y_max - y_min < 0.1:
                    y_max = y_min + 0.1
                
                ax.set_ylim(y_min, y_max)
                
                # Add a subtle horizontal line at the average for reference
                ax.axhline(y=avg_value, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)
                # Add text annotation for average value
                ax.text(full_x_min + (full_x_max - full_x_min) * 0.02, avg_value, f"avg: {avg_value:.3f}", 
                      fontsize=9, color='gray', verticalalignment='bottom',
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                
                # Also show min value for frame-to-frame plots
                if plot_type == "diff":
                    ax.text(full_x_min + (full_x_max - full_x_min) * 0.02, min_value, f"min: {min_value:.3f}", 
                          fontsize=9, color='gray', verticalalignment='top',
                          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                
                # Also keep the focus threshold line
                if focus_threshold < avg_value:
                    ax.axhline(y=focus_threshold, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)
            
            # Add a legend to explain the plot elements
            legend_elements = [
                Line2D([0], [0], color=color, marker='o', markersize=6, linewidth=2.5, 
                     markerfacecolor=color, markeredgecolor='black',
                     label='Movement Magnitude'),
                Line2D([0], [0], color='tomato', marker='o', markersize=8, linewidth=3.0, 
                     markerfacecolor='tomato', markeredgecolor='darkred',
                     label='Significant Movement')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            # Add informative text about the current frame
            if current_bin_idx >= 0 and current_bin_idx < len(binned_frames):
                current_frame = binned_frames[current_bin_idx]
                frame_info = f"Current: Frame {frame_idx+1}/{total_frames}"
                fig.text(0.02, 0.02, frame_info, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
            
            # Adjust layout 
            plt.tight_layout()
            
            # Save the figure to a temporary buffer
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # Use PIL/Pillow to open the image from the buffer
            img_pil = Image.open(buf)
            # Convert to RGB explicitly to remove alpha channel
            if img_pil.mode == 'RGBA':
                img_pil = img_pil.convert('RGB')
            plot_img = np.array(img_pil)
            
            # Close buffer
            buf.close()
            
            # Close figure to free memory
            plt.close(fig)
            
            # Create a padded image with white background
            padding = max(int(plot_width * 0.02), 5)  # Reduce padding to 2% or minimum 5px
            padded_img = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
            
            # Resize to fit in the padded area while maintaining aspect ratio
            h, w = plot_img.shape[:2]
            target_w = plot_width - (2 * padding)
            target_h = plot_height - (2 * padding)
            
            # Calculate the scaling factor to fit the image within the padded area
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize the plot image
            resized_plot = cv2.resize(plot_img, (new_w, new_h))
            
            # Calculate position to center the image
            x_offset = (plot_width - new_w) // 2
            y_offset = (plot_height - new_h) // 2
            
            # Place the resized image in the padded frame
            padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_plot
            
            # Convert from RGB to BGR for OpenCV if needed
            if len(padded_img.shape) == 3 and padded_img.shape[2] == 3:
                padded_img = cv2.cvtColor(padded_img, cv2.COLOR_RGB2BGR)
            
            # Ensure the output is uint8
            if padded_img.dtype != np.uint8:
                padded_img = padded_img.astype(np.uint8)
            
            return padded_img
            
        except Exception as e:
            # If anything fails, return a blank frame with error message
            print(f"Error creating plot for frame {frame_idx}: {str(e)}")
            blank = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
            cv2.putText(blank, f"Error: {str(e)[:50]}", 
                      (20, plot_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return blank
    
    def create_region_plot_pair(self, phase_changes, frame_idx, region_name, plot_width, plot_height):
        """Creates a pair of plots (raw and diff) for a single facial region
        
        Args:
            phase_changes: Dictionary of phase changes for all regions
            frame_idx: Current frame index
            region_name: Specific region key to plot
            plot_width: Width of each individual plot
            plot_height: Height of each individual plot
            
        Returns:
            Side-by-side raw and diff plots for the region
        """
        try:
            # Check if we have data for this region
            if not phase_changes or region_name not in phase_changes:
                # Return empty black frame if no phase changes for this region
                return np.zeros((plot_height, plot_width * 2, 3), dtype=np.uint8)
            
            # Get the data for this region
            values = phase_changes.get(region_name, [])
            if len(values) == 0:
                return np.zeros((plot_height, plot_width * 2, 3), dtype=np.uint8)
            
            # Format the region name for display with better human-readable names
            if "left_eye" in region_name:
                display_name = "Left Eye"
            elif "right_eye" in region_name:
                display_name = "Right Eye"
            elif "nose_tip" in region_name or "nose" in region_name:
                display_name = "Nose"
            else:
                # Default formatting for any other regions
                display_name = region_name.replace('face1_', '').replace('_', ' ').title()
            
            # Generate raw phase plot
            raw_plot = self.create_single_plot(values, frame_idx, plot_width, plot_height, 
                                           "raw", display_name)
            
            # Generate frame-to-frame diff plot
            diff_plot = self.create_single_plot(values, frame_idx, plot_width, plot_height, 
                                            "diff", display_name)
            
            # Combine side by side
            combined = np.hstack([raw_plot, diff_plot])
            
            return combined
            
        except Exception as e:
            # If anything fails, return a blank frame with error message
            print(f"Error creating plot pair for {region_name} at index {frame_idx}: {str(e)}")
            blank = np.ones((plot_height, plot_width * 2, 3), dtype=np.uint8) * 255
            cv2.putText(blank, f"Error: {str(e)[:50]}", 
                      (20, plot_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return blank
    
    def calculate_color_changes(self, frames):
        """Calculate color changes across frames for Eulerian color magnification
        
        Args:
            frames: List of region frames
            
        Returns:
            Array of color changes across frames
        """
        try:
            if not frames or len(frames) < 2:
                return np.zeros(len(frames) if frames else 0)
                
            # Extract the green channel which is most sensitive to blood flow
            green_vals = []
            for frame in frames:
                # Extract green channel
                green = frame[:, :, 1].astype(np.float32)
                # Calculate mean green value
                green_vals.append(np.mean(green))
                
            # Convert to numpy array
            signal_data = np.array(green_vals)
            
            # Normalize signal
            signal_data = signal_data - np.mean(signal_data)
            if np.std(signal_data) > 0:
                signal_data = signal_data / np.std(signal_data)
                
            # Apply bandpass filter to focus on heart rate frequencies (0.7-4Hz)
            # Assuming 30fps video
            fps = 30
            low_cutoff = 0.7 / (fps/2)  # ~42 BPM
            high_cutoff = 4.0 / (fps/2)  # ~240 BPM
            
            # Create butterworth bandpass filter
            b, a = signal.butter(2, [low_cutoff, high_cutoff], btype='band')
            filtered_signal = signal.filtfilt(b, a, signal_data)
            
            return filtered_signal
            
        except Exception as e:
            print(f"Error in calculate_color_changes: {str(e)}")
            # Return zeros with the same length as input
            return np.zeros(len(frames) if frames else 0)
    
    def calculate_bpm(self, color_signals, fps=30):
        """Calculate heart rate (BPM) from color signals
        
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
            
            # Calculate instantaneous heart rate using peaks
            peaks, _ = signal.find_peaks(combined_signal, distance=7)  # Minimum distance between peaks
            
            # Calculate instantaneous BPM for each frame
            inst_bpm = np.zeros(len(combined_signal))
            
            if len(peaks) >= 2:
                # Calculate for each peak
                for i in range(len(peaks)-1):
                    # Calculate time between peaks in seconds
                    time_between_peaks = (peaks[i+1] - peaks[i]) / fps
                    # Convert to BPM
                    if time_between_peaks > 0:
                        curr_bpm = 60 / time_between_peaks
                        # Only assign valid BPM values (between 40-240)
                        if 40 <= curr_bpm <= 240:
                            # Assign to all frames between these peaks
                            start_idx = peaks[i]
                            end_idx = peaks[i+1]
                            inst_bpm[start_idx:end_idx] = curr_bpm
                
                # Fill in any remaining frames at the end
                if peaks[-1] < len(inst_bpm):
                    inst_bpm[peaks[-1]:] = inst_bpm[peaks[-1]-1] if peaks[-1] > 0 else 0
                    
                # Fill in any blank frames at the beginning
                if peaks[0] > 0 and len(peaks) > 1 and peaks[1] < len(inst_bpm) and inst_bpm[peaks[1]] > 0:
                    inst_bpm[:peaks[0]] = inst_bpm[peaks[1]]
            
            # Calculate windowed average BPM (smoother signal)
            avg_bpm = np.zeros_like(inst_bpm)
            window_len = int(fps * 3)  # 3-second window
            
            for i in range(len(inst_bpm)):
                start_idx = max(0, i - window_len//2)
                end_idx = min(len(inst_bpm), i + window_len//2)
                values = inst_bpm[start_idx:end_idx]
                values = values[values > 0]  # Only consider non-zero values
                if len(values) > 0:
                    avg_bpm[i] = np.mean(values)
                
            return inst_bpm, avg_bpm, combined_signal
            
        except Exception as e:
            print(f"Error in calculate_bpm: {str(e)}")
            return None
    
    def create_heart_rate_plot(self, bpm_data, frame_idx, plot_width, plot_height, x_min=None, x_max=None):
        """Creates a plot showing heart rate (BPM)"""
        try:
            # Create figure with improved aspect ratio - wider for better horizontal stretching
            # Use a fixed figure size that scales well regardless of data length
            fig = plt.figure(figsize=(10, 6), dpi=100, facecolor='white')
            ax = fig.add_subplot(1, 1, 1)
            plt.subplots_adjust(left=0.12, right=0.88, top=0.85, bottom=0.15)
            
            # Extract data
            inst_bpm, avg_bpm, combined_signal = bpm_data
            
            # Get max length
            max_len = len(inst_bpm)
            
            # Time points
            x = np.arange(max_len)
            
            # Plot BPM data
            ax.plot(x, avg_bpm, '-', color='red', linewidth=2.5, label='Average BPM')
            ax.plot(x, inst_bpm, '-', color='gray', alpha=0.6, linewidth=1, label='Instantaneous BPM')
            
            # Add current frame indicator
            if frame_idx > 0 and frame_idx < max_len:
                # Scale frame_idx if necessary to fit within the data length
                scaled_frame_idx = min(frame_idx, max_len-1)
                
                ax.axvline(x=scaled_frame_idx, color='green', linestyle='--', alpha=0.7)
                # Add a marker for current BPM
                current_bpm = avg_bpm[scaled_frame_idx]
                if current_bpm > 0:
                    ax.plot(scaled_frame_idx, current_bpm, 'o', color='lime', markersize=12, 
                          markeredgecolor='black', markeredgewidth=2, zorder=10)
                    ax.text(scaled_frame_idx+5, current_bpm, f"{current_bpm:.1f} BPM", 
                          fontsize=12, color='black', fontweight='bold',
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
            
            # Set BPM plot properties with increased font sizes
            ax.set_title("Heart Rate Estimation", fontsize=20, fontweight='bold')
            ax.set_ylabel("BPM", fontsize=14)
            ax.set_xlabel("Frame Number", fontsize=14)
            
            # Use consistent x-axis limits if provided
            # For heart rate, we might need to scale the global limits to match the data length
            if x_min is not None and x_max is not None:
                if max_len > 0:
                    # Scale the global limits to match the heart rate data range
                    # (since heart rate data might not have same length as phase data)
                    scaled_x_min = 0
                    scaled_x_max = max_len
                    ax.set_xlim(scaled_x_min, scaled_x_max)
                else:
                    ax.set_xlim(x_min, x_max)
            else:
                # Always use consistent x-axis limits regardless of data length
                ax.set_xlim(0, max_len-1)
                # Always use consistent x-axis limits regardless of data length
                ax.set_xlim(0, max_len-1)
            
            # Set tick positions at intervals appropriate for frame numbers
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            if x_range > 1000:
                tick_spacing = 200
            elif x_range > 500:
                tick_spacing = 100
            elif x_range > 200:
                tick_spacing = 50
            else:
                tick_spacing = 20
                
            # Set x-axis ticks at regular intervals
            tick_positions = np.arange(
                int(ax.get_xlim()[0] / tick_spacing) * tick_spacing,  # Round to nearest tick_spacing
                ax.get_xlim()[1] + tick_spacing,
                tick_spacing
            )
            ax.set_xticks(tick_positions)
            
            # Use fixed y-axis range for consistent BPM display (40-140 BPM is a normal human range)
            # This ensures the scale doesn't change between videos
            max_bpm = max(140, np.max(avg_bpm) * 1.1 if len(avg_bpm) > 0 and np.max(avg_bpm) > 0 else 140)
            ax.set_ylim(40, max_bpm)
            
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper right', fontsize=12)
            
            plt.tight_layout()
            
            # Save to image
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # Convert to numpy array
            img_pil = Image.open(buf)
            # Convert to RGB explicitly to remove alpha channel
            if img_pil.mode == 'RGBA':
                img_pil = img_pil.convert('RGB')
            plot_img = np.array(img_pil)
            
            # Close buffer and figure
            buf.close()
            plt.close(fig)
            
            # Create a padded image with white background
            padding = max(int(plot_width * 0.02), 5)  # Reduce padding to 2% or minimum 5px
            padded_img = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
            
            # Resize to fit in the padded area while maintaining aspect ratio
            h, w = plot_img.shape[:2]
            target_w = plot_width - (2 * padding)
            target_h = plot_height - (2 * padding)
            
            # Calculate the scaling factor to fit the image within the padded area
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize the plot image
            resized_plot = cv2.resize(plot_img, (new_w, new_h))
            
            # Calculate position to center the image
            x_offset = (plot_width - new_w) // 2
            y_offset = (plot_height - new_h) // 2
            
            # Place the resized image in the padded frame
            padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_plot
            
            # Convert from RGB to BGR for OpenCV
            padded_img = cv2.cvtColor(padded_img, cv2.COLOR_RGB2BGR)
            
            return padded_img
            
        except Exception as e:
            print(f"Error creating heart rate plot for frame {frame_idx}: {str(e)}")
            blank = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
            cv2.putText(blank, f"Error: {str(e)[:50]}", 
                      (20, plot_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return blank
    
    def create_pulse_signal_plot(self, bpm_data, frame_idx, plot_width, plot_height, x_min=None, x_max=None):
        """Creates a plot showing the blood volume pulse signal"""
        try:
            # Create figure with improved aspect ratio - wider for better horizontal stretching
            # Use a fixed figure size that scales well regardless of data length
            fig = plt.figure(figsize=(10, 6), dpi=100, facecolor='white')
            ax = fig.add_subplot(1, 1, 1)
            plt.subplots_adjust(left=0.12, right=0.88, top=0.85, bottom=0.15)
            
            # Extract data
            _, _, combined_signal = bpm_data
            
            # Get max length
            max_len = len(combined_signal)
            
            # Time points
            x = np.arange(max_len)
            
            # Plot signal
            ax.plot(x, combined_signal, '-', color='green', linewidth=1.5)
            
            # Add current frame indicator with proper scaling
            if frame_idx > 0 and frame_idx < max_len:
                # Scale frame_idx if necessary to fit within the data length
                scaled_frame_idx = min(frame_idx, max_len-1)
                
                ax.axvline(x=scaled_frame_idx, color='green', linestyle='--', alpha=0.7)
                ax.plot(scaled_frame_idx, combined_signal[scaled_frame_idx], 'o', color='lime', markersize=10, 
                      markeredgecolor='black', markeredgewidth=1.5, zorder=10)
            
            # Set signal plot properties with increased font sizes
            ax.set_title("Blood Volume Pulse Signal", fontsize=20, fontweight='bold')
            ax.set_xlabel("Frame Number", fontsize=14)
            ax.set_ylabel("Amplitude", fontsize=14)
            
            # Use consistent x-axis limits if provided
            # For pulse signal, we might need to scale the global limits to match the data length
            if x_min is not None and x_max is not None:
                if max_len > 0:
                    # Scale the global limits to match the pulse data range
                    # (since pulse data might not have same length as phase data)
                    scaled_x_min = 0
                    scaled_x_max = max_len
                    ax.set_xlim(scaled_x_min, scaled_x_max)
                else:
                    # Always use consistent x-axis limits regardless of data length
                    ax.set_xlim(x_min, x_max)
            else:
                # Always use consistent x-axis limits regardless of data length
                ax.set_xlim(0, max_len-1)
                # Always use consistent x-axis limits regardless of data length
                ax.set_xlim(0, max_len-1)
                
            # Set tick positions at intervals appropriate for frame numbers
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            if x_range > 1000:
                tick_spacing = 200
            elif x_range > 500:
                tick_spacing = 100
            elif x_range > 200:
                tick_spacing = 50
            else:
                tick_spacing = 20
                
            # Set x-axis ticks at regular intervals
            tick_positions = np.arange(
                int(ax.get_xlim()[0] / tick_spacing) * tick_spacing,  # Round to nearest tick_spacing
                ax.get_xlim()[1] + tick_spacing,
                tick_spacing
            )
            ax.set_xticks(tick_positions)
            
            # Use consistent y-axis scale for pulse signal
            # Find reasonable y-limits based on the signal or use defaults
            if len(combined_signal) > 0:
                signal_std = np.std(combined_signal)
                if signal_std > 0:
                    y_max = max(1.0, np.max(combined_signal) * 1.2)
                    y_min = min(-1.0, np.min(combined_signal) * 1.2)
                    # Ensure symmetric to show the oscillation pattern clearly
                    max_abs = max(abs(y_min), abs(y_max))
                    ax.set_ylim(-max_abs, max_abs)
                else:
                    ax.set_ylim(-1, 1)  # Default if signal has no variation
            else:
                ax.set_ylim(-1, 1)  # Default if no signal
            
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Increase tick font size
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            plt.tight_layout()
            
            # Save to image
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # Convert to numpy array
            img_pil = Image.open(buf)
            # Convert to RGB explicitly to remove alpha channel
            if img_pil.mode == 'RGBA':
                img_pil = img_pil.convert('RGB')
            plot_img = np.array(img_pil)
            
            # Close buffer and figure
            buf.close()
            plt.close(fig)
            
            # Create a padded image with white background
            padding = max(int(plot_width * 0.02), 5)  # Reduce padding to 2% or minimum 5px
            padded_img = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
            
            # Resize to fit in the padded area while maintaining aspect ratio
            h, w = plot_img.shape[:2]
            target_w = plot_width - (2 * padding)
            target_h = plot_height - (2 * padding)
            
            # Calculate the scaling factor to fit the image within the padded area
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize the plot image
            resized_plot = cv2.resize(plot_img, (new_w, new_h))
            
            # Calculate position to center the image
            x_offset = (plot_width - new_w) // 2
            y_offset = (plot_height - new_h) // 2
            
            # Place the resized image in the padded frame
            padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_plot
            
            # Convert from RGB to BGR for OpenCV
            padded_img = cv2.cvtColor(padded_img, cv2.COLOR_RGB2BGR)
            
            return padded_img
            
        except Exception as e:
            print(f"Error creating pulse signal plot for frame {frame_idx}: {str(e)}")
            blank = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
            cv2.putText(blank, f"Error: {str(e)[:50]}", 
                      (20, plot_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return blank
    
    def compute_phase_change(self, current_frame, prev_frame):
        """Compute the phase change between two frames
        
        This method calculates motion changes between PBM-magnified frames. Since
        the frames were already magnified by the PBM process, we use a simple
        approach to quantify the magnified motion, with enhancements to better
        highlight significant movements.
        
        Args:
            current_frame: Current PBM-magnified frame 
            prev_frame: Previous PBM-magnified frame
            
        Returns:
            Float value representing phase change magnitude
        """
        try:
            # Convert to grayscale if not already
            if len(current_frame.shape) == 3:
                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            else:
                current_gray = current_frame
                
            if len(prev_frame.shape) == 3:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            else:
                prev_gray = prev_frame
            
            # Ensure same size
            if current_gray.shape != prev_gray.shape:
                current_gray = cv2.resize(current_gray, (prev_gray.shape[1], prev_gray.shape[0]))
            
            # Apply minimal Gaussian blur to reduce noise while preserving important movements
            current_gray = cv2.GaussianBlur(current_gray, (3, 3), 0.5)
            prev_gray = cv2.GaussianBlur(prev_gray, (3, 3), 0.5)
            
            # Calculate absolute difference between frames - this directly measures 
            # the magnified motion from the PBM-magnified frames
            diff = cv2.absdiff(current_gray, prev_gray)
            
            # Use a more non-linear approach to accentuate significant movements
            # This will make deceptive micro-expressions stand out better
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_diff = clahe.apply(diff)
            
            # Calculate statistics on the enhanced difference
            avg_diff = np.mean(enhanced_diff) 
            max_diff = np.max(enhanced_diff)
            
            # Calculate a weighted score that emphasizes larger differences
            # This gives more weight to potentially deceptive movements
            # (movements that are significantly larger than average)
            if max_diff > 0:
                # Calculate a weight factor: higher for frames with larger max differences
                weight_factor = np.log1p(max_diff / (avg_diff + 1e-5))
                weighted_diff = avg_diff * (1 + weight_factor)
                
                # Further normalize using a power function to enhance larger spikes
                # This ensures small natural movements are de-emphasized while
                # preserving pronounced movements that may indicate deception
                normalized_diff = np.power(weighted_diff / 50.0, 0.7) * 1.2
            else:
                normalized_diff = 0.0
            
            # Return normalized value, with a minimum floor to avoid zeros
            # and a maximum cap to prevent extreme outliers
            return max(min(normalized_diff, 1.0), 0.01)
            
        except Exception as e:
            print(f"Error in compute_phase_change: {str(e)}")
            return 0.01  # Return small non-zero value on error for visualization
    
    def detect_micro_expression_regions(self, frame: np.ndarray) -> Dict:
        """Detect facial regions for micro-expression analysis"""
        height, width = frame.shape[:2]
        regions_dict = {}
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Extract landmarks for the first face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert normalized landmarks to image coordinates
            landmarks = [(int(l.x * width), int(l.y * height)) 
                        for l in face_landmarks.landmark]
            
            # Extract regions based on configuration
            if FACIAL_REGIONS.get('track_eyebrows', True):
                for region_name, indices in self.EYEBROW_LANDMARKS.items():
                    center = self.get_region_center(landmarks, indices)
                    region_img, bounds = self.extract_rectangle_region(
                        frame, center, self.REGION_DIMENSIONS['eyebrow']
                    )
                    if region_img.size > 0:
                        regions_dict[f'eyebrow_{region_name}'] = {
                            'image': region_img,
                            'bounds': bounds,
                            'original_size': self.REGION_DIMENSIONS['eyebrow']
                        }
            
            if FACIAL_REGIONS.get('track_upper_eyelids', True):
                for region_name, indices in self.EYELID_LANDMARKS.items():
                    center = self.get_region_center(landmarks, indices)
                    region_img, bounds = self.extract_rectangle_region(
                        frame, center, self.REGION_DIMENSIONS['eyelid']
                    )
                    if region_img.size > 0:
                        regions_dict[f'eyelid_{region_name}'] = {
                            'image': region_img,
                            'bounds': bounds,
                            'original_size': self.REGION_DIMENSIONS['eyelid']
                        }
            
            if FACIAL_REGIONS.get('track_eye_corners', True):
                for region_name, indices in self.EYE_CORNER_LANDMARKS.items():
                    center = self.get_region_center(landmarks, indices)
                    region_img, bounds = self.extract_rectangle_region(
                        frame, center, self.REGION_DIMENSIONS['eye_corner']
                    )
                    if region_img.size > 0:
                        regions_dict[f'eye_corner_{region_name}'] = {
                            'image': region_img,
                            'bounds': bounds,
                            'original_size': self.REGION_DIMENSIONS['eye_corner']
                        }
            
            # Add explicit nose landmark indices
            # This is a critical part for tracking nose movements
            for region_name, indices in self.NOSE_LANDMARKS.items():
                center = self.get_region_center(landmarks, indices)
                # For nose tip, use a larger region to capture more area
                if region_name == 'nose_tip':
                    # Size needs to be large enough to track but not too large to include irrelevant areas
                    nose_size = (80, 60)  # Larger width than height for nose tip
                else:
                    nose_size = (70, 70)  # Square region for other nose parts
                    
                region_img, bounds = self.extract_rectangle_region(
                    frame, center, nose_size
                )
                if region_img.size > 0:
                    # Store nose regions directly with standardized names
                    if region_name == 'nose_tip':
                        # Use the standard name for nose_tip that the code expects
                        regions_dict['nose_tip'] = {
                            'image': region_img,
                            'bounds': bounds,
                            'original_size': nose_size
                        }
                    else:
                        # Store other nose regions with their specific names
                        regions_dict[f'nose_{region_name}'] = {
                            'image': region_img,
                            'bounds': bounds,
                            'original_size': nose_size
                        }
            
            if FACIAL_REGIONS.get('track_nasolabial_fold', True):
                for region_name, indices in self.NASOLABIAL_LANDMARKS.items():
                    center = self.get_region_center(landmarks, indices)
                    region_img, bounds = self.extract_rectangle_region(
                        frame, center, self.REGION_DIMENSIONS['nasolabial']
                    )
                    if region_img.size > 0:
                        regions_dict[f'nasolabial_{region_name}'] = {
                            'image': region_img,
                            'bounds': bounds,
                            'original_size': self.REGION_DIMENSIONS['nasolabial']
                        }
            
            if FACIAL_REGIONS.get('track_forehead', True):
                for region_name, indices in self.FOREHEAD_LANDMARKS.items():
                    center = self.get_region_center(landmarks, indices)
                    # Move the forehead center up by 15 pixels
                    center = (center[0], max(0, center[1] - 15))
                    region_img, bounds = self.extract_rectangle_region(
                        frame, center, self.REGION_DIMENSIONS['forehead']
                    )
                    if region_img.size > 0:
                        regions_dict[f'forehead_{region_name}'] = {
                            'image': region_img,
                            'bounds': bounds,
                            'original_size': self.REGION_DIMENSIONS['forehead']
                        }
            
            return {'regions': regions_dict, 'landmarks': landmarks}
        
        # Fallback to basic face detection if Face Mesh fails
        return self.detect_upper_face(frame)
    
    def detect_upper_face(self, frame: np.ndarray) -> Dict:
        """Detect face and get upper face region (fallback method)"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        
        if not results.detections:
            return {'regions': {}}
        
        regions_dict = {}
        h, w, _ = frame.shape
        
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)
            
            # Define eye region (upper 40% of face)
            eye_y = y
            eye_height = int(height * 0.4)
            eye_region, eye_bounds = self.extract_rectangle_region(
                frame, (x + width//2, eye_y + eye_height//2), (width, eye_height)
            )
            
            if eye_region.size > 0:
                regions_dict['eyes_region'] = {
                    'image': eye_region,
                    'bounds': eye_bounds,
                    'original_size': (width, eye_height)
                }
                
            # Define nose region (middle 30% of face)
            nose_y = y + eye_height
            nose_height = int(height * 0.3)
            nose_region, nose_bounds = self.extract_rectangle_region(
                frame, (x + width//2, nose_y + nose_height//2), (width, nose_height)
            )
            
            if nose_region.size > 0:
                regions_dict['nose_region'] = {
                    'image': nose_region,
                    'bounds': nose_bounds,
                    'original_size': (width, nose_height)
                }
                
                # Also add a specific nose_tip region for consistency with other detection
                nose_tip_y = nose_y + nose_height // 4
                nose_tip_size = min(width // 3, nose_height)
                nose_tip_region, nose_tip_bounds = self.extract_rectangle_region(
                    frame, (x + width//2, nose_tip_y), (nose_tip_size, nose_tip_size)
                )
                
                if nose_tip_region.size > 0:
                    regions_dict['nose_tip'] = {
                        'image': nose_tip_region,
                        'bounds': nose_tip_bounds,
                        'original_size': (nose_tip_size, nose_tip_size)
                    }
            
            # Only process the first face
            break
        
        return {'regions': regions_dict}
    
    def process_motion_video_no_mouth(self, input_path, output_path):
        """Process video with facial phase-based motion magnification region"""
        # This is a specialized version of the motion processor's process_video method
        # that explicitly prevents the mouth region from being magnified
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use half speed (0.5x) for output
        output_fps = fps / 2
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        print("Reading frames and detecting faces for motion magnification...")
        all_frames = []
        all_faces = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            faces = self.motion_processor.face_detector.detect_faces(frame)
            all_frames.append(frame)
            all_faces.append(faces)
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        print("Processing facial regions...")
        processed_frames = all_frames.copy()
        
        # Process each face
        for face_idx in range(len(all_faces[0]) if all_faces and all_faces[0] else 0):
            # Process each region, EXCLUDING mouth
            for region_name in ['left_eye', 'right_eye', 'nose_tip']:  # Explicitly exclude mouth
                print(f"Processing {region_name} for face {face_idx + 1}...")
                
                # Collect region frames
                region_frames = []
                for frame_idx in range(len(all_faces)):
                    if all_faces[frame_idx] and len(all_faces[frame_idx]) > face_idx:
                        face = all_faces[frame_idx][face_idx]
                        if region_name in face['regions']:
                            region_frames.append(face['regions'][region_name]['image'])
                
                if region_frames:
                    # Magnify the region
                    magnified_frames, _ = self.motion_processor.phase_magnifier.magnify(region_frames)
                    
                    # Replace the regions in the processed frames
                    for frame_idx, magnified in enumerate(magnified_frames):
                        if (all_faces[frame_idx] and 
                            len(all_faces[frame_idx]) > face_idx and 
                            region_name in all_faces[frame_idx][face_idx]['regions']):
                            
                            face = all_faces[frame_idx][face_idx]
                            region_info = face['regions'][region_name]
                            bounds = region_info['bounds']
                            original_size = region_info['original_size']
                            
                            # Resize magnified region back to original size if needed
                            if magnified.shape[:2] != (bounds[3]-bounds[1], bounds[2]-bounds[0]):
                                magnified = cv2.resize(magnified, 
                                                     (bounds[2]-bounds[0], bounds[3]-bounds[1]))
                            
                            # Replace region in frame
                            processed_frames[frame_idx][bounds[1]:bounds[3], 
                                                      bounds[0]:bounds[2]] = magnified
        
        # Write processed frames
        print("Writing motion magnification output video...")
        for frame in processed_frames:
            out.write(frame)
        
        cap.release()
        out.release()
        print("Motion magnification complete!")
    
    def process_color_magnification(self, frames: List[np.ndarray], fps: float = 30.0, alpha_blend: float = 0.5) -> List[np.ndarray]:
        """Apply color magnification to the entire face region above the mouth for heart rate detection.
        
        Args:
            frames: List of input frames
            fps: Frames per second of the video
            alpha_blend: Blend factor for color magnification (0.0-1.0)
            
        Returns:
            List of color-magnified frames
        """
        if not frames:
            return []
        
        # Use our upper face detector to get the face region above the mouth
        detected, face_regions = self.upper_face_detector.detect_upper_face(frames[0])
        if not detected:
            print("No face detected for color magnification")
            return frames.copy()
        
        # Get upper face region
        face_region = face_regions[0]['regions']['upper_face']
        x_min, y_min, x_max, y_max = face_region['bounds']
        
        # Create debug visualization
        debug_frame = frames[0].copy()
        debug_frame = self.upper_face_detector.draw_upper_face_region(debug_frame, face_regions[0])
        cv2.imwrite("Separate_Color_Motion_Magnification/output_videos/heart_rate_region_debug.jpg", debug_frame)
        
        # Extract upper face regions from all frames
        upper_face_frames = []
        for frame in frames:
            upper_face = frame[y_min:y_max, x_min:x_max].copy()
            upper_face_frames.append(upper_face)
        
        # Apply color magnification to the upper face regions
        print(f"Processing upper face region for heart rate detection ({len(upper_face_frames)} frames at {fps} fps)")
        
        # Set magnification parameters from config - following the article's recommendations
        self.color_processor.alpha = COLOR_MAG_PARAMS['alpha']
        self.color_processor.level = COLOR_MAG_PARAMS['level']
        self.color_processor.f_lo = COLOR_MAG_PARAMS['f_lo']
        self.color_processor.f_hi = COLOR_MAG_PARAMS['f_hi']
        
        # Apply magnification to face region using ColorMagnification directly
        magnified_regions = self.color_processor.magnify(upper_face_frames)
        
        # Create output frames by reintegrating the magnified regions
        magnified_frames = []
        for i, frame in enumerate(frames):
            # Create a copy of the original frame
            magnified = frame.copy()
            
            # Check if we have a valid magnified region
            if i < len(magnified_regions):
                # Get the original region
                original_region = magnified[y_min:y_max, x_min:x_max]
                
                # Get the magnified region
                magnified_region = magnified_regions[i]
                
                # Ensure both have the same shape
                if magnified_region.shape[:2] == original_region.shape[:2]:
                    # Use a lower alpha for blending to avoid color artifacts
                    # The article suggests not overamplifying to avoid unwanted artifacts
                    blend_factor = min(alpha_blend, 0.5)
                    
                    # Blend the original and magnified regions instead of direct replacement
                    blended_region = cv2.addWeighted(
                        magnified_region, blend_factor,
                        original_region, 1.0 - blend_factor,
                        0
                    )
                    
                    # Replace the region in the output frame
                    magnified[y_min:y_max, x_min:x_max] = blended_region
                else:
                    # Resize if needed (shouldn't happen but just in case)
                    resized_magnified = cv2.resize(magnified_region, (original_region.shape[1], original_region.shape[0]))
                    blend_factor = min(alpha_blend, 0.5)
                    blended_region = cv2.addWeighted(
                        resized_magnified, blend_factor,
                        original_region, 1.0 - blend_factor,
                        0
                    )
                    magnified[y_min:y_max, x_min:x_max] = blended_region
                    print(f"Resized region for frame {i} due to shape mismatch.")
            else:
                print(f"Warning: No magnified region for frame {i}. Skipping magnification.")
            
            magnified_frames.append(magnified)
        
        return magnified_frames
    
    def process_video(self, input_path=None, output_path=None):
        """Process video to create side-by-side magnification with tracking graphs"""
        # Use provided paths or default from config
        input_path = input_path or INPUT_VIDEO_PATH
        output_path = output_path or OUTPUT_VIDEO_PATH
        
        # Create temp file paths
        motion_output_path = output_path.replace(".mp4", "_motion_only.mp4")
        color_output_path = output_path.replace(".mp4", "_color_only.mp4")
        
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
        
        # Store fps as instance variable for use in other methods
        self.fps = fps
        
        # Set half speed by halving the fps for output (0.5x speed)
        output_fps = fps / 2
        print(f"Input video: {fps} fps, Output video: {output_fps} fps (0.5x speed)")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print("Reading frames...")
        all_frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            all_frames.append(frame)
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"Read {frame_count}/{total_frames} frames")
        
        # Close the input video
        cap.release()
        
        # Dictionaries to store phase and color changes
        all_phase_changes = {}
        all_color_changes = {}
        
        print("Applying motion magnification to micro-expression regions...")
        # Custom modification to the motion magnification to focus on micro-expression regions
        # This overrides the default regions in the FacialPhaseMagnification class
        self.motion_processor.detect_face_regions = self.detect_micro_expression_regions
        
        # STEP 1: Process facial regions directly with PBM and collect phase changes
        print("Processing facial regions with Phase-Based Motion (PBM) magnification...")
        
        # First detect faces in the initial frame
        face_data = self.detect_micro_expression_regions(all_frames[0])
        
        # Track key regions we want to analyze
        motion_regions = ['left_eye', 'right_eye', 'nose_tip']
        
        # Only process if we have detected regions
        if 'regions' in face_data and len(face_data['regions']) > 0:
            # Process each key region separately
            for region_name, region_info in face_data['regions'].items():
                # Check if this is one of the key regions we want to track
                key_region = None
                if 'left' in region_name and 'eye' in region_name:
                    key_region = 'left_eye'
                elif 'right' in region_name and 'eye' in region_name:
                    key_region = 'right_eye'
                elif region_name == 'nose_tip' or 'nose' in region_name:
                    key_region = 'nose_tip'  # Standardize all nose regions to nose_tip
                
                # Only process key regions
                if key_region and key_region in motion_regions:
                    # Create region key for storage
                    region_key = f"face1_{key_region}"
                    print(f"Processing {region_key} with PBM magnification...")
                    
                    # Collect region frames
                    region_frames = []
                    valid_frames = []
                    for frame_idx, frame in enumerate(all_frames):
                        # Detect regions for this frame
                        current_face_data = self.detect_micro_expression_regions(frame)
                        if 'regions' in current_face_data:
                            # Find matching region in current frame
                            for curr_name, curr_info in current_face_data['regions'].items():
                                if (('left' in curr_name and 'eye' in curr_name and key_region == 'left_eye') or
                                    ('right' in curr_name and 'eye' in curr_name and key_region == 'right_eye') or
                                    ('nose' in curr_name and key_region == 'nose_tip')):
                                    region_frames.append(curr_info['image'])
                                    valid_frames.append(frame_idx)
                                    break
                    
                    if len(region_frames) > 0:
                        print(f"Collected {len(region_frames)} frames for {region_key}")
                        
                        # Apply PBM magnification directly to get phase changes
                        magnified_frames, phase_changes = self.motion_processor.phase_magnifier.magnify(region_frames)
                        
                        # Store phase changes for this region
                        all_phase_changes[region_key] = phase_changes
                        print(f"Extracted phase changes for {region_key}: {len(phase_changes)} values")
                        
                        # Replace regions in frames with magnified versions
                        for i, (frame_idx, magnified) in enumerate(zip(valid_frames, magnified_frames)):
                            if frame_idx < len(all_frames):
                                # Get current face data for this frame
                                current_face_data = self.detect_micro_expression_regions(all_frames[frame_idx])
                                if 'regions' in current_face_data:
                                    # Find matching region to replace
                                    for curr_name, curr_info in current_face_data['regions'].items():
                                        if (('left' in curr_name and 'eye' in curr_name and key_region == 'left_eye') or
                                            ('right' in curr_name and 'eye' in curr_name and key_region == 'right_eye') or
                                            ('nose' in curr_name and key_region == 'nose_tip')):
                                            # Get bounds
                                            bounds = curr_info['bounds']
                                            # Replace region
                                            all_frames[frame_idx][bounds[1]:bounds[3], bounds[0]:bounds[2]] = magnified
                                            break
        
        # STEP 2: Save motion magnified output
        print("Saving motion magnified video...")
        motion_writer = cv2.VideoWriter(motion_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                                      output_fps, (width, height))
        for frame in all_frames:
            motion_writer.write(frame)
        motion_writer.release()
        
        # Create a copy of motion frames for the side-by-side view
        motion_frames = all_frames.copy()
        
        print("Applying color magnification for heart rate detection...")
        # Process with color magnification using a more conservative blend factor
        # The article recommends not over-amplifying to avoid color artifacts
        color_frames = self.process_color_magnification(all_frames, fps, alpha_blend=0.5)
        
        print("Creating output with side-by-side videos and graphs...")
        # Ensure all videos have the same number of frames
        min_frames = min(len(all_frames), len(motion_frames), len(color_frames))
        
        # Configure layout dimensions
        
        # Use fixed dimensions for consistent output regardless of input video size
        # Set a fixed base width for each video column - increased width for better graph visibility
        fixed_base_width = 720  # Increased from 640 to 720 for wider graphs
        # Calculate the side by side width based on fixed size
        side_by_side_width = fixed_base_width * 3
        
        # Set a fixed height for video display - standard HD half-height (maintain 16:9 aspect)
        video_height = 360
        # Calculate video width based on original aspect ratio, but enforce minimum width
        aspect_ratio = width / height
        video_width = max(int(video_height * aspect_ratio), int(fixed_base_width * 0.85))
        # Make sure video_width doesn't exceed the column width
        video_width = min(video_width, fixed_base_width)
        
        # Configure fixed plot dimensions - always the same size regardless of input video
        plot_width = fixed_base_width  # Each plot will be as wide as a video column
        plot_height = 320   # Increased from 300 to 320 for taller graphs
        
        # Ensure we only plot the three main regions plus heart rate
        main_regions = ['face1_left_eye', 'face1_right_eye', 'face1_nose_tip']
        # Also check for alternative nose region names
        nose_alternatives = ['face1_nose_region', 'face1_nose', 'nose_tip', 'nose_region']
        
        motion_regions_to_plot = []
        
        # Filter to include only the main regions that actually have data
        # and explicitly exclude any mouth-related data
        for region in main_regions:
            # Skip mouth regions
            if 'mouth' in region:
                continue
                
            if region in all_phase_changes and len(all_phase_changes[region]) > 0:
                motion_regions_to_plot.append(region)
            # Special case for nose - try alternative names
            elif region == 'face1_nose_tip':
                found_nose = False
                for alt_nose in nose_alternatives:
                    if alt_nose in all_phase_changes and len(all_phase_changes[alt_nose]) > 0:
                        # Use the alternative nose region but keep the standard key
                        all_phase_changes['face1_nose_tip'] = all_phase_changes[alt_nose]
                        motion_regions_to_plot.append('face1_nose_tip')
                        found_nose = True
                        print(f"Using alternative nose region: {alt_nose}")
                        break
                if not found_nose:
                    print(f"Warning: No data collected for {region} or alternatives, will not display graph")
            else:
                print(f"Warning: No data collected for {region}, will not display graph")
        
        # Clean up any mouth data that might have been collected
        for key in list(all_phase_changes.keys()):
            if 'mouth' in key:
                del all_phase_changes[key]
                print(f"Removed mouth region data: {key}")
        
        # Ensure we have a nose plot placeholder even if no nose data is detected
        # This checks if a nose_tip is in our regions to plot
        has_nose_plot = any('nose_tip' in region for region in motion_regions_to_plot)
        
        if not has_nose_plot:
            print("No nose region detected for graphing. Creating placeholder graph.")
            # Create synthetic data for the nose plot (small random values instead of zeros)
            # Use float values for compatibility with plotting
            nose_data = np.ones(min_frames) * 0.01  # Small non-zero values
            all_phase_changes['face1_nose_tip'] = nose_data.astype(np.float64)
            motion_regions_to_plot.append('face1_nose_tip')
        
        # Heart rate plots
        pulse_plots_needed = 2 if bpm_data else 0  # Heart rate and pulse signal
        
        # New layout:
        # - First column: All frame-to-frame graphs (nose, left eye, right eye)
        # - Second column: All raw graphs (nose, left eye, right eye)
        # - Third column: Heart rate estimation and pulse signal
        
        # We now have 3 rows (nose, left eye, right eye) and 3 columns
        plots_per_row = 3  # Number of columns
        num_motion_regions = len(motion_regions_to_plot)
        
        # Calculate how many rows we need - each row will contain one region's "raw" and "diff" plots
        total_plot_rows = max(num_motion_regions, 3)  # Ensure at least 3 rows for consistent layout
        
        # Calculate total height needed for all plots
        total_plot_height = total_plot_rows * plot_height
        
        # Calculate combined frame dimensions: videos on top, plots below
        combined_height = video_height + total_plot_height
        combined_width = side_by_side_width
        
        # Create video writer for combined output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (combined_width, combined_height))
        
        print(f"Creating output video: {combined_width}x{combined_height}, with {total_plot_rows} rows of plots")
        print(f"Displaying graphs for regions: {[r.replace('face1_', '') for r in motion_regions_to_plot]}")
        
        # Add this new section to calculate global frame limits for all plots
        # This needs to be done before creating any plots
        print("Calculating global frame limits for consistent x-axis across all graphs...")
        # Initialize global min and max frame values
        global_frame_min = float('inf')
        global_frame_max = float('-inf')
        
        # Temporary dictionary to store binned frames for each region
        all_binned_frames = {}
        
        # Calculate binned frames for each region to find global limits
        bin_size_frames = int(fps)  # 1-second bins in frames
        
        for region_name in motion_regions_to_plot:
            if region_name in all_phase_changes:
                # Get data for this region
                phase_data = all_phase_changes[region_name]
                max_len = len(phase_data)
                
                # Calculate number of bins for this region
                num_bins = max(1, int(np.ceil(max_len / bin_size_frames)))
                
                # Calculate binned frames
                binned_frames = []
                for bin_idx in range(num_bins):
                    start_idx = bin_idx * bin_size_frames
                    end_idx = min(max_len, (bin_idx + 1) * bin_size_frames)
                    
                    if start_idx < max_len:
                        # Calculate bin center frame
                        bin_center_frame = start_idx + (end_idx - start_idx) // 2
                        binned_frames.append(bin_center_frame)
                
                if binned_frames:
                    # Update global min and max
                    global_frame_min = min(global_frame_min, binned_frames[0])
                    global_frame_max = max(global_frame_max, binned_frames[-1])
                    
                    # Store binned frames for this region
                    all_binned_frames[region_name] = np.array(binned_frames)
        
        # Safety check in case no valid regions were found
        if global_frame_min == float('inf'):
            global_frame_min = 0
        if global_frame_max == float('-inf'):
            global_frame_max = total_frames
            
        # Make sure the x-axis extends to the full video length
        global_frame_min = 0
        global_frame_max = total_frames
        
        print(f"Using global frame limits for all graphs: {global_frame_min} to {global_frame_max}")
        
        # Process frames with motion magnification and generate combined output video
        for i in range(min_frames):
            # Create a blank combined frame
            combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            
            # Resize the three video frames to the fixed dimensions
            original_resized = cv2.resize(all_frames[i], (video_width, video_height))
            motion_resized = cv2.resize(motion_frames[i], (video_width, video_height))
            color_resized = cv2.resize(color_frames[i], (video_width, video_height))
            
            # Add labels to the video frames - include slow motion indication (0.5x speed)
            original_labeled = self.add_label(original_resized, f"{ORIGINAL_LABEL} (0.5x)")
            motion_labeled = self.add_label(motion_resized, f"{MOTION_LABEL} (0.5x)")
            color_labeled = self.add_label(color_resized, f"{COLOR_LABEL} (0.5x)")
            
            # Calculate horizontal positions to center videos within their columns
            x_offset1 = (fixed_base_width - video_width) // 2
            x_offset2 = fixed_base_width + (fixed_base_width - video_width) // 2
            x_offset3 = fixed_base_width * 2 + (fixed_base_width - video_width) // 2
            
            # Place the three videos at the top of the combined frame
            try:
                combined_frame[:video_height, x_offset1:x_offset1+video_width, :] = original_labeled
                combined_frame[:video_height, x_offset2:x_offset2+video_width, :] = motion_labeled
                combined_frame[:video_height, x_offset3:x_offset3+video_width, :] = color_labeled
            except Exception as e:
                print(f"Error placing videos in frame {i}: {str(e)}")
            
            # Store the current position for tracking
            current_frame_position = i
            total_video_frames = min_frames
            
            # Create individual plots for frame-to-frame and raw changes for each region
            # Define the ordering of regions to match image: Nose, Left Eye, Right Eye
            ordered_regions = []
            # First ensure we have nose_tip
            nose_region = next((r for r in motion_regions_to_plot if 'nose_tip' in r), None)
            if nose_region:
                ordered_regions.append(nose_region)
            
            # Then add left eye
            left_eye_region = next((r for r in motion_regions_to_plot if 'left_eye' in r), None)
            if left_eye_region:
                ordered_regions.append(left_eye_region)
                
            # Then add right eye
            right_eye_region = next((r for r in motion_regions_to_plot if 'right_eye' in r), None)
            if right_eye_region:
                ordered_regions.append(right_eye_region)
                
            # Add any remaining regions not explicitly ordered
            # But explicitly exclude any mouth-related regions
            for region in motion_regions_to_plot:
                if region not in ordered_regions and 'mouth' not in region:
                    ordered_regions.append(region)
            
            # Create plots for each region and place them in the grid
            for row_idx, region_name in enumerate(ordered_regions):
                # Skip any mouth-related regions
                if 'mouth' in region_name:
                    continue
                    
                y_start = video_height + (row_idx * plot_height)
                y_end = y_start + plot_height
                
                # Get data for this region
                region_data = all_phase_changes.get(region_name, [])
                
                # Get display name for the region
                if "left_eye" in region_name:
                    display_name = "Left Eye"
                elif "right_eye" in region_name:
                    display_name = "Right Eye"
                elif "nose_tip" in region_name or "nose" in region_name:
                    display_name = "Nose"
                else:
                    display_name = region_name.replace('face1_', '').replace('_', ' ').title()
                
                try:
                    # Column 1: Frame-to-Frame changes plot (left position)
                    ftf_plot = self.create_single_plot(
                        region_data, current_frame_position, plot_width, plot_height, 
                        "diff", f"{display_name} Frame-to-Frame", total_video_frames,
                        x_min=global_frame_min, x_max=global_frame_max
                    )
                    # Place Frame-to-Frame plot in column 1 - distribute evenly across width
                    x_start_ftf = fixed_base_width * 0
                    x_end_ftf = fixed_base_width * 1
                    combined_frame[y_start:y_end, x_start_ftf:x_end_ftf, :] = ftf_plot
                    
                    # Column 2: Raw phase plot (middle position)
                    raw_plot = self.create_single_plot(
                        region_data, current_frame_position, plot_width, plot_height, 
                        "raw", f"{display_name} Raw", total_video_frames,
                        x_min=global_frame_min, x_max=global_frame_max
                    )
                    # Place Raw plot in column 2 - distribute evenly across width
                    x_start_raw = fixed_base_width * 1
                    x_end_raw = fixed_base_width * 2
                    combined_frame[y_start:y_end, x_start_raw:x_end_raw, :] = raw_plot
                except Exception as e:
                    print(f"Error creating plots for region {region_name} at frame {i}: {str(e)}")
            
            # Add heart rate and pulse signal plots if available in the third column
            if bpm_data:
                try:
                    # Heart Rate plot (first row, third column)
                    heart_rate_plot = self.create_heart_rate_plot(
                        bpm_data, current_frame_position, plot_width, plot_height,
                        x_min=global_frame_min, x_max=global_frame_max
                    )
                    
                    # Place in first row, third column - distribute evenly across width
                    hr_y_start = video_height
                    hr_y_end = hr_y_start + plot_height
                    hr_x_start = fixed_base_width * 2
                    hr_x_end = fixed_base_width * 3
                    combined_frame[hr_y_start:hr_y_end, hr_x_start:hr_x_end, :] = heart_rate_plot
                    
                    # Pulse Signal plot (second row, third column)
                    pulse_signal_plot = self.create_pulse_signal_plot(
                        bpm_data, current_frame_position, plot_width, plot_height,
                        x_min=global_frame_min, x_max=global_frame_max
                    )
                    
                    # Place in second row, third column - distribute evenly across width
                    pulse_y_start = video_height + plot_height
                    pulse_y_end = pulse_y_start + plot_height
                    pulse_x_start = fixed_base_width * 2
                    pulse_x_end = fixed_base_width * 3
                    
                    if pulse_y_end <= combined_height:  # Make sure it's not out of bounds
                        combined_frame[pulse_y_start:pulse_y_end, pulse_x_start:pulse_x_end, :] = pulse_signal_plot
                        
                    # Add information panel in the bottom right slot (third row, third column)
                    info_panel = self.create_info_panel(
                        current_frame_position, total_video_frames, bpm_data, plot_width, plot_height
                    )
                    
                    # Place in third row, third column - distribute evenly across width
                    info_y_start = video_height + (2 * plot_height)
                    info_y_end = info_y_start + plot_height
                    info_x_start = fixed_base_width * 2
                    info_x_end = fixed_base_width * 3
                    
                    if info_y_end <= combined_height:  # Make sure it's not out of bounds
                        combined_frame[info_y_start:info_y_end, info_x_start:info_x_end, :] = info_panel
                        
                except Exception as e:
                    print(f"Error creating plots at frame {i}: {str(e)}")
            
            # Write the combined frame
            out.write(combined_frame)
            
            if i % 10 == 0:
                print(f"Processed {i}/{min_frames} frames")
            
            # Clean up matplotlib figures after every frame
            plt.close('all')
        
        # Release resources
        out.release()
        
        # Delete temporary files to save space
        if not KEEP_TEMP_FILES:
            if os.path.exists(motion_output_path):
                os.remove(motion_output_path)
            if os.path.exists(color_output_path) and os.path.exists(color_output_path):
                os.remove(color_output_path)
        
        print("Processing complete!")
    
    def create_info_panel(self, frame_idx, total_frames, bpm_data, plot_width, plot_height):
        """Creates an information panel with useful stats and parameters
        
        Args:
            frame_idx: Current frame index
            total_frames: Total frames in video
            bpm_data: Heart rate data tuple
            plot_width: Width of the plot in pixels
            plot_height: Height of the plot in pixels
            
        Returns:
            Info panel image as numpy array
        """
        try:
            # Create figure with the same fixed size as other plots
            fig = plt.figure(figsize=(10, 6), dpi=100, facecolor='white')
            ax = fig.add_subplot(1, 1, 1)
            
            # Remove axis for cleaner look
            ax.axis('off')
            
            # Extract BPM data if available
            current_bpm = "N/A"
            if bpm_data is not None:
                inst_bpm, avg_bpm, _ = bpm_data
                if frame_idx < len(avg_bpm):
                    current_bpm = f"{avg_bpm[frame_idx]:.1f}"
            
            # Calculate progress percentage
            progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            
            # Get motion magnification parameters from config
            motion_params = {
                "Phase Mag": f"{MOTION_MAG_PARAMS.get('phase_mag', 'N/A')}",
                "Freq Range": f"{MOTION_MAG_PARAMS.get('f_lo', 'N/A'):.2f}-{MOTION_MAG_PARAMS.get('f_hi', 'N/A'):.2f} Hz",
                "Sigma": f"{MOTION_MAG_PARAMS.get('sigma', 'N/A')}"
            }
            
            # Get color magnification parameters from config
            color_params = {
                "Alpha": f"{COLOR_MAG_PARAMS.get('alpha', 'N/A')}",
                "Low Freq": f"{COLOR_MAG_PARAMS.get('f_lo', 'N/A'):.2f} Hz",
                "High Freq": f"{COLOR_MAG_PARAMS.get('f_hi', 'N/A'):.2f} Hz"
            }
            
            # Create text content with improved formatting and larger font size
            info_text = (
                f"Frame: {frame_idx+1}/{total_frames} ({progress:.1f}%)\n"
                f"Current Heart Rate: {current_bpm} BPM\n"
                f"Playback Speed: 0.5x\n\n"
                "MOTION MAGNIFICATION (PBM)\n"
                f"• Phase Amp: {motion_params['Phase Mag']}x\n"
                f"• Freq Band: {motion_params['Freq Range']}\n"
                f"• Filter: σ={motion_params['Sigma']}\n\n"
                "PHASE CHANGE ANALYSIS\n"
                "• Red points indicate significant\n  phase changes in regions\n"
                "• Graphs show direct phase changes\n  amplified by PBM algorithm\n"
                "• X-axis shows frame numbers\n  across full video range\n\n"
                "COLOR MAGNIFICATION (EVM)\n"
                f"• Alpha: {color_params['Alpha']}\n"
                f"• Freq Band: {color_params['Low Freq']} -\n  {color_params['High Freq']}"
            )
            
            # Add the text content with even larger font size directly
            # No separate title, centered in the panel
            ax.text(0.5, 0.5, info_text, 
                  horizontalalignment='center', 
                  verticalalignment='center',  # Center vertically instead of top alignment
                  transform=ax.transAxes,
                  fontsize=22,  # Significantly increased font size from 16 to 22
                  family='monospace',
                  weight='bold',  
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
            
            # Save to image
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # Convert to numpy array
            img_pil = Image.open(buf)
            # Convert to RGB explicitly to remove alpha channel
            if img_pil.mode == 'RGBA':
                img_pil = img_pil.convert('RGB')
            plot_img = np.array(img_pil)
            
            # Close buffer and figure
            buf.close()
            plt.close(fig)
            
            # Create a padded image with white background
            padding = max(int(plot_width * 0.02), 5)  # Reduce padding to 2% or minimum 5px
            padded_img = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
            
            # Resize to fit in the padded area while maintaining aspect ratio
            h, w = plot_img.shape[:2]
            target_w = plot_width - (2 * padding)
            target_h = plot_height - (2 * padding)
            
            # Calculate resize dimensions
            if h/w > target_h/target_w:  # Height limited
                resize_h = target_h
                resize_w = min(int(w * (resize_h / h)), target_w)
            else:  # Width limited
                resize_w = target_w
                resize_h = min(int(h * (resize_w / w)), target_h)
                
            # Resize image
            if h != resize_h or w != resize_w:
                plot_img = cv2.resize(plot_img, (resize_w, resize_h))
            
            # Calculate position to center in padded image
            x_offset = padding + (target_w - resize_w) // 2
            y_offset = padding + (target_h - resize_h) // 2
            
            # Place in padded image
            padded_img[y_offset:y_offset+resize_h, x_offset:x_offset+resize_w] = plot_img
            
            return padded_img
            
        except Exception as e:
            print(f"Error creating info panel: {str(e)}")
            # Return blank white image on error
            return np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255


if __name__ == "__main__":
    # Process the video using settings from config.py
    processor = SideBySideMagnification()
    
    # Check if input and output paths are specified in config
    if 'INPUT_VIDEO_PATH' in globals() and 'OUTPUT_VIDEO_PATH' in globals():
        processor.process_video(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)
    else:
        # Use default test path if config doesn't have paths
        test_input = "/Users/naveenmirapuri/VideoProcessing/test_videos/face.mp4"
        test_output = "/Users/naveenmirapuri/VideoProcessing/Separate_Color_Motion_Magnification/output_videos/side_by_side_output.mp4"
        
        if os.path.exists(test_input):
            processor.process_video(test_input, test_output)
        else:
            print("Error: No valid input video path specified.")
            print("Please set INPUT_VIDEO_PATH in config.py or modify this script to use your video file.") 