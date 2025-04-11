import cv2
import numpy as np
import os
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from scipy import signal
import matplotlib

# Set up matplotlib for non-interactive use
matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 10
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
        
        # Add text
        cv2.putText(
            frame_with_label, 
            text, 
            (int(w/2 - 140), h-15), 
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
    
    def create_single_plot(self, phase_data, frame_idx, plot_width, plot_height, plot_type="raw", title=None):
        """Creates a single square plot of either raw phase data or frame-to-frame changes
        
        Args:
            phase_data: Array of phase change values
            frame_idx: Current frame index
            plot_width: Width of the plot in pixels
            plot_height: Height of the plot in pixels
            plot_type: "raw" or "diff" for raw phase changes or frame-to-frame differences
            title: Title to display on the plot
            
        Returns:
            Square plot image as numpy array
        """
        try:
            # Create a figure with square aspect ratio
            fig = plt.figure(figsize=(6, 6), dpi=100, facecolor='white')
            ax = fig.add_subplot(1, 1, 1)
            
            # Get array length
            max_len = len(phase_data)
            
            # Don't try to process if frame_idx is beyond available data
            if frame_idx >= max_len:
                frame_idx = max_len - 1
            
            # Get full timeline data
            x = np.arange(len(phase_data))
            
            if plot_type == "diff" and len(phase_data) > 1:
                # Calculate derivatives for frame-to-frame changes
                y = np.abs(np.diff(phase_data))
                y = np.insert(y, 0, 0)  # Add zero at the beginning
                title_prefix = "Frame-to-Frame Change Rate"
                color = 'green'
            else:
                # Raw phase data
                y = phase_data
                title_prefix = "Raw Phase Changes"
                color = 'blue'
            
            # Plot with assigned color
            ax.plot(x, y, color=color, linewidth=2)
            
            # Find and mark peaks
            if plot_type == "diff":
                peaks, _ = signal.find_peaks(y, height=max(0.1, np.max(y) * 0.3))
            else:
                peaks, _ = signal.find_peaks(y, height=0.3)  # Adjust height as needed
                
            if len(peaks) > 0:
                ax.plot(peaks, y[peaks], 'ro', markersize=6)
            
            # Add a bright green dot for current frame position - make it larger and more visible
            if frame_idx > 0 and frame_idx < len(y):
                ax.plot(frame_idx, y[frame_idx], 'o', color='lime', markersize=12, 
                       markeredgecolor='black', markeredgewidth=2, zorder=10)
            
            # Set title
            if title:
                ax.set_title(f"{title}: {title_prefix}", fontsize=12, fontweight='bold')
            else:
                ax.set_title(title_prefix, fontsize=12, fontweight='bold')
                
            # Set labels and grid
            ax.set_xlabel("Frame Number", fontsize=10)
            ax.set_ylabel("Magnitude", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Always show full timeline
            ax.set_xlim(0, max_len-1)
            
            # Add a bit of padding to y-axis
            if ax.get_ylim()[1] > 0:
                ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
            
            # Add a frame counter
            frame_text = f"Frame: {frame_idx}/{max_len-1}"
            fig.text(0.02, 0.02, frame_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
            
            # Adjust layout 
            plt.tight_layout()
            
            # Save the figure to a temporary buffer
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # Use PIL/Pillow to open the image from the buffer
            img_pil = Image.open(buf)
            plot_img = np.array(img_pil)
            
            # Close buffer
            buf.close()
            
            # Close figure to free memory
            plt.close(fig)
            
            # Resize to match requested dimensions
            plot_img = cv2.resize(plot_img, (plot_width, plot_height))
            
            # Convert from RGB to BGR for OpenCV
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
            
            # Ensure the output is uint8
            if plot_img.dtype != np.uint8:
                plot_img = plot_img.astype(np.uint8)
            
            return plot_img
            
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
    
    def create_heart_rate_plot(self, bpm_data, frame_idx, plot_width, plot_height):
        """Creates a plot showing heart rate (BPM)"""
        try:
            # Create figure 
            fig = plt.figure(figsize=(10, 6), dpi=100, facecolor='white')
            ax = fig.add_subplot(1, 1, 1)
            
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
            if frame_idx > 0 and frame_idx < len(avg_bpm):
                ax.axvline(x=frame_idx, color='green', linestyle='--', alpha=0.7)
                # Add a marker for current BPM
                current_bpm = avg_bpm[frame_idx]
                if current_bpm > 0:
                    ax.plot(frame_idx, current_bpm, 'o', color='lime', markersize=12, 
                          markeredgecolor='black', markeredgewidth=2, zorder=10)
                    ax.text(frame_idx+5, current_bpm, f"{current_bpm:.1f} BPM", 
                          fontsize=12, color='black', fontweight='bold',
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
            
            # Set BPM plot properties
            ax.set_title("Heart Rate Estimation", fontsize=14, fontweight='bold')
            ax.set_ylabel("BPM", fontsize=12)
            ax.set_xlabel("Frame Number", fontsize=12)
            ax.set_xlim(0, max_len-1)
            
            # Fix the cut-off by increasing the upper limit
            max_bpm = max(140, np.max(avg_bpm) * 1.2 if len(avg_bpm) > 0 else 140)
            ax.set_ylim(40, max_bpm)  # Dynamically adjust to fit the data
            
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            
            # Save to image
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # Convert to numpy array
            img_pil = Image.open(buf)
            plot_img = np.array(img_pil)
            
            # Close buffer and figure
            buf.close()
            plt.close(fig)
            
            # Resize to match requested dimensions
            plot_img = cv2.resize(plot_img, (plot_width, plot_height))
            
            # Convert from RGB to BGR for OpenCV
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
            
            return plot_img
            
        except Exception as e:
            print(f"Error creating heart rate plot for frame {frame_idx}: {str(e)}")
            blank = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
            cv2.putText(blank, f"Error: {str(e)[:50]}", 
                      (20, plot_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return blank
    
    def create_pulse_signal_plot(self, bpm_data, frame_idx, plot_width, plot_height):
        """Creates a plot showing the blood volume pulse signal"""
        try:
            # Create figure
            fig = plt.figure(figsize=(10, 6), dpi=100, facecolor='white')
            ax = fig.add_subplot(1, 1, 1)
            
            # Extract data
            _, _, combined_signal = bpm_data
            
            # Get max length
            max_len = len(combined_signal)
            
            # Time points
            x = np.arange(max_len)
            
            # Plot signal
            ax.plot(x, combined_signal, '-', color='green', linewidth=1.5)
            
            # Add current frame indicator
            if frame_idx > 0 and frame_idx < len(combined_signal):
                ax.axvline(x=frame_idx, color='green', linestyle='--', alpha=0.7)
                ax.plot(frame_idx, combined_signal[frame_idx], 'o', color='lime', markersize=10, 
                      markeredgecolor='black', markeredgewidth=1.5, zorder=10)
            
            # Set signal plot properties
            ax.set_title("Blood Volume Pulse Signal", fontsize=14, fontweight='bold')
            ax.set_xlabel("Frame Number", fontsize=12)
            ax.set_ylabel("Amplitude", fontsize=12)
            ax.set_xlim(0, max_len-1)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Save to image
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # Convert to numpy array
            img_pil = Image.open(buf)
            plot_img = np.array(img_pil)
            
            # Close buffer and figure
            buf.close()
            plt.close(fig)
            
            # Resize to match requested dimensions
            plot_img = cv2.resize(plot_img, (plot_width, plot_height))
            
            # Convert from RGB to BGR for OpenCV
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
            
            return plot_img
            
        except Exception as e:
            print(f"Error creating pulse signal plot for frame {frame_idx}: {str(e)}")
            blank = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
            cv2.putText(blank, f"Error: {str(e)[:50]}", 
                      (20, plot_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return blank
    
    def compute_phase_change(self, current_frame, prev_frame):
        """Compute the phase change between two frames
        
        This is a simplified method to compute approximate phase changes between frames
        for visualization purposes, when we don't have direct access to the internal
        phase data from the motion magnification algorithm.
        
        Args:
            current_frame: Current frame image
            prev_frame: Previous frame image
            
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
                
            # Calculate absolute difference
            diff = cv2.absdiff(current_gray, prev_gray)
            
            # Return mean of differences as a simple measure of change
            return np.mean(diff) / 255.0  # Normalize to 0-1 range
            
        except Exception as e:
            print(f"Error in compute_phase_change: {str(e)}")
            return 0.0
    
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
            
            # Only process the first face
            break
        
        return {'regions': regions_dict}
    
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
        
        # Create deep copies for motion and color magnification
        motion_frames = deepcopy(all_frames)
        
        # Dictionaries to store phase and color changes
        all_phase_changes = {}
        all_color_changes = {}
        
        print("Applying motion magnification to micro-expression regions...")
        # Process with motion magnification - this will focus on micro-expression regions
        
        # Custom modification to the motion magnification to focus on micro-expression regions
        # This overrides the default regions in the FacialPhaseMagnification class
        self.motion_processor.detect_face_regions = self.detect_micro_expression_regions
        
        # Process frames with motion magnification
        # Get motion regions and phase changes during magnification
        # Limit to only the three important regions: left eye, right eye, and nose
        motion_regions = ['left_eye', 'right_eye', 'nose_tip']
        
        print("Processing motion for facial regions...")
        for i, frame in enumerate(all_frames):
            # Detect face regions for the current frame
            face_data = self.detect_micro_expression_regions(frame)
            
            if 'regions' in face_data and len(face_data['regions']) > 0:
                # Process each motion region
                for region_name, region_info in face_data['regions'].items():
                    # Check if this is one of the three key regions we want to track
                    # Map arbitrary region names to our standardized key regions
                    key_region = None
                    if 'left' in region_name and 'eye' in region_name:
                        key_region = 'left_eye'
                    elif 'right' in region_name and 'eye' in region_name:
                        key_region = 'right_eye'
                    elif 'nose' in region_name:
                        key_region = 'nose_tip'  # Standardize all nose regions to nose_tip
                    
                    # Only track the three key regions
                    if key_region:
                        region_key = f"face1_{key_region}"
                        # Extract the region image
                        region_img = region_info['image']
                        
                        # Apply the motion magnifier to this region
                        if i == 0:
                            # Initialize phase tracking for this region
                            if region_key not in all_phase_changes:
                                all_phase_changes[region_key] = []
                                print(f"Tracking motion for {region_key}")
                        
                        # Calculate phase changes between frames
                        if i > 0:  # Need at least 2 frames for phase change
                            # Get previous region if available
                            prev_frame_idx = max(0, i-1)
                            prev_face_data = self.detect_micro_expression_regions(all_frames[prev_frame_idx])
                            
                            if 'regions' in prev_face_data:
                                # Find matching region in previous frame
                                prev_region_img = None
                                for prev_name, prev_info in prev_face_data['regions'].items():
                                    if (('left' in prev_name and 'eye' in prev_name and key_region == 'left_eye') or
                                        ('right' in prev_name and 'eye' in prev_name and key_region == 'right_eye') or
                                        ('nose' in prev_name and key_region == 'nose_tip')):
                                        prev_region_img = prev_info['image']
                                        break
                                
                                if prev_region_img is not None:
                                    phase_change = self.compute_phase_change(region_img, prev_region_img)
                                    
                                    # Ensure the region key exists in the dictionary
                                    if region_key not in all_phase_changes:
                                        all_phase_changes[region_key] = [0.0] * i  # Fill with zeros for previous frames
                                    
                                    all_phase_changes[region_key].append(phase_change)
                                else:
                                    # If previous region not found, use a small random value to avoid flat line
                                    if region_key not in all_phase_changes:
                                        all_phase_changes[region_key] = [0.0] * i
                                    all_phase_changes[region_key].append(0.01)
                            else:
                                # No previous regions at all
                                if region_key not in all_phase_changes:
                                    all_phase_changes[region_key] = [0.0] * i
                                all_phase_changes[region_key].append(0.01)
                        else:
                            # First frame has no phase change
                            all_phase_changes[region_key].append(0.0)
            
            if i % 10 == 0:
                print(f"Processed motion for {i}/{len(all_frames)} frames")
        
        # Process the full video again for motion magnification (actual visual magnification)
        self.motion_processor.process_video(input_path, motion_output_path)
        
        # Read the motion magnified video
        motion_cap = cv2.VideoCapture(motion_output_path)
        motion_frames = []
        
        while True:
            ret, frame = motion_cap.read()
            if not ret:
                break
            motion_frames.append(frame)
        
        motion_cap.release()
        
        print("Applying color magnification to entire face above mouth for heart rate detection...")
        # Process with color magnification using a more conservative blend factor
        # The article recommends not over-amplifying to avoid color artifacts
        color_frames = self.process_color_magnification(all_frames, fps, alpha_blend=0.5)
        
        # Calculate color changes and heart rate
        print("Processing color changes for heart rate detection...")
        
        # Detect upper face region for heart rate analysis
        face_regions = self.upper_face_detector.detect_upper_face(all_frames[0])
        
        if face_regions and len(face_regions) > 0:
            upper_face_frames = []
            
            for frame in all_frames:
                detected, regions = self.upper_face_detector.detect_upper_face(frame)
                if detected and 'upper_face' in regions[0]['regions']:
                    region = regions[0]['regions']['upper_face']
                    upper_face_frames.append(region['image'])
            
            # Calculate color changes for upper face region
            if upper_face_frames:
                print(f"Calculating color changes for {len(upper_face_frames)} upper face frames")
                upper_face_changes = self.calculate_color_changes(upper_face_frames)
                all_color_changes['face1_upper_face'] = upper_face_changes
                
                # Calculate BPM from color signals
                bpm_data = self.calculate_bpm(all_color_changes, fps)
            else:
                print("No upper face frames detected for color analysis")
                bpm_data = None
        else:
            print("No face detected for heart rate analysis")
            bpm_data = None
        
        print("Creating output with side-by-side videos and graphs...")
        # Ensure all videos have the same number of frames
        min_frames = min(len(all_frames), len(motion_frames), len(color_frames))
        
        # Configure layout dimensions
        
        # For the videos: maintain original aspect ratio but reduce size
        # Calculate the scaling factor to fit three videos side by side
        video_scale_factor = 0.9  # Leave some margin
        side_by_side_width = width * 3
        
        # Reduce height to fit graphs below
        video_height = int(height * 0.6)  # Use 60% of the original height for videos
        video_width = int((width * video_height) / height)  # Maintain aspect ratio
        
        # Configure graph layout
        plot_size = 250  # Size of each square plot (both width and height)
        
        # Ensure we only plot the three main regions plus heart rate
        main_regions = ['face1_left_eye', 'face1_right_eye', 'face1_nose_tip']
        # Also check for alternative nose region names
        nose_alternatives = ['face1_nose_region', 'face1_nose']
        
        motion_regions_to_plot = []
        
        # Filter to include only the main regions that actually have data
        for region in main_regions:
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
        
        # Heart rate plots
        pulse_plots_needed = 2 if bpm_data else 0  # Heart rate and pulse signal
        
        # Calculate total plots and organize them in rows
        plots_per_row = 3  # Number of plot pairs per row (one for each video column)
        num_motion_plots = len(motion_regions_to_plot)
        
        # Calculate how many rows we need
        # One row for motion plots and one row for pulse if needed
        total_plot_rows = 1  # At least one row for motion plots
        
        # Add a row for pulse/heart rate if needed
        if pulse_plots_needed > 0:
            total_plot_rows += 1
        
        total_plot_height = total_plot_rows * plot_size
        
        # Calculate combined frame dimensions: videos on top, plots below
        combined_height = video_height + total_plot_height
        combined_width = side_by_side_width
        
        # Create video writer for combined output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))
        
        print(f"Creating output video: {combined_width}x{combined_height}, with {total_plot_rows} rows of plots")
        print(f"Displaying graphs for regions: {[r.replace('face1_', '') for r in motion_regions_to_plot]}")
        
        for i in range(min_frames):
            # Create a blank combined frame
            combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            
            # Resize the three video frames while maintaining aspect ratio
            original_resized = cv2.resize(all_frames[i], (video_width, video_height))
            motion_resized = cv2.resize(motion_frames[i], (video_width, video_height))
            color_resized = cv2.resize(color_frames[i], (video_width, video_height))
            
            # Add labels to the video frames
            original_labeled = self.add_label(original_resized, ORIGINAL_LABEL)
            motion_labeled = self.add_label(motion_resized, MOTION_LABEL)
            color_labeled = self.add_label(color_resized, COLOR_LABEL)
            
            # Calculate horizontal positions to center videos
            video_margin = (width - video_width) // 2
            x_offset1 = video_margin
            x_offset2 = width + video_margin
            x_offset3 = width * 2 + video_margin
            
            # Place the three videos at the top of the combined frame
            try:
                combined_frame[:video_height, x_offset1:x_offset1+video_width, :] = original_labeled
                combined_frame[:video_height, x_offset2:x_offset2+video_width, :] = motion_labeled
                combined_frame[:video_height, x_offset3:x_offset3+video_width, :] = color_labeled
            except Exception as e:
                print(f"Error placing videos in frame {i}: {str(e)}")
            
            # Add motion region plots below videos - one per video column
            for idx, region_name in enumerate(motion_regions_to_plot):
                if idx >= plots_per_row:  # Only show up to 3 regions
                    break
                    
                # Calculate position for this plot - place directly under corresponding video
                # Left eye under original, right eye under motion, nose under color
                plot_col = idx
                y_start = video_height
                y_end = y_start + plot_size
                x_start = plot_col * width
                x_end = x_start + width
                
                # Generate plot for this region
                try:
                    region_plot = self.create_region_plot_pair(
                        all_phase_changes, i, region_name,
                        width // 2, plot_size
                    )
                    
                    # Add plot to the combined frame
                    if y_end <= combined_height and x_end <= combined_width:
                        combined_frame[y_start:y_end, x_start:x_end, :] = region_plot
                except Exception as e:
                    print(f"Error creating plot for region {region_name} at frame {i}: {str(e)}")
            
            # Add heart rate and pulse signal plots if available
            if bpm_data:
                # Position for heart rate plots (on the second row)
                y_start = video_height + plot_size if num_motion_plots > 0 else video_height
                y_end = y_start + plot_size
                
                # Position for Heart Rate plot (left half)
                x_start = 0
                x_end = combined_width // 2
                
                # Generate and add heart rate plot
                try:
                    heart_rate_plot = self.create_heart_rate_plot(
                        bpm_data, i, combined_width // 2, plot_size
                    )
                    combined_frame[y_start:y_end, x_start:x_end, :] = heart_rate_plot
                except Exception as e:
                    print(f"Error creating heart rate plot at frame {i}: {str(e)}")
                
                # Position for Pulse Signal plot (right half)
                x_start = combined_width // 2
                x_end = combined_width
                
                # Generate and add pulse signal plot
                try:
                    pulse_signal_plot = self.create_pulse_signal_plot(
                        bpm_data, i, combined_width // 2, plot_size
                    )
                    combined_frame[y_start:y_end, x_start:x_end, :] = pulse_signal_plot
                except Exception as e:
                    print(f"Error creating pulse signal plot at frame {i}: {str(e)}")
            
            # Write the combined frame
            out.write(combined_frame)
            
            if i % 10 == 0:
                print(f"Processed {i}/{min_frames} frames")
            
            # Clean up matplotlib figures periodically
            if i % 30 == 0:
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