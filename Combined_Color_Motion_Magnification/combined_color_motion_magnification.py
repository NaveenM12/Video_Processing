# export PYTHONPATH=$PYTHONPATH:/Users/naveenmirapuri/VideoProcessing

import cv2
import numpy as np
import matplotlib
# Use Agg backend to avoid requiring a GUI
matplotlib.use('Agg')
# Set a figure limit to avoid memory issues
matplotlib.rcParams['figure.max_open_warning'] = 10
# Set larger default figure size to avoid tiny plots
matplotlib.rcParams['figure.figsize'] = [10, 6]
matplotlib.rcParams['figure.dpi'] = 100
import matplotlib.pyplot as plt
from Face_Motion_Magnification.face_region_motion_magnification import FacialPhaseMagnification
from Face_Color_Magnification.face_region_color_magnification import FacialColorMagnification
import os
from io import BytesIO
from PIL import Image
from scipy import signal

class CombinedFacialMagnification:
    def __init__(self):
        """Initialize both motion and color magnification processors"""
        self.motion_processor = FacialPhaseMagnification()
        self.color_processor = FacialColorMagnification()
    
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
            
            # Format the region name for display
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
            
            # Use scipy.signal.butter, not calling butter on the signal variable
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
            
    def create_bpm_plot(self, bpm_data, signal_data, frame_idx, plot_width, plot_height):
        """
        Legacy method to create a combined BPM plot - kept for backward compatibility
        Now using create_heart_rate_plot and create_pulse_signal_plot separately
        """
        try:
            # Create figure
            fig = plt.figure(figsize=(10, 8), dpi=100, facecolor='white')
            
            # Check if we have valid BPM data
            if bpm_data is None:
                # If no BPM data, create an informational plot
                ax = fig.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, "BPM data not available\nCheck that color region detection is working",
                      horizontalalignment='center', verticalalignment='center',
                      fontsize=14, color='red', transform=ax.transAxes)
                ax.set_title("Heart Rate Estimation", fontsize=14, fontweight='bold')
                ax.axis('off')
            else:
                inst_bpm, avg_bpm, combined_signal = bpm_data
                
                # Create two subplots - BPM on top, raw signal at bottom
                ax1 = fig.add_subplot(2, 1, 1)  # BPM plot
                ax2 = fig.add_subplot(2, 1, 2)  # Raw signal plot
                
                # Get max length
                max_len = len(inst_bpm)
                
                # Don't try to process if frame_idx is beyond available data
                if frame_idx >= max_len:
                    frame_idx = max_len - 1
                    
                # Time points
                x = np.arange(max_len)
                
                # Plot BPM data
                ax1.plot(x, avg_bpm, '-', color='red', linewidth=2.5, label='Average BPM')
                ax1.plot(x, inst_bpm, '-', color='gray', alpha=0.6, linewidth=1, label='Instantaneous BPM')
                
                # Add current frame indicator
                if frame_idx > 0 and frame_idx < len(avg_bpm):
                    ax1.axvline(x=frame_idx, color='green', linestyle='--', alpha=0.7)
                    # Add a marker for current BPM
                    current_bpm = avg_bpm[frame_idx]
                    if current_bpm > 0:
                        ax1.plot(frame_idx, current_bpm, 'o', color='lime', markersize=12, 
                              markeredgecolor='black', markeredgewidth=2, zorder=10)
                        ax1.text(frame_idx+5, current_bpm, f"{current_bpm:.1f} BPM", 
                              fontsize=12, color='black', fontweight='bold',
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
                
                # Set BPM plot properties
                ax1.set_title("Heart Rate Estimation", fontsize=14, fontweight='bold')
                ax1.set_ylabel("BPM", fontsize=12)
                ax1.set_xlim(0, max_len-1)
                ax1.set_ylim(40, 140)  # Typical heart rate range
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.legend(loc='upper right')
                
                # Plot raw color signal
                ax2.plot(x, combined_signal, '-', color='green', linewidth=1.5)
                
                # Add current frame indicator
                if frame_idx > 0 and frame_idx < len(combined_signal):
                    ax2.axvline(x=frame_idx, color='green', linestyle='--', alpha=0.7)
                    ax2.plot(frame_idx, combined_signal[frame_idx], 'o', color='lime', markersize=10, 
                          markeredgecolor='black', markeredgewidth=1.5, zorder=10)
                
                # Set signal plot properties
                ax2.set_title("Blood Volume Pulse Signal", fontsize=14, fontweight='bold')
                ax2.set_xlabel("Frame Number", fontsize=12)
                ax2.set_ylabel("Amplitude", fontsize=12)
                ax2.set_xlim(0, max_len-1)
                ax2.grid(True, linestyle='--', alpha=0.7)
            
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
            print(f"Error creating BPM plot for frame {frame_idx}: {str(e)}")
            blank = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
            cv2.putText(blank, f"Error: {str(e)[:50]}", 
                      (20, plot_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return blank

    def process_video(self, input_path: str, output_path: str, alpha_color: float = 0.5, plot_dir: str = None):
        """Process video with both motion and color magnification"""
        # Set default plot directory if not provided
        if plot_dir is None:
            plot_dir = os.path.join(os.path.dirname(output_path), 'phase_plots')
            
        # Read input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate slow motion fps (0.5x speed)
        slowmo_fps = fps / 2
        
        # Define regions to process
        motion_regions = ['left_eye', 'right_eye', 'nose_tip']  # Removed 'mouth'
        color_regions = ['left_cheek', 'right_cheek']  # Removed 'forehead'
        
        # Make the video smaller and graphs larger
        # Reduce video size to 1/3 of original height
        video_display_height = height // 3
        video_display_width = int((width * video_display_height) / height)
        
        # Configure graph layout
        plot_size = 350  # Increased from 250 - Size of each square plot (both width and height)
        plots_per_row = 2  # Two plot pairs per row
        
        # Calculate plot sizes 
        # Each region gets TWO side-by-side plots (raw + diff)
        total_plot_width = width  # Use full video width
        plot_pair_width = total_plot_width // plots_per_row  # Width for each pair of plots
        single_plot_width = plot_pair_width // 2  # Width for a single plot
        
        # Calculate grid layout based on number of regions
        # +1 for the BPM plot
        all_regions = motion_regions  # Only motion regions get individual plots
        num_regions = len(all_regions)  # Motion regions get individual plots
        num_rows = (num_regions + plots_per_row - 1) // plots_per_row  # Ceiling division for motion regions
        num_rows += 1  # Add one more row for the BPM plots
        
        # Total height needed for all plots
        total_plot_height = num_rows * plot_size
        
        # Calculate dimensions for combined video (video + plots grid)
        combined_height = video_display_height + total_plot_height
        
        # Print debug info
        print(f"Input video: {input_path}")
        print(f"Original video dimensions: {width}x{height}, FPS: {fps}")
        print(f"Display video dimensions: {video_display_width}x{video_display_height}")
        print(f"Combined output dimensions: {width}x{combined_height}")
        
        # Use simple MP4V codec for maximum compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create video writers
        out = None
        out_video_only = None
        
        try:
            # Main output with plots
            out = cv2.VideoWriter(output_path, fourcc, slowmo_fps, (width, combined_height))
            if not out.isOpened():
                raise RuntimeError(f"Failed to create video writer for {output_path}")
                
            # Video-only output
            magnified_only_path = output_path.replace('.mp4', '_video_only.mp4')
            out_video_only = cv2.VideoWriter(magnified_only_path, fourcc, slowmo_fps, (width, height))
            if not out_video_only.isOpened():
                raise RuntimeError(f"Failed to create video writer for {magnified_only_path}")
                
            print("Successfully created video writers")
        except Exception as e:
            print(f"Error creating video writers: {e}")
            if out is not None and out.isOpened():
                out.release()
            if out_video_only is not None and out_video_only.isOpened():
                out_video_only.release()
            return
        
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
        
        cap.release()
        
        if len(all_frames) == 0:
            print("Error: No frames were read from the input video")
            if out is not None and out.isOpened():
                out.release()
            if out_video_only is not None and out_video_only.isOpened():
                out_video_only.release()
            return
            
        print(f"Successfully read {len(all_frames)} frames")
        
        # Initialize output frames with original frames
        output_frames = all_frames.copy()
        
        # Dictionary to store phase changes for motion regions
        all_phase_changes = {}
        
        # Dictionary to store color changes for color regions
        all_color_changes = {}
        
        print("Processing motion magnification...")
        # Process motion magnification
        motion_processor = FacialPhaseMagnification()
        face_motion_results = motion_processor.face_detector.detect_faces(all_frames[0])
        
        # Check if face detection is working properly
        print(f"Face detection found {len(face_motion_results) if face_motion_results else 0} faces in the first frame")
        if face_motion_results:
            # Report detected regions in first frame only (avoid log spam)
            for face_idx, face in enumerate(face_motion_results):
                detected_regions = list(face['regions'].keys())
                print(f"Face {face_idx+1} has regions: {detected_regions}")

            for face_idx in range(len(face_motion_results)):
                # Process defined motion regions (except mouth which is commented out)
                for region_name in motion_regions:
                    print(f"Processing motion for {region_name} in face {face_idx + 1}...")
                    region_frames = []
                    face_data = []
                    
                    for frame_idx, frame in enumerate(all_frames):
                        faces = motion_processor.face_detector.detect_faces(frame)
                        if faces and len(faces) > face_idx:
                            face = faces[face_idx]
                            if region_name in face['regions']:
                                region_frames.append(face['regions'][region_name]['image'])
                                face_data.append((face, frame_idx))
                            elif frame_idx == 0:  # Only log for the first frame to avoid spam
                                print(f"WARNING: {region_name} not found in face {face_idx+1} for frame {frame_idx}")
                    
                    print(f"Collected {len(region_frames)} frames for {region_name}")
                    if region_frames:
                        print(f"Processing motion magnification for {region_name}...")
                        magnified_frames, phase_changes = motion_processor.phase_magnifier.magnify(region_frames)
                        print(f"Completed motion magnification for {region_name}")
                        
                        # Store phase changes for this region
                        region_key = f"face{face_idx+1}_{region_name}"
                        all_phase_changes[region_key] = phase_changes
                        print(f"Added phase changes for {region_key}: {len(phase_changes)} data points")
                        
                        print(f"Applying magnification to {len(magnified_frames)} frames for {region_name}...")
                        frame_counter = 0
                        for (face, frame_idx), magnified in zip(face_data, magnified_frames):
                            region_info = face['regions'][region_name]
                            bounds = region_info['bounds']
                            original_size = region_info['original_size']
                            
                            if magnified.shape[:2] != (original_size[1], original_size[0]):
                                magnified = cv2.resize(magnified, 
                                                     (original_size[0], original_size[1]))
                            
                            # Directly apply motion magnification to output frames
                            output_frames[frame_idx][bounds[1]:bounds[3], 
                                                   bounds[0]:bounds[2]] = magnified
                            
                            # Show progress occasionally
                            frame_counter += 1
                            if frame_counter % 50 == 0:
                                print(f"Applied {frame_counter}/{len(magnified_frames)} frames for {region_name}")
                        
                        print(f"Completed applying magnification for {region_name}")
                    else:
                        print(f"WARNING: No frames collected for {region_name}")
        
        # Generate static phase change plots if we have data
        if all_phase_changes and plot_dir:
            print("Generating phase change plots...")
            motion_processor.plot_phase_changes(all_phase_changes, plot_dir)
        
        print("Processing color magnification...")
        # Process color magnification
        color_processor = FacialColorMagnification()
        face_color_results = color_processor.face_detector.detect_faces(all_frames[0])
        
        if face_color_results:
            for face_idx in range(len(face_color_results)):
                # Process only cheek regions for color (forehead is commented out)
                for region_name in color_regions:
                    print(f"Processing color for {region_name} in face {face_idx + 1}...")
                    region_frames = []
                    face_data = []
                    
                    for frame_idx, frame in enumerate(all_frames):
                        faces = color_processor.face_detector.detect_faces(frame)
                        if faces and len(faces) > face_idx:
                            face = faces[face_idx]
                            if region_name in face['regions']:
                                region_frames.append(face['regions'][region_name]['image'])
                                face_data.append((face, frame_idx))
                    
                    if region_frames:
                        # Calculate color changes for this region
                        print(f"Processing color changes for {region_name}...")
                        color_changes = self.calculate_color_changes(region_frames)
                        print(f"Completed processing color changes for {region_name}")
                        
                        # Store color changes for this region
                        region_key = f"face{face_idx+1}_{region_name}"
                        all_color_changes[region_key] = color_changes
                        
                        # Magnify the region
                        print(f"Applying color magnification to {len(region_frames)} frames for {region_name}...")
                        magnified_frames = color_processor.color_magnifier.magnify(region_frames)
                        print(f"Completed color magnification for {region_name}")
                        
                        frame_counter = 0
                        for (face, frame_idx), magnified in zip(face_data, magnified_frames):
                            region_info = face['regions'][region_name]
                            bounds = region_info['bounds']
                            original_size = region_info['original_size']
                            
                            magnified_resized = cv2.resize(magnified, 
                                                         (original_size[0], original_size[1]))
                            
                            # Blend only color magnification with already motion-magnified frames
                            region = output_frames[frame_idx][bounds[1]:bounds[3], 
                                                            bounds[0]:bounds[2]]
                            blended_region = cv2.addWeighted(
                                magnified_resized, alpha_color,
                                region, 1.0 - alpha_color,
                                0
                            )
                            output_frames[frame_idx][bounds[1]:bounds[3], 
                                                   bounds[0]:bounds[2]] = blended_region
                            
                            # Show progress occasionally
                            frame_counter += 1
                            if frame_counter % 50 == 0:
                                print(f"Applied {frame_counter}/{len(magnified_frames)} frames for {region_name}")
                        
                        print(f"Completed applying color magnification for {region_name}")
        
                '''
                # Commented out forehead processing
                region_name = 'forehead'
                print(f"Processing color for {region_name} in face {face_idx + 1}...")
                region_frames = []
                face_data = []
                
                for frame_idx, frame in enumerate(all_frames):
                    faces = color_processor.face_detector.detect_faces(frame)
                    if faces and len(faces) > face_idx:
                        face = faces[face_idx]
                        if region_name in face['regions']:
                            region_frames.append(face['regions'][region_name]['image'])
                            face_data.append((face, frame_idx))
                
                if region_frames:
                    magnified_frames = color_processor.color_magnifier.magnify(region_frames)
                    
                    for (face, frame_idx), magnified in zip(face_data, magnified_frames):
                        region_info = face['regions'][region_name]
                        bounds = region_info['bounds']
                        original_size = region_info['original_size']
                        
                        magnified_resized = cv2.resize(magnified, 
                                                     (original_size[0], original_size[1]))
                        
                        # Blend only color magnification with already motion-magnified frames
                        region = output_frames[frame_idx][bounds[1]:bounds[3], 
                                                        bounds[0]:bounds[2]]
                        blended_region = cv2.addWeighted(
                            magnified_resized, alpha_color,
                            region, 1.0 - alpha_color,
                            0
                        )
                        output_frames[frame_idx][bounds[1]:bounds[3], 
                                               bounds[0]:bounds[2]] = blended_region
                '''
        
        # Calculate BPM from color signals if we have data
        bpm_data = None
        if all_color_changes:
            print("Calculating heart rate (BPM) from color signals...")
            try:
                bpm_data = self.calculate_bpm(all_color_changes, fps=fps)
                if bpm_data is None:
                    print("Failed to calculate BPM. Will not show BPM plot.")
            except Exception as e:
                print(f"Error calculating BPM: {str(e)}")
                bpm_data = None
        
        # Print debug info about detected phase changes
        print("Detected phase changes for these regions:")
        for region_key in all_phase_changes.keys():
            print(f"  - {region_key}: {len(all_phase_changes[region_key])} data points")
        
        # Prepare the grid layout
        print("Setting up plot grid layout...")
        # Collect all region keys from face1 (we assume one face for simplicity)
        face1_regions = [key for key in all_phase_changes.keys() if key.startswith('face1_')]
        
        print(f"Face1 regions found: {face1_regions}")
        
        # Prioritize the order: eyes, nose
        ordered_regions = []
        # Explicitly check for each expected region to ensure they're included if present
        expected_regions = [f"face1_{region}" for region in motion_regions]
        for expected_region in expected_regions:
            if expected_region in all_phase_changes:
                ordered_regions.append(expected_region)
                print(f"Added region to display: {expected_region}")
            else:
                print(f"WARNING: Expected region {expected_region} not found in phase data")
        
        # Add any remaining regions
        for region in face1_regions:
            if region not in ordered_regions:
                ordered_regions.append(region)
                print(f"Added additional region: {region}")
        
        print(f"Final ordered regions for display: {ordered_regions}")
        
        print("Writing output video with synchronized plots...")
        # Create combined frames (video + plots) and write to output
        try:
            frame_count = 0
            for i, frame in enumerate(output_frames):
                # Ensure frame is the right shape and type
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    print(f"Warning: Frame {i} has unexpected shape {frame.shape}, skipping")
                    continue
                
                # Ensure the frame is exactly the expected dimensions
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))
                
                # Ensure the frame is uint8
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                
                # Write magnified video only 
                out_video_only.write(frame)
                
                # Create a fresh combined frame with exactly the right dimensions
                combined_frame = np.zeros((combined_height, width, 3), dtype=np.uint8)
                
                # Resize and center the video frame at the top
                resized_frame = cv2.resize(frame, (video_display_width, video_display_height))
                
                # Calculate x offset to center the video
                x_offset = (width - video_display_width) // 2
                
                # Copy the resized video frame to the top part (centered)
                combined_frame[:video_display_height, x_offset:x_offset+video_display_width, :] = resized_frame
                
                # Fill in remaining top area with black background
                # This is already handled by initializing combined_frame with zeros
                
                # Create the grid of plots below the video
                # First, add motion region plots (eyes and nose in proper positions)
                # Calculate total plots needed for motion regions
                eye_plots_count = 0  # Count of plots for eye regions
                nose_plot_found = False
                # Pre-count to determine positions
                for region_name in ordered_regions:
                    if 'eye' in region_name:
                        eye_plots_count += 1
                    elif 'nose_tip' in region_name:
                        nose_plot_found = True
                
                # Process motion region plots
                for idx, region_name in enumerate(ordered_regions):
                    # Special positioning logic
                    if 'eye' in region_name:
                        # Eyes go in the first row
                        row = 0
                        # Left eye on left, right eye on right
                        col = 0 if 'left' in region_name else 1
                    elif 'nose_tip' in region_name:
                        # Nose goes in second row, left side
                        row = 1
                        col = 0
                    else:
                        # Default positioning for any other regions
                        row = idx // plots_per_row
                        col = idx % plots_per_row
                    
                    # Calculate pixel position in the combined frame
                    y_start = video_display_height + (row * plot_size)
                    y_end = y_start + plot_size
                    x_start = col * plot_pair_width
                    x_end = x_start + plot_pair_width
                    
                    # Generate side-by-side plots for this region with current frame indicator
                    plot_pair = self.create_region_plot_pair(
                        all_phase_changes, i, region_name, 
                        single_plot_width, plot_size
                    )
                    
                    # Debug information to track missing nose plot (only show every 50th frame to avoid spam)
                    if region_name.endswith('nose_tip') and i % 50 == 0:
                        if region_name not in all_phase_changes:
                            print(f"Frame {i} - No nose_tip data in all_phase_changes")
                        else:
                            print(f"Frame {i} - Displaying nose_tip plots at position: row={row}, col={col}, y={y_start}-{y_end}, x={x_start}-{x_end}")
                    
                    # Add to combined frame at the right position
                    if y_end <= combined_height and x_end <= width:
                        try:
                            combined_frame[y_start:y_end, x_start:x_end, :] = plot_pair
                        except ValueError as e:
                            print(f"Error placing plot in grid at position ({row},{col}): {e}")
                
                # Add BPM plots in the bottom right corner
                if bpm_data:
                    # Heart rate estimation goes in second row, right side
                    # Only if nose tip is found, otherwise default positioning
                    if nose_plot_found:
                        # Position heart rate graphs in second row, right position (next to nose tip)
                        hr_row = 1
                        hr_col = 1
                        
                        # Position for Heart Rate Estimation
                        y_start = video_display_height + (hr_row * plot_size)
                        y_end = y_start + plot_size
                        x_start = hr_col * plot_pair_width
                        x_end = x_start + plot_pair_width
                        
                        # Generate BPM plots (heart rate and pulse signal side by side)
                        heart_rate_plot = self.create_heart_rate_plot(
                            bpm_data, i,
                            single_plot_width, plot_size
                        )
                        
                        pulse_signal_plot = self.create_pulse_signal_plot(
                            bpm_data, i,
                            single_plot_width, plot_size
                        )
                        
                        # Combine the two plots side by side
                        combined_heart_plots = np.hstack([heart_rate_plot, pulse_signal_plot])
                        
                        # Add the combined heart plots to the frame
                        if y_end <= combined_height and x_end <= width:
                            try:
                                combined_frame[y_start:y_end, x_start:x_end, :] = combined_heart_plots
                                if i % 50 == 0:
                                    print(f"Frame {i} - Placed heart rate plots at position: row={hr_row}, col={hr_col}, y={y_start}-{y_end}, x={x_start}-{x_end}")
                            except ValueError as e:
                                print(f"Error placing heart plots: {e}")
                                print(f"Heart plots shape: {combined_heart_plots.shape}, Target area: {y_end-y_start}x{x_end-x_start}")
                    else:
                        # Fall back to default positioning if no nose tip
                        bpm_row = (len(ordered_regions) + plots_per_row - 1) // plots_per_row
                        y_start = video_display_height + (bpm_row * plot_size)
                        y_end = y_start + plot_size
                        
                        # Position for Heart Rate Estimation (left half)
                        x_start = 0
                        x_end = width // 2
                        
                        # Generate BPM plot for heart rate
                        heart_rate_plot = self.create_heart_rate_plot(
                            bpm_data, i,
                            width // 2, plot_size
                        )
                        
                        # Add Heart Rate plot to left side
                        if y_end <= combined_height and x_end <= width:
                            try:
                                combined_frame[y_start:y_end, x_start:x_end, :] = heart_rate_plot
                            except ValueError as e:
                                print(f"Error placing heart rate plot: {e}")
                        
                        # Position for Blood Volume Pulse Signal (right half)
                        x_start = width // 2
                        x_end = width
                        
                        # Generate BPM plot for blood volume pulse
                        pulse_signal_plot = self.create_pulse_signal_plot(
                            bpm_data, i,
                            width // 2, plot_size
                        )
                        
                        # Add Pulse Signal plot to right side
                        if y_end <= combined_height and x_end <= width:
                            try:
                                combined_frame[y_start:y_end, x_start:x_end, :] = pulse_signal_plot
                            except ValueError as e:
                                print(f"Error placing pulse signal plot: {e}")
                else:
                    # Display a message if BPM data is not available
                    bpm_row = (len(ordered_regions) + plots_per_row - 1) // plots_per_row
                    y_start = video_display_height + (bpm_row * plot_size)
                    y_end = min(y_start + plot_size, combined_height)
                    
                    # Create a blank image with text
                    blank = np.ones((plot_size, width, 3), dtype=np.uint8) * 255
                    cv2.putText(blank, "BPM data not available", 
                              (width//4, plot_size//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    
                    # Add to combined frame
                    if y_end <= combined_height:
                        combined_frame[y_start:y_end, :, :] = blank
                
                # Ensure the combined frame is uint8
                if combined_frame.dtype != np.uint8:
                    combined_frame = combined_frame.astype(np.uint8)
                
                # Write the combined output
                out.write(combined_frame)
                
                frame_count += 1
                if i % 50 == 0:  # Changed from 10 to 50 to reduce log spam
                    print(f"Written {i}/{len(output_frames)} frames")
                    # Do a reality check on file size periodically
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        print(f"Current output file size: {file_size/1024/1024:.2f} MB")
                    
                # Clean up open figures periodically to avoid memory issues
                if i % 30 == 0:
                    plt.close('all')
            
            # Final count
            print(f"Total frames written: {frame_count}")
            
        except Exception as e:
            print(f"Error during video writing: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Ensure resources are released
            if out is not None:
                out.release()
            if out_video_only is not None:
                out_video_only.release()
            plt.close('all')  # Close all matplotlib figures
            
            # Verify the output files exist and have content
            for file_path in [output_path, magnified_only_path]:
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    print(f"Output file: {file_path}")
                    print(f"File size: {size} bytes ({size/1024/1024:.2f} MB)")
                    if size < 1000:
                        print("WARNING: Output file is suspiciously small!")
                    else:
                        print(f"Successfully created output video")
                else:
                    print(f"WARNING: Output file {file_path} does not exist!")
            
        print("Processing complete!")

if __name__ == "__main__":
    # Define input and output paths
    input_video_path = "Videos_Misc/Converted_MP4_Videos/00038 39.mp4"  # Use a known existing path
    output_video_path = "Combined_Color_Motion_Magnification/output_videos/TEST_00038_39.mp4"
    plot_dir = "Combined_Color_Motion_Magnification/output_videos/phase_plots"
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Process the video
    processor = CombinedFacialMagnification()
    processor.process_video(input_video_path, output_video_path, plot_dir=plot_dir)