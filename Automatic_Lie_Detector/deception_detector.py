"""
Deception detector module that implements an unsupervised anomaly detection approach
to identify potential deception in videos based on micro-expressions and heart rate patterns.
"""

import numpy as np
import cv2
import os
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import IsolationForest
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import io
from PIL import Image
from matplotlib.lines import Line2D

# Set up matplotlib for non-interactive use
matplotlib.use('Agg')

# Import from our configuration
import config

class DeceptionDetector:
    """
    Implements a specialized PBM-focused deception detection method.
    This class prioritizes phase-based micro-expressions as the primary indicator,
    with heart rate from EVM as optional secondary confirmation.
    """
    
    def __init__(self, detection_params=None):
        """
        Initialize the deception detector with specified parameters.
        
        Args:
            detection_params: Optional custom detection parameters
        """
        # Use provided parameters or default from config
        self.params = detection_params or config.DETECTION_PARAMS
        
        # Store video metadata
        self.fps = 30  # Default, will be updated with actual video fps
        self.total_frames = 0
        
        # Store detection results
        self.windows = []               # All analysis windows 
        self.deception_windows = []     # Selected deception window(s)
        self.peak_regions = []          # Regions with significant peaks
        
        # Init state for later use
        self.micro_expression_data = None
        self.heart_rate_data = None
    
    def fit(self, phase_changes: Dict, heart_rate_data: Optional[Tuple], fps: float, total_frames: int):
        """
        Analyze phase changes from PBM and optionally heart rate from EVM
        to identify potential deception windows.
        
        This completely reworked method uses a transparent approach focused directly
        on PBM micro-expression peaks.
        
        Args:
            phase_changes: Dictionary of phase changes from PBM
            heart_rate_data: Optional tuple of heart rate data from EVM
            fps: Frames per second of the video
            total_frames: Total number of frames in the video
        """
        # Store metadata
        self.fps = fps
        self.total_frames = total_frames
        self.micro_expression_data = phase_changes
        self.heart_rate_data = heart_rate_data
        
        # Reset detection results
        self.windows = []
        self.deception_windows = []
        self.peak_regions = []
        
        # Calculate window parameters
        window_size_frames = int(self.params['window_size_seconds'] * fps)
        window_stride = int(window_size_frames * (1 - self.params['window_overlap']))
        
        # Get detection thresholds
        min_peaks = self.params.get('peak_detection', {}).get('min_peaks_required', 2)
        max_gap = self.params.get('peak_detection', {}).get('max_gap_frames', 30)
        threshold_multiplier = self.params.get('peak_detection', {}).get('threshold_multiplier', 1.3)
        
        print("\n--- DIRECT PBM-FOCUSED DECEPTION DETECTION ---")
        
        # STEP 1: Get the combined phase change data (core PBM signal)
        if 'combined' in phase_changes and len(phase_changes['combined']) > 0:
            # Use pre-computed combined data
            pbm_data = phase_changes['combined']
            print(f"Using pre-computed combined PBM data ({len(pbm_data)} frames)")
        else:
            # If no pre-computed combined data, try to find individual PBM data
            available_regions = []
            for region_name, data in phase_changes.items():
                if region_name != 'combined' and len(data) > 0:
                    available_regions.append((region_name, data))
            
            if not available_regions:
                print("ERROR: No PBM phase change data available!")
                return self
            
            # Combine available regions
            print(f"Combining {len(available_regions)} PBM regions...")
            # Use the longest available region as reference
            longest_region = max(available_regions, key=lambda x: len(x[1]))
            pbm_data = np.zeros(len(longest_region[1]))
            
            # Add each region
            for region_name, data in available_regions:
                # Resize to match if needed
                if len(data) != len(pbm_data):
                    temp = np.zeros(len(pbm_data))
                    temp[:min(len(data), len(temp))] = data[:min(len(data), len(temp))]
                    data = temp
                pbm_data += data
            
            # Average
            pbm_data /= len(available_regions)
        
        # STEP 2: Find significant micro-expression peaks across the entire PBM signal
        print(f"Analyzing PBM signal for micro-expression peaks...")
        
        # Calculate dynamic threshold based on the signal characteristics
        mean_value = np.mean(pbm_data)
        std_value = np.std(pbm_data)
        peak_threshold = mean_value + (threshold_multiplier * std_value)
        
        # Find all peaks 
        peak_mask = pbm_data > peak_threshold
        peak_indices = np.where(peak_mask)[0]
        
        if len(peak_indices) == 0:
            print("No significant micro-expression peaks found in PBM data")
            return self
            
        print(f"Found {len(peak_indices)} significant micro-expression peaks in PBM data")
        
        # STEP 3: Sliding window analysis of the entire video
        print(f"Performing sliding window analysis with window size {window_size_frames} frames...")
        
        window_centers = []
        for center_frame in range(window_size_frames//2, total_frames - window_size_frames//2, window_stride):
            window_centers.append(center_frame)
        
        # Initialize all windows with their frame ranges
        for center_frame in window_centers:
            start_frame = max(0, center_frame - window_size_frames//2) 
            end_frame = min(total_frames-1, center_frame + window_size_frames//2)
            
            window = {
                'center_frame': center_frame,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'pbm_peaks': 0,
                'pbm_peak_indices': [],
                'pbm_peak_density': 0,
                'pbm_peak_max': 0,
                'heart_rate_change': 0,
                'deception_score': 0
            }
            
            self.windows.append(window)
        
        # STEP 4: Count peaks in each window
        print("Analyzing peaks in each window...")
        
        for window in self.windows:
            # Extract window boundaries
            start_frame = window['start_frame']
            end_frame = window['end_frame']
            
            # Count peaks in this window
            window_peak_indices = [i for i in peak_indices if start_frame <= i <= end_frame]
            window_peak_count = len(window_peak_indices)
            
            # Skip if no peaks
            if window_peak_count == 0:
                continue
                
            # Store peak information
            window['pbm_peaks'] = window_peak_count
            window['pbm_peak_indices'] = window_peak_indices
            
            # Calculate peak density (higher when peaks are clustered)
            if window_peak_count > 1:
                peak_distances = np.diff(window_peak_indices)
                mean_distance = np.mean(peak_distances)
                window['pbm_peak_density'] = window_peak_count / (mean_distance + 1)
            
            # Find max peak value
            peak_values = [pbm_data[i] for i in window_peak_indices]
            window['pbm_peak_max'] = max(peak_values) if peak_values else 0
            
            # Calculate additional peak metrics for scoring
            window['clustered'] = True  # Assume clustered by default
            if window_peak_count > 1:
                max_distance = np.max(peak_distances)
                # Mark as non-clustered if peaks are too far apart
                if max_distance > max_gap:
                    window['clustered'] = False
            
            # Mark windows with significant peaks 
            if window_peak_count >= min_peaks and window['clustered']:
                self.peak_regions.append(window)
        
        # STEP 5: Analyze heart rate if available (as secondary confirmation only)
        if heart_rate_data is not None and len(heart_rate_data) >= 2:
            print("Analyzing heart rate data as secondary confirmation...")
            
            # Extract averaged heart rate signal
            _, avg_bpm, _ = heart_rate_data
            
            if len(avg_bpm) > 0:
                # Calculate heart rate changes
                for window in self.windows:
                    # Extract window boundaries
                    start_frame = window['start_frame']
                    end_frame = window['end_frame']
                    
                    # Skip if outside heart rate data range
                    if end_frame >= len(avg_bpm) or start_frame >= len(avg_bpm):
                        continue
                    
                    # Get heart rate data for this window
                    window_hr = avg_bpm[start_frame:end_frame+1]
                    
                    if len(window_hr) > 1:
                        # Calculate heart rate variability in this window
                        hr_std = np.std(window_hr)
                        hr_change = np.max(window_hr) - np.min(window_hr)
                        hr_diff = np.diff(window_hr)
                        hr_max_change = np.max(np.abs(hr_diff)) if len(hr_diff) > 0 else 0
                        
                        # Store heart rate change metric
                        window['heart_rate_change'] = max(hr_std, hr_max_change)
        
        # STEP 6: Calculate deception scores
        # Now create a completely transparent scoring function focused on PBM peaks
        print("Calculating final deception scores...")
        
        # First prioritize windows with significant peaks
        if self.peak_regions:
            print(f"Found {len(self.peak_regions)} windows with significant micro-expression peaks")
            
            # Sort peak regions by:
            # 1. Peak count (most important - more peaks means stronger evidence)
            # 2. Peak density (second - clustered peaks are more significant)
            # 3. Heart rate change (tertiary - only as confirmation)
            for window in self.peak_regions:
                # Calculate final deception score with clear weighting
                pbm_weight = self.params['feature_weights']['phase_change']
                hr_weight = self.params['feature_weights']['heart_rate']
                
                # Base score is primarily driven by peak count
                peak_score = window['pbm_peaks'] * pbm_weight
                
                # Add density bonus for clustered peaks
                density_bonus = window['pbm_peak_density'] * 0.1 * pbm_weight
                
                # Add small heart rate confirmation bonus if available
                hr_bonus = window['heart_rate_change'] * hr_weight * 0.1
                
                # Calculate final transparent score
                window['deception_score'] = peak_score + density_bonus + hr_bonus
            
            # Sort by deception score
            self.peak_regions.sort(key=lambda w: w['deception_score'], reverse=True)
            
            # Print top candidates
            print("\nTop deception window candidates:")
            for i, window in enumerate(self.peak_regions[:3]):
                print(f"Candidate {i+1}:")
                print(f"  - Frames: {window['start_frame']}-{window['end_frame']} (center: {window['center_frame']})")
                print(f"  - Peaks: {window['pbm_peaks']}")
                print(f"  - Peak density: {window['pbm_peak_density']:.2f}")
                print(f"  - Heart rate change: {window['heart_rate_change']:.2f}")
                print(f"  - Deception score: {window['deception_score']:.2f}")
            
            # Select the top peak region as the deception window
            top_window = self.peak_regions[0]
            self.deception_windows = [top_window]
            
            # Add time information
            top_window['time_info'] = {
                'start_time': top_window['start_frame'] / self.fps,
                'end_time': top_window['end_frame'] / self.fps,
                'center_time': top_window['center_frame'] / self.fps,
                'duration': (top_window['end_frame'] - top_window['start_frame']) / self.fps
            }
            
            print(f"\nSELECTED DECEPTION WINDOW:")
            print(f"  - Frames: {top_window['start_frame']}-{top_window['end_frame']} (center: {top_window['center_frame']})")
            print(f"  - {top_window['pbm_peaks']} micro-expression peaks detected")
            print(f"  - Duration: {top_window['time_info']['duration']:.2f}s")
            print(f"  - Deception score: {top_window['deception_score']:.2f}")
        else:
            print("No windows with significant micro-expression peaks found")
            self.deception_windows = []
        
        return self
    
    def get_frame_deception_score(self, frame_idx: int) -> Tuple[float, bool]:
        """
        Get the deception score for a specific frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Tuple of (normalized_score, is_deception)
        """
        if not self.deception_windows:
            return 0.0, False
        
        # Check if frame is in any deception window
        is_deception = any(
            window['start_frame'] <= frame_idx <= window['end_frame']
            for window in self.deception_windows
        )
        
        # Find matching window for this frame
        matching_windows = [
            window for window in self.windows
            if window['start_frame'] <= frame_idx <= window['end_frame']
        ]
        
        if not matching_windows:
            return 0.0, is_deception
            
        # Get max score from matching windows
        max_score = max(window.get('deception_score', 0) for window in matching_windows)
        
        # Normalize score to 0-1 range if needed
        if self.windows:
            all_scores = [w.get('deception_score', 0) for w in self.windows]
            max_possible = max(all_scores) if all_scores else 1.0
            normalized_score = max_score / max_possible if max_possible > 0 else 0.0
        else:
            normalized_score = 0.0
            
        return min(normalized_score, 1.0), is_deception
    
    def highlight_deception_regions(self, plot_img: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Highlight regions of potential deception on a plot image.
        
        Args:
            plot_img: Plot image as numpy array
            frame_idx: Current frame index
            
        Returns:
            Image with highlighted deception regions
        """
        if not self.deception_windows:
            return plot_img
        
        result = plot_img.copy()
        h, w = result.shape[:2]
        
        # Create a mask for deception regions
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Calculate timeline position
        timeline_y = int(h * 0.95)  # Position at 95% from top
        timeline_start_x = int(w * 0.05)
        timeline_end_x = int(w * 0.95)
        timeline_width = timeline_end_x - timeline_start_x
        
        # Draw deception regions on the mask
        for window in self.deception_windows:
            # Calculate positions on timeline
            start_x = timeline_start_x + int(timeline_width * (window['start_frame'] / self.total_frames))
            end_x = timeline_start_x + int(timeline_width * (window['end_frame'] / self.total_frames))
            
            # Check if current frame is in this window
            if window['start_frame'] <= frame_idx <= window['end_frame']:
                # Use a stronger highlight for the current window
                color = self.params['highlight_color']
                cv2.rectangle(result, (start_x, timeline_y-10), (end_x, timeline_y+10), color, -1)
            else:
                # Use a semi-transparent rectangle for other deception windows
                alpha = self.params['highlight_alpha']
                color = self.params['highlight_color']
                cv2.rectangle(mask, (start_x, timeline_y-10), (end_x, timeline_y+10), color, -1)
        
        # Blend the mask with the original image
        cv2.addWeighted(mask, self.params['highlight_alpha'], result, 1, 0, result)
        
        # Add a marker for the current frame position
        current_x = timeline_start_x + int(timeline_width * (frame_idx / self.total_frames))
        cv2.circle(result, (current_x, timeline_y), 5, (0, 255, 0), -1)
        
        return result
    
    def create_deception_timeline_plot(self, frame_idx: int, width: int, height: int) -> np.ndarray:
        """
        Create a deception detection timeline visualization.
        This plots the micro-expression peaks, heart rate changes, and the deception window.
        
        Args:
            frame_idx: Current frame index
            width: Plot width in pixels
            height: Plot height in pixels
            
        Returns:
            The plot as a numpy array
        """
        # Create figure with specific size
        dpi = 100  # Higher DPI for better quality
        fig_width = width / dpi
        fig_height = height / dpi
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # Plot timeline data if available
        if self.windows:
            # Extract data for plotting
            frame_centers = [w['center_frame'] for w in self.windows]
            times = [frame / self.fps for frame in frame_centers]
            peak_counts = [w.get('pbm_peaks', 0) for w in self.windows]
            scores = []
            
            # Normalize scores for visualization
            max_score = max(w.get('deception_score', 0) for w in self.windows) if self.windows else 1.0
            if max_score > 0:
                scores = [min(w.get('deception_score', 0) / max_score, 1.0) for w in self.windows]
            else:
                scores = [0.0] * len(self.windows)
            
            # Current time marker
            current_time = frame_idx / self.fps
            ax.axvline(x=current_time, color='green', linestyle='-', linewidth=3, label='Current time')
            
            # Plot the scores using a color gradient
            cmap = LinearSegmentedColormap.from_list("custom", [(0, 'green'), (0.5, 'yellow'), (1, 'red')])
            sc = ax.scatter(times, scores, c=scores, cmap=cmap, s=80, alpha=0.8, label='Deception score')
            
            # Connect points with a line for better visualization
            ax.plot(times, scores, '-', color='#888888', alpha=0.3, linewidth=1.5)
            
            # Mark windows with significant peaks using diamond markers
            peak_times = []
            peak_scores = []
            
            for i, count in enumerate(peak_counts):
                # Mark windows with 2+ peaks (matches min_peaks_required default)
                if count >= 2:
                    peak_times.append(times[i])
                    peak_scores.append(scores[i])
                    
                    # Add text label showing exact peak count
                    ax.text(times[i], scores[i] + 0.05, f"{count}", 
                           ha='center', va='bottom', fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Plot diamond markers for windows with significant peaks
            if peak_times:
                ax.scatter(peak_times, peak_scores, color='yellow', marker='D', s=160, 
                          edgecolor='black', linewidth=1.5, alpha=0.9, label='Significant peaks')
            
            # Highlight the selected deception region with a RED rectangle
            if self.deception_windows:
                for window in self.deception_windows:
                    if 'time_info' in window:
                        start_time = window['time_info']['start_time']
                        end_time = window['time_info']['end_time']
                        
                        # Add RED deception highlight rectangle
                        rect = plt.Rectangle((start_time, 0), end_time - start_time, 1.0, 
                                            color='red', alpha=0.5, label='Deception region')
                        ax.add_patch(rect)
                        
                        # Get peak count and add a label
                        peak_count = window.get('pbm_peaks', 0)
                        if peak_count > 0:
                            ax.text((start_time + end_time) / 2, 0.7, f"{peak_count} peaks", 
                                   ha='center', va='center', fontsize=16, fontweight='bold',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                                            edgecolor='red', linewidth=2))
            
            # Configure plot appearance
            ax.set_xlabel('Time (seconds)', fontsize=14)
            ax.set_ylabel('Normalized Score (0-1)', fontsize=14)
            ax.set_title('Deception Detection Timeline', fontsize=18, fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.set_xlim(0, times[-1] if times else 15)  # End at last frame or 15s if empty
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Create custom legend
            legend_elements = [
                Line2D([0], [0], color='green', lw=3, label='Current time'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Low score'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Medium score'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='High score'),
                Line2D([0], [0], marker='D', color='w', markerfacecolor='yellow', markersize=10, label='Significant peaks'),
                plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.5, label='Deception region')
            ]
            
            # Add the legend
            ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        else:
            ax.text(0.5, 0.5, "No detection data available", 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
        
        # Draw the figure
        fig.tight_layout()
        fig.canvas.draw()
        
        # Convert to numpy array
        buffer = fig.canvas.tostring_argb()
        img = np.frombuffer(buffer, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        
        # Convert ARGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Close figure to free memory
        plt.close(fig)
        
        return img 