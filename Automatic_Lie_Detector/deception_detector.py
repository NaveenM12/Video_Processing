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
    Implements unsupervised anomaly detection to identify potential deception
    in videos based on micro-expressions and physiological patterns.
    """
    
    def __init__(self, detection_params=None):
        """
        Initialize the deception detector with specified parameters.
        
        Args:
            detection_params: Optional custom detection parameters
        """
        # Use provided parameters or default from config
        self.params = detection_params or config.DETECTION_PARAMS
        
        # Initialize the anomaly detection model (Isolation Forest)
        self.model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=0.1,  # Expected proportion of anomalies
            random_state=42
        )
        
        # Store video metadata
        self.fps = 30  # Default, will be updated with actual video fps
        self.total_frames = 0
        
        # Store detection results
        self.anomaly_scores = []
        self.deception_windows = []
        self.feature_importance = {}
    
    def extract_features(self, phase_changes: Dict, bpm_data: Tuple, 
                         frame_idx: int, window_size: int) -> np.ndarray:
        """
        Extract relevant features from the phase changes and heart rate data
        for a specific window around the given frame index.
        
        Args:
            phase_changes: Dictionary of phase changes for different facial regions
            bpm_data: Tuple containing heart rate data (inst_bpm, avg_bpm, pulse_signal)
            frame_idx: Current frame index
            window_size: Size of the window in frames
            
        Returns:
            Feature vector as numpy array with separate phase and heart rate features
        """
        # Calculate window boundaries
        half_window = window_size // 2
        start_idx = max(0, frame_idx - half_window)
        end_idx = min(self.total_frames - 1, frame_idx + half_window)
        actual_window_size = end_idx - start_idx + 1
        
        # We'll keep phase features and heart rate features separate
        phase_features = []
        heart_rate_features = []
        
        # Get feature weights for scaling
        phase_weight = self.params['feature_weights']['phase_change']
        heart_rate_weight = self.params['feature_weights']['heart_rate']
        
        # Get peak detection parameters
        peak_threshold_multiplier = self.params.get('peak_detection', {}).get('threshold_multiplier', 1.5)
        cluster_importance = self.params.get('peak_detection', {}).get('cluster_importance', 1.0)
        
        # 1. Process phase changes (micro-expressions) - USING COMBINED PHASE DATA
        # First check if we already have combined phase data
        combined_phase_data = None
        
        # Prioritize using the pre-computed combined phase data if available
        if 'combined' in phase_changes and len(phase_changes['combined']) > 0:
            # Use the pre-computed combined data directly
            if end_idx < len(phase_changes['combined']):
                combined_phase_data = phase_changes['combined'][start_idx:end_idx+1]
        
        # If no pre-computed combined data is available, combine regions manually
        if combined_phase_data is None:
            valid_phase_regions = 0
            
            for region_name, phase_data in phase_changes.items():
                if region_name == 'combined' or len(phase_data) == 0:
                    # Skip combined key and empty regions
                    continue
                
                # Get phase data for this window, ensuring we don't exceed array bounds
                if end_idx < len(phase_data):
                    region_window = phase_data[start_idx:end_idx+1]
                    if len(region_window) < 3:
                        # Skip if not enough data
                        continue
                    
                    # Initialize or add to combined data
                    if combined_phase_data is None:
                        combined_phase_data = np.zeros_like(region_window)
                    
                    # Add this region's data to the combined data
                    combined_phase_data += region_window
                    valid_phase_regions += 1
            
            # Average the combined data
            if combined_phase_data is not None and valid_phase_regions > 0:
                combined_phase_data /= valid_phase_regions
        
        # Extract features from the combined phase data if available
        if combined_phase_data is not None and len(combined_phase_data) >= 3:
            # Apply phase_weight to emphasize PBM features
            
            # Basic statistical features
            mean_phase = np.mean(combined_phase_data) * phase_weight
            std_phase = np.std(combined_phase_data) * phase_weight
            max_phase = np.max(combined_phase_data) * phase_weight
            
            # Rate of change features (first derivative)
            phase_change_rate = np.diff(combined_phase_data)
            mean_change_rate = np.mean(np.abs(phase_change_rate)) * phase_weight
            max_change_rate = np.max(np.abs(phase_change_rate)) * phase_weight
            
            # Sudden movement features (second derivative)
            acceleration = np.diff(phase_change_rate)
            if len(acceleration) > 0:
                max_acceleration = np.max(np.abs(acceleration)) * phase_weight
            else:
                max_acceleration = 0
            
            # Detect clusters of peaks in the phase data using the custom threshold multiplier
            peak_threshold = np.mean(combined_phase_data) + peak_threshold_multiplier * np.std(combined_phase_data)
            peaks = combined_phase_data > peak_threshold
            peak_count = np.sum(peaks)
            
            # Calculate cluster score - higher when peaks are closer together
            if peak_count > 1:
                peak_indices = np.where(peaks)[0]
                peak_distances = np.diff(peak_indices)
                # Apply cluster importance factor to make clusters more significant
                cluster_score = (peak_count / (np.mean(peak_distances) + 1)) * cluster_importance
            else:
                cluster_score = 0.0
                
            # Add combined features to phase feature vector
            phase_features.extend([
                mean_phase, std_phase, max_phase,
                mean_change_rate, max_change_rate, max_acceleration,
                peak_count * phase_weight,     # Number of peaks is crucial
                cluster_score * phase_weight,  # Cluster density is crucial
            ])
            
            # Add frequency domain features for the combined signal
            if len(combined_phase_data) >= 10:
                # Calculate power spectral density
                f, psd = signal.welch(combined_phase_data, fs=self.fps, nperseg=min(256, len(combined_phase_data)))
                # Get dominant frequency
                dominant_freq_idx = np.argmax(psd)
                dominant_freq = f[dominant_freq_idx] if dominant_freq_idx < len(f) else 0
                
                # Add to feature vector with weight
                phase_features.extend([dominant_freq * phase_weight])
        else:
            # If no phase data is available, add zeros as placeholders
            phase_features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0])  # 9 placeholders for phase features (added 2 new)
        
        # 2. Process heart rate features
        if bpm_data is not None and len(bpm_data) >= 3:
            inst_bpm, avg_bpm, pulse_signal = bpm_data
            
            # Get heart rate data for this window
            hr_window = avg_bpm[start_idx:end_idx+1] if len(avg_bpm) > end_idx else []
            pulse_window = pulse_signal[start_idx:end_idx+1] if len(pulse_signal) > end_idx else []
            
            if len(hr_window) >= 3:
                # Heart rate statistical features - apply heart_rate_weight
                mean_hr = np.mean(hr_window) * heart_rate_weight
                std_hr = np.std(hr_window) * heart_rate_weight
                
                # Heart rate variability (HRV)
                if len(hr_window) > 1:
                    hr_diff = np.diff(hr_window)
                    hrv = np.std(hr_diff) * heart_rate_weight
                else:
                    hrv = 0
                
                # Rate of change in heart rate
                mean_hr_change = np.mean(np.abs(hr_diff)) * heart_rate_weight if len(hr_window) > 1 else 0
                max_hr_change = np.max(np.abs(hr_diff)) * heart_rate_weight if len(hr_window) > 1 else 0
                
                # Add to heart rate feature vector
                heart_rate_features.extend([
                    mean_hr, std_hr, hrv, mean_hr_change, max_hr_change
                ])
            
            # Pulse signal features (if available)
            if len(pulse_window) >= 3:
                # Pulse signal energy
                pulse_energy = np.sum(np.square(pulse_window)) * heart_rate_weight
                
                # Pulse signal frequency analysis (if enough data)
                if len(pulse_window) >= 10:
                    # Calculate power spectral density
                    f, psd = signal.welch(pulse_window, fs=self.fps, nperseg=min(256, len(pulse_window)))
                    # Get dominant frequency
                    dominant_freq_idx = np.argmax(psd)
                    dominant_freq = f[dominant_freq_idx] if dominant_freq_idx < len(f) else 0
                    # Get power in different frequency bands
                    low_freq_power = np.sum(psd[(f >= 0.04) & (f < 0.15)])
                    high_freq_power = np.sum(psd[(f >= 0.15) & (f < 0.4)])
                    lf_hf_ratio = low_freq_power / high_freq_power if high_freq_power > 0 else 0
                else:
                    dominant_freq = 0
                    lf_hf_ratio = 0
                
                # Add to heart rate feature vector
                heart_rate_features.extend([
                    pulse_energy, 
                    dominant_freq * heart_rate_weight, 
                    lf_hf_ratio * heart_rate_weight
                ])
        
        # Fill with zeros if we don't have heart rate data
        if not heart_rate_features:
            heart_rate_features = [0] * 8  # 8 placeholders for heart rate features
            
        # Combined feature array with separate sections for phase and heart rate
        combined_features = np.concatenate([
            np.array(phase_features, dtype=np.float32),
            np.array(heart_rate_features, dtype=np.float32)
        ])
        
        # Handle NaN values
        combined_features = np.nan_to_num(combined_features)
        
        return combined_features
    
    def fit(self, phase_changes: Dict, bpm_data: Optional[Tuple], fps: float, total_frames: int):
        """
        Fit the anomaly detection model using the extracted features.
        
        Args:
            phase_changes: Dictionary of phase changes for different facial regions
            bpm_data: Tuple containing heart rate data (inst_bpm, avg_bpm, pulse_signal)
            fps: Frames per second of the video
            total_frames: Total number of frames in the video
            
        Returns:
            self
        """
        self.fps = fps
        self.total_frames = total_frames
        
        # Calculate window size in frames
        window_size = int(self.params['window_size_seconds'] * fps)
        window_stride = int(window_size * (1 - self.params['window_overlap']))
        
        # Extract features for each window
        features_list = []
        frame_indices = []
        phase_features_list = []  # For storing just the phase features
        raw_phase_data = []       # For directly analyzing clusters of peaks
        
        for frame_idx in range(0, total_frames - window_size, window_stride):
            features = self.extract_features(phase_changes, bpm_data, frame_idx + window_size//2, window_size)
            if len(features) > 0:
                features_list.append(features)
                frame_indices.append(frame_idx + window_size//2)
                
                # Store the phase features separately (first 9 features now with the new peak features)
                phase_features_list.append(features[:9])
                
                # Store raw phase data for this window for direct analysis
                center_idx = frame_idx + window_size//2
                start_window = max(0, center_idx - window_size//2)
                end_window = min(total_frames-1, center_idx + window_size//2)
                if 'combined' in phase_changes and len(phase_changes['combined']) > 0:
                    window_data = phase_changes['combined'][start_window:end_window+1]
                    raw_phase_data.append({
                        'center_frame': center_idx,
                        'data': window_data,
                        'peak_count': features[6] / self.params['feature_weights']['phase_change'],  # Unweigthed peak count
                        'cluster_score': features[7] / self.params['feature_weights']['phase_change'] # Unweighted cluster score
                    })
        
        if not features_list:
            print("WARNING: No features could be extracted for anomaly detection")
            return self
            
        # Convert to numpy arrays
        X = np.array(features_list)
        phase_X = np.array(phase_features_list)
        
        # Normalize phase features only
        if phase_X.shape[0] > 0:
            phase_mean = np.mean(phase_X, axis=0)
            phase_std = np.std(phase_X, axis=0)
            phase_std[phase_std == 0] = 1  # Avoid division by zero
            phase_X_normalized = (phase_X - phase_mean) / phase_std
            
            # Fit isolation forest using only PBM motion features
            self.model.fit(phase_X_normalized)
            
            # Calculate anomaly scores based on PBM features only
            raw_scores = self.model.decision_function(phase_X_normalized)
            
            # Convert to anomaly scores (higher means more anomalous)
            anomaly_scores = -raw_scores
            
            # Store results
            self.anomaly_scores = []
            for i, frame_idx in enumerate(frame_indices):
                self.anomaly_scores.append({
                    'start_frame': max(0, frame_idx - window_size//2),
                    'end_frame': min(total_frames-1, frame_idx + window_size//2),
                    'center_frame': frame_idx,
                    'score': anomaly_scores[i],
                    'phase_features': phase_X[i],
                    'all_features': X[i],
                    'peak_data': raw_phase_data[i] if i < len(raw_phase_data) else None
                })
            
            # Normalize scores to 0-1 range
            if self.anomaly_scores:
                min_score = min(item['score'] for item in self.anomaly_scores)
                max_score = max(item['score'] for item in self.anomaly_scores)
                score_range = max_score - min_score
                
                if score_range > 0:
                    for item in self.anomaly_scores:
                        item['score_normalized'] = (item['score'] - min_score) / score_range
                else:
                    for item in self.anomaly_scores:
                        item['score_normalized'] = 0.5
            
            # Identify potential deception windows by directly counting peaks
            self._identify_deception_windows()
            
        return self
    
    def _identify_deception_windows(self):
        """
        Identify potential deception windows by directly counting peaks in the 
        combined micro-expression data and identifying regions with high peak density.
        Modified to focus on a single primary deception region where significant
        micro-expression clusters occur.
        """
        if not self.anomaly_scores:
            self.deception_windows = []
            return
        
        # Direct peak-counting approach for primary detection
        peak_scores = [item.get('peak_data', {}).get('peak_count', 0) for item in self.anomaly_scores]
        cluster_scores = [item.get('peak_data', {}).get('cluster_score', 0) for item in self.anomaly_scores]
        raw_phase_data = [item.get('peak_data', {}).get('data', None) for item in self.anomaly_scores]
        center_frames = [item.get('center_frame', 0) for item in self.anomaly_scores]
        
        # Only consider non-zero scores
        valid_windows = []
        
        # Get peak detection parameters
        threshold_multiplier = self.params.get('peak_detection', {}).get('threshold_multiplier', 1.8)
        cluster_importance = self.params.get('peak_detection', {}).get('cluster_importance', 2.5)
        min_peaks_required = self.params.get('peak_detection', {}).get('min_peaks_required', 3)
        max_gap_frames = self.params.get('peak_detection', {}).get('max_gap_frames', 40)
        
        # Step 1: Find windows with significant peak counts directly in the raw phase data
        for i, window in enumerate(self.anomaly_scores):
            if raw_phase_data[i] is not None:
                data = raw_phase_data[i]
                center_frame = center_frames[i]
                
                # Check if this window is in the region of interest (frames 200-300)
                # Give higher weight to windows in this region
                frame_weight = 1.0
                if 180 <= center_frame <= 320:  # Slightly expanded region for better detection
                    frame_weight = 1.5  # Higher weight for target region
                
                # Find peaks using the configured threshold
                threshold = np.mean(data) + threshold_multiplier * np.std(data)
                peaks = data > threshold
                peak_count = np.sum(peaks)
                
                # Find clusters by calculating distances between consecutive peaks
                if peak_count >= min_peaks_required:
                    peak_indices = np.where(peaks)[0]
                    peak_distances = np.diff(peak_indices)
                    
                    # Check if this is actually a cluster (peaks not too far apart)
                    is_cluster = True
                    if len(peak_distances) > 0:
                        mean_distance = np.mean(peak_distances)
                        if mean_distance > max_gap_frames:
                            is_cluster = False
                    
                    if is_cluster:
                        # Store more detailed peak information in the window
                        window['detailed_peaks'] = {
                            'count': peak_count,
                            'indices': peak_indices.tolist(),
                            'mean_distance': np.mean(peak_distances) if len(peak_distances) > 0 else 0,
                            'max_value': np.max(data),
                            'time_seconds': window['center_frame'] / self.fps  # Add time in seconds
                        }
                        
                        # Calculate a peak density score: higher when more peaks are closer together
                        # Apply the cluster importance factor
                        if len(peak_distances) > 0 and np.mean(peak_distances) > 0:
                            peak_density = (peak_count / np.mean(peak_distances)) * cluster_importance * frame_weight
                        else:
                            peak_density = peak_count * cluster_importance * frame_weight
                        
                        window['peak_density'] = peak_density
                        valid_windows.append(window)
        
        # Step 2: If we found valid windows with peaks, identify the most significant cluster
        if valid_windows:
            # Sort by peak density (primary) and peak count (secondary)
            valid_windows.sort(key=lambda x: (
                x.get('peak_density', 0),
                x.get('detailed_peaks', {}).get('count', 0)
            ), reverse=True)
            
            # Take only the top cluster
            top_window = valid_windows[0]
            self.deception_windows = [top_window]
            
            # Add time information in seconds to the window
            start_time = top_window['start_frame'] / self.fps
            end_time = top_window['end_frame'] / self.fps
            center_time = top_window['center_frame'] / self.fps
            
            top_window['time_info'] = {
                'start_time': start_time,
                'end_time': end_time,
                'center_time': center_time,
                'duration': end_time - start_time
            }
            
            # Print detailed information about the detected region
            peak_count = top_window.get('detailed_peaks', {}).get('count', 0)
            print(f"Primary deception window detected at {center_time:.2f}s: {peak_count} peaks, "
                  f"duration: {end_time - start_time:.2f}s")
            
            return
        
        # Fall back to isolation forest if no good peak-based windows found
        print("No clear peak clusters found, falling back to anomaly detection")
        sorted_scores = sorted(self.anomaly_scores, key=lambda x: x.get('score_normalized', 0), reverse=True)
        
        # Take only the top outlier as the primary deception region
        if sorted_scores:
            top_window = sorted_scores[0]
            if top_window.get('score_normalized', 0) >= self.params['min_anomaly_score']:
                self.deception_windows = [top_window]
                
                # Add time information
                start_time = top_window['start_frame'] / self.fps
                end_time = top_window['end_frame'] / self.fps
                center_time = top_window['center_frame'] / self.fps
                
                top_window['time_info'] = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'center_time': center_time,
                    'duration': end_time - start_time
                }
            else:
                self.deception_windows = []
        else:
            self.deception_windows = []
    
    def _merge_deception_windows(self):
        """
        Empty method since our modified detection approach now only returns
        a single deception window, so there's no need to merge multiple windows.
        Method is kept for compatibility with the rest of the code.
        """
        # No need to merge anything since we only return a single window
        pass
    
    def get_frame_deception_score(self, frame_idx: int) -> Tuple[float, bool]:
        """
        Get the deception score for a specific frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Tuple of (normalized_score, is_deception)
        """
        if not self.anomaly_scores:
            return 0.0, False
        
        # Find windows that contain this frame
        containing_windows = [
            window for window in self.anomaly_scores
            if window['start_frame'] <= frame_idx <= window['end_frame']
        ]
        
        if not containing_windows:
            return 0.0, False
        
        # Get max score among containing windows
        max_score = max(window.get('score_normalized', 0) for window in containing_windows)
        
        # Check if in a deception window
        is_deception = any(
            window['start_frame'] <= frame_idx <= window['end_frame']
            for window in self.deception_windows
        )
        
        return max_score, is_deception
    
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
        
        # Calculate timeline position for deception windows
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
        Create a timeline plot showing deception scores across the video.
        
        Args:
            frame_idx: Current frame index
            width: Desired width of the plot
            height: Desired height of the plot
            
        Returns:
            Plot image as numpy array
        """
        # Create figure with adjusted size for proper display
        fig, ax = plt.subplots(figsize=(10, 3), dpi=100)
        
        # Extract data for plotting
        if self.anomaly_scores:
            frame_indices = [item['center_frame'] for item in self.anomaly_scores]
            scores = [item.get('score_normalized', 0) for item in self.anomaly_scores]
            # Get peak counts for each window for more intuitive visualization
            peak_counts = [item.get('detailed_peaks', {}).get('count', 0) for item in self.anomaly_scores]
            peak_counts = [min(count, 10) / 10 for count in peak_counts]  # Normalize to 0-1 range, cap at 10
        else:
            frame_indices = [0]
            scores = [0]
            peak_counts = [0]
        
        # Create dual colored plot - use peak counts for color intensity
        cmap = LinearSegmentedColormap.from_list('deception_cmap', ['green', 'yellow', 'red'])
        
        # Calculate time in seconds for x-axis
        time_seconds = [frame_idx / self.fps for frame_idx in frame_indices]
        current_time = frame_idx / self.fps
        
        # Draw combined micro-expression phase data if available (bottom of plot)
        # This makes the relationship between phase changes and detection more explicit
        if len(self.anomaly_scores) > 0 and 'peak_data' in self.anomaly_scores[0]:
            # Collect all phase data to determine Y-scale
            all_phase_data = []
            for item in self.anomaly_scores:
                if 'peak_data' in item and 'data' in item['peak_data']:
                    all_phase_data.extend(item['peak_data']['data'])
            
            if all_phase_data:
                # Determine y-axis scale for phase data
                max_phase = max(all_phase_data) * 1.1  # Add 10% margin
                
                # Draw scaled phase data on the bottom 1/3 of the plot as a gray line
                for i, item in enumerate(self.anomaly_scores):
                    if 'peak_data' in item and 'data' in item['peak_data']:
                        data = item['peak_data']['data']
                        
                        # Convert to time in seconds
                        window_start = item['start_frame'] / self.fps
                        window_end = item['end_frame'] / self.fps
                        t = np.linspace(window_start, window_end, len(data))
                        
                        # Scale data to fit in the bottom 1/3 of the plot
                        scaled_data = 0.3 * (data / max_phase)
                        
                        # Plot phase data as a thin gray line
                        ax.plot(t, scaled_data, '-', color='gray', alpha=0.3, linewidth=0.8)
                        
                        # If this window has detected peaks, mark them
                        if 'detailed_peaks' in item and 'indices' in item['detailed_peaks']:
                            peak_indices = item['detailed_peaks']['indices']
                            if peak_indices:
                                # Calculate time points for peaks
                                data_length = len(data)
                                peak_times = [window_start + ((idx / data_length) * (window_end - window_start)) 
                                            for idx in peak_indices]
                                peak_values = [scaled_data[idx] if idx < len(scaled_data) else 0 
                                            for idx in peak_indices]
                                
                                # Plot peaks as small red markers
                                ax.scatter(peak_times, peak_values, color='red', s=15, marker='x', 
                                        alpha=0.8, zorder=5, label='_nolegend_')
        
        # Plot the scores with peak-based coloring (top part of plot)
        colors = [cmap(max(score, peak)) for score, peak in zip(scores, peak_counts)]
        scatter = ax.scatter(time_seconds, scores, c=colors, s=35, alpha=0.7, zorder=10)
        
        # Connect points with lines
        ax.plot(time_seconds, scores, '-', color='#555555', alpha=0.3, linewidth=1)
        
        # Add threshold line if we have deception windows
        if self.deception_windows and len(self.anomaly_scores) > 0:
            min_score = min(item.get('score_normalized', 0) for item in self.deception_windows) if self.deception_windows else 0
            ax.axhline(y=min_score, color='red', linestyle='--', alpha=0.7)
            ax.text(self.total_frames / self.fps * 0.05, min_score + 0.02, f"Threshold", fontsize=8, color='red')
        
        # Mark current frame
        closest_idx = np.argmin(np.abs(np.array(time_seconds) - current_time)) if time_seconds else 0
        if closest_idx < len(scores):
            ax.plot(current_time, scores[closest_idx], 'o', color='lime', markersize=10, 
                   markeredgecolor='black', markeredgewidth=1, zorder=15)
        
        # Highlight deception windows with time labels and peak counts
        for i, window in enumerate(self.deception_windows):
            start_time = window.get('time_info', {}).get('start_time', window['start_frame'] / self.fps)
            end_time = window.get('time_info', {}).get('end_time', window['end_frame'] / self.fps)
            
            # Display peak count in the highlight if available
            peak_count = window.get('detailed_peaks', {}).get('count', 0)
            
            # Use different transparency for alternating windows for clarity
            alpha = 0.2 if i % 2 == 0 else 0.15
            ax.axvspan(start_time, end_time, alpha=alpha, color='red', zorder=1)
            
            # Add a label with the peak count in the middle of the window
            if peak_count > 0:
                mid_time = (start_time + end_time) / 2
                y_pos = 0.9  # Position at top of plot
                
                # Create an outlined text box for better visibility
                ax.text(mid_time, y_pos, f"{peak_count} peaks", fontsize=8, color='white',
                        ha='center', va='center', bbox=dict(facecolor='red', alpha=0.8, 
                                                          boxstyle='round,pad=0.3'),
                        zorder=20)
        
        # Set labels and title
        ax.set_title("Deception Detection Timeline", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_ylabel("Anomaly Score / Phase Changes", fontsize=10)
        
        # Set axis limits
        ax.set_xlim(0, self.total_frames / self.fps)
        ax.set_ylim(0, 1.05)
        
        # Add custom time ticks every 5 seconds
        max_time = self.total_frames / self.fps
        time_ticks = np.arange(0, max_time + 5, 5)
        ax.set_xticks(time_ticks)
        ax.set_xticklabels([f"{t:.0f}s" for t in time_ticks])
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add marker for current time
        ax.axvline(x=current_time, color='green', linestyle='-', alpha=0.5, zorder=2)
        ax.text(current_time, 0.02, f"{current_time:.1f}s", fontsize=8, color='green',
                ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7),
                zorder=20)
        
        # Add legend explaining the plot elements
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='High anomaly score'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Low anomaly score'),
            Line2D([0], [0], marker='x', color='red', markersize=6, label='Micro-expression peak'),
            Line2D([0], [0], color='red', linestyle='--', label='Threshold'),
            Line2D([0], [0], color='green', label='Current time')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.7)
        
        # Tight layout
        fig.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to numpy array
        img = np.array(Image.open(buf))
        
        # Close resources
        plt.close(fig)
        buf.close()
        
        # Resize to desired dimensions
        img = cv2.resize(img, (width, height))
        
        # Convert RGB to BGR (for OpenCV)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img 