import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import signal
from typing import Dict, List, Tuple
from Face_Motion_Magnification.utils.steerable_pyramid import SteerablePyramid
from Face_Motion_Magnification.utils.phase_utils import rgb2yiq, yiq2rgb
import os

class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        """Initialize MediaPipe Face Detection"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=4,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        
        # Define landmarks for finding region centers
        self.LEFT_EYE_LANDMARKS = [252, 254, 442, 443]  # Left eye area
        self.RIGHT_EYE_LANDMARKS = [223, 222, 24, 22]  # Right eye area
        self.NOSE_TIP_LANDMARKS = [195, 5, 4, 1, 19]  # Nose tip area
        # Replace separate mouth landmarks with single mouth landmarks
        self.MOUTH_LANDMARKS = [61, 291, 0, 17, 78, 308]  # Left and right corners + additional mouth points
        
        # Define rectangle dimensions for each region (width, height)
        self.REGION_DIMENSIONS = {
            'left_eye': (180, 170),    # Wider than tall for eyes
            'right_eye': (180, 170),   # Wider than tall for eyes
            'nose_tip': (196, 100),    # Square for nose
            'mouth': (186, 80)        # Wide rectangle for full mouth
        }

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

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces and extract rectangular regions around facial features"""
        height, width = frame.shape[:2]
        results = []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp.solutions.face_mesh.FaceMesh.IMAGE_DIMENSIONS = (height, width)  # Add this
        
        # Detect faces
        face_results = self.face_mesh.process(rgb_frame)
        
        if not face_results.multi_face_landmarks:
            return results
            
        for face_landmarks in face_results.multi_face_landmarks:
            # Convert landmarks to image coordinates
            landmarks = [(int(l.x * width), int(l.y * height)) 
                        for l in face_landmarks.landmark]
            
            regions = {}
            
            # Process left eye region
            left_eye_center = self.get_region_center(landmarks, self.LEFT_EYE_LANDMARKS)
            # Move the left eye center upward to include eyebrows
            left_eye_center = (left_eye_center[0], left_eye_center[1] - 20)
            left_eye_img, left_eye_bounds = self.extract_rectangle_region(
                frame, left_eye_center, self.REGION_DIMENSIONS['left_eye']
            )
            
            # Process right eye region
            right_eye_center = self.get_region_center(landmarks, self.RIGHT_EYE_LANDMARKS)
            # Move the right eye center upward to include eyebrows
            right_eye_center = (right_eye_center[0], right_eye_center[1] - 20)
            right_eye_img, right_eye_bounds = self.extract_rectangle_region(
                frame, right_eye_center, self.REGION_DIMENSIONS['right_eye']
            )
            
            # Process nose tip region
            nose_tip_center = self.get_region_center(landmarks, self.NOSE_TIP_LANDMARKS)
            nose_tip_img, nose_tip_bounds = self.extract_rectangle_region(
                frame, nose_tip_center, self.REGION_DIMENSIONS['nose_tip']
            )
            
            # Process unified mouth region (replacing separate left/right mouth regions)
            mouth_center = self.get_region_center(landmarks, self.MOUTH_LANDMARKS)
            mouth_img, mouth_bounds = self.extract_rectangle_region(
                frame, mouth_center, self.REGION_DIMENSIONS['mouth']
            )
            
            # Store regions if they're valid
            if left_eye_img.size > 0:
                regions['left_eye'] = {
                    'image': left_eye_img,
                    'bounds': left_eye_bounds,
                    'original_size': self.REGION_DIMENSIONS['left_eye']
                }
            
            if right_eye_img.size > 0:
                regions['right_eye'] = {
                    'image': right_eye_img,
                    'bounds': right_eye_bounds,
                    'original_size': self.REGION_DIMENSIONS['right_eye']
                }
                
            if nose_tip_img.size > 0:
                regions['nose_tip'] = {
                    'image': nose_tip_img,
                    'bounds': nose_tip_bounds,
                    'original_size': self.REGION_DIMENSIONS['nose_tip']
                }
                
            if mouth_img.size > 0:
                regions['mouth'] = {
                    'image': mouth_img,
                    'bounds': mouth_bounds,
                    'original_size': self.REGION_DIMENSIONS['mouth']
                }
            
            if regions:
                results.append({
                    'regions': regions,
                    'landmarks': landmarks
                })
            
        return results

class PhaseMagnification:
    def __init__(self, 
                 phase_mag: float = 15.0,
                 f_lo: float = 0.25,
                 f_hi: float = 0.3,
                 sigma: float = 1.0,
                 attenuate: bool = True):
        """Initialize Phase-Based Motion Magnification parameters"""
        self.phase_mag = phase_mag
        self.f_lo = f_lo
        self.f_hi = f_hi
        self.sigma = sigma
        self.attenuate = attenuate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize Gaussian kernel for amplitude weighted blurring
        self.setup_gaussian_kernel()
        
        # Initialize temporal bandpass filter
        self.setup_temporal_filter()
        
        # Initialize Complex Steerable Pyramid
        self.setup_csp()

    def setup_gaussian_kernel(self):
        """Setup Gaussian kernel for amplitude weighted blurring"""
        ksize = np.max((3, np.ceil(4*self.sigma) - 1)).astype(int)
        if ((ksize % 2) != 1):
            ksize += 1
            
        gk = cv2.getGaussianKernel(ksize=ksize, sigma=self.sigma)
        self.gauss_kernel = torch.tensor(gk @ gk.T).type(torch.float32) \
                                                  .to(self.device) \
                                                  .unsqueeze(0) \
                                                  .unsqueeze(0)
        
        # Setup conv2d filter
        self.filter2D = nn.Conv2d(in_channels=1, 
                                 out_channels=1,
                                 kernel_size=self.gauss_kernel.shape[2:],
                                 padding='same',
                                 padding_mode='circular',
                                 groups=1,
                                 bias=False)
        
        self.filter2D.weight.data = self.gauss_kernel
        self.filter2D.weight.requires_grad = False

    def setup_temporal_filter(self):
        """Setup temporal bandpass filter"""
        # Note: This is just initialization. The actual filter will be created in magnify()
        self.fs = 30  # Video sampling rate (fps)
        
    def create_temporal_filter(self, num_frames: int):
        """Create temporal bandpass filter for specific number of frames"""
        from scipy import signal
        
        # Normalize frequencies to Nyquist rate range [0, 1]
        norm_f_lo = self.f_lo / self.fs * 2
        norm_f_hi = self.f_hi / self.fs * 2
        
        # Get bandpass filter impulse response matching the number of frames
        return signal.firwin(numtaps=num_frames,
                           cutoff=[norm_f_lo, norm_f_hi],
                           pass_zero=False)
        
    def setup_csp(self):
        """Setup Complex Steerable Pyramid"""
        self.max_depth = 4  # Adjust based on region size
        self.csp = SteerablePyramid(depth=self.max_depth,
                                   orientations=8,
                                   filters_per_octave=2,
                                   twidth=0.75,
                                   complex_pyr=True)

    def build_level(self, dft, filt):
        """Build pyramid level"""
        return torch.fft.ifft2(torch.fft.ifftshift(dft * filt))

    def build_level_batch(self, dft, filter_batch):
        """Build pyramid level for a batch of filters"""
        return torch.fft.ifft2(torch.fft.ifftshift(
            dft.unsqueeze(0).repeat(filter_batch.shape[0], 1, 1) * filter_batch
        ))

    def recon_level_batch(self, pyr, filter_batch):
        """Reconstruct pyramid level for a batch of filters"""
        return torch.fft.fftshift(torch.fft.fft2(pyr)) * filter_batch

    def magnify(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        """Apply Phase-Based Motion Magnification to a sequence of frames and track phase changes
        
        Returns:
            Tuple containing:
                - List of magnified frames
                - Array of phase change magnitudes over time
        """
        if not frames or len(frames) < 2:
            return frames, np.zeros((len(frames),))
            
        # Convert frames to YIQ color space
        yiq_frames = []
        for frame in frames:
            rgb = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)/255)
            yiq_frames.append(rgb2yiq(rgb))
            
        # Get dimensions from first frame
        h, w = yiq_frames[0].shape[:2]
        
        # Convert frames to tensor and extract Luma channel
        frames_tensor = torch.tensor(np.array([frame[:,:,0] for frame in yiq_frames])) \
                            .type(torch.float32).to(self.device)
        
        # Compute DFT for each frame
        video_dft = torch.fft.fftshift(torch.fft.fft2(frames_tensor, dim=(1,2))) \
                        .type(torch.complex64).to(self.device)
        
        # Get filters and move to device
        filters, _ = self.csp.get_filters(h, w, cropped=False)
        filters_tensor = torch.tensor(np.array(filters)).type(torch.float32).to(self.device)
        
        # Store DFT of motion magnified frames
        recon_dft = torch.zeros((len(frames), h, w), dtype=torch.complex64).to(self.device)
        
        # Create a proper storage for capturing the raw phase changes before magnification
        # This is what we will use for visualization
        raw_phase_changes = np.zeros(len(frames))
        
        # Reference frame (first frame)
        ref_idx = 0
        
        # Batch processing parameters
        batch_size = 4
        
        # Running total for normalization
        total_amplitude_weight = 0.0
        
        # Process pyramid levels
        for level in range(1, len(filters) - 1, batch_size):
            # Get batch indices
            idx1 = level
            idx2 = min(level + batch_size, len(filters) - 1)
            
            # Get current filter batch
            filter_batch = filters_tensor[idx1:idx2]
            
            # Get reference frame pyramid and phase
            ref_pyr = self.build_level_batch(video_dft[ref_idx], filter_batch)
            ref_phase = torch.angle(ref_pyr)
            
            # Process each frame
            phase_deltas = torch.zeros((idx2-idx1, len(frames), h, w),
                                     dtype=torch.complex64).to(self.device)
            
            # Track amplitude for weighting
            avg_amplitude = torch.zeros((idx2-idx1), dtype=torch.float32).to(self.device)
            
            for vid_idx in range(len(frames)):
                curr_pyr = self.build_level_batch(video_dft[vid_idx], filter_batch)
                
                # Get unwrapped phase delta
                _delta = torch.angle(curr_pyr) - ref_phase
                
                # Wrap phase delta to [-pi, pi]
                phase_deltas[:, vid_idx] = ((torch.pi + _delta) % (2*torch.pi)) - torch.pi
                
                # Store average amplitude for this level
                if vid_idx == 0:
                    # Use amplitude from first frame as reference weighting
                    avg_amplitude = torch.mean(torch.abs(curr_pyr), dim=(1, 2))
            
            # Create and apply temporal filter matching the number of frames
            bandpass = self.create_temporal_filter(len(frames))
            transfer_function = torch.fft.fft(
                torch.fft.ifftshift(torch.tensor(bandpass))
            ).to(self.device).type(torch.complex64)
            
            transfer_function = torch.tile(transfer_function,
                                         [idx2-idx1, 1, 1, 1]).permute(0, 3, 1, 2)
            
            # Filter in frequency domain
            filtered_phase_deltas = torch.fft.ifft(
                transfer_function * torch.fft.fft(phase_deltas, dim=1),
                dim=1
            ).real
            
            # Apply motion magnification
            for vid_idx in range(len(frames)):
                curr_pyr = self.build_level_batch(video_dft[vid_idx], filter_batch)
                delta = filtered_phase_deltas[:, vid_idx]
                
                # Amplitude weighted blurring
                if self.sigma != 0:
                    amplitude_weight = torch.abs(curr_pyr) + 1e-6
                    
                    weight = F.conv2d(
                        input=amplitude_weight.unsqueeze(1),
                        weight=self.gauss_kernel,
                        padding='same'
                    ).squeeze(1)
                    
                    delta = F.conv2d(
                        input=(amplitude_weight * delta).unsqueeze(1),
                        weight=self.gauss_kernel,
                        padding='same'
                    ).squeeze(1)
                    
                    delta /= weight
                
                # UPDATED: Track the raw phase changes before magnification
                # This better represents the actual motion being detected (not magnified yet)
                for i in range(len(avg_amplitude)):
                    # Get significant phase changes in this level/orientation
                    level_delta = torch.abs(delta[i])
                    
                    # Use 95th percentile to capture significant motion while avoiding outliers
                    percentile_value = torch.quantile(level_delta.flatten(), 0.95)
                    
                    # Weight by amplitude to prioritize stronger signals
                    weight = avg_amplitude[i].item()
                    raw_phase_changes[vid_idx] += percentile_value.item() * weight
                    
                    # Add to total weight only on first frame for normalization
                    if vid_idx == 0:
                        total_amplitude_weight += weight
                
                # Modify phase variation
                modified_phase = delta * self.phase_mag
                
                # Attenuate other frequencies
                if self.attenuate:
                    curr_pyr = torch.abs(curr_pyr) * (ref_pyr/torch.abs(ref_pyr))
                
                # Apply modified phase
                curr_pyr = curr_pyr * torch.exp(1.0j*modified_phase)
                
                # Accumulate reconstructed levels
                recon_dft[vid_idx] += self.recon_level_batch(curr_pyr, filter_batch).sum(dim=0)
        
        # Add back lo and hi pass components
        hipass = filters_tensor[0]
        lopass = filters_tensor[-1]
        
        for vid_idx in range(len(frames)):
            # Add lo pass components
            curr_pyr_lo = self.build_level_batch(video_dft[vid_idx], lopass.unsqueeze(0))
            dft_lo = torch.fft.fftshift(torch.fft.fft2(curr_pyr_lo))
            recon_dft[vid_idx] += dft_lo.squeeze(0) * lopass
        
        # Inverse DFT of results
        result_video = torch.fft.ifft2(
            torch.fft.ifftshift(recon_dft, dim=(1,2)), 
            dim=(1,2)
        ).real.cpu().numpy()
        
        # Reconstruct with RGB channels
        magnified_frames = []
        for vid_idx in range(len(frames)):
            # Get current YIQ frame and replace Luma channel
            yiq_frame = yiq_frames[vid_idx].copy()
            yiq_frame[:, :, 0] = result_video[vid_idx]
            
            # Convert to RGB and normalize
            rgb_frame = yiq2rgb(yiq_frame)
            rgb_frame = cv2.convertScaleAbs(rgb_frame, alpha=255, beta=0)
            
            # Convert back to BGR
            magnified_frames.append(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        
        # Normalize by total amplitude weight for consistent scaling
        if total_amplitude_weight > 0:
            raw_phase_changes = raw_phase_changes / total_amplitude_weight
            
        # Normalize phase change magnitudes for better visualization
        if np.max(raw_phase_changes) > 0:
            raw_phase_changes = raw_phase_changes / np.max(raw_phase_changes)
        
        return magnified_frames, raw_phase_changes

class FacialPhaseMagnification:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.phase_magnifier = PhaseMagnification()
        
    def plot_phase_changes(self, region_phase_changes, output_dir):
        """Plot phase changes for each facial region with improved visualization
        
        Args:
            region_phase_changes: Dictionary with region names as keys and phase change arrays as values
            output_dir: Directory to save the plots
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a single figure with subplots for each region
        fig, axes = plt.subplots(len(region_phase_changes), 1, figsize=(12, 10), sharex=True)
        
        # If there's only one region, make axes iterable
        if len(region_phase_changes) == 1:
            axes = [axes]
            
        for idx, (region_name, phase_changes) in enumerate(region_phase_changes.items()):
            # Skip if no data
            if len(phase_changes) == 0:
                continue
                
            # Time points (x-axis)
            time_points = np.arange(len(phase_changes))
            
            # Plot
            axes[idx].plot(time_points, phase_changes, '-', linewidth=2)
            axes[idx].set_title(f'{region_name.replace("_", " ").title()} - Pre-Magnification Phase Changes')
            axes[idx].set_ylabel('Phase Change\nMagnitude')
            
            # Add grid
            axes[idx].grid(True, linestyle='--', alpha=0.7)
            
            # Highlight peaks
            peaks, _ = signal.find_peaks(phase_changes, height=0.5)  # Adjust height as needed
            if len(peaks) > 0:
                axes[idx].plot(peaks, phase_changes[peaks], 'ro')
        
        # Set common labels
        plt.xlabel('Frame Number')
        plt.suptitle('Local Phase Changes Detected by Complex Steerable Pyramid', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, 'phase_changes.png'), dpi=150)
        plt.close()
        
        # Create more detailed individual plots for each region
        for region_name, phase_changes in region_phase_changes.items():
            # Skip if no data or too few frames
            if len(phase_changes) < 3:
                continue
            
            # Create a figure with multiple subplots for detailed analysis
            fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            
            # Time points (x-axis)
            time_points = np.arange(len(phase_changes))
            
            # 1. Raw phase changes plot
            axes[0].plot(time_points, phase_changes, '-', linewidth=2, color='blue')
            axes[0].set_title(f'{region_name.replace("_", " ").title()} - Pre-Magnification Phase Changes')
            axes[0].set_ylabel('Phase Change\nMagnitude')
            axes[0].grid(True, linestyle='--', alpha=0.7)
            
            # Highlight peaks in raw data
            peaks, _ = signal.find_peaks(phase_changes, height=0.5)  # Adjust height as needed
            if len(peaks) > 0:
                axes[0].plot(peaks, phase_changes[peaks], 'ro', label='Peaks')
                axes[0].legend()
            
            # 2. Frame-to-frame derivative (better shows motion spikes)
            derivatives = np.abs(np.diff(phase_changes))
            # Add zero at the beginning to maintain same length
            derivatives = np.insert(derivatives, 0, 0)
            
            axes[1].plot(time_points, derivatives, '-', linewidth=2, color='green')
            axes[1].set_title(f'{region_name.replace("_", " ").title()} - Rate of Phase Change')
            axes[1].set_ylabel('Rate of Change')
            axes[1].grid(True, linestyle='--', alpha=0.7)
            
            # Highlight spikes in derivative
            derivative_peaks, _ = signal.find_peaks(derivatives, height=np.max(derivatives) * 0.4)
            if len(derivative_peaks) > 0:
                axes[1].plot(derivative_peaks, derivatives[derivative_peaks], 'ro', label='Phase Spikes')
                axes[1].legend()
            
            # 3. Frequency domain analysis (shows periodic motion)
            if len(phase_changes) > 10:  # Need enough data for FFT
                # Compute FFT
                phase_fft = np.abs(np.fft.rfft(phase_changes - np.mean(phase_changes)))
                freqs = np.fft.rfftfreq(len(phase_changes))
                
                # Exclude DC component (first value)
                axes[2].plot(freqs[1:], phase_fft[1:], '-', linewidth=2, color='purple')
                axes[2].set_title(f'{region_name.replace("_", " ").title()} - Frequency Analysis')
                axes[2].set_xlabel('Frequency (cycles/frame)')
                axes[2].set_ylabel('Phase Component\nAmplitude')
                axes[2].grid(True, linestyle='--', alpha=0.7)
                
                # Find dominant frequencies
                if len(phase_fft) > 3:
                    freq_peaks, _ = signal.find_peaks(phase_fft[1:], height=np.max(phase_fft[1:]) * 0.3)
                    if len(freq_peaks) > 0:
                        axes[2].plot(freqs[1:][freq_peaks], phase_fft[1:][freq_peaks], 'ro', 
                                  label='Dominant Frequencies')
                        axes[2].legend()
            else:
                axes[2].text(0.5, 0.5, 'Not enough frames for frequency analysis', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axes[2].transAxes)
                axes[2].set_title(f'{region_name.replace("_", " ").title()} - Frequency Analysis')
                axes[2].set_xlabel('Frequency (cycles/frame)')
            
            plt.suptitle(f'Phase-Based Motion Analysis: {region_name.replace("_", " ").title()}', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.savefig(os.path.join(output_dir, f'{region_name}_detailed_analysis.png'), dpi=150)
            plt.close()
            
            # Also create a cleaner single plot just showing raw phase changes (original implementation)
            plt.figure(figsize=(10, 6))
            plt.plot(time_points, phase_changes, '-', linewidth=2)
            plt.title(f'Pre-Magnification Phase Changes: {region_name.replace("_", " ").title()}')
            plt.xlabel('Frame Number')
            plt.ylabel('Phase Change Magnitude')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Highlight peaks
            if len(peaks) > 0:
                plt.plot(peaks, phase_changes[peaks], 'ro', label='Peaks')
                plt.legend()
            
            plt.savefig(os.path.join(output_dir, f'{region_name}_phase_changes.png'), dpi=150)
            plt.close()
        
    def process_video(self, input_path: str, output_path: str, plot_dir: str = None):
        """Process video with facial phase-based motion magnification and generate phase change plots
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            plot_dir: Directory to save phase change plots. If None, will use the directory of output_path.
        """
        # Set default plot directory
        if plot_dir is None:
            plot_dir = os.path.join(os.path.dirname(output_path), 'phase_plots')
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Reading frames and detecting faces...")
        all_frames = []
        all_faces = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            faces = self.face_detector.detect_faces(frame)
            all_frames.append(frame)
            all_faces.append(faces)
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        print("Processing facial regions...")
        processed_frames = all_frames.copy()
        
        # Dictionary to store phase changes for each region
        all_phase_changes = {}
        
        # Process each face
        for face_idx in range(len(all_faces[0])):  # For each face detected in first frame
            # Process each region
            # for region_name in ['left_eye', 'right_eye', 'nose_tip', 'mouth']:
            for region_name in ['left_eye', 'right_eye', 'nose_tip']:
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
                    magnified_frames, phase_changes = self.phase_magnifier.magnify(region_frames)
                    
                    # Store phase changes for this region
                    region_key = f"face{face_idx+1}_{region_name}"
                    all_phase_changes[region_key] = phase_changes
                    
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
                            if magnified.shape[:2] != (original_size[1], original_size[0]):
                                magnified = cv2.resize(magnified, 
                                                     (original_size[0], original_size[1]))
                            
                            # Replace region in frame
                            processed_frames[frame_idx][bounds[1]:bounds[3], 
                                                      bounds[0]:bounds[2]] = magnified
        
        # Generate phase change plots
        if all_phase_changes:
            print("Generating phase change plots...")
            self.plot_phase_changes(all_phase_changes, plot_dir)
        
        # Write processed frames
        print("Writing output video...")
        for frame in processed_frames:
            out.write(frame)
        
        cap.release()
        out.release()
        print("Processing complete!")

if __name__ == "__main__":
    # Define input and output paths
    input_video_path = "test_videos/face.mp4"
    output_video_path = "Face_Motion_Magnification/output_videos/output.mp4"
    plot_dir = "Face_Motion_Magnification/output_videos/phase_plots"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # Process the video
    processor = FacialPhaseMagnification()
    processor.process_video(input_video_path, output_video_path, plot_dir)



