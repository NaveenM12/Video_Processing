import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, List, Tuple
from Face_Motion_Magnification.utils.steerable_pyramid import SteerablePyramid
from Face_Motion_Magnification.utils.phase_utils import rgb2yiq, yiq2rgb

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
        # Split mouth landmarks into left and right sides
        self.LEFT_MOUTH_LANDMARKS = [61]  # Left mouth corner area
        self.RIGHT_MOUTH_LANDMARKS = [291]  # Right mouth corner area
        
        # Define rectangle dimensions for each region (width, height)
        self.REGION_DIMENSIONS = {
            'left_eye': (116, 84),    # Wider than tall for eyes
            'right_eye': (116, 84),   # Wider than tall for eyes
            'nose_tip': (100, 80),  # Square for nose
            'left_mouth': (70, 54),  # Square for mouth corners
            'right_mouth': (70, 54)  # Square for mouth corners
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
            left_eye_img, left_eye_bounds = self.extract_rectangle_region(
                frame, left_eye_center, self.REGION_DIMENSIONS['left_eye']
            )
            
            # Process right eye region
            right_eye_center = self.get_region_center(landmarks, self.RIGHT_EYE_LANDMARKS)
            right_eye_img, right_eye_bounds = self.extract_rectangle_region(
                frame, right_eye_center, self.REGION_DIMENSIONS['right_eye']
            )
            
            # Process nose tip region
            nose_tip_center = self.get_region_center(landmarks, self.NOSE_TIP_LANDMARKS)
            nose_tip_img, nose_tip_bounds = self.extract_rectangle_region(
                frame, nose_tip_center, self.REGION_DIMENSIONS['nose_tip']
            )
            
            # Process left mouth region
            left_mouth_center = self.get_region_center(landmarks, self.LEFT_MOUTH_LANDMARKS)
            left_mouth_img, left_mouth_bounds = self.extract_rectangle_region(
                frame, left_mouth_center, self.REGION_DIMENSIONS['left_mouth']
            )
            
            # Process right mouth region
            right_mouth_center = self.get_region_center(landmarks, self.RIGHT_MOUTH_LANDMARKS)
            right_mouth_img, right_mouth_bounds = self.extract_rectangle_region(
                frame, right_mouth_center, self.REGION_DIMENSIONS['right_mouth']
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
                
            if left_mouth_img.size > 0:
                regions['left_mouth'] = {
                    'image': left_mouth_img,
                    'bounds': left_mouth_bounds,
                    'original_size': self.REGION_DIMENSIONS['left_mouth']
                }
                
            if right_mouth_img.size > 0:
                regions['right_mouth'] = {
                    'image': right_mouth_img,
                    'bounds': right_mouth_bounds,
                    'original_size': self.REGION_DIMENSIONS['right_mouth']
                }
            
            if regions:
                results.append({
                    'regions': regions,
                    'landmarks': landmarks
                })
            
        return results

class PhaseMagnification:
    def __init__(self, 
                 phase_mag: float = 25.0,
                 f_lo: float = 0.2,
                 f_hi: float = 0.25,
                 sigma: float = 5.0,
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

    def magnify(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply Phase-Based Motion Magnification to a sequence of frames"""
        if not frames or len(frames) < 2:
            return frames
            
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
        
        # Reference frame (first frame)
        ref_idx = 0
        
        # Batch processing parameters
        batch_size = 4
        
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
            
            for vid_idx in range(len(frames)):
                curr_pyr = self.build_level_batch(video_dft[vid_idx], filter_batch)
                
                # Get unwrapped phase delta
                _delta = torch.angle(curr_pyr) - ref_phase
                
                # Wrap phase delta to [-pi, pi]
                phase_deltas[:, vid_idx] = ((torch.pi + _delta) % (2*torch.pi)) - torch.pi
            
            # Create and apply temporal filter matching the number of frames
            bandpass = self.create_temporal_filter(len(frames))
            transfer_function = torch.fft.fft(
                torch.fft.ifftshift(torch.tensor(bandpass))
            ).to(self.device).type(torch.complex64)
            
            transfer_function = torch.tile(transfer_function,
                                         [idx2-idx1, 1, 1, 1]).permute(0, 3, 1, 2)
            
            # Filter in frequency domain
            phase_deltas = torch.fft.ifft(
                transfer_function * torch.fft.fft(phase_deltas, dim=1),
                dim=1
            ).real
            
            # Apply motion magnification
            for vid_idx in range(len(frames)):
                curr_pyr = self.build_level_batch(video_dft[vid_idx], filter_batch)
                delta = phase_deltas[:, vid_idx]
                
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
        
        return magnified_frames

class FacialPhaseMagnification:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.phase_magnifier = PhaseMagnification()
        
    def process_video(self, input_path: str, output_path: str):
        """Process video with facial phase-based motion magnification"""
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
        
        # Process each face
        for face_idx in range(len(all_faces[0])):  # For each face detected in first frame
            # Process each region
            for region_name in ['left_eye', 'right_eye', 'nose_tip', 'left_mouth', 'right_mouth']:
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
                    magnified_frames = self.phase_magnifier.magnify(region_frames)
                    
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
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # Process the video
    processor = FacialPhaseMagnification()
    processor.process_video(input_video_path, output_video_path)



