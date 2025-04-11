import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple
import scipy.signal
import traceback


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
        # For forehead: points above eyebrows, below hairline
        self.FOREHEAD_CENTER_LANDMARKS = [10, 151, 9]  # Center forehead points
        
        # For cheeks: points on upper cheek area
        self.LEFT_CHEEK_CENTER_LANDMARKS = [330, 347, 280, 425, 266]  # Left cheek center area
        self.RIGHT_CHEEK_CENTER_LANDMARKS = [101, 118, 50, 36, 205]  # Right cheek center area
        
        # Define rectangle dimensions for each region (width, height)
        self.REGION_DIMENSIONS = {
            'forehead': (90, 70),     # Wide rectangle for forehead
            'left_cheek': (80, 84),   # Taller rectangle for cheeks
            'right_cheek': (80, 84)   # Taller rectangle for cheeks
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
        """Detect faces and extract rectangular regions"""
        height, width = frame.shape[:2]
        results = []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp.solutions.face_mesh.FaceMesh.IMAGE_DIMENSIONS = (height, width) 
        
        # Detect faces
        face_results = self.face_mesh.process(rgb_frame)
        
        if not face_results.multi_face_landmarks:
            return results
            
        for face_landmarks in face_results.multi_face_landmarks:
            # Convert landmarks to image coordinates
            landmarks = [(int(l.x * width), int(l.y * height)) for l in face_landmarks.landmark]
            
            regions = {}
            
            # Process forehead region
            forehead_center = self.get_region_center(landmarks, self.FOREHEAD_CENTER_LANDMARKS)
            # Move the forehead center up by 20 pixels to avoid eyebrows
            forehead_center = (forehead_center[0], forehead_center[1] - 20)
            forehead_img, forehead_bounds = self.extract_rectangle_region(
                frame, forehead_center, self.REGION_DIMENSIONS['forehead']
            )
            
            # Process cheek regions
            left_cheek_center = self.get_region_center(landmarks, self.LEFT_CHEEK_CENTER_LANDMARKS)
            right_cheek_center = self.get_region_center(landmarks, self.RIGHT_CHEEK_CENTER_LANDMARKS)
            
            left_cheek_img, left_cheek_bounds = self.extract_rectangle_region(
                frame, left_cheek_center, self.REGION_DIMENSIONS['left_cheek']
            )
            right_cheek_img, right_cheek_bounds = self.extract_rectangle_region(
                frame, right_cheek_center, self.REGION_DIMENSIONS['right_cheek']
            )
            
            # Store regions if they're valid
            if forehead_img.size > 0:
                regions['forehead'] = {
                    'image': forehead_img,
                    'bounds': forehead_bounds,
                    'original_size': self.REGION_DIMENSIONS['forehead']
                }
            
            if left_cheek_img.size > 0:
                regions['left_cheek'] = {
                    'image': left_cheek_img,
                    'bounds': left_cheek_bounds,
                    'original_size': self.REGION_DIMENSIONS['left_cheek']
                }
                
            if right_cheek_img.size > 0:
                regions['right_cheek'] = {
                    'image': right_cheek_img,
                    'bounds': right_cheek_bounds,
                    'original_size': self.REGION_DIMENSIONS['right_cheek']
                }
            
            if regions:
                results.append({
                    'regions': regions,
                    'landmarks': landmarks
                })
            
        return results

class ColorMagnification:
    def __init__(self, alpha: float = 50.0, level: int = 4, 
                 f_lo: float = 50/60, f_hi: float = 60/60):
        self.alpha = alpha
        self.level = level
        self.f_lo = f_lo
        self.f_hi = f_hi

    def bgr2yiq(self, bgr: np.ndarray) -> np.ndarray:
        """Convert BGR to YIQ, matching the original implementation"""
        # Normalize to float32 in range [0,1]
        frame = np.float32(bgr) / 255.0
        
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to YIQ
        y = rgb @ np.array([[0.30], [0.59], [0.11]])
        rby = rgb[:, :, (0,2)] - y
        i = np.sum(rby * np.array([[[0.74, -0.27]]]), axis=-1)
        q = np.sum(rby * np.array([[[0.48, 0.41]]]), axis=-1)
        
        return np.dstack((y.squeeze(), i, q))

    def yiq2rgb(self, yiq: np.ndarray) -> np.ndarray:
        """Convert YIQ to RGB, then normalize to uint8"""
        r = yiq @ np.array([1.0, 0.9468822170900693, 0.6235565819861433])
        g = yiq @ np.array([1.0, -0.27478764629897834, -0.6356910791873801])
        b = yiq @ np.array([1.0, -1.1085450346420322, 1.7090069284064666])
        rgb = np.clip(np.dstack((r, g, b)), 0, 1)
        
        # Convert to uint8 BGR format
        rgb_uint8 = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
        return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    def build_gaussian_pyramid(self, image: np.ndarray) -> List[np.ndarray]:
        """Build Gaussian pyramid"""
        # Ensure image is 2D
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
            
        pyramid = [image.copy()]
        current_level = image.copy()
        
        for _ in range(self.level):
            current_level = cv2.pyrDown(current_level)
            pyramid.append(current_level.copy())
            
        return pyramid

    def temporal_bandpass_filter(self, frames: np.ndarray) -> np.ndarray:
        """Apply temporal bandpass filter following the article's approach"""
        fs = 30  # Sampling rate (30fps)
        
        # Implement FIR filter similar to the article rather than IIR (Butterworth)
        # This provides better frequency isolation for heart rate detection
        num_frames = frames.shape[0]
        bandpass = scipy.signal.firwin(
            numtaps=num_frames,
            cutoff=(self.f_lo, self.f_hi),
            fs=fs,
            pass_zero=False
        )
        
        # Get transfer function
        transfer_function = np.fft.rfft(bandpass)
        
        # Apply filter in frequency domain for each pixel
        filtered_frames = np.zeros_like(frames)
        
        # For each pixel position
        for y in range(frames.shape[1]):
            for x in range(frames.shape[2]):
                # Get temporal signal for this pixel
                pixel_values = frames[:, y, x]
                
                # Apply filter in frequency domain
                pixel_fft = np.fft.rfft(pixel_values)
                filtered_fft = pixel_fft * transfer_function
                filtered_signal = np.fft.irfft(filtered_fft, num_frames)
                
                # Store filtered result
                filtered_frames[:, y, x] = filtered_signal
        
        return filtered_frames

    def magnify(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply Eulerian Video Magnification optimized for heart rate detection"""
        if not frames or len(frames) < 2:
            return frames

        # Verify all frames have the same shape
        first_shape = frames[0].shape
        for i, frame in enumerate(frames):
            if frame.shape != first_shape:
                print(f"WARNING: Frame {i} has shape {frame.shape} which differs from the first frame shape {first_shape}")
                # Resize to match the first frame
                frames[i] = cv2.resize(frame, (first_shape[1], first_shape[0]))
        
        try:
            # Convert all frames to YIQ color space
            yiq_frames = np.array([self.bgr2yiq(frame) for frame in frames])
            
            # Get dimensions for the pyramid
            height, width = frames[0].shape[:2]
            pyr_height = height
            pyr_width = width
            for _ in range(self.level):
                pyr_height = (pyr_height + 1) // 2
                pyr_width = (pyr_width + 1) // 2
            
            # Process each channel separately - focus on Y channel for heart rate
            magnified_frames = []
            magnified_yiq = np.zeros_like(yiq_frames)
            
            for channel in range(3):  # For each YIQ channel
                # Extract channel frames
                channel_frames = yiq_frames[:, :, :, channel]
                
                # Build pyramid for each frame
                pyramid_frames = np.zeros((len(frames), pyr_height, pyr_width))
                
                for i, frame in enumerate(channel_frames):
                    # Create 2D frame
                    frame_2d = frame.reshape(height, width)
                    pyramid = self.build_gaussian_pyramid(frame_2d)
                    # Get the last level (most downsampled)
                    pyramid_frames[i] = cv2.resize(pyramid[-1], (pyr_width, pyr_height))
                
                # Apply temporal filter to pyramid level
                filtered = self.temporal_bandpass_filter(pyramid_frames)
                
                # Apply magnification - use lower alpha factor for I and Q channels
                # to prevent color artifacts (as per the article's approach)
                channel_alpha = self.alpha if channel == 0 else self.alpha * 0.5
                
                # Amplify and reconstruct
                for i in range(len(frames)):
                    # Amplify using appropriate factor for each channel
                    magnified = filtered[i] * channel_alpha
                    
                    # Add back to original
                    magnified_yiq[i, :, :, channel] = yiq_frames[i, :, :, channel] + \
                        cv2.resize(magnified, (width, height))
            
            # Convert back to BGR
            for i in range(len(frames)):
                magnified_frames.append(self.yiq2rgb(magnified_yiq[i]))
            
            return magnified_frames
            
        except Exception as e:
            print(f"Error in color magnification: {str(e)}")
            print("Returning original frames without magnification")
            traceback.print_exc()
            return frames.copy()
    

class FacialColorMagnification:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.color_magnifier = ColorMagnification()
        
    def process_video(self, input_path: str, output_path: str):
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
        for face_idx in range(len(all_faces[0])):  # For each face detected in the first frame
            # Process each region
            for region_name in ['forehead', 'left_cheek', 'right_cheek']:
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
                    magnified_frames = self.color_magnifier.magnify(region_frames)
                    
                    # Replace the regions in the processed frames
                    for frame_idx, magnified in enumerate(magnified_frames):
                        if (all_faces[frame_idx] and 
                            len(all_faces[frame_idx]) > face_idx and 
                            region_name in all_faces[frame_idx][face_idx]['regions']):
                            
                            face = all_faces[frame_idx][face_idx]
                            region_info = face['regions'][region_name]
                            bounds = region_info['bounds']
                            original_size = region_info['original_size']
                            
                            # Resize magnified region back to original size
                            magnified_resized = cv2.resize(magnified, 
                                                         (original_size[0], original_size[1]))
                            
                            # Replace region in frame
                            processed_frames[frame_idx][bounds[1]:bounds[3], 
                                                      bounds[0]:bounds[2]] = magnified_resized
        
        # Write processed frames
        print("Writing output video...")
        for frame in processed_frames:
            out.write(frame)
        
        cap.release()
        out.release()
        print("Processing complete!")

if __name__ == "__main__":
    # Define input and output paths
    # input_video_path = "test_videos/face.mp4"  # Path to your input video
    input_video_path = "test_videos/face.mp4"
    output_video_path = "Face_Color_Magnification/output_videos/output.mp4"  # Path where output will be saved
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # Process the video
    processor = FacialColorMagnification()
    processor.process_video(input_video_path, output_video_path)