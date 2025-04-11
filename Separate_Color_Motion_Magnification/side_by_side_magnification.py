import cv2
import numpy as np
import os
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

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
        """Process video to create side-by-side magnification"""
        # Use provided paths or default from config
        input_path = input_path or INPUT_VIDEO_PATH
        output_path = output_path or OUTPUT_VIDEO_PATH
        
        # Create temp file paths
        motion_output_path = output_path.replace(".mp4", "_motion_only.mp4")
        
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
        
        # Create video writer for combined output
        # Output will be 3x the width of the original
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width*3, height))
        
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
        
        # Create deep copies for motion and color magnification
        motion_frames = deepcopy(all_frames)
        
        print("Applying motion magnification to micro-expression regions...")
        # Process with motion magnification - this will focus on micro-expression regions
        
        # Custom modification to the motion magnification to focus on micro-expression regions
        # This overrides the default regions in the FacialPhaseMagnification class
        self.motion_processor.detect_face_regions = self.detect_micro_expression_regions
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
        
        print("Creating side-by-side output...")
        # Ensure all videos have the same number of frames
        min_frames = min(len(all_frames), len(motion_frames), len(color_frames))
        
        for i in range(min_frames):
            # Stitch frames side by side with labels
            stitched_frame = self.stitch_frames(
                all_frames[i], 
                motion_frames[i], 
                color_frames[i]
            )
            
            # Write to output
            out.write(stitched_frame)
            
            if i % 10 == 0:
                print(f"Processed {i}/{min_frames} frames")
        
        # Release resources
        cap.release()
        out.release()
        
        # Delete temporary files to save space
        if not KEEP_TEMP_FILES:
            if os.path.exists(motion_output_path):
                os.remove(motion_output_path)
        
        print("Processing complete!")


if __name__ == "__main__":
    # Process the video using settings from config.py
    processor = SideBySideMagnification()
    processor.process_video() 