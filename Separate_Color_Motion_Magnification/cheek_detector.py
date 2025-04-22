import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Dict

class CheekDetector:
    """Detects the left and right cheek regions for heart rate detection using MediaPipe Face Mesh."""
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh for cheek detection."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        
        # Define landmarks for finding cheek centers
        # For cheeks: points on upper cheek area as specified in the face_region_color_magnification.py
        self.LEFT_CHEEK_CENTER_LANDMARKS = [330, 347, 280, 425, 266]  # Left cheek center area
        self.RIGHT_CHEEK_CENTER_LANDMARKS = [101, 118, 50, 36, 205]  # Right cheek center area
        
        # Define rectangle dimensions for each region (width, height)
        self.REGION_DIMENSIONS = {
            'left_cheek': (110, 110),   # Same dimensions used in face_region_color_magnification.py
            'right_cheek': (110, 110)   # Same dimensions used in face_region_color_magnification.py
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
    
    def detect_cheeks(self, frame: np.ndarray) -> Tuple[bool, List[Dict]]:
        """Detect face and extract left and right cheek regions.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Tuple containing:
                - Boolean indicating if face was detected
                - List of dictionaries with cheek regions and landmarks
        """
        height, width = frame.shape[:2]
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return False, []
        
        face_regions = []
        
        # Get the first detected face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to image coordinates
        landmarks = [(int(l.x * width), int(l.y * height)) for l in face_landmarks.landmark]
        
        regions = {}
        
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
            face_regions.append({
                'regions': regions,
                'landmarks': landmarks
            })
        
        return True, face_regions
    
    def draw_cheek_regions(self, frame: np.ndarray, face_regions: List[Dict]) -> np.ndarray:
        """Draw the detected cheek regions on a frame with improved visualization.
        
        Args:
            frame: Input frame
            face_regions: Face region dictionary
            
        Returns:
            Frame with cheek regions drawn
        """
        result = frame.copy()
        
        if not face_regions:
            return result
            
        for face_region in face_regions:
            if 'regions' not in face_region:
                continue
                
            for region_name, region_info in face_region['regions'].items():
                bounds = region_info['bounds']
                x_min, y_min, x_max, y_max = bounds
                
                # Create a semi-transparent overlay for the region
                overlay = result.copy()
                
                # Use different colors for left and right cheek
                if region_name == 'left_cheek':
                    color = (0, 255, 0)  # Green for left cheek
                else:
                    color = (0, 255, 255)  # Yellow for right cheek
                
                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
                alpha = 0.2  # Transparency factor
                cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
                
                # Draw a solid border around the region
                cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Add more visible label
                label_bg = result.copy()
                text = f"Color Magnification: {region_name.replace('_', ' ').title()}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                
                # Create background for text
                text_x = x_min
                text_y = max(y_min - 10, text_size[1] + 10)  # Ensure it's on screen
                cv2.rectangle(
                    label_bg, 
                    (text_x, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 10, text_y + 5),
                    (color[0]//2, color[1]//2, color[2]//2),
                    -1
                )
                
                # Blend text background
                cv2.addWeighted(label_bg, 0.7, result, 0.3, 0, result)
                
                # Add text
                cv2.putText(
                    result,
                    text,
                    (text_x + 5, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )
        
        return result 