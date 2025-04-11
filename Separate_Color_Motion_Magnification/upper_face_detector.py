import cv2
import numpy as np
from typing import List, Tuple, Dict

class UpperFaceDetector:
    """Detects the upper face region for heart rate detection using OpenCV."""
    
    def __init__(self):
        """Initialize OpenCV face and eye detectors."""
        # Load Haar cascades for face and eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def detect_upper_face(self, frame: np.ndarray) -> Tuple[bool, List[Dict]]:
        """Detect face and extract expanded region from forehead to below the nose.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Tuple containing:
                - Boolean indicating if face was detected
                - List of dictionaries with face regions and landmarks
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return False, []
        
        face_regions = []
        
        # Process only the first (largest) face
        x, y, w, h = faces[0]
        
        # Extend the face region upward to capture more forehead
        # Add 20% more space above the detected face if possible
        extended_y = max(0, y - int(h * 0.2))
        
        # Extend downward to include the nose area (approximately 70% of face height)
        # This typically covers down to just below the nose
        extended_height = y - extended_y + int(h * 0.7)
        
        # Ensure we don't go out of bounds
        frame_height = frame.shape[0]
        if extended_y + extended_height > frame_height:
            extended_height = frame_height - extended_y
        
        # Extract the expanded face region
        upper_face_img = frame[extended_y:extended_y+extended_height, x:x+w].copy()
        
        # Find eyes to refine the upper region if possible
        face_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(face_gray)
        
        upper_face_top = extended_y
        upper_face_bottom = extended_y + extended_height
        
        # If eyes are detected, ensure we include enough area above them
        if len(eyes) > 0:
            # Find the top-most position of eyes
            min_eye_y = min([eye[1] for eye in eyes])
            eye_y_in_frame = y + min_eye_y
            
            # Make sure we have enough area above the eyes (at least 3x the eye-to-face-top distance)
            min_forehead_height = min_eye_y * 3
            
            # If our current region doesn't include enough forehead, adjust it
            if eye_y_in_frame - upper_face_top < min_forehead_height:
                # Try to extend upward more
                new_top = max(0, eye_y_in_frame - min_forehead_height)
                upper_face_top = new_top
                
                # But maintain the bottom position to keep the nose area
                # No change to upper_face_bottom here
                
                # Update the image with the refined region
                if upper_face_bottom > upper_face_top + 20:  # Ensure reasonable size
                    upper_face_img = frame[upper_face_top:upper_face_bottom, x:x+w].copy()
        
        # Create region info with the updated bounds
        region_info = {
            'image': upper_face_img,
            'bounds': (x, upper_face_top, x+w, upper_face_bottom),
            'original_size': (w, upper_face_bottom - upper_face_top)
        }
        
        # Add to results
        face_regions.append({
            'regions': {'upper_face': region_info},
            'landmarks': []  # No landmarks with traditional CV
        })
        
        return True, face_regions
    
    def draw_upper_face_region(self, frame: np.ndarray, face_region: Dict) -> np.ndarray:
        """Draw the detected expanded face region on a frame with improved visualization.
        
        Args:
            frame: Input frame
            face_region: Face region dictionary
            
        Returns:
            Frame with expanded face region drawn
        """
        result = frame.copy()
        
        if 'regions' in face_region and 'upper_face' in face_region['regions']:
            bounds = face_region['regions']['upper_face']['bounds']
            x_min, y_min, x_max, y_max = bounds
            
            # Create a semi-transparent overlay for the region
            overlay = result.copy()
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), -1)
            alpha = 0.2  # Transparency factor
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
            
            # Draw a solid border around the region
            cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Add more visible label
            label_bg = result.copy()
            text = "Color Magnification Region"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Create background for text
            text_x = x_min
            text_y = max(y_min - 10, text_size[1] + 10)  # Ensure it's on screen
            cv2.rectangle(
                label_bg, 
                (text_x, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 10, text_y + 5),
                (0, 100, 0),
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
            
            # Add size information
            region_height = y_max - y_min
            region_width = x_max - x_min
            size_text = f"Size: {region_width}x{region_height} px"
            
            cv2.putText(
                result,
                size_text,
                (x_min + 5, y_max + 20),
                font,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
        
        return result 