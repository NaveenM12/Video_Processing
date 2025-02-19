import cv2
import numpy as np
from Face_Motion_Magnification.face_region_motion_magnification import FacialPhaseMagnification
from Face_Color_Magnification.face_region_color_magnification import FacialColorMagnification

class CombinedFacialMagnification:
    def __init__(self):
        """Initialize both motion and color magnification processors"""
        self.motion_processor = FacialPhaseMagnification()
        self.color_processor = FacialColorMagnification()
    
    def process_video(self, input_path: str, output_path: str, alpha_color: float = 0.5):
        """Process video with both motion and color magnification"""
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
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
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
        
        # Initialize output frames with original frames
        output_frames = all_frames.copy()
        
        print("Processing motion magnification...")
        # Process motion magnification
        motion_processor = FacialPhaseMagnification()
        face_motion_results = motion_processor.face_detector.detect_faces(all_frames[0])
        
        if face_motion_results:
            for face_idx in range(len(face_motion_results)):
                for region_name in ['left_eye', 'right_eye', 'nose_tip', 'left_mouth', 'right_mouth']:
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
                    
                    if region_frames:
                        magnified_frames = motion_processor.phase_magnifier.magnify(region_frames)
                        
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
        
        print("Processing color magnification...")
        # Process color magnification
        color_processor = FacialColorMagnification()
        face_color_results = color_processor.face_detector.detect_faces(all_frames[0])
        
        if face_color_results:
            for face_idx in range(len(face_color_results)):
                for region_name in ['forehead', 'left_cheek', 'right_cheek']:
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
        
        print("Writing output video...")
        # Write output frames
        for i, frame in enumerate(output_frames):
            out.write(frame)
            if i % 10 == 0:
                print(f"Written {i}/{len(output_frames)} frames")
        
        out.release()
        print("Processing complete!")

if __name__ == "__main__":
    # Define input and output paths
    input_video_path = "test_videos/face.mp4"
    output_video_path = "Combined_Magnification/output_videos/output.mp4"
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # Process the video
    processor = CombinedFacialMagnification()
    processor.process_video(input_video_path, output_video_path)