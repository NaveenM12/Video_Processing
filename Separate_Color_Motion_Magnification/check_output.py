import cv2
import os
from config import OUTPUT_VIDEO_PATH

def check_video_dimensions(video_path):
    """Check the dimensions of a video file"""
    
    # Get the full absolute path
    abs_path = os.path.abspath(video_path)
    print(f"Checking video dimensions for: {abs_path}")
    
    if not os.path.exists(abs_path):
        print(f"Error: Video file does not exist at {abs_path}")
        return
    
    try:
        cap = cv2.VideoCapture(abs_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {abs_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video dimensions:")
        print(f"- Width: {width}px")
        print(f"- Height: {height}px")
        print(f"- Aspect ratio: {width/height:.2f}")
        print(f"- FPS: {fps}")
        print(f"- Total frames: {frame_count}")
        print(f"- Duration: {frame_count/fps:.2f} seconds")
        
        # Check if the dimensions match the expected fixed values
        expected_width = 1920  # 640 * 3
        expected_height = 1260  # 360 + (300 * 3)
        
        if width == expected_width and height == expected_height:
            print("\nSUCCESS: Video dimensions match the expected fixed values!")
            print("The issue has been resolved - the output size is consistent and fixed.")
        else:
            print("\nWARNING: Video dimensions do not match the expected values.")
            print(f"Expected: {expected_width}x{expected_height}")
            print(f"Actual: {width}x{height}")
        
        # Read the first frame to confirm we can access the video
        ret, frame = cap.read()
        if ret:
            print(f"\nSuccessfully read the first frame with dimensions: {frame.shape}")
        else:
            print("\nCould not read the first frame")
        
        cap.release()
    except Exception as e:
        print(f"Error checking video: {e}")

if __name__ == "__main__":
    check_video_dimensions(OUTPUT_VIDEO_PATH) 