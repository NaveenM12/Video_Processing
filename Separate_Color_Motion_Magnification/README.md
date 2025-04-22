# Side-by-Side Video Magnification for Micro-Expression and Heart Rate Detection

This module processes videos to produce a side-by-side comparison of:

1. The original video
2. Motion-magnified video (phase-based, optimized for micro-expressions)
3. Color-magnified video (Eulerian, optimized for heart rate detection)

## Features

- **Optimized Micro-Expression Detection**: Specially tuned to detect subtle facial movements during speech that may indicate deception
- **Heart Rate Detection**: Visual amplification of blood flow variations in facial skin to reveal heart rate
- **Precision Region Tracking**: Focuses on smaller, more targeted facial regions for better micro-expression detection
- **Direct Implementation**: Uses the existing implementations from Face_Motion_Magnification and Face_Color_Magnification directly

## Structure

The implementation consists of:

- `side_by_side_magnification.py`: Main implementation that coordinates motion and color magnification
- `upper_face_detector.py`: Utility for detecting the upper face region for heart rate visualization
- `config.py`: Configuration parameters for both color and motion magnification
- `run_magnification.py`: Entry point script to run the magnification

## Usage

### Configuration-Based Approach

Simply edit the `config.py` file to set your parameters, then run the script:

```bash
python run_magnification.py
```

No command-line arguments needed! All settings are read from the configuration file.

### Configuration Options

Edit `config.py` to customize:

```python
# Motion Magnification Parameters - Optimized for detecting micro-expressions
MOTION_MAG_PARAMS = {
    'phase_mag': 25.0,       # Higher magnification factor for subtle movements
    'f_lo': 0.4,             # Higher low cutoff to filter out normal speech movements
    'f_hi': 1.2,             # Higher high cutoff to capture quick micro-expressions
    'sigma': 3.0,            # Increased sigma for better amplitude-weighted blurring
    'attenuate': True        # Attenuate other frequencies to focus on the band of interest
}

# Color Magnification Parameters - Optimized for heart rate detection
COLOR_MAG_PARAMS = {
    'alpha': 50.0,           # Color magnification factor
    'level': 4,              # Gaussian pyramid level (4-6 works well)
    'f_lo': 0.83,            # Lower boundary ~50 BPM (0.83 Hz)
    'f_hi': 2.0,             # Upper boundary ~120 BPM (2.0 Hz)
}
```

## Implementation Details

This module directly uses:

- `FacialPhaseMagnification` from the Face_Motion_Magnification module for motion magnification
- `ColorMagnification` from the Face_Color_Magnification module for color magnification
- Custom `UpperFaceDetector` class to isolate the face area above the mouth
- MediaPipe Face Mesh for precise facial landmark tracking

## Output Format

The output video is a horizontal concatenation of three videos:

1. Left side: Original video labeled "Original"
2. Middle: Motion-magnified video labeled "Motion Magnified" (optimized for micro-expressions)
3. Right side: Color-magnified video labeled "Color Magnified" (optimized for heart rate detection)

All three videos maintain their original aspect ratio, and labels are added with semi-transparent backgrounds for better visibility.

## Heart Rate Detection

The color magnification component has been updated to use the cheek regions instead of the upper face region for heart rate detection. This follows the approach outlined in ["How to detect Heart Rate in Videos" by Isaac Berrios](https://medium.com/@itberrios6/how-to-detect-heart-rate-in-videos-3dbbf1eb62fd).

### Why Cheeks?

Cheeks are ideal for heart rate detection because:

1. They have good blood perfusion due to facial arteries
2. They have less movement compared to other facial regions
3. They are less affected by expressions than forehead or lips
4. They have thinner skin allowing better visibility of blood volume changes

The system now:

- Detects left and right cheek regions using MediaPipe Face Mesh
- Applies color magnification to enhance subtle color changes in the cheeks
- Combines signals from both cheeks for improved heart rate estimation
- Visualizes both the magnified video and the pulse signal

## Dependencies

- OpenCV
- NumPy
- Matplotlib
- MediaPipe
- SciPy
