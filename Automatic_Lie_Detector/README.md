# Automatic Lie Detector

This module processes videos through micro-expression (PBM) and heart rate (EVM) magnification to detect potential regions of deception using an unsupervised anomaly detection algorithm.

## Features

- **Integrated Video Magnification**: Leverages both Phase-Based Magnification (PBM) for micro-expressions and Eulerian Video Magnification (EVM) for heart rate detection
- **Unsupervised Anomaly Detection**: Uses Isolation Forest algorithm to identify unusual physiological patterns without requiring labeled training data
- **Multi-Feature Deception Analysis**: Combines facial micro-expressions, heart rate variability, and cross-signal correlations
- **Visual Timeline**: Provides a clear visualization of potential deception regions with confidence scores
- **Real-time Analysis**: Processes videos frame-by-frame with annotations showing where deception may occur
- **Side-by-Side Comparison**: Shows original video alongside magnified versions for human verification

## How It Works

The Automatic Lie Detector uses a multi-stage approach:

1. **Magnification**: Applies PBM to amplify subtle facial movements and EVM to amplify blood flow changes related to heart rate
2. **Feature Extraction**: Extracts temporal features related to micro-expressions, heart rate, and physiological patterns
3. **Anomaly Detection**: Uses an Isolation Forest model to identify frames with unusual physiological responses
4. **Visualization**: Creates a comprehensive output video with annotated regions of potential deception

## Usage

### Basic Usage

To process all videos in the default input folder:

```bash
python run_detector.py
```

### Custom Input/Output Folders

```bash
python run_detector.py --input /path/to/videos --output /path/to/output
```

### Process a Single Video

```bash
python run_detector.py --single /path/to/video.mp4
```

## Output Format

The output video consists of:

1. Top section:

   - Original video (left)
   - Motion-magnified video (middle)
   - Color-magnified video (right)

2. Middle section:

   - Micro-expression plot showing PBM phase changes
   - Heart rate plot showing detected BPM

3. Bottom section:
   - Deception timeline showing anomaly scores
   - Visual indicators for frames with potential deception

Regions of potential deception are highlighted in red on the timeline and in the plots.

## Implementation Details

This module builds upon the existing video magnification framework:

- Uses `Face_Motion_Magnification` for micro-expression detection
- Uses `Face_Color_Magnification` for heart rate monitoring
- Extends `Separate_Color_Motion_Magnification` for combined visualization

The deception detection algorithm analyzes:

- Sudden spikes in micro-expressions
- Heart rate acceleration or variability
- Temporal correlation between different signals
- Consistency of physiological responses

## Requirements

This module has the same dependencies as the Separate_Color_Motion_Magnification module:

- OpenCV
- NumPy
- Matplotlib
- MediaPipe
- SciPy
- scikit-learn (for anomaly detection)
- PIL

## Configuration

All parameters can be adjusted in `config.py`:

- Input/output paths
- Magnification parameters
- Deception detection thresholds
- Visualization settings

## Limitations

- Works best with face-focused videos
- Requires consistent lighting for optimal performance
- False positives may occur during natural emotional responses
- Limited by the quality of facial landmark detection

## Deception Theory

The deception detection is based on leaked micro-expressions and physiological responses that occur when someone is being deceptive:

1. Brief facial expressions that contradict stated emotions
2. Elevated heart rate during deceptive statements
3. Unusual patterns of movement or stillness
4. Temporal anomalies between different physiological signals
