# PBM/EVM Exclusive Deception Detector

This detector is a specialized implementation that exclusively uses:

1. **Phase-Based Magnification (PBM)** for micro-expression detection
2. **Eulerian Video Magnification (EVM)** for heart rate analysis

No other motion detection or physiological measurement techniques are used. The purpose of this implementation is to evaluate how effectively these specific technologies can be used for deception detection.

## Implementation Details

This implementation is fully contained within the `Automatic_Lie_Detector` directory. It does not rely on any code outside this directory except for the underlying PBM and EVM libraries from the parent project.

## Key Files

- **pbm_evm_detection.py** - Main implementation that uses PBM/EVM exclusively
- **deception_config.py** - Configuration file with parameters for PBM and EVM
- **run_pbm_evm_detector.sh** - Shell script to run the detector

## How It Works

The detector operates by:

1. **Micro-Expression Detection**: Uses PBM to detect subtle facial movements

   - Applies steerable pyramid decomposition to extract phase changes
   - Processes key facial regions: eyes and nose tip
   - Measures phase changes between consecutive frames

2. **Heart Rate Analysis**: Uses EVM to detect subtle color changes in the skin

   - Amplifies color changes in skin regions (cheeks)
   - Reveals blood flow patterns
   - Calculates heart rate variations

3. **Detection Process**:
   - Detects facial regions using MediaPipe
   - Applies PBM to extract phase changes from selected regions
   - Applies EVM to extract heart rate data
   - Combines these signals with feature weighting: 100% PBM for motion, 50% EVM for heart rate, 0% for cross-correlation
   - Identifies regions with significant micro-expressions and heart rate changes

## How to Run

> **IMPORTANT**: This detector must be run from within the Automatic_Lie_Detector directory.

### Single Video Processing

```bash
cd /Users/naveenmirapuri/VideoProcessing/Automatic_Lie_Detector
./run_pbm_evm_detector.sh ../trimmed_mp4/1.mp4
```

Note that all paths should be relative to the Automatic_Lie_Detector directory.

## Configuration

Key parameters in `deception_config.py`:

### PBM Parameters (Micro-Expressions)

```python
PBM_PARAMS = {
    'phase_mag': 15.0,       # Magnification factor
    'f_lo': 0.25,            # Low cutoff frequency
    'f_hi': 0.3,             # High cutoff frequency
    'sigma': 1.0,            # Sigma for Gaussian smoothing
    'attenuate': True        # Attenuate frequencies outside band
}
```

### EVM Parameters (Heart Rate)

```python
EVM_PARAMS = {
    'alpha': 50,             # Amplification factor
    'level': 3,              # Pyramid level
    'f_lo': 0.83,            # Low cutoff frequency (0.83/30 Hz = ~50 BPM)
    'f_hi': 1.0              # High cutoff frequency (1.0/30 Hz = ~60 BPM)
}
```

### Feature Weights

```python
FEATURE_WEIGHTS = {
    'phase_change': 1.0,      # Full weight for PBM micro-expression features
    'heart_rate': 0.5,        # Half weight for EVM heart rate features
    'cross_correlation': 0.0  # No weight for correlation - not using any other technique
}
```

## Output

The detector produces a video with:

- Original video feed
- PBM-magnified regions showing micro-expressions
- EVM-magnified regions showing heart rate
- Timeline of detected micro-expressions and heart rate changes
- Highlighted deception region based on PBM and EVM signals

## Research Purpose

This implementation is specifically designed for a research study evaluating how well PBM and EVM can detect deception through micro-expressions and physiological changes. It is not intended to be a comprehensive lie detector but rather a focused study on these specific technologies.
