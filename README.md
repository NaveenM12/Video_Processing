# VideoProcessing

### [Modernizing the Polygraph: Combining Computer-Vision Magnification Techniques to Detect Lies by Naveen Mirapuri](https://docs.google.com/document/d/10ZHX9YoqdBPSwx63Rndku3zwC49oYc_GGo8EmWTmyio/edit?usp=sharing) 

Please read the paper above for detailed explanation of approach and implementation for this project. This repository contains various implementations of deception detection approaches using facial micro-expressions and physiological signals.

## Directory Structure

- **`/Automatic_Lie_Detector`** - Contains the main deception detection implementations

  - All detector code is contained exclusively in this directory
  - Implementation uses PBM and EVM techniques for deception detection
  - To run the detector, use the script within this directory

- **`/Face_Motion_Magnification`** - Contains PBM implementation for micro-expression detection

  - Supporting library used by the main detector

- **`/Face_Color_Magnification`** - Contains EVM implementation for heart rate analysis

  - Supporting library used by the main detector

- **`/Separate_Color_Motion_Magnification`** - Contains shared utilities and configuration
  - Supporting library used by the main detector

## Running the PBM/EVM Detector

To run the detector, navigate to the Automatic_Lie_Detector directory and use the included script:

```bash
cd Automatic_Lie_Detector
./run_pbm_evm_detector.sh ../trimmed_mp4/1.mp4
```

All detector-related files are stored exclusively in the Automatic_Lie_Detector directory, and the detector should only be run from within that directory.

## Understanding the PBM/EVM Detector

The PBM/EVM detector brings together multiple aspects of video processing into a single coherent framework:

1. Uses Phase-Based Magnification (PBM) exclusively for micro-expression detection
2. Uses Eulerian Video Magnification (EVM) exclusively for heart rate analysis
3. Configuration parameters are centralized in a single file (deception_config.py)
4. Feature weighting is explicitly set to ensure only PBM is used for motion and only EVM for heart rate

For detailed information about the PBM/EVM detector, see:
`/Automatic_Lie_Detector/PBM_EVM_DETECTOR_README.md`

## Legacy Files

Several files have been backed up with `.bak` extensions:

- `deception_config.py.bak`
- `pbm_evm_detector.py.bak`
- `run_pbm_evm_detector.py.bak`
- `run_pbm_evm.sh.bak`

These are preserved for reference but should not be used directly, as they have been consolidated into the `/Automatic_Lie_Detector` implementation.
