"""
Automatic Lie Detector package for detecting potential deception in videos.

This package processes videos through micro-expression and heart rate magnification,
then applies unsupervised anomaly detection to identify regions of potential deception.
"""

import os
import sys
import importlib.util

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import Separate_Color_Motion_Magnification config to sys.modules as 'config'
# This fixes the unqualified import in side_by_side_magnification.py
scmm_dir = os.path.join(parent_dir, 'Separate_Color_Motion_Magnification')
config_path = os.path.join(scmm_dir, 'config.py')
spec = importlib.util.spec_from_file_location('config', config_path)
config_module = importlib.util.module_from_spec(spec)
sys.modules['config'] = config_module
spec.loader.exec_module(config_module)

# Expose main classes
from .automatic_lie_detector import AutomaticLieDetector
from .deception_detector import DeceptionDetector

__all__ = ['AutomaticLieDetector', 'DeceptionDetector'] 