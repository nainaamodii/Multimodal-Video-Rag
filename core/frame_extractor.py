"""Frame extraction from video files with intelligent keyframe selection.

This module provides functionality to extract keyframes from video files using
multiple strategies: scene detection, transcript alignment, or a hybrid approach.
It intelligently selects the most important visual moments from tutorial videos.

Example:
    Basic usage::

        from framewise import FrameExtractor, TranscriptExtractor
        
        # Extract transcript first
        transcript = TranscriptExtractor().extract("video.mp4")
        
        # Extract frames using hybrid strategy
        extractor = FrameExtractor(strategy="hybrid")
        frames = extractor.extract("video.mp4", transcript=transcript)
        
        for frame in frames:
            print(f"{frame.timestamp}s: {frame.extraction_reason}")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass
import json
import cv2
import numpy as np
from PIL import Image
from loguru import logger

from framewise.core.transcript_extractor import Transcript, TranscriptSegment