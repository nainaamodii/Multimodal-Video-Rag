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

@dataclass
class ExtractedFrame:
    
    frame_id: str
    path: Path
    timestamp: float
    transcript_segment: Optional[TranscriptSegment] = None
    extraction_reason: str = "unknown"
    scene_change_score: float = 0.0
    quality_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Union[str, float, Dict, None]]:
        """Convert frame to dictionary format.
        
        Returns:
            Dictionary containing all frame metadata.
            
        Example:
            >>> frame.to_dict()
            {
                'frame_id': 'frame_0001',
                'path': 'frames/frame_0001.jpg',
                'timestamp': 12.5,
                'extraction_reason': 'keyword:click',
                'quality_score': 0.85
            }
        """
        return {
            "frame_id": self.frame_id,
            "path": str(self.path),
            "timestamp": self.timestamp,
            "transcript_segment": self.transcript_segment.to_dict() if self.transcript_segment else None,
            "extraction_reason": self.extraction_reason,
            "scene_change_score": self.scene_change_score,
            "quality_score": self.quality_score,
        }