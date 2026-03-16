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
    
class FrameExtractor:
    """Extract keyframes from videos using intelligent strategies.
    
    This class provides multiple strategies for extracting important frames
    from tutorial videos:
    
    - **Scene Detection**: Identifies visual transitions and changes
    - **Transcript Alignment**: Extracts frames when action keywords are mentioned
    - **Hybrid**: Combines both approaches for optimal coverage
    
    The extractor also performs quality assessment to filter out blurry or
    low-quality frames.
    
    Attributes:
        strategy: The extraction strategy being used.
        max_frames_per_video: Maximum number of frames to extract per video.
        scene_threshold: Threshold for scene change detection (0-1).
        quality_threshold: Minimum quality score for frames (0-1).
        ACTION_KEYWORDS: List of keywords indicating important moments.
    
    Example:
        Scene detection only::
        
            extractor = FrameExtractor(strategy="scene", scene_threshold=0.3)
            frames = extractor.extract("video.mp4")
        
        Transcript-based extraction::
        
            transcript = TranscriptExtractor().extract("video.mp4")
            extractor = FrameExtractor(strategy="transcript")
            frames = extractor.extract("video.mp4", transcript=transcript)
        
        Hybrid approach (recommended)::
        
            extractor = FrameExtractor(
                strategy="hybrid",
                max_frames_per_video=20,
                quality_threshold=0.5
            )
            frames = extractor.extract("video.mp4", transcript=transcript)
    """
    
    # Action keywords that indicate important moments in tutorials
    ACTION_KEYWORDS = [
        "click", "select", "choose", "open", "close", "press", "tap",
        "drag", "drop", "scroll", "type", "enter", "delete", "save",
        "export", "import", "upload", "download", "copy", "paste",
        "button", "menu", "icon", "tab", "window", "dialog", "popup"
    ]
    
    def __init__(
        self,
        strategy: str = "hybrid",
        max_frames_per_video: int = 20,
        scene_threshold: float = 0.3,
        quality_threshold: float = 0.5,
    ) -> None:
        """Initialize the frame extractor.
        
        Args:
            strategy: Extraction strategy to use. Options:
                - 'scene': Extract frames at scene changes only
                - 'transcript': Extract frames when action keywords are mentioned
                - 'hybrid': Combine both strategies (recommended)
                Defaults to 'hybrid'.
            max_frames_per_video: Maximum number of frames to extract from each
                video. If more candidates are found, they will be evenly sampled.
                Defaults to 20.
            scene_threshold: Threshold for scene change detection (0-1).
                Higher values = only major scene changes. Lower values = more
                sensitive to changes. Defaults to 0.3.
            quality_threshold: Minimum quality score for frames (0-1).
                Frames below this threshold will be filtered out. Higher values
                = stricter quality requirements. Defaults to 0.5.
        
        Raises:
            ValueError: If strategy is not one of 'scene', 'transcript', or 'hybrid'.
        
        Example:
            >>> # Strict quality, fewer frames
            >>> extractor = FrameExtractor(
            ...     strategy="hybrid",
            ...     max_frames_per_video=10,
            ...     quality_threshold=0.7
            ... )
            
            >>> # More sensitive to scene changes
            >>> extractor = FrameExtractor(
            ...     strategy="scene",
            ...     scene_threshold=0.2
            ... )
        """
        if strategy not in ["scene", "transcript", "hybrid"]:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                "Must be 'scene', 'transcript', or 'hybrid'"
            )
        
        self.strategy = strategy
        self.max_frames_per_video = max_frames_per_video
        self.scene_threshold = scene_threshold
        self.quality_threshold = quality_threshold