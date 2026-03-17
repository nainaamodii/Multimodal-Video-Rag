"""Transcript extraction from video files using OpenAI Whisper.

This module provides functionality to extract audio transcripts from video files
using the Whisper speech recognition model. It supports multiple languages,
different model sizes, and batch processing.

Example:
    Basic usage::

        from framewise import TranscriptExtractor
        
        extractor = TranscriptExtractor(model_size="base")
        transcript = extractor.extract("video.mp4")
        
        print(f"Language: {transcript.language}")
        print(f"Text: {transcript.full_text}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
import json

@dataclass
class TranscriptSegment:
    """A segment of transcribed text with timing information.
    
    Represents a single segment of transcribed speech with start and end timestamps.
    Segments are typically sentence or phrase-level chunks of the transcript.
    
    Attributes:
        start: Start time of the segment in seconds.
        end: End time of the segment in seconds.
        text: The transcribed text content for this segment.
    
    Example:
        >>> segment = TranscriptSegment(start=0.0, end=2.5, text="Hello world")
        >>> print(f"[{segment.start}s - {segment.end}s]: {segment.text}")
        [0.0s - 2.5s]: Hello world
    """
    
    start: float
    end: float
    text: str
    
    def to_dict(self) -> Dict[str, Union[float, str]]:
        """Convert segment to dictionary format.
        
        Returns:
            Dictionary containing start, end, and text fields.
            
        Example:
            >>> segment = TranscriptSegment(0.0, 2.5, "Hello")
            >>> segment.to_dict()
            {'start': 0.0, 'end': 2.5, 'text': 'Hello'}
        """
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text
        }

@dataclass
class Transcript:
    """Complete transcript with metadata and segments.
    
    Represents a full video transcript including the source video path,
    detected language, individual segments with timestamps, and the
    complete transcribed text.
    
    Attributes:
        video_path: Path to the source video file.
        language: Detected or specified language code (e.g., 'en', 'es').
        segments: List of transcript segments with timing information.
        full_text: Complete transcribed text without timestamps.
    
    Example:
        >>> transcript = Transcript(
        ...     video_path=Path("video.mp4"),
        ...     language="en",
        ...     segments=[TranscriptSegment(0.0, 2.5, "Hello world")],
        ...     full_text="Hello world"
        ... )
        >>> transcript.save("transcript.json")
    """
    
    video_path: Path
    language: str
    segments: List[TranscriptSegment]
    full_text: str
    
    def to_dict(self) -> Dict[str, Union[str, List[Dict]]]:
        """Convert transcript to dictionary format.
        
        Returns:
            Dictionary containing all transcript data including video path,
            language, segments, and full text.
            
        Example:
            >>> transcript.to_dict()
            {
                'video_path': 'video.mp4',
                'language': 'en',
                'segments': [{'start': 0.0, 'end': 2.5, 'text': 'Hello'}],
                'full_text': 'Hello'
            }
        """
        return {
            "video_path": str(self.video_path),
            "language": self.language,
            "segments": [seg.to_dict() for seg in self.segments],
            "full_text": self.full_text
        }
    
    