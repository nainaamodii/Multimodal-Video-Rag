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