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
        
        if strategy not in ["scene", "transcript", "hybrid"]:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                "Must be 'scene', 'transcript', or 'hybrid'"
            )
        
        self.strategy = strategy
        self.max_frames_per_video = max_frames_per_video
        self.scene_threshold = scene_threshold
        self.quality_threshold = quality_threshold
    
    def extract(
        self,
        video_path: Union[str, Path],
        transcript: Optional[Transcript] = None,
        output_dir: Union[str, Path] = "frames",
    ) -> List[ExtractedFrame]:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Extracting frames from: {video_path.name}")
        logger.info(f"Strategy: {self.strategy}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {duration:.1f}s, {fps:.1f} fps, {total_frames} frames")
        
        if self.strategy == "scene":
            candidate_timestamps = self._extract_by_scene_change(cap, fps)
        elif self.strategy == "transcript":
            if transcript is None:
                raise ValueError("Transcript required for 'transcript' strategy")
            candidate_timestamps = self._extract_by_transcript(transcript)
        elif self.strategy == "hybrid":
            scene_timestamps = self._extract_by_scene_change(cap, fps)
            transcript_timestamps = self._extract_by_transcript(transcript) if transcript else []
            candidate_timestamps = self._merge_timestamps(scene_timestamps, transcript_timestamps)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        if len(candidate_timestamps) > self.max_frames_per_video:
            step = len(candidate_timestamps) / self.max_frames_per_video
            candidate_timestamps = [
                candidate_timestamps[int(i * step)] 
                for i in range(self.max_frames_per_video)
            ]
        
        logger.info(f"Extracting {len(candidate_timestamps)} frames")
        
        extracted_frames = []
        for idx, timestamp_info in enumerate(candidate_timestamps):
            if isinstance(timestamp_info, tuple):
                timestamp, reason, score = timestamp_info
            else:
                timestamp, reason, score = timestamp_info, "unknown", 0.0
            
            frame = self._extract_frame_at_timestamp(cap, timestamp, fps)
            if frame is None:
                continue
            
            quality = self._assess_frame_quality(frame)
            if quality < self.quality_threshold:
                logger.debug(f"Skipping low quality frame at {timestamp:.1f}s")
                continue
            
            frame_id = f"frame_{idx:04d}"
            frame_filename = f"{frame_id}_t{timestamp:07.1f}s.jpg"
            frame_path = output_dir / frame_filename
            
            cv2.imwrite(str(frame_path), frame)
            segment = self._find_transcript_segment(transcript, timestamp) if transcript else None
            
            extracted_frame = ExtractedFrame(
                frame_id=frame_id,
                path=frame_path,
                timestamp=timestamp,
                transcript_segment=segment,
                extraction_reason=reason,
                scene_change_score=score,
                quality_score=quality,
            )
            
            extracted_frames.append(extracted_frame)
            logger.debug(f"Extracted: {frame_filename}")
        
        cap.release()
        
        self._save_metadata(extracted_frames, output_dir)
        
        logger.success(f"Extracted {len(extracted_frames)} frames to {output_dir}")
        return extracted_frames
    
    def _extract_by_scene_change(
        self,
        cap: cv2.VideoCapture,
        fps: float
    ) -> List[Tuple[float, str, float]]:
        
        timestamps = []
        prev_frame = None
        frame_idx = 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if prev_frame is not None:
                score = self._calculate_scene_change(prev_frame, frame)
                
                if score > self.scene_threshold:
                    timestamp = frame_idx / fps
                    timestamps.append((timestamp, "scene_change", score))
            
            prev_frame = frame
            frame_idx += 1
            
            if frame_idx % 30 == 0: 
                continue
        
        return timestamps
    
    def _extract_by_transcript(
        self,
        transcript: Transcript
    ) -> List[Tuple[float, str, float]]:
        
        timestamps = []
        
        for segment in transcript.segments:
            text_lower = segment.text.lower()
            
            for keyword in self.ACTION_KEYWORDS:
                if keyword in text_lower:
                    timestamp = (segment.start + segment.end) / 2
                    timestamps.append((timestamp, f"keyword:{keyword}", 1.0))
                    break 
        
        return timestamps
    
    def _merge_timestamps(
        self,
        scene_timestamps: List[Tuple[float, str, float]],
        transcript_timestamps: List[Tuple[float, str, float]],
        merge_window: float = 2.0
    ) -> List[Tuple[float, str, float]]:
        
        all_timestamps = scene_timestamps + transcript_timestamps
        all_timestamps.sort(key=lambda x: x[0])
        
        merged = []
        for timestamp, reason, score in all_timestamps:
            if merged and abs(merged[-1][0] - timestamp) < merge_window:
                if score > merged[-1][2]:
                    merged[-1] = (timestamp, reason, score)
            else:
                merged.append((timestamp, reason, score))
        
        return merged
    
    def _extract_frame_at_timestamp(
        self,
        cap: cv2.VideoCapture,
        timestamp: float,
        fps: float
    ) -> Optional[np.ndarray]:
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        if not ret:
            return None
        
        return frame
    
    def _calculate_scene_change(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> float:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        gray1 = cv2.resize(gray1, (320, 240))
        gray2 = cv2.resize(gray2, (320, 240))
        
        diff = cv2.absdiff(gray1, gray2)
        
        score = np.mean(diff) / 255.0
        
        return score
    
    def _assess_frame_quality(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        quality = min(laplacian_var / 500.0, 1.0)
        
        return quality
    
    def _find_transcript_segment(
        self,
        transcript: Transcript,
        timestamp: float
    ) -> Optional[TranscriptSegment]:
        for segment in transcript.segments:
            if segment.start <= timestamp <= segment.end:
                return segment
        
        closest = min(
            transcript.segments,
            key=lambda s: min(abs(s.start - timestamp), abs(s.end - timestamp))
        )
        
        if min(abs(closest.start - timestamp), abs(closest.end - timestamp)) < 2.0:
            return closest
        
        return None
    
    def _save_metadata(
        self,
        frames: List[ExtractedFrame],
        output_dir: Path
    ) -> None:
        metadata = {
            "total_frames": len(frames),
            "frames": [frame.to_dict() for frame in frames]
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved metadata to {metadata_path}")