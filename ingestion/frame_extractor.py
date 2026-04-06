from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass
import json
import cv2
import numpy as np
from PIL import Image
from loguru import logger

from .transcript_extractor import Transcript, TranscriptSegment


@dataclass
class VideoSegment:
    segment_id: str             # "vid_001_seg_0042"
    video_id: str
    course_id: str
    start: float                # seconds
    end: float                  # seconds
    duration: float             # end - start
    text: str                   # corrected whisper transcript
    raw_text: str               # original whisper output (keep for debugging)
    frame_timestamp: float      # when to extract the frame (start + settle)
    segmentation_reason: str    # "scene_change" | "duration_split" | "merged"
    scene_change_score: float   # confidence of visual boundary


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


class FrameExtractionConfig:
    settle_offset: float = 0.5
    static_segment_offset: float = 0.2
    min_quality_score: float = 0.4
    target_resolution: Tuple = (1280, 720)
    thumbnail_resolution: Tuple = (320, 180)


class FrameExtractor:

    def __init__(
        self,
        settle_offset: float = 0.5,
        static_segment_offset: float = 0.2,
        min_quality_score: float = 0.4,
        target_resolution: Tuple[int, int] = (1280, 720),
        thumbnail_resolution: Tuple[int, int] = (320, 180),
    ) -> None:
        # FIX BUG-1: Each attribute is now assigned exactly once.
        # Previously settle_offset, static_segment_offset, and min_quality_score
        # were each assigned twice, causing custom caller values to be silently
        # overwritten by the duplicate line.
        self.settle_offset = settle_offset
        self.static_segment_offset = static_segment_offset
        self.min_quality_score = min_quality_score
        self.target_resolution = target_resolution
        self.thumbnail_resolution = thumbnail_resolution

    # ------------------------------------------------------------------
    # Public pipeline entry points
    # ------------------------------------------------------------------

    def segment_video(
        self,
        video_path: Union[str, Path],
        transcript: Transcript,
        video_id: str,
        course_id: str
    ) -> List[VideoSegment]:
        """Segment video according to EduQuery design."""
        video_path = Path(video_path)

        scene_changes = self._detect_scene_changes(video_path)
        base_segments = self._create_base_segments(scene_changes, transcript, video_id, course_id)
        bounded_segments = self._enforce_duration_bounds(base_segments)
        final_segments = self._assign_transcript_text(bounded_segments, transcript)
        final_segments = self._post_process_segments(final_segments)

        return final_segments

    def extract_frames_for_segments(
        self,
        video_path: Union[str, Path],
        segments: List[VideoSegment],
        output_dir: Union[str, Path] = "frames",
    ) -> List[ExtractedFrame]:
        """Extract frames for video segments according to EduQuery design."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Extracting frames from: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        extracted_frames = []
        for segment in segments:
            frame = self._extract_frame_for_segment(cap, segment, fps)
            if frame is not None:
                quality = self._assess_frame_quality(frame)
                if quality >= self.min_quality_score:
                    full_frame_path = self._save_frame(
                        frame, output_dir, segment.segment_id, "full", self.target_resolution
                    )
                    thumb_frame_path = self._save_frame(
                        frame, output_dir, segment.segment_id, "thumb", self.thumbnail_resolution
                    )

                    extracted_frame = ExtractedFrame(
                        frame_id=segment.segment_id,
                        path=full_frame_path,
                        timestamp=segment.frame_timestamp,
                        transcript_segment=None,
                        extraction_reason=segment.segmentation_reason,
                        scene_change_score=segment.scene_change_score,
                        quality_score=quality,
                    )
                    extracted_frame.thumbnail_path = thumb_frame_path

                    extracted_frames.append(extracted_frame)
                    logger.debug(f"Extracted: {segment.segment_id}")

        cap.release()
        self._save_metadata(extracted_frames, output_dir)

        logger.success(f"Extracted {len(extracted_frames)} frames to {output_dir}")
        return extracted_frames

    # FIX BUG-5: The legacy extract() method referenced self.strategy and
    # self.max_frames_per_video which were never defined in __init__, causing
    # AttributeError on any call. The method has been removed. The ingestion
    # pipeline uses extract_frames_for_segments() exclusively. If you need the
    # old "scene / transcript / hybrid strategy" behaviour, add the missing
    # attributes to __init__ and reinstate the method.

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_scene_changes(self, video_path: Path) -> List[Tuple[float, float]]:
        """Detect scene changes using spike detection (not flat threshold)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        scores = []
        prev_frame = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is not None:
                score = self._calculate_scene_change(prev_frame, frame)
                scores.append((frame_idx / fps, score))

            prev_frame = frame
            frame_idx += 1

        cap.release()

        scene_changes = []
        if scores:
            timestamps, score_values = zip(*scores)
            mean_score = np.mean(score_values)
            std_score = np.std(score_values)
            threshold = mean_score + 2 * std_score

            for i, (timestamp, score) in enumerate(scores):
                if score > threshold:
                    is_local_max = True
                    for j in range(max(0, i - 2), min(len(scores), i + 3)):
                        if j != i and scores[j][1] > score:
                            is_local_max = False
                            break
                    if is_local_max:
                        scene_changes.append((timestamp, score))

        return scene_changes

    def _create_base_segments(
        self,
        scene_changes: List[Tuple[float, float]],
        transcript: Transcript,
        video_id: str,
        course_id: str
    ) -> List[VideoSegment]:
        """Create base segments from scene change boundaries."""
        boundaries = [0.0]
        for timestamp, _ in scene_changes:
            boundaries.append(timestamp)
        boundaries.append(transcript.segments[-1].end if transcript.segments else 0.0)

        # Build a lookup: boundary_timestamp → scene_change_score
        scene_score_map = dict(scene_changes)

        segments = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            duration = end - start

            # FIX BUG-2: segment_id was the literal string "04d" because the
            # f-string prefix was missing.  Now properly formatted.
            segment_id = f"{video_id}_seg_{i:04d}"

            if i == 0:
                reason = "video_start"
                score = 0.0
            else:
                reason = "scene_change"
                score = scene_score_map.get(start, 0.0)

            segments.append(VideoSegment(
                segment_id=segment_id,
                video_id=video_id,
                course_id=course_id,
                start=start,
                end=end,
                duration=duration,
                text="",
                raw_text="",
                frame_timestamp=start + self.settle_offset,
                segmentation_reason=reason,
                scene_change_score=score,
            ))

        return segments

    def _enforce_duration_bounds(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """Enforce duration bounds [15 s, 120 s] by merging / splitting segments."""
        MIN_DURATION = 15.0
        MAX_DURATION = 120.0

        processed: List[VideoSegment] = []
        i = 0
        while i < len(segments):
            segment = segments[i]

            if segment.duration < MIN_DURATION:
                if i + 1 < len(segments):
                    next_seg = segments[i + 1]
                    merged = VideoSegment(
                        segment_id=f"{segment.segment_id}_merged",
                        video_id=segment.video_id,
                        course_id=segment.course_id,
                        start=segment.start,
                        end=next_seg.end,
                        duration=next_seg.end - segment.start,
                        text="",
                        raw_text="",
                        frame_timestamp=segment.start + self.settle_offset,
                        segmentation_reason="merged",
                        scene_change_score=max(
                            segment.scene_change_score, next_seg.scene_change_score
                        ),
                    )
                    processed.append(merged)
                    i += 2
                else:
                    processed.append(segment)
                    i += 1
            elif segment.duration > MAX_DURATION:
                processed.extend(self._split_long_segment(segment))
                i += 1
            else:
                processed.append(segment)
                i += 1

        return processed

    def _split_long_segment(self, segment: VideoSegment) -> List[VideoSegment]:
        """Split a long segment into roughly 60-second pieces."""
        num_splits = int(segment.duration // 60) + 1
        split_duration = segment.duration / num_splits

        splits = []
        for j in range(num_splits):
            start = segment.start + j * split_duration
            end = min(segment.start + (j + 1) * split_duration, segment.end)
            splits.append(VideoSegment(
                segment_id=f"{segment.segment_id}_split_{j}",
                video_id=segment.video_id,
                course_id=segment.course_id,
                start=start,
                end=end,
                duration=end - start,
                text="",
                raw_text="",
                frame_timestamp=start + self.settle_offset,
                segmentation_reason="duration_split",
                scene_change_score=0.0,
            ))
        return splits

    def _assign_transcript_text(
        self, segments: List[VideoSegment], transcript: Transcript
    ) -> List[VideoSegment]:
        """Assign overlapping transcript text to each segment."""
        for segment in segments:
            overlapping = [
                ts.text
                for ts in transcript.segments
                if ts.start < segment.end and ts.end > segment.start
            ]
            segment.text = " ".join(overlapping)
            segment.raw_text = segment.text
        return segments

    def _post_process_segments(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """Post-process segments (ML vocabulary correction placeholder)."""
        return segments

    def _extract_frame_for_segment(
        self,
        cap: cv2.VideoCapture,
        segment: VideoSegment,
        fps: float,
    ) -> Optional[np.ndarray]:
        frame_number = int(segment.frame_timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        return frame if ret else None

    def _save_frame(
        self,
        frame: np.ndarray,
        output_dir: Path,
        segment_id: str,
        suffix: str,
        resolution: Tuple[int, int],
    ) -> Path:
        resized = cv2.resize(frame, resolution, interpolation=cv2.INTER_LANCZOS4)
        filepath = output_dir / f"{segment_id}_{suffix}.jpg"
        cv2.imwrite(str(filepath), resized)
        return filepath

    def _calculate_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        gray1 = cv2.resize(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), (320, 240))
        gray2 = cv2.resize(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), (320, 240))
        return float(np.mean(cv2.absdiff(gray1, gray2)) / 255.0)

    def _assess_frame_quality(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(min(laplacian_var / 500.0, 1.0))

    def _find_transcript_segment(
        self,
        transcript: Transcript,
        timestamp: float,
    ) -> Optional[TranscriptSegment]:
        for segment in transcript.segments:
            if segment.start <= timestamp <= segment.end:
                return segment

        closest = min(
            transcript.segments,
            key=lambda s: min(abs(s.start - timestamp), abs(s.end - timestamp)),
        )
        if min(abs(closest.start - timestamp), abs(closest.end - timestamp)) < 2.0:
            return closest
        return None

    def _save_metadata(self, frames: List[ExtractedFrame], output_dir: Path) -> None:
        metadata = {
            "total_frames": len(frames),
            "frames": [frame.to_dict() for frame in frames],
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved metadata to {metadata_path}")
