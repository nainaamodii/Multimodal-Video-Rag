"""
EduQuery Ingester - Main ingestion pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import json
import os
from loguru import logger

from .transcript_extractor import TranscriptExtractor
from .frame_extractor import FrameExtractor, VideoSegment
from query.embeddings.embedder import EduQueryEmbedder
from storage.retrieval.vector_store import EduQueryVectorStore


class EduQueryIngester:
    """Main ingestion pipeline for EduQuery."""
    
    def __init__(
        self,
        transcript_extractor: Optional[TranscriptExtractor] = None,
        frame_extractor: Optional[FrameExtractor] = None,
        embedder: Optional[EduQueryEmbedder] = None,
        vector_store: Optional[EduQueryVectorStore] = None,
    ):
        self.transcript_extractor = transcript_extractor or TranscriptExtractor()
        self.frame_extractor = frame_extractor or FrameExtractor()
        self.embedder = embedder or EduQueryEmbedder()
        self.vector_store = vector_store or EduQueryVectorStore()
    
    def ingest_video(
        self,
        video_path: Union[str, Path],
        course_id: str,
        video_id: str,
        output_dir: Union[str, Path] = "processed_videos"
    ) -> List[VideoSegment]:
        """Ingest  video according """
        video_path = Path(video_path)
        output_dir = Path(output_dir) / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting ingestion of video: {video_id}")
        
        # Step 1: Extract transcript
        logger.info("Step 1: Extracting transcript...")
        transcript_path = output_dir / "transcript.json"
        transcript = self.transcript_extractor.extract(
            video_path, 
            output_path=transcript_path
        )
        
        # Step 2: Segment video
        logger.info("Step 2: Segmenting video...")
        segments = self.frame_extractor.segment_video(
            video_path, transcript, video_id, course_id
        )
        
        # Step 3: Extract frames
        logger.info("Step 3: Extracting frames...")
        frames_dir = output_dir / "frames"
        extracted_frames = self.frame_extractor.extract_frames_for_segments(
            video_path, segments, frames_dir
        )
        
        # Step 4: Embed segments
        logger.info("Step 4: Generating embeddings...")
        embedded_segments = self._embed_segments(segments, extracted_frames)
        
        # Step 5: Store in vector database
        logger.info("Step 5: Storing in vector database...")
        self._store_segments(embedded_segments)
        
        logger.success(f"Successfully ingested video {video_id} with {len(segments)} segments")
        return segments
    
    def _embed_segments(
        self, 
        segments: List[VideoSegment], 
        frames: List["ExtractedFrame"]
    ) -> List[Dict]:
        """Generate embeddings for segments."""
        embedded_data = []
        
        for segment in segments:
            # Find corresponding frame
            frame = next((f for f in frames if f.frame_id == segment.segment_id), None)
            if not frame:
                continue
            
            # Generate embeddings
            text_emb = self.embedder.embed_text(segment.text)
            image_emb = self.embedder.embed_image(frame.path)
            
            # Create embedded segment data.
            # frame_id and frame_path are required by vector_store.create_table
            # (core display fields). All other keys feed the design §2 payload.
            embedded_data.append({
                # --- vector_store required core fields ---
                "frame_id":         segment.segment_id,
                "frame_path":       str(frame.path),
                "timestamp":        segment.start,
                "extraction_reason": segment.segmentation_reason,
                "quality_score":    frame.quality_score,
                # --- text / embeddings ---
                "text":             segment.text,
                "text_embedding":   text_emb,
                "image_embedding":  image_emb,
                # --- design §2 payload fields ---
                "segment_id":       segment.segment_id,
                "video_id":         segment.video_id,
                "course_id":        segment.course_id,
                "start_time":       segment.start,
                "end_time":         segment.end,
                "thumb_url":        str(frame.thumbnail_path) if hasattr(frame, "thumbnail_path") else "",
                "video_url":        f"https://youtube.com/watch?v={segment.video_id}",
                "video_title":      f"Video {segment.video_id}",
                "course_title":     f"Course {segment.course_id}",
            })
        
        return embedded_data
    
    def _store_segments(self, embedded_segments: List[Dict]) -> None:
        """Store embedded segments in vector database."""
        self.vector_store.create_table(embedded_segments, mode="append")