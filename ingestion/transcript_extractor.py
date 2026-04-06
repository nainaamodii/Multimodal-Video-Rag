from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
import json


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    
    def to_dict(self) -> Dict[str, Union[float, str]]:
        
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text
        }


@dataclass
class Transcript:
    video_path: Path
    language: str
    segments: List[TranscriptSegment]
    full_text: str
    
    def to_dict(self) -> Dict[str, Union[str, List[Dict]]]:
        return {
            "video_path": str(self.video_path),
            "language": self.language,
            "segments": [seg.to_dict() for seg in self.segments],
            "full_text": self.full_text
        }
    
    def save(self, output_path: Union[str, Path]) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> Transcript:
        path = Path(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = [
            TranscriptSegment(**seg) for seg in data['segments']
        ]
        
        return cls(
            video_path=Path(data['video_path']),
            language=data['language'],
            segments=segments,
            full_text=data['full_text']
        )


class TranscriptExtractor:
    def __init__(
        self,
        model_size: str ="base",
        device: Optional[str] = None,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None
    ) -> None:
        
        self.model_size = model_size
        self.device = device
        self.language = language
        self.initial_prompt = initial_prompt or (
            "Machine learning lecture. Terms include: gradient descent, "
            "backpropagation, sigmoid, ReLU, softmax, regularization, "
            "hyperparameter, Jacobian, neural network, cost function."
        )
        self._model = None
    
    def _load_model(self) -> None:
        
        if self._model is None:
            try:
                import whisper
                self._model = whisper.load_model(
                    self.model_size,
                    device=self.device
                )
            except ImportError:
                raise ImportError(
                    "openai-whisper is not installed. "
                    "Install it with: pip install openai-whisper"
                )
            
    def chunk_transcript(
        self, 
        transcript: Transcript, 
        target_word_count: int = 150, 
        overlap_count: int = 30
    ) -> List[TranscriptSegment]:
        """
        Groups raw Whisper segments into uniform chunks with overlap.
        
        Args:
            transcript: The Transcript object containing raw segments.
            target_word_count: Ideal number of words per chunk.
            overlap_count: How many words to keep from the previous chunk for context.
        """
        raw_segments = transcript.segments
        chunks = []
        
        current_text = []
        current_word_count = 0
        start_time = raw_segments[0].start if raw_segments else 0.0

        for i, seg in enumerate(raw_segments):
            words = seg.text.split()
            current_text.extend(words)
            current_word_count += len(words)

            # If we reached the target size, or it's the last segment
            if current_word_count >= target_word_count or i == len(raw_segments) - 1:
                chunk_text = " ".join(current_text)
                
                chunks.append(TranscriptSegment(
                    start=start_time,
                    end=seg.end,
                    text=chunk_text
                ))

                # Handle Overlap: Keep the last 'overlap_count' words for the next chunk
                overlap_words = current_text[-overlap_count:] if len(current_text) > overlap_count else []
                current_text = overlap_words
                current_word_count = len(current_text)
                
                # The start time of the next chunk is roughly the start of the current segment
                start_time = seg.start 

        return chunks

    def extract(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        do_chunking: bool = True
    ) -> Transcript:
        """
        Transcribe a video file using Whisper.

        Args:
            video_path: Path to the video (or audio) file.
            output_path: Optional path to save the transcript JSON.
            do_chunking: If True, group raw Whisper segments into
                uniform chunks with overlap (default True).

        Returns:
            Transcript object with segments and full text.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self._load_model()
        
        result = self._model.transcribe(
            str(video_path),
            language=self.language,
            verbose=False,
            initial_prompt=self.initial_prompt
        )
        
        # Initial raw segments from Whisper
        segments = [
            TranscriptSegment(
                start=seg['start'],
                end=seg['end'],
                text=seg['text'].strip()
            )
            for seg in result['segments']
        ]
        
        temp_transcript = Transcript(
            video_path=video_path,
            language=result['language'],
            segments=segments,
            full_text=result['text'].strip()
        )

        # Apply the uniform chunking logic
        if do_chunking:
            uniform_segments = self.chunk_transcript(temp_transcript)
            temp_transcript.segments = uniform_segments
        
        if output_path:
            temp_transcript.save(Path(output_path))
        
        return temp_transcript

    # FIX BUG-6: The old extract() copy was wrapped in a triple-quoted string
    # ("""""…""") that turned extract_batch into an unreachable string literal.
    # It has been removed entirely. The active extract() above supersedes it.

    def extract_batch(
        self,
        video_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Transcript]:
        transcripts = []
        
        for video_path in video_paths:
            video_path = Path(video_path)
            
            output_path = None
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{video_path.stem}_transcript.json"
            
            transcript = self.extract(video_path, output_path)
            transcripts.append(transcript)
        
        return transcripts
