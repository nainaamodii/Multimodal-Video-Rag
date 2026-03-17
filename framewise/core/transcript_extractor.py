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
        model_size: str = "base",
        device: Optional[str] = None,
        language: Optional[str] = None
    ) -> None:
        
        self.model_size = model_size
        self.device = device
        self.language = language
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
    
    def extract(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Transcript:
        
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self._load_model()
        
        result = self._model.transcribe(
            str(video_path),
            language=self.language,
            verbose=False
        )
        
        segments = [
            TranscriptSegment(
                start=seg['start'],
                end=seg['end'],
                text=seg['text'].strip()
            )
            for seg in result['segments']
        ]
        
        transcript = Transcript(
            video_path=video_path,
            language=result['language'],
            segments=segments,
            full_text=result['text'].strip()
        )
        
        if output_path:
            transcript.save(Path(output_path))
        
        return transcript
    
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