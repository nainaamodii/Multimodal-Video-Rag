"""
This module provides functionality to generate embeddings for both images and text
using state-of-the-art models (CLIP for images, Sentence Transformers for text).
These embeddings enable semantic search across video content.

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Union
import numpy as np
from PIL import Image
import torch
from loguru import logger

from framewise.core.frame_extractor import ExtractedFrame
from framewise.core.transcript_extractor import TranscriptSegment


class FrameWiseEmbedder:
    
    
    def __init__(
        self,
        text_model: str = "all-MiniLM-L6-v2",
        vision_model: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ) -> None:
        
        self.text_model_name = text_model
        self.vision_model_name = vision_model
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing FrameWiseEmbedder on {self.device}")
        
        # Lazy load models
        self._text_model = None
        self._vision_model = None
        self._vision_processor = None
    
    def _load_text_model(self) -> None:
        
        if self._text_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading text model: {self.text_model_name}")
                self._text_model = SentenceTransformer(
                    self.text_model_name,
                    device=self.device
                )
                logger.success("Text model loaded")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Install it with: pip install sentence-transformers"
                )
    
    def _load_vision_model(self) -> None:
        
        if self._vision_model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
                logger.info(f"Loading vision model: {self.vision_model_name}")
                self._vision_model = CLIPModel.from_pretrained(self.vision_model_name)
                self._vision_processor = CLIPProcessor.from_pretrained(self.vision_model_name)
                self._vision_model = self._vision_model.to(self.device)
                logger.success("Vision model loaded")
            except ImportError:
                raise ImportError(
                    "transformers is not installed. "
                    "Install it with: pip install transformers"
                )
    
    def embed_text(self, text: str) -> np.ndarray:
        
        self._load_text_model()
        embedding = self._text_model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_text_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        
        self._load_text_model()
        embeddings = self._text_model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings
    
