from __future__ import annotations

from typing import Dict, List , Optional
import re
from loguru import logger

from framewise.core.transcript_extractor import Transcript, TranscriptSegment


class TranscriptCorrector:

    
    def __init__(self, corrections: Optional[Dict[str, str]] = None) -> None:

        self.corrections = corrections or {}
    
    def add_correction(self, incorrect: str, correct: str) -> None:
        self.corrections[incorrect] = correct
        logger.debug(f"Added correction: '{incorrect}' → '{correct}'")
    
    def add_corrections(self, corrections: Dict[str, str]) -> None:
        self.corrections.update(corrections)
        logger.debug(f"Added {len(corrections)} corrections")
    
    def correct_text(self, text: str) -> str:
        corrected = text
        
        for incorrect, correct in self.corrections.items():
            # Case-insensitive replacement with case preservation
            def replace_preserve_case(match):
                original = match.group(0)
                if original.isupper():
                    return correct.upper()
                elif original[0].isupper():
                    return correct.capitalize()
                else:
                    return correct.lower()
            
            pattern = re.compile(re.escape(incorrect), re.IGNORECASE)
            corrected = pattern.sub(replace_preserve_case, corrected)
        
        return corrected
    
    def correct_segment(self, segment: TranscriptSegment) -> TranscriptSegment:
        corrected_text = self.correct_text(segment.text)
        
        return TranscriptSegment(
            start=segment.start,
            end=segment.end,
            text=corrected_text
        )
    
    def correct_transcript(self, transcript: Transcript) -> Transcript:
        logger.info(f"Correcting transcript with {len(self.corrections)} rules")
        
        corrected_segments = [
            self.correct_segment(seg) for seg in transcript.segments
        ]
        
        corrected_full_text = self.correct_text(transcript.full_text)
        
        # Count corrections made
        corrections_made = sum(
            1 for orig, corr in zip(transcript.segments, corrected_segments)
            if orig.text != corr.text
        )
        
        if corrections_made > 0:
            logger.success(f"Applied corrections to {corrections_made} segments")
        else:
            logger.info("No corrections needed")
        
        return Transcript(
            video_path=transcript.video_path,
            language=transcript.language,
            segments=corrected_segments,
            full_text=corrected_full_text
        )


# Common product/brand name corrections
COMMON_CORRECTIONS = {
    # Add your product-specific corrections here
    "Defali": "Definely",
    "DefaliDraft": "DefinelyDraft",
    # Add more as needed
}


def create_product_corrector(product_terms: Optional[Dict[str, str]] = None) -> TranscriptCorrector:
    corrections = COMMON_CORRECTIONS.copy()
    if product_terms:
        corrections.update(product_terms)
    
    return TranscriptCorrector(corrections)