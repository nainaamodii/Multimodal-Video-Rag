"""
EduQuery - Final Q&A Pipeline
==============================
Usage:
    python test_qa.py --video "path/to/lecture.mp4"
    python test_qa.py --video "path/to/lecture.mp4" --question "What is gradient descent?"
    python test_qa.py --video "path/to/lecture.mp4" --summary
    python test_qa.py --video "path/to/lecture.mp4" --force-reprocess

All storage is derived from the video filename — no hardcoding anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from loguru import logger
from ingestion.transcript_extractor import TranscriptExtractor
from ingestion.frame_extractor import FrameExtractor
from query.embeddings.embedder import EduQueryEmbedder
from storage.retrieval.vector_store import EduQueryVectorStore
from storage.retrieval.qa_system import EduQueryQA
from dotenv import load_dotenv

load_dotenv()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    """Convert arbitrary filename text to a safe slug (lowercase, hyphens)."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text[:64]  # cap length


def derive_video_meta(video_path: Path) -> dict:
    """
    Derive all video metadata purely from the file path — no hardcoding.

    Returns:
        video_id   : short slug + 8-char hash  (e.g. "lecture4-a1b2c3d4")
        course_id  : parent folder slug         (e.g. "ml-course")
        video_title: cleaned filename           (e.g. "Lecture 4")
        db_path    : Path to per-video LanceDB
        frames_dir : Path to per-video frames folder
    """
    stem      = video_path.stem                          # filename without extension
    parent    = video_path.parent.name or "default"      # immediate parent folder
    file_hash = hashlib.md5(str(video_path.resolve()).encode()).hexdigest()[:8]

    video_slug  = _slugify(stem)
    course_slug = _slugify(parent)
    video_id    = f"{video_slug[:48]}-{file_hash}"       # unique even if names clash
    course_id   = course_slug

    # Human-readable title: strip bracketed YouTube IDs, clean up separators
    clean_title = re.sub(r"\[.*?\]", "", stem).strip()
    clean_title = re.sub(r"[\s_-]+", " ", clean_title).strip()

    # Storage roots — everything lives under processed_videos/<video_id>/
    storage_root = Path("processed_videos") / video_id
    db_path      = storage_root / "vector_store.db"
    frames_dir   = storage_root / "frames"

    return {
        "video_id":    video_id,
        "course_id":   course_id,
        "video_title": clean_title,
        "db_path":     db_path,
        "frames_dir":  frames_dir,
        "storage_root": storage_root,
    }


def is_already_processed(meta: dict) -> bool:
    """Return True only if BOTH the DB and frames folder exist and are non-empty."""
    db_path    = meta["db_path"]
    frames_dir = meta["frames_dir"]

    db_exists     = db_path.exists() and any(db_path.iterdir())
    frames_exist  = frames_dir.exists() and any(frames_dir.iterdir())

    return db_exists and frames_exist


# ── Processing pipeline ───────────────────────────────────────────────────────

def process_video(video_path: Path, meta: dict) -> tuple:
    """
    Run full ingestion pipeline for a video.
    Returns (vector_store, embedder) ready for querying.
    """
    from ingestion.transcript_extractor import TranscriptExtractor
    from ingestion.frame_extractor import FrameExtractor
    from query.embeddings.embedder import EduQueryEmbedder
    from storage.retrieval.vector_store import EduQueryVectorStore

    storage_root = meta["storage_root"]
    frames_dir   = meta["frames_dir"]
    db_path      = meta["db_path"]
    video_id     = meta["video_id"]
    course_id    = meta["course_id"]
    video_title  = meta["video_title"]

    storage_root.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Transcript ────────────────────────────────────────────────────
    logger.info("Step 1/5 — Extracting transcript...")
    transcript_extractor = TranscriptExtractor(model_size="base")
    transcript_path      = storage_root / "transcript.json"
    transcript = transcript_extractor.extract(
        str(video_path),
        output_path=transcript_path,
    )
    logger.success(f"  ✓ {len(transcript.segments)} transcript segments")

    # ── Step 2: Segment video ─────────────────────────────────────────────────
    logger.info("Step 2/5 — Segmenting video...")
    frame_extractor = FrameExtractor(
        settle_offset=0.5,
        static_segment_offset=0.2,
        min_quality_score=0.4,
        target_resolution=(1280, 720),
        thumbnail_resolution=(320, 180),
    )
    segments = frame_extractor.segment_video(
        video_path=str(video_path),
        transcript=transcript,
        video_id=video_id,
        course_id=course_id,
    )
    logger.success(f"  ✓ {len(segments)} video segments")

    # ── Step 3: Extract frames ────────────────────────────────────────────────
    logger.info("Step 3/5 — Extracting frames...")
    frames = frame_extractor.extract_frames_for_segments(
        video_path=str(video_path),
        segments=segments,
        output_dir=str(frames_dir),
    )
    logger.success(f"  ✓ {len(frames)} frames extracted")

    if not frames:
        logger.error("No frames extracted — cannot proceed.")
        sys.exit(1)

    # ── Step 4: Embeddings ────────────────────────────────────────────────────
    logger.info("Step 4/5 — Generating embeddings...")
    embedder = EduQueryEmbedder(
        text_model="all-MiniLM-L6-v2",
        vision_model="openai/clip-vit-base-patch32",
        device="cpu",
    )

    embedded_segments = []
    for segment in segments:
        frame = next((f for f in frames if f.frame_id == segment.segment_id), None)
        if frame is None:
            continue

        text_emb  = embedder.embed_text(segment.text)
        image_emb = embedder.embed_image(frame.path)

        embedded_segments.append({
            # Core vector store fields
            "frame_id":          segment.segment_id,
            "frame_path":        str(frame.path),
            "timestamp":         segment.start,
            "extraction_reason": segment.segmentation_reason,
            "quality_score":     frame.quality_score,
            # Text + embeddings
            "text":              segment.text,
            "text_embedding":    text_emb,
            "image_embedding":   image_emb,
            # Metadata
            "segment_id":        segment.segment_id,
            "video_id":          video_id,
            "course_id":         course_id,
            "start_time":        segment.start,
            "end_time":          segment.end,
            "thumb_url":         str(frame.thumbnail_path) if hasattr(frame, "thumbnail_path") else "",
            "video_url":         "",          # local file, no URL
            "video_title":       video_title,
            "course_title":      course_id,
        })

    logger.success(f"  ✓ {len(embedded_segments)} embeddings generated")

    # ── Step 5: Vector store ──────────────────────────────────────────────────
    logger.info("Step 5/5 — Building vector database...")
    vector_store = EduQueryVectorStore(
        db_path=str(db_path),
        table_name="frames",
    )
    vector_store.create_table(embedded_segments, mode="overwrite")
    logger.success("  ✓ Vector database ready")

    return vector_store, embedder


def load_existing(meta: dict) -> tuple:
    """Load an already-processed video's vector store and embedder."""
    from query.embeddings.embedder import EduQueryEmbedder
    from storage.retrieval.vector_store import EduQueryVectorStore

    logger.info("Loading existing processed data...")
    embedder = EduQueryEmbedder(
        text_model="all-MiniLM-L6-v2",
        vision_model="openai/clip-vit-base-patch32",
        device="cpu",
    )
    vector_store = EduQueryVectorStore(
        db_path=str(meta["db_path"]),
        table_name="frames",
    )
    logger.success("  ✓ Loaded from cache")
    return vector_store, embedder


# ── Summary ───────────────────────────────────────────────────────────────────

def get_summary(qa) -> str:
    """Generate a structured, comprehensive summary of the video."""
    summary_question = (
        "You are an expert tutor. Please provide a detailed, comprehensive summary of this lecture. "
        "1. Start with a brief 'Overview' paragraph. "
        "2. Break down the core concepts into bulleted sections. "
        "3. Explain the technical processes mentioned (e.g., how the algorithm works). "
        "4. Summarize the key takeaways for a student. "
        "Use enough detail to act as a study guide."
    )
    # Increase top_k to include more video segments for context
    response = qa.ask(summary_question, top_k=20) 
    return response["answer"]


# ── QA ────────────────────────────────────────────────────────────────────────

def ask_question(qa, question: str) -> dict:
    """
    Ask a question scoped strictly to this video.
    Returns the full response dict (answer + sources).
    """
    return qa.ask(question, top_k=5)


def print_response(response: dict) -> None:
    """Pretty-print a QA response."""
    logger.info("\n" + "=" * 60)
    logger.success("Answer:")
    print(f"\n{response['answer']}\n")

    sources = response.get("sources", [])
    if sources:
        logger.info(f"Sources ({len(sources)}):")
        for i, src in enumerate(sources, 1):
            start = src.get("start_time", 0)
            end   = src.get("end_time", 0)
            title = src.get("video_title", "")
            preview = src.get("text_preview", "")[:100]
            print(f"  [{i}] {title} | {int(start//60)}:{int(start%60):02d} – {int(end//60)}:{int(end%60):02d}")
            print(f"       {preview}...")


# ── Interactive QA loop ───────────────────────────────────────────────────────

def interactive_loop(qa, meta: dict) -> None:
    """Run an interactive Q&A session in the terminal."""
    logger.info(f"\nInteractive Q&A — Video: {meta['video_title']}")
    logger.info("Type your question and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        response = ask_question(qa, question)
        print_response(response)
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def build_qa(vector_store, embedder):
    """Initialise EduQueryQA — raises ValueError if GOOGLE_API_KEY missing."""
    from storage.retrieval.qa_system import EduQueryQA
    return EduQueryQA(
        vector_store=vector_store,
        embedder=embedder,
        model="gemini-2.5-flash",
        max_tokens=1024,
        temperature=0.7,
        api_key=None,   # reads GOOGLE_API_KEY from .env
    )


def main():
    parser = argparse.ArgumentParser(
        description="EduQuery — Video Q&A Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to the video file (e.g. dataset/lecture4.mp4)",
    )
    parser.add_argument(
        "--question", default=None,
        help="Ask a single question and exit.",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print video summary and exit.",
    )
    parser.add_argument(
        "--force-reprocess", action="store_true",
        help="Re-process the video even if already cached.",
    )
    args = parser.parse_args()

    # ── Resolve video path ────────────────────────────────────────────────────
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    # ── Derive all metadata from filename — zero hardcoding ──────────────────
    meta = derive_video_meta(video_path)

    logger.info("=" * 60)
    logger.info(f"Video      : {video_path.name}")
    logger.info(f"Video ID   : {meta['video_id']}")
    logger.info(f"Course ID  : {meta['course_id']}")
    logger.info(f"Storage    : {meta['storage_root']}")
    logger.info("=" * 60)

    # ── Process or load ───────────────────────────────────────────────────────
    if args.force_reprocess or not is_already_processed(meta):
        if args.force_reprocess:
            logger.info("Force-reprocess flag set — re-processing video...")
        else:
            logger.info("Video not yet processed — starting ingestion pipeline...")
        vector_store, embedder = process_video(video_path, meta)
    else:
        logger.info("Video already processed — loading from cache...")
        vector_store, embedder = load_existing(meta)

    # ── Build QA system ───────────────────────────────────────────────────────
    try:
        qa = build_qa(vector_store, embedder)
        logger.success("Q&A system ready.\n")
    except ValueError as e:
        logger.error(str(e))
        logger.info("Set GOOGLE_API_KEY in your .env file and retry.")
        sys.exit(1)

    # ── Summary mode ──────────────────────────────────────────────────────────
    if args.summary:
        logger.info("Generating video summary...")
        summary = get_summary(qa)
        print("\n" + "=" * 60)
        print(f"SUMMARY — {meta['video_title']}")
        print("=" * 60)
        print(summary)
        print("=" * 60)
        return

    # ── Single-question mode ──────────────────────────────────────────────────
    if args.question:
        response = ask_question(qa, args.question)
        print_response(response)
        return

    # ── Default: interactive loop ─────────────────────────────────────────────
    # Show summary first, then open Q&A
    logger.info("Generating video summary...")
    summary = get_summary(qa)
    print("\n" + "=" * 60)
    print(f"SUMMARY — {meta['video_title']}")
    print("=" * 60)
    print(summary)
    print("=" * 60 + "\n")

    interactive_loop(qa, meta)


if __name__ == "__main__":
    main()