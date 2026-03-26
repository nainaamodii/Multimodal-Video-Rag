"""
Example: Complete FrameWise Pipeline

This demonstrates the full pipeline:
1. Extract transcript from video
2. Extract keyframes
3. Generate embeddings
4. Store in vector database
5. Search for relevant frames
"""

from pathlib import Path
from loguru import logger
from framewise import (
    TranscriptExtractor,
    FrameExtractor,
    FrameWiseEmbedder,
    FrameWiseVectorStore,
)


def main():
    """Run the complete FrameWise pipeline"""
    
    
    # CONFIGURE THIS
    
    VIDEO_PATH = "dataset\#3 Machine Learning Specialization [Course 1, Week 1, Lesson 2] [XtlwSmJfUs4].mp4"
    
    if not Path(VIDEO_PATH).exists():
        logger.error(f" Video not found: {VIDEO_PATH}")
        logger.info("Update VIDEO_PATH and run again")
        return
    
    logger.info(" FrameWise Complete Pipeline Demo")
    logger.info("=" * 60)
    
    
    # STEP 1: Extract Transcript
    
    logger.info("\n Step 1: Extracting Transcript")
    logger.info("-" * 60)
    
    transcript_extractor = TranscriptExtractor(model_size="base")
    transcript = transcript_extractor.extract(VIDEO_PATH)
    
    logger.success(f" Extracted {len(transcript.segments)} transcript segments")
    
    
    # STEP 2: Extract Frames
    
    logger.info("\n Step 2: Extracting Keyframes")
    logger.info("-" * 60)
    
    frame_extractor = FrameExtractor(
        strategy="hybrid",
        max_frames_per_video=15,
        scene_threshold=0.3,
        quality_threshold=0.3
    )
    
    frames = frame_extractor.extract(
        video_path=VIDEO_PATH,
        transcript=transcript,
        output_dir="pipeline_frames"
    )
    
    logger.success(f" Extracted {len(frames)} keyframes")
    
    
    # STEP 3: Generate Embeddings
    
    logger.info("\n Step 3: Generating Embeddings")
    logger.info("-" * 60)
    
    embedder = FrameWiseEmbedder(
        text_model="all-MiniLM-L6-v2",
        vision_model="openai/clip-vit-base-patch32",
        device="cpu"  # Use "cuda" if you have GPU
    )
    
    # Get embedding dimensions
    dims = embedder.get_embedding_dimensions()
    logger.info(f"Text embedding dimension: {dims['text_embedding_dim']}")
    logger.info(f"Image embedding dimension: {dims['image_embedding_dim']}")
    
    # Generate embeddings for all frames
    embeddings = embedder.embed_frames_batch(frames, batch_size=4)
    
    logger.success(f"✓ Generated embeddings for {len(embeddings)} frames")
    
    
    # STEP 4: Store in Vector Database
    
    logger.info("\n Step 4: Storing in Vector Database")
    logger.info("-" * 60)
    
    vector_store = FrameWiseVectorStore(
        db_path="pipeline_demo.db",
        table_name="tutorial_frames"
    )
    
    vector_store.create_table(embeddings, mode="overwrite")
    
    stats = vector_store.get_stats()
    logger.success(f"✓ Stored {stats['total_frames']} frames in database")
    

    # STEP 5: Search for Relevant Frames
    logger.info("\n Step 5: Semantic Search")
    logger.info("-" * 60)
    
    # Example queries
    queries = [
        "What is machine learning?",
        "What is checkers program?",
        "How to better the performance of a model?",
    ]
    
    for query in queries:
        logger.info(f"\n Query: '{query}'")
        
        results = vector_store.search_by_text(
            query_text=query,
            embedder=embedder,
            limit=3,
            search_type="hybrid"
        )
        
        if results:
            logger.info(f"   Found {len(results)} relevant frames:")
            for i, result in enumerate(results, 1):
                logger.info(f"\n   Result {i}:")
                logger.info(f"     Time: {result['timestamp']:.1f}s")
                logger.info(f"     Text: '{result['text']}'")
                logger.info(f"     Frame: {Path(result['frame_path']).name}")
                logger.info(f"     Similarity: {result.get('_distance', 'N/A')}")
        else:
            logger.warning("   No results found")
    

    # SUMMARY
    
    logger.info("\n" + "=" * 60)
    logger.success(" Complete Pipeline Executed Successfully!")
    logger.info("\n Summary:")
    logger.info(f"   Transcript segments: {len(transcript.segments)}")
    logger.info(f"   Keyframes extracted: {len(frames)}")
    logger.info(f"   Embeddings generated: {len(embeddings)}")
    logger.info(f"   Database entries: {stats['total_frames']}")
    
    logger.info("\n Output files:")
    logger.info("   - pipeline_frames/ (extracted frames)")
    logger.info("   - pipeline_demo.db/ (vector database)")
    


if __name__ == "__main__":
    logger.info(" FrameWise - Complete Pipeline Demo")
    logger.info("=" * 60)
    
    
    logger.info("\n" + "=" * 60)
    logger.info(" Running Pipeline...")
    logger.info("=" * 60)
    
    main()


# fusion 