"""Vector database integration using LanceDB for fast similarity search.

This module provides a vector database interface for storing and searching
frame embeddings. It supports hybrid search combining both image and text
embeddings for powerful multimodal retrieval.

Example:
    Basic usage::

        from EduQuery import EduQueryVectorStore, EduQueryEmbedder
        
        # Create embeddings
        embedder = EduQueryEmbedder()
        embeddings = embedder.embed_frames_batch(frames)
        
        # Store in vector database
        store = EduQueryVectorStore(db_path="tutorials.db")
        store.create_table(embeddings)
        
        # Search
        results = store.search_by_text("How do I export?", embedder, limit=5)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import numpy as np
import lancedb
from loguru import logger


class EduQueryVectorStore:
    """Vector database for storing and searching frame embeddings.
    
    This class provides a high-level interface to LanceDB for storing multimodal
    embeddings and performing similarity search. It supports three search modes:
    
    - **Hybrid**: Searches using combined image + text embeddings
    - **Text**: Searches using text embeddings only
    - **Image**: Searches using image embeddings only
    
    The database connection is established lazily on first use.
    
    Attributes:
        db_path: Path to the LanceDB database directory.
        table_name: Name of the table storing frame embeddings.
    
    Example:
        Create and populate database::
        
            store = EduQueryVectorStore(db_path="my_videos.db")
            store.create_table(embeddings, mode="overwrite")
        
        Search for similar frames::
        
            results = store.search_by_text(
                "How do I save?",
                embedder=embedder,
                limit=3,
                search_type="hybrid"
            )
        
        Get database statistics::
        
            stats = store.get_stats()
            print(f"Total frames: {stats['total_frames']}")
    """
    
    def __init__(
        self,
        db_path: Union[str, Path] = "EduQuery.db",
        table_name: str = "frames"
    ) -> None:
        """Initialize the vector store.
        
        Args:
            db_path: Path to the LanceDB database directory. Will be created
                if it doesn't exist. Defaults to "EduQuery.db".
            table_name: Name of the table to store frame embeddings.
                Defaults to "frames".
        
        Example:
            >>> # Use default database
            >>> store = EduQueryVectorStore()
            
            >>> # Use custom database and table
            >>> store = EduQueryVectorStore(
            ...     db_path="my_tutorials.db",
            ...     table_name="tutorial_frames"
            ... )
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._db = None
        self._table = None
        
        logger.info(f"Initializing vector store at: {self.db_path}")
    
    def _connect(self) -> None:
        """Connect to the LanceDB database.
        
        Establishes connection to the database if not already connected.
        The connection is lazy - only created when needed.
        
        Raises:
            RuntimeError: If connection to the database fails.
        """
        if self._db is None:
            self._db = lancedb.connect(str(self.db_path))
            logger.debug(f"Connected to database: {self.db_path}")
    
    def create_table(
        self,
        embeddings: List[Dict[str, Any]],
        mode: str = "overwrite"
    ) -> None:
        """Create or update the frames table with embeddings.
        
        Stores frame embeddings in the database. Each embedding includes both
        image and text vectors, enabling hybrid search. The combined vector
        is created by concatenating image and text embeddings.
        
        Args:
            embeddings: List of embedding dictionaries from EduQueryEmbedder.
                Each dictionary should contain:
                - frame_id: Unique identifier
                - timestamp: Time in seconds
                - image_embedding: Image embedding vector
                - text_embedding: Text embedding vector
                - text: Transcript text
                - frame_path: Path to frame image
                - extraction_reason: Why frame was extracted
                - quality_score: Frame quality score
            mode: Table creation mode. Options:
                - 'overwrite': Replace existing table (default)
                - 'append': Add to existing table
                Defaults to 'overwrite'.
        
        Raises:
            ValueError: If embeddings list is empty or has invalid format.
            RuntimeError: If table creation fails.
        
        Example:
            >>> embedder = EduQueryEmbedder()
            >>> embeddings = embedder.embed_frames_batch(frames)
            >>> store = EduQueryVectorStore()
            >>> store.create_table(embeddings, mode="overwrite")
            
            >>> # Append more embeddings later
            >>> new_embeddings = embedder.embed_frames_batch(new_frames)
            >>> store.create_table(new_embeddings, mode="append")
        """
        self._connect()
        
        if not embeddings:
            logger.warning("No embeddings provided")
            return
        
        logger.info(f"Creating table '{self.table_name}' with {len(embeddings)} entries")
        
        # Prepare data for LanceDB
        # LanceDB expects a list of dictionaries with consistent schema.
        # FIX BUG-4 (partial): also store all design-required display fields so
        # search results can be rendered without a relational DB roundtrip (§2).
        data = []
        for emb in embeddings:
            # Combine image and text embeddings into a single vector
            combined_embedding = np.concatenate([
                emb["image_embedding"],
                emb["text_embedding"]
            ])

            record = {
                # Core frame fields
                "frame_id": emb["frame_id"],
                "timestamp": emb["timestamp"],
                "text": emb["text"],
                "frame_path": emb["frame_path"],
                "extraction_reason": emb["extraction_reason"],
                "quality_score": emb["quality_score"],
                # Vector columns
                "vector": combined_embedding.tolist(),
                "image_vector": emb["image_embedding"].tolist(),
                "text_vector": emb["text_embedding"].tolist(),
                # Design §2 payload fields – needed to render results without DB roundtrip
                "segment_id": emb.get("segment_id", emb["frame_id"]),
                "video_id": emb.get("video_id", ""),
                "course_id": emb.get("course_id", ""),
                "start_time": emb.get("start_time", emb["timestamp"]),
                "end_time": emb.get("end_time", emb["timestamp"]),
                "thumb_url": emb.get("thumb_url", ""),
                "video_url": emb.get("video_url", ""),
                "video_title": emb.get("video_title", ""),
                "course_title": emb.get("course_title", ""),
            }
            data.append(record)
        
        # Create or overwrite table
        if mode == "overwrite":
            self._table = self._db.create_table(
                self.table_name,
                data=data,
                mode="overwrite"
            )
        else:
            # Append to existing table
            if self.table_name in self._db.table_names():
                self._table = self._db.open_table(self.table_name)
                self._table.add(data)
            else:
                self._table = self._db.create_table(self.table_name, data=data)
        
        logger.success(f"Table '{self.table_name}' created with {len(data)} entries")
    
    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        search_type: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """Search for similar segments using vector similarity.
        
        Performs approximate nearest neighbor search to find segments with
        embeddings most similar to the query embedding.
        
        Args:
            query_embedding: Query embedding vector (fused according to weights).
            limit: Maximum number of results to return. Defaults to 5.
            search_type: Type of search (for compatibility, but now always hybrid).
        
        Returns:
            List of segment payloads sorted by similarity.
        """
        self._connect()
        
        if self._table is None:
            if self.table_name not in self._db.table_names():
                raise ValueError(f"Table '{self.table_name}' does not exist")
            self._table = self._db.open_table(self.table_name)
        
        # Perform search on the fused vector
        results = (
            self._table
            .search(query_embedding.tolist(), vector_column_name="vector")
            .limit(limit)
            .to_list()
        )

        # FIX BUG-4: previously did result["payload"] but records are stored
        # flat – there is no "payload" key.  Return each row directly,
        # dropping the internal LanceDB distance key for cleanliness.
        payloads = []
        for result in results:
            row = {k: v for k, v in result.items() if k != "_distance"}
            payloads.append(row)

        logger.debug(f"Found {len(payloads)} results")
        return payloads
    
    def search_by_text(
        self,
        query_text: str,
        embedder: Any,  # EduQueryEmbedder type
        limit: int = 5,
        search_type: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """Search using a text query.
        
        Generates fused embedding from text query using both Sentence Transformers
        and CLIP text encoding, then searches the concatenated vector.
        
        Args:
            query_text: Natural language query.
            embedder: EduQueryEmbedder instance.
            limit: Maximum number of results to return. Defaults to 5.
            search_type: Type of search ('hybrid', 'text', 'image').
        
        Returns:
            List of segment payloads sorted by similarity.
        """
        logger.info(f"Searching for: '{query_text}'")
        
        # Generate embeddings for both modalities
        text_emb = embedder.embed_text(query_text)  # Sentence Transformers (384-dim)
        image_emb = embedder.embed_text_clip(query_text)  # CLIP text encoding (512-dim)
        
        # Apply fusion weights based on query type
        if search_type == "text":
            text_weight = 0.9
            image_weight = 0.1
        elif search_type == "image":
            text_weight = 0.1
            image_weight = 0.9
        else:
            text_weight = embedder.text_weight
            image_weight = embedder.image_weight
        
        # Create fused query embedding matching stored concatenated vectors
        query_embedding = np.concatenate([
            image_weight * image_emb,  # CLIP image part (512-dim)
            text_weight * text_emb     # Sentence Transformers text part (384-dim)
        ])
        
        return self.search(query_embedding, limit, search_type)
    
    def search_by_text_scoped(
        self,
        query_text: str,
        embedder: Any,
        limit: int = 5,
        video_id: Optional[str] = None,
        course_id: Optional[str] = None,
        search_type: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """Search using text query with optional scope filtering.
        
        Same as search_by_text but filters results to specific video or course.
        
        Args:
            query_text: Natural language query.
            embedder: EduQueryEmbedder instance.
            limit: Maximum number of results to return.
            video_id: If provided, only search within this video.
            course_id: If provided, only search within this course.
            search_type: Type of search ('hybrid', 'text', 'image').
        
        Returns:
            List of segment payloads filtered by scope, sorted by similarity.
        """
        # First get unscoped results
        all_results = self.search_by_text(query_text, embedder, limit=limit*3, search_type=search_type)
        
        # Filter by scope
        filtered = all_results
        if video_id:
            filtered = [r for r in filtered if r.get("video_id") == video_id]
        if course_id:
            filtered = [r for r in filtered if r.get("course_id") == course_id]
        
        # Return top k after filtering
        return filtered[:limit]
    
    def search_scoped(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        video_id: Optional[str] = None,
        course_id: Optional[str] = None,
        search_type: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """Search with optional scope filtering by video or course.
        
        Filters results to specific video or course after vector search.
        Useful when you want to search within a specific video/course only.
        
        Args:
            query_embedding: Query embedding vector.
            limit: Maximum results to return after filtering.
            video_id: Only return results from this video_id.
            course_id: Only return results from this course_id.
            search_type: Type of search.
        
        Returns:
            Filtered search results.
        """
        # Get more results initially to account for filtering loss
        all_results = self.search(query_embedding, limit=limit*3, search_type=search_type)
        
        # Apply scope filters
        if video_id:
            all_results = [r for r in all_results if r.get("video_id") == video_id]
        if course_id:
            all_results = [r for r in all_results if r.get("course_id") == course_id]
        
        return all_results[:limit]
    
    def get_stats(self, course_id: Optional[str] = None) -> Dict[str, Union[bool, int, str]]:
        """Get statistics about the vector store.
        
        Provides information about the database state, including whether
        the table exists and how many frames are stored. Optionally filter
        stats to a specific course.
        
        Args:
            course_id: Optional course ID to filter statistics.
                      If provided, only frames from this course are counted.
        
        Returns:
            Dictionary containing:
            - exists: Whether the table exists (bool)
            - total_frames: Number of frames in the table (int, optionally filtered)
            - table_name: Name of the table (str, if exists)
            - db_path: Path to the database (str, if exists)
            - course_id: The course filter applied (str, if provided)
        
        Example:
            >>> store = EduQueryVectorStore()
            >>> # Global stats
            >>> stats = store.get_stats()
            >>> if stats['exists']:
            ...     print(f"Database has {stats['total_frames']} frames")
            
            >>> # Course-specific stats
            >>> course_stats = store.get_stats(course_id="ml-specialization")
            >>> print(f"ML course has {course_stats['total_frames']} frames")
        """
        self._connect()
        
        if self.table_name not in self._db.table_names():
            return {
                "exists": False,
                "total_frames": 0
            }
        
        table = self._db.open_table(self.table_name)
        df = table.to_pandas()
        
        # Filter by course if provided
        if course_id:
            df = df[df.get("course_id", "") == course_id]
        
        count = len(df)
        
        result = {
            "exists": True,
            "total_frames": count,
            "table_name": self.table_name,
            "db_path": str(self.db_path)
        }
        
        if course_id:
            result["course_id"] = course_id
        
        return result
    
    def delete_table(self) -> None:
        """Delete the frames table from the database.
        
        Permanently removes the table and all its data. Use with caution.
        
        Raises:
            RuntimeError: If table deletion fails.
        
        Example:
            >>> store = EduQueryVectorStore()
            >>> store.delete_table()  # Removes all data
            >>> stats = store.get_stats()
            >>> print(stats['exists'])
            False
        """
        self._connect()
        
        if self.table_name in self._db.table_names():
            self._db.drop_table(self.table_name)
            logger.info(f"Deleted table: {self.table_name}")
        else:
            logger.warning(f"Table '{self.table_name}' does not exist")
