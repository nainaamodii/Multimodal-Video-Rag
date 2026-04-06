"""
Question-Answering system using LLM with retrieved context segments.
"""

import os
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from query.embeddings.embedder import EduQueryEmbedder
from .vector_store import EduQueryVectorStore


# FIX: QueryType was defined as a nested class-body statement with an inline
# `from enum import Enum` import — both illegal at class scope in Python.
# Moved to module level where it belongs.
class QueryType(Enum):
    VISUAL = "visual"
    CONCEPTUAL = "conceptual"
    MATHEMATICAL = "mathematical"


class EduQueryQA:
    """Question-Answering system for educational videos using Gemini."""

    SYSTEM_PROMPT = """
You are EduQuery, an AI assistant that helps students understand educational video content.

You have access to:
1. Video transcripts with timestamps
2. Screenshots from key moments in educational videos
3. Context about when things were said and shown

Your role:
- Answer student questions about educational content clearly and concisely
- Reference specific timestamps when relevant
- Describe what's shown in screenshots when helpful
- Be precise and actionable
- Cite sources with video titles and timestamps

Guidelines:
- Keep answers focused and educational
- Mention timestamps in format: "at 1:23 in [Video Title]"
- If you're not sure, say so - don't make up information
- Always cite your sources
"""

    def __init__(
        self,
        vector_store: EduQueryVectorStore,
        embedder: EduQueryEmbedder,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder

        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.llm = ChatGoogleGenerativeAI(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            google_api_key=api_key,
        )

        logger.info(f"Initialized EduQueryQA with {model}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        course_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict:
        """
        Ask a question about educational video content.

        Args:
            question:  The student's natural-language question.
            course_id: Optional course ID to scope the search.
            top_k:     Number of source segments to return (default 5).

        Returns:
            Dict with keys: answer, sources, query_type, num_sources.
        """
        logger.info(f"Question: '{question}'")

        # Step 1: Classify query type
        query_type = self._classify_query(question)
        logger.info(f"Query classified as: {query_type.value}")

        # Step 2: Retrieve candidates (3× top_k, capped at 20)
        num_candidates = min(top_k * 3, 20)
        
        # Use scoped search if course_id is provided
        if course_id:
            candidates = self.vector_store.search_by_text_scoped(
                query_text=question,
                embedder=self.embedder,
                limit=num_candidates,
                course_id=course_id,
                search_type="hybrid",
            )
        else:
            candidates = self.vector_store.search_by_text(
                query_text=question,
                embedder=self.embedder,
                limit=num_candidates,
                search_type="hybrid",
            )

        if not candidates:
            logger.warning("No relevant segments found")
            return {
                "answer": (
                    "I couldn't find any relevant information in the "
                    "educational content for that question."
                ),
                "sources": [],
                "query_type": query_type.value,
                "num_sources": 0,
            }

        # Step 3: Rerank
        reranked = self._rerank_candidates(question, candidates, top_k)
        logger.success(f"Reranked to top {len(reranked)} segments")

        # Step 4: Generate answer
        logger.info("Generating answer...")
        answer = self._generate_answer(question, reranked)

        # Step 5: Build source list
        sources = []
        for segment in reranked:
            sources.append({
                "segment_id":   segment.get("segment_id", ""),
                "video_title":  segment.get("video_title", ""),
                "start_time":   segment.get("start_time", 0.0),
                "end_time":     segment.get("end_time", 0.0),
                "text_preview": segment.get("text", ""),
                "thumb_url":    segment.get("thumb_url", ""),
                "deep_link":    segment.get("video_url", ""),
                # FIX: segment is a plain dict, not an object — use .get(), not getattr()
                "score":        segment.get("score", 0.0),
            })

        logger.success("Answer generated")
        return {
            "answer":     answer,
            "sources":    sources,
            "query_type": query_type.value,
            "num_sources": len(sources),
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        course_id: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict:
        """
        Multi-turn conversation: extract the last user message and call ask().

        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."}.
            course_id: Optional course ID to scope the search.
            top_k:    Number of source segments (FIX: was 'num_results', now
                      matches ask() signature).
        """
        last_user_message = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break

        if not last_user_message:
            raise ValueError("No user message found in conversation history.")

        return self.ask(last_user_message, course_id=course_id, top_k=top_k)

    def batch_ask(
        self,
        questions: List[str],
        course_id: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Answer a list of questions in sequence.

        Args:
            questions: List of question strings.
            course_id: Optional course ID to scope the search.
            top_k:     Number of source segments per question (FIX: was
                       'num_results', now matches ask() signature).
        """
        logger.info(f"Answering {len(questions)} questions...")
        answers = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Question {i}/{len(questions)}: {question}")
            answers.append(self.ask(question, course_id=course_id, top_k=top_k))
        logger.success(f"Answered {len(questions)} questions")
        return answers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_query(self, query: str) -> QueryType:
        """Classify a query into VISUAL, MATHEMATICAL, or CONCEPTUAL."""
        visual_keywords = [
            "show", "diagram", "slide", "figure",
            "what does", "looks like", "draw", "display",
        ]
        mathematical_keywords = [
            "formula", "equation", "derive", "proof",
            "calculate", "compute", "solve",
        ]
        query_lower = query.lower()
        if any(k in query_lower for k in visual_keywords):
            return QueryType.VISUAL
        if any(k in query_lower for k in mathematical_keywords):
            return QueryType.MATHEMATICAL
        return QueryType.CONCEPTUAL

    def _rerank_candidates(
        self, query: str, candidates: List[Dict], top_k: int
    ) -> List[Dict]:
        """Rerank candidates using a cross-encoder; fall back to vector order."""
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query, cand.get("text", "")) for cand in candidates]
            scores = reranker.predict(pairs)
            for cand, score in zip(candidates, scores):
                cand["score"] = float(score)
            reranked = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
            return reranked[:top_k]
        except ImportError:
            logger.warning("sentence-transformers not available — using vector ranking")
            return candidates[:top_k]

    def _build_context(self, segments: List[Dict]) -> str:
        """Assemble a readable context block from retrieved segments."""
        parts = []
        for segment in segments:
            video_title = segment.get("video_title", "Unknown")
            start = segment.get("start_time", 0.0)
            end   = segment.get("end_time",   0.0)
            text  = segment.get("text", "")
            parts.append(
                f"[{video_title} | "
                f"{self._format_time(start)}-{self._format_time(end)}]\n{text}"
            )
        return "\n\n".join(parts)

    def _generate_answer(self, question: str, segments: List[Dict]) -> str:
        """Generate a natural-language answer via the configured LLM."""
        context = self._build_context(segments)

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", (
                "Based on the following transcript segments from educational videos, "
                "please answer the user's question.\n\n"
                "Context from videos:\n{context}\n\n"
                "User Question: {question}\n\n"
                "Provide a clear, educational answer that cites specific timestamps "
                "and video titles. If the context doesn't contain enough information, "
                "acknowledge this clearly."
            )),
        ])

        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": question})
        return response.content

    def _format_time(self, seconds: float) -> str:
        """Format a float number of seconds as MM:SS."""
        minutes = int(seconds) // 60
        secs    = int(seconds) % 60
        return f"{minutes}:{secs:02d}"
