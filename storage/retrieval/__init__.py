"""
Vector search and retrieval functionality
"""

from .vector_store import EduQueryVectorStore

# Optional Q&A system (requires LLM dependencies)
try:
    from .qa_system import EduQueryQA
    _has_qa = True
except ImportError:
    _has_qa = False
    EduQueryQA = None

__all__ = [
    "EduQueryVectorStore",
]

if _has_qa:
    __all__.append("EduQueryQA")

if _has_qa:
    __all__.extend(["EduQueryQA", "EduQueryQA"])
