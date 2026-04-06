"""
pytest configuration for EduQuery test suite.
Adds the project root to sys.path so imports resolve correctly.
"""
import sys
from pathlib import Path

# Project root = two levels up from this conftest
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
