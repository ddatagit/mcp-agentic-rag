"""Type definitions for MCP Agentic RAG models."""

from enum import Enum
from typing import Literal


class SearchStrategy(str, Enum):
    """Search strategy options for query routing."""
    VECTOR_ONLY = "vector_only"
    WEB_ONLY = "web_only"
    VECTOR_FIRST = "vector_first"
    COMBINED = "combined"


class ConfidenceLevel(str, Enum):
    """Confidence level classifications."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Type aliases for better readability
SourceType = Literal["vector", "web"]
ScoreRange = float  # Should be between 0.0 and 1.0
