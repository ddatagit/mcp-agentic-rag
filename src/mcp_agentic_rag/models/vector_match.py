"""VectorMatch model for MCP Agentic RAG system."""

from typing import Dict, Any
from pydantic import Field, field_validator
import uuid as uuid_module

from .base import BaseModel
from .types import ScoreRange


class VectorMatch(BaseModel):
    """
    Results from vector database search with relevance scoring.

    Represents a single document match from the vector database,
    including similarity score and metadata.
    """

    document_id: str = Field(..., description="Unique identifier for the source document")
    content: str = Field(..., min_length=1, description="The matching text content")
    score: ScoreRange = Field(..., ge=0.0, le=1.0, description="Similarity score from vector search")
    source_document: str = Field(..., description="Original document reference")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata")

    @field_validator('document_id')
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document ID is a valid UUID format."""
        try:
            # Validate it's a proper UUID
            uuid_module.UUID(v)
            return v
        except ValueError:
            raise ValueError("document_id must be a valid UUID format")

    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content is non-empty after stripping."""
        if not v.strip():
            raise ValueError("content cannot be empty")
        return v.strip()

    @field_validator('source_document')
    @classmethod
    def validate_source_document(cls, v: str) -> str:
        """Validate source document reference is non-empty."""
        if not v.strip():
            raise ValueError("source_document cannot be empty")
        return v.strip()

    def get_relevance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of this match's relevance information.

        Returns:
            Dictionary with relevance metrics and metadata.
        """
        return {
            'document_id': self.document_id,
            'score': self.score,
            'source': self.source_document,
            'content_length': len(self.content),
            'metadata_keys': list(self.metadata.keys()) if self.metadata else [],
            'relevance_category': self._categorize_relevance()
        }

    def _categorize_relevance(self) -> str:
        """Categorize relevance based on score."""
        if self.score >= 0.8:
            return "high"
        elif self.score >= 0.6:
            return "medium"
        else:
            return "low"

    def truncate_content(self, max_length: int = 200) -> str:
        """
        Get truncated version of content for display.

        Args:
            max_length: Maximum length of content to return.

        Returns:
            Truncated content with ellipsis if needed.
        """
        if len(self.content) <= max_length:
            return self.content

        # Try to break at word boundary
        truncated = self.content[:max_length]
        last_space = truncated.rfind(' ')

        if last_space > max_length * 0.8:  # If we can break reasonably close to limit
            return truncated[:last_space] + "..."

        return truncated + "..."