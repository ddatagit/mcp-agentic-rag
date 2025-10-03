"""Query model for MCP Agentic RAG system."""

from datetime import datetime

from pydantic import Field, field_validator

from .base import BaseModel


class Query(BaseModel):
    """
    Represents a user query with metadata for intelligent routing decisions.

    This model encapsulates all information needed to process a user's query,
    including the text, routing parameters, and metadata extraction hints.
    """

    id: str = Field(default_factory=lambda: Query.generate_id())
    text: str = Field(..., min_length=1, max_length=1000)
    timestamp: datetime = Field(default_factory=lambda: Query.current_timestamp())
    domain_hints: list[str] = Field(default_factory=list)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_response_time: float = Field(default=5.0, gt=0.0)

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate query text is not empty after stripping."""
        if not v.strip():
            raise ValueError("query cannot be empty")
        return v.strip()

    @field_validator('domain_hints')
    @classmethod
    def validate_domain_hints(cls, v: list[str]) -> list[str]:
        """Validate and clean domain hints."""
        return [hint.strip().lower() for hint in v if hint.strip()]

    def extract_ml_keywords(self) -> list[str]:
        """
        Extract machine learning keywords from the query text.

        Returns:
            List of ML-related keywords found in the query.
        """
        ml_keywords = [
            'machine learning', 'ml', 'neural network', 'deep learning',
            'ai', 'artificial intelligence', 'algorithm', 'model',
            'training', 'supervised', 'unsupervised', 'reinforcement',
            'gradient', 'backpropagation', 'cnn', 'rnn', 'lstm',
            'transformer', 'attention', 'overfitting', 'regularization',
            'classification', 'regression', 'clustering', 'feature',
            'dataset', 'tensor', 'pytorch', 'tensorflow', 'scikit'
        ]

        text_lower = self.text.lower()
        found_keywords = []

        for keyword in ml_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)

        return found_keywords

    def has_ml_context(self) -> bool:
        """
        Determine if the query has machine learning context.

        Returns:
            True if ML keywords are detected or domain hints indicate ML context.
        """
        ml_keywords = self.extract_ml_keywords()
        ml_domain_hints = any('ml' in hint or 'machine' in hint for hint in self.domain_hints)

        return len(ml_keywords) > 0 or ml_domain_hints

    def to_routing_context(self) -> dict:
        """
        Generate routing context for the intelligent router.

        Returns:
            Dictionary containing routing-relevant information.
        """
        return {
            'query_id': self.id,
            'text': self.text,
            'ml_keywords': self.extract_ml_keywords(),
            'has_ml_context': self.has_ml_context(),
            'confidence_threshold': self.confidence_threshold,
            'max_response_time': self.max_response_time,
            'domain_hints': self.domain_hints
        }
