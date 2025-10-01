"""WebResult model for MCP Agentic RAG system."""

from datetime import datetime
from urllib.parse import urlparse

from pydantic import Field, field_validator

from .base import BaseModel
from .types import ScoreRange


class WebResult(BaseModel):
    """Results from web search with relevance and source information."""

    title: str = Field(..., min_length=1, description="Title of the web page/result")
    snippet: str = Field(..., min_length=1, description="Brief excerpt from the content")
    url: str = Field(..., description="Source URL")
    relevance_score: ScoreRange = Field(..., ge=0.0, le=1.0, description="Search engine relevance score")
    source_domain: str = Field(..., description="Domain of the source website")
    timestamp: datetime = Field(default_factory=lambda: WebResult.current_timestamp())

    @field_validator('title', 'snippet')
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        """Validate title and snippet are non-empty after stripping."""
        if not v.strip():
            raise ValueError("field cannot be empty")
        return v.strip()

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        if not v.strip():
            raise ValueError("URL cannot be empty")

        parsed = urlparse(v.strip())
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("URL must be a valid HTTP/HTTPS URL")

        if parsed.scheme not in ['http', 'https']:
            raise ValueError("URL must use HTTP or HTTPS protocol")

        return v.strip()

    @field_validator('source_domain')
    @classmethod
    def validate_source_domain(cls, v: str) -> str:
        """Validate source domain is non-empty."""
        if not v.strip():
            raise ValueError("source_domain cannot be empty")
        return v.strip().lower()

    @classmethod
    def from_search_result(cls, search_data: dict) -> 'WebResult':
        """Create WebResult from search API response."""
        parsed_url = urlparse(search_data['url'])

        return cls(
            title=search_data['title'],
            snippet=search_data['snippet'],
            url=search_data['url'],
            relevance_score=search_data.get('relevance_score', 0.5),
            source_domain=parsed_url.netloc
        )
