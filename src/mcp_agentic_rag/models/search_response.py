"""SearchResponse model for MCP Agentic RAG system."""


from pydantic import Field, field_validator

from .base import BaseModel
from .types import ConfidenceLevel, SourceType
from .vector_match import VectorMatch
from .web_result import WebResult


class SearchResponse(BaseModel):
    """Unified response containing results from either or both sources."""

    query_id: str = Field(..., description="Reference to the original query")
    results: list[VectorMatch | WebResult] = Field(default_factory=list, description="Combined and ranked results")
    sources_used: list[SourceType] = Field(..., description="Which sources were queried")
    total_results: int = Field(..., ge=0, description="Total number of results returned")
    response_time: float = Field(..., gt=0, description="Total time taken to generate response")
    confidence_level: ConfidenceLevel = Field(..., description="Overall confidence level")
    fallback_triggered: bool = Field(..., description="Whether web search fallback was used")

    @field_validator('results')
    @classmethod
    def validate_results_not_empty_if_total_positive(cls, v: list, info) -> list:
        """Validate results consistency with total_results."""
        if hasattr(info, 'data') and info.data.get('total_results', 0) > 0:
            if not v:
                raise ValueError("results cannot be empty when total_results > 0")
        return v

    @field_validator('sources_used')
    @classmethod
    def validate_sources_used(cls, v: list[SourceType]) -> list[SourceType]:
        """Validate sources_used contains valid source names."""
        if not v:
            raise ValueError("sources_used cannot be empty")

        valid_sources = {"vector", "web"}
        for source in v:
            if source not in valid_sources:
                raise ValueError(f"Invalid source: {source}. Must be one of {valid_sources}")

        return v

    def get_vector_results(self) -> list[VectorMatch]:
        """Get only vector search results."""
        return [r for r in self.results if isinstance(r, VectorMatch)]

    def get_web_results(self) -> list[WebResult]:
        """Get only web search results."""
        return [r for r in self.results if isinstance(r, WebResult)]

    def get_sorted_results(self, by_score: bool = True) -> list[VectorMatch | WebResult]:
        """Get results sorted by relevance score."""
        if not by_score:
            return self.results

        def get_score(result):
            if isinstance(result, VectorMatch):
                return result.score
            elif isinstance(result, WebResult):
                return result.relevance_score
            return 0.0

        return sorted(self.results, key=get_score, reverse=True)
