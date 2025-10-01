"""RoutingDecision model for MCP Agentic RAG system."""


from pydantic import Field, field_validator

from .base import BaseModel
from .types import SearchStrategy


class RoutingDecision(BaseModel):
    """Logic and metadata for determining which search method(s) to use."""

    query_id: str = Field(..., description="Reference to the query being routed")
    strategy: SearchStrategy = Field(..., description="Chosen routing strategy")
    reasoning: str = Field(..., min_length=1, description="Human-readable explanation of the routing decision")
    domain_detected: bool = Field(..., description="Whether domain-specific terminology was detected")
    ml_keywords_found: list[str] = Field(default_factory=list, description="Specific ML terms that influenced routing")
    estimated_local_coverage: float = Field(..., ge=0.0, le=1.0, description="Estimated likelihood of finding results in vector DB")

    @field_validator('reasoning')
    @classmethod
    def validate_reasoning(cls, v: str) -> str:
        """Validate reasoning is non-empty explanatory text."""
        if not v.strip():
            raise ValueError("reasoning cannot be empty")
        return v.strip()

    @field_validator('ml_keywords_found')
    @classmethod
    def validate_ml_keywords(cls, v: list[str]) -> list[str]:
        """Validate and clean ML keywords."""
        return [keyword.strip().lower() for keyword in v if keyword.strip()]

    def should_use_vector(self) -> bool:
        """Determine if vector search should be used based on strategy."""
        return self.strategy in [SearchStrategy.VECTOR_ONLY, SearchStrategy.VECTOR_FIRST, SearchStrategy.COMBINED]

    def should_use_web(self) -> bool:
        """Determine if web search should be used based on strategy."""
        return self.strategy in [SearchStrategy.WEB_ONLY, SearchStrategy.VECTOR_FIRST, SearchStrategy.COMBINED]

    def get_confidence_category(self) -> str:
        """Get confidence category based on local coverage estimate."""
        if self.estimated_local_coverage >= 0.8:
            return "high"
        elif self.estimated_local_coverage >= 0.5:
            return "medium"
        else:
            return "low"
