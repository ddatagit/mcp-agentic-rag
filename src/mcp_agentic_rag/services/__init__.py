"""Services for MCP Agentic RAG system."""

# Conditional imports to handle missing dependencies gracefully
try:
    from .vector_retrieval import (
        EmbedData,
        QdrantVDB,
        Retriever,
        VectorRetrievalService,
    )
    _VECTOR_AVAILABLE = True
except ImportError:
    _VECTOR_AVAILABLE = False
    VectorRetrievalService = None
    Retriever = None
    QdrantVDB = None
    EmbedData = None

try:
    from .web_search import FallbackSearch, WebSearchService
    _WEB_AVAILABLE = True
except ImportError:
    _WEB_AVAILABLE = False
    WebSearchService = None
    FallbackSearch = None

try:
    from .intelligent_router import IntelligentRouter
    _ROUTER_AVAILABLE = True
except ImportError:
    _ROUTER_AVAILABLE = False
    IntelligentRouter = None

try:
    from .exceptions import (
        APIError,
        MCPToolError,
        NoResultsError,
        ServiceError,
        TimeoutError,
        ValidationError,
    )
    _EXCEPTIONS_AVAILABLE = True
except ImportError:
    _EXCEPTIONS_AVAILABLE = False
    MCPToolError = None
    ValidationError = None
    ServiceError = None
    TimeoutError = None
    APIError = None
    NoResultsError = None

__all__ = []

# Add available services to __all__
if _VECTOR_AVAILABLE:
    __all__.extend(["VectorRetrievalService", "Retriever", "QdrantVDB", "EmbedData"])

if _WEB_AVAILABLE:
    __all__.extend(["WebSearchService", "FallbackSearch"])

if _ROUTER_AVAILABLE:
    __all__.append("IntelligentRouter")

if _EXCEPTIONS_AVAILABLE:
    __all__.extend([
        "MCPToolError",
        "ValidationError",
        "ServiceError",
        "TimeoutError",
        "APIError",
        "NoResultsError",
    ])
