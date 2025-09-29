"""MCP Agentic RAG - Intelligent retrieval-augmented generation with vector and web search."""

__version__ = "0.1.0"
__author__ = "MCP Agentic RAG Team"
__description__ = "MCP server for intelligent RAG with vector database and web search fallback"

# Import models (these don't require external dependencies)
from .models import Query, VectorMatch, WebResult, SearchResponse, RoutingDecision
from .config import Settings

# Conditional imports for services and server (require external dependencies)
try:
    from .services import VectorRetrievalService, WebSearchService
    _SERVICES_AVAILABLE = True
except ImportError:
    _SERVICES_AVAILABLE = False
    VectorRetrievalService = None
    WebSearchService = None

try:
    from .server import MCPServer
    _SERVER_AVAILABLE = True
except ImportError:
    _SERVER_AVAILABLE = False
    MCPServer = None

__all__ = [
    "Query",
    "VectorMatch",
    "WebResult",
    "SearchResponse",
    "RoutingDecision",
    "Settings",
]

# Add services to __all__ if available
if _SERVICES_AVAILABLE:
    __all__.extend(["VectorRetrievalService", "WebSearchService"])

if _SERVER_AVAILABLE:
    __all__.extend(["MCPServer"])