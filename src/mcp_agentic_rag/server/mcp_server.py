"""MCP Server for Agentic RAG with intelligent routing."""

from mcp.server.fastmcp import FastMCP
import json
import uuid
import time
from typing import Dict, Any

# Import from new package structure
from ..models.query import Query
from ..config.settings import load_config

# Conditional imports for services
try:
    from ..services.vector_retrieval import VectorRetrievalService, Retriever, QdrantVDB, EmbedData
    _VECTOR_AVAILABLE = True
except ImportError:
    _VECTOR_AVAILABLE = False
    VectorRetrievalService = None
    Retriever = None
    QdrantVDB = None
    EmbedData = None

try:
    from ..services.web_search import WebSearchService, FallbackSearch
    _WEB_AVAILABLE = True
except ImportError:
    _WEB_AVAILABLE = False
    WebSearchService = None
    FallbackSearch = None

# Create MCP server
mcp = FastMCP(name="mcp-agentic-rag", host="127.0.0.1", port=8080)

# Initialize services conditionally
config = load_config()
vector_service = VectorRetrievalService(config.vector_collection_name) if _VECTOR_AVAILABLE else None
web_service = WebSearchService() if _WEB_AVAILABLE else None

# For backward compatibility, also initialize the original components
intelligent_router = None  # Will be initialized when intelligent_router service is migrated
fallback_search = FallbackSearch() if _WEB_AVAILABLE else None
retriever = Retriever(QdrantVDB("ml_faq_collection"), EmbedData()) if _VECTOR_AVAILABLE else None


@mcp.tool()
def machine_learning_faq_retrieval_tool(query: str) -> str:
    """
    Retrieve the most relevant documents from the machine learning FAQ collection.

    Use this tool when the user asks about ML concepts, algorithms, best practices,
    or any machine learning related questions.

    Args:
        query: The user query to retrieve the most relevant documents

    Returns:
        Most relevant documents retrieved from vector database, formatted as text

    Raises:
        ValueError: When query is not a string or is empty
        ConnectionError: When vector database is unreachable
        TimeoutError: When search exceeds time limits
    """
    # Input validation
    if not isinstance(query, str):
        raise ValueError("query must be a string")

    if not query.strip():
        raise ValueError("query cannot be empty")

    if len(query) > 1000:
        raise ValueError("query exceeds maximum length of 1000 characters")

    try:
        # Check if vector service is available
        if not _VECTOR_AVAILABLE or vector_service is None:
            raise Exception("Vector search service not available - missing dependencies")

        # Use the new service for enhanced functionality
        response = vector_service.search(query.strip())
        return response

    except ConnectionError as e:
        raise ConnectionError(f"Database unreachable: {str(e)}")

    except TimeoutError as e:
        raise TimeoutError(f"Search timed out: {str(e)}")

    except Exception as e:
        raise Exception(f"Vector search failed: {str(e)}")


@mcp.tool()
def bright_data_web_search_tool(query: str) -> dict:
    """
    Search for information using Bright Data (Google Custom Search API).

    Use this tool when the user asks about topics not covered in the ML FAQ,
    or when vector database results have low confidence scores.

    Args:
        query: The user query to search for information

    Returns:
        Dictionary containing search results with metadata

    Raises:
        ValueError: When query is invalid
        APIError: When Bright Data API returns error
        RateLimitError: When API rate limits are exceeded
        TimeoutError: When search exceeds time limits
    """
    # Input validation
    if not isinstance(query, str):
        raise ValueError("query must be a string")

    if not query.strip():
        raise ValueError("query cannot be empty")

    if len(query) > 1000:
        raise ValueError("query exceeds maximum length of 1000 characters")

    try:
        # Check if web service is available
        if not _WEB_AVAILABLE or web_service is None:
            raise Exception("Web search service not available - missing dependencies")

        # Use the new web service
        start_time = time.time()
        results = web_service.search(query.strip(), num_results=5)
        response_time = time.time() - start_time

        # Format response to match expected contract
        return {
            "results": results["results"],
            "total_results": results["total_results"],
            "response_time": response_time
        }

    except Exception as e:
        # Check if it's a known error type with error_type attribute
        if hasattr(e, 'error_type'):
            if e.error_type == "API_KEY_MISSING":
                raise Exception("API key missing or invalid")
            elif e.error_type == "API_QUOTA_EXCEEDED":
                raise Exception("Rate limit exceeded")
            elif e.error_type == "SEARCH_TIMEOUT":
                raise TimeoutError("Search request timed out")
            elif e.error_type == "NO_RESULTS":
                # Return empty results instead of raising for no results
                return {
                    "results": [],
                    "total_results": 0,
                    "response_time": 0.0
                }

        # For other errors, raise with context
        raise Exception(f"Web search failed: {str(e)}")


@mcp.tool()
def intelligent_query_router_tool(query: str, confidence_threshold: float = 0.7) -> dict:
    """
    Intelligently route queries between vector database and web search.

    Analyzes the query to determine the best search strategy, performs the search,
    and returns unified results with routing decision metadata.

    Args:
        query: The user query to analyze and route
        confidence_threshold: Minimum confidence threshold for vector results (default: 0.7)

    Returns:
        Dictionary containing unified search results and routing decision

    Raises:
        ValueError: When query or confidence_threshold is invalid
        ServiceError: When underlying services fail
        TimeoutError: When total response time exceeds 5 seconds
        NoResultsError: When no results found from any source
    """
    # Input validation
    if not isinstance(query, str):
        raise ValueError("query must be a string")

    if not query.strip():
        raise ValueError("query cannot be empty")

    if len(query) > 1000:
        raise ValueError("query exceeds maximum length of 1000 characters")

    if not isinstance(confidence_threshold, (int, float)) or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be a number between 0.0 and 1.0")

    start_time = time.time()

    try:
        # Create Query object for analysis
        query_obj = Query(text=query.strip(), confidence_threshold=confidence_threshold)

        # Determine routing strategy based on ML context
        ml_keywords = query_obj.extract_ml_keywords()
        has_ml_context = query_obj.has_ml_context()

        results = []
        sources_used = []
        fallback_triggered = False
        strategy = "vector_only"

        # Try vector search first if ML context detected
        if has_ml_context and _VECTOR_AVAILABLE and vector_service is not None:
            try:
                vector_matches = vector_service.advanced_search(query, confidence_threshold)

                if vector_matches:
                    # Convert to unified result format
                    for match in vector_matches:
                        results.append({
                            "type": "vector",
                            "content": match.content,
                            "score": match.score,
                            "source_document": match.source_document
                        })
                    sources_used.append("vector")

                    # Check if we need fallback
                    avg_confidence = sum(m.score for m in vector_matches) / len(vector_matches)
                    if avg_confidence < confidence_threshold:
                        fallback_triggered = True
                        strategy = "vector_first"

            except Exception as e:
                print(f"Vector search failed: {e}")
                fallback_triggered = True
                strategy = "web_only"

        else:
            fallback_triggered = True
            strategy = "web_only"

        # Perform web search if needed
        if (fallback_triggered or not has_ml_context) and _WEB_AVAILABLE and web_service is not None:
            try:
                web_results = web_service.search(query, num_results=3)

                for result in web_results["results"]:
                    results.append({
                        "type": "web",
                        "title": result["title"],
                        "snippet": result["snippet"],
                        "url": result["url"],
                        "relevance_score": result.get("confidence", 0.5)
                    })

                if "web" not in sources_used:
                    sources_used.append("web")

            except Exception as e:
                print(f"Web search failed: {e}")
                # Continue with vector results if available

        # Calculate response time
        response_time = time.time() - start_time

        # Determine confidence level
        if results:
            if has_ml_context and not fallback_triggered:
                confidence_level = "high"
            elif results and len(sources_used) == 1:
                confidence_level = "medium"
            else:
                confidence_level = "low"
        else:
            confidence_level = "low"

        # Prepare routing decision
        routing_decision = {
            "strategy": strategy,
            "reasoning": f"{'ML terminology detected' if has_ml_context else 'General query'}, {'fallback triggered' if fallback_triggered else 'vector sufficient'}",
            "domain_detected": has_ml_context,
            "ml_keywords_found": ml_keywords
        }

        return {
            "results": results,
            "sources_used": sources_used,
            "routing_decision": routing_decision,
            "total_results": len(results),
            "response_time": response_time,
            "confidence_level": confidence_level,
            "fallback_triggered": fallback_triggered
        }

    except Exception as e:
        response_time = time.time() - start_time

        # Check timeout
        if response_time > config.max_response_time:
            raise TimeoutError(f"Total response time {response_time:.2f}s exceeds limit of {config.max_response_time}s")

        raise Exception(f"Query routing failed: {str(e)}")


def main():
    """Main entry point for the MCP server."""
    print("Starting MCP Agentic RAG Server...")
    print(f"Server running on http://127.0.0.1:8080")
    print("Available tools:")
    print("- machine_learning_faq_retrieval_tool")
    print("- bright_data_web_search_tool")
    print("- intelligent_query_router_tool")

    # Start the server
    mcp.run()


if __name__ == "__main__":
    main()


# Export for external use
class MCPServer:
    """MCP Server wrapper class."""

    def __init__(self):
        self.mcp = mcp
        self.vector_service = vector_service
        self.web_service = web_service
        self.vector_available = _VECTOR_AVAILABLE
        self.web_available = _WEB_AVAILABLE

    def start(self):
        """Start the MCP server."""
        return self.mcp.run()

    def get_tools(self):
        """Get list of available tools."""
        return [
            "machine_learning_faq_retrieval_tool",
            "bright_data_web_search_tool",
            "intelligent_query_router_tool"
        ]