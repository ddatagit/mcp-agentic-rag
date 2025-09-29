"""Web search service for MCP Agentic RAG system."""

import os
import httpx
import time
from typing import List, Dict, Any, Optional
from pydantic import ValidationError

from ..config.settings import get_setting
from ..models.web_result import WebResult


class FallbackSearch:
    """Google Custom Search API integration for fallback search."""

    def __init__(self):
        """Initialize fallback search with configuration."""
        self.api_key = get_setting("GOOGLE_API_KEY")
        self.cx = get_setting("GOOGLE_CX")
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self._validated = False

    def _validate_configuration(self):
        """Validate API configuration when needed."""
        if self._validated:
            return

        if not self.api_key:
            raise ValueError("Google API key not configured. Set GOOGLE_API_KEY environment variable.")
        if not self.cx:
            raise ValueError("Google Custom Search CX not configured. Set GOOGLE_CX environment variable.")

        self._validated = True

    def validate_web_input(self, input_data: Dict[str, Any]) -> None:
        """Validate web search input parameters."""
        if not input_data.get("query"):
            raise ValueError("Query is required")

        query = input_data["query"]
        if len(query) == 0 or len(query) > 1000:
            raise ValueError(f"Query length {len(query)} not in range 1-1000")

        if "num_results" in input_data:
            num_results = input_data["num_results"]
            if not isinstance(num_results, int) or num_results < 1 or num_results > 10:
                raise ValueError("num_results must be between 1 and 10")

        if "safe_search" in input_data:
            safe_search = input_data["safe_search"]
            if safe_search not in ["off", "medium", "high"]:
                raise ValueError("safe_search must be one of: off, medium, high")

    def search(self, query: str, num_results: int = 5, safe_search: str = "medium") -> List[Dict[str, Any]]:
        """Perform Google Custom Search and return normalized results."""
        start_time = time.time()

        try:
            # Validate configuration and inputs
            self._validate_configuration()
            self.validate_web_input({
                "query": query,
                "num_results": num_results,
                "safe_search": safe_search
            })

            # Prepare search parameters
            params = {
                "key": self.api_key,
                "cx": self.cx,
                "q": query,
                "num": num_results,
                "safe": safe_search
            }

            # Perform search with timeout
            with httpx.Client(timeout=30.0) as client:
                response = client.get(self.base_url, params=params)

            # Handle HTTP errors
            if response.status_code == 401:
                error = type('APIKeyError', (), {
                    'error_type': 'API_KEY_MISSING',
                    'config_variable': 'GOOGLE_API_KEY',
                    'http_status_code': 500
                })()
                raise error

            elif response.status_code == 429:
                error = type('QuotaError', (), {
                    'error_type': 'API_QUOTA_EXCEEDED',
                    'daily_limit': 100,  # Default Google Custom Search limit
                    'reset_time': "24 hours",
                    'http_status_code': 429
                })()
                raise error

            elif response.status_code != 200:
                error = type('NetworkError', (), {
                    'error_type': 'NETWORK_ERROR',
                    'error_details': f"HTTP {response.status_code}: {response.text}",
                    'status_code': response.status_code,
                    'http_status_code': 502
                })()
                raise error

            # Parse response
            data = response.json()
            items = data.get("items", [])

            if not items:
                error = type('NoResultsError', (), {
                    'error_type': 'NO_RESULTS',
                    'query': query,
                    'http_status_code': 404
                })()
                raise error

            # Normalize results
            normalized_results = []
            for i, item in enumerate(items):
                result = {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                    "display_url": self._create_display_url(item.get("link", "")),
                    "confidence": self._calculate_confidence(i, len(items))
                }
                normalized_results.append(result)

            return normalized_results

        except httpx.TimeoutException:
            error = type('TimeoutError', (), {
                'error_type': 'SEARCH_TIMEOUT',
                'timeout_seconds': 30,
                'http_status_code': 408
            })()
            raise error

        except httpx.ConnectError as e:
            error = type('NetworkError', (), {
                'error_type': 'NETWORK_ERROR',
                'error_details': str(e),
                'status_code': None,
                'http_status_code': 502
            })()
            raise error

        except Exception as e:
            # Re-raise known errors
            if hasattr(e, 'error_type'):
                raise e

            # Handle unexpected errors
            error = type('WebSearchError', (), {
                'error_type': 'WEB_SEARCH_ERROR',
                'error_details': str(e),
                'http_status_code': 502
            })()
            raise error

    def search_with_metadata(
        self,
        query: str,
        num_results: int = 5,
        safe_search: str = "medium"
    ) -> Dict[str, Any]:
        """Perform search and return results with metadata."""
        start_time = time.time()

        try:
            results = self.search(query, num_results, safe_search)
            search_time = time.time() - start_time

            return {
                "query_id": self._generate_query_id(),
                "results": results,
                "total_found": len(results),
                "search_time_seconds": search_time,
                "search_engine": "google_custom_search"
            }

        except Exception as e:
            search_time = time.time() - start_time

            # Return error with timing information
            error_response = {
                "query_id": self._generate_query_id(),
                "results": [],
                "total_found": 0,
                "search_time_seconds": search_time,
                "search_engine": "google_custom_search",
                "error": str(e)
            }

            # Re-raise the error but with additional context
            raise e

    def _create_display_url(self, url: str) -> str:
        """Create user-friendly display URL."""
        if not url:
            return ""

        # Remove protocol
        display_url = url.replace("https://", "").replace("http://", "")

        # Remove www.
        if display_url.startswith("www."):
            display_url = display_url[4:]

        # Truncate if too long
        if len(display_url) > 50:
            display_url = display_url[:47] + "..."

        return display_url

    def _calculate_confidence(self, position: int, total_results: int) -> float:
        """Calculate normalized confidence score based on result position."""
        # Higher confidence for earlier results
        # Scale from 0.9 (first result) to 0.5 (last result)
        if total_results <= 1:
            return 0.9

        confidence = 0.9 - (position * (0.4 / (total_results - 1)))
        return max(0.5, min(0.9, confidence))

    def _generate_query_id(self) -> str:
        """Generate unique query ID."""
        import uuid
        return str(uuid.uuid4())

    def get_health_status(self) -> Dict[str, Any]:
        """Check health status of Google Custom Search API."""
        try:
            # Perform minimal test search
            test_result = self.search("test", num_results=1)
            return {
                "status": "healthy",
                "api_accessible": True,
                "last_check": time.time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "api_accessible": False,
                "error": str(e),
                "last_check": time.time()
            }


class WebSearchService:
    """
    High-level web search service with enhanced functionality.

    This service provides a clean interface for web search operations
    with proper result modeling and error handling.
    """

    def __init__(self):
        self.fallback_search = FallbackSearch()

    def search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Perform web search and return structured results.

        Args:
            query: Search query text
            num_results: Maximum number of results to return

        Returns:
            Dictionary with search results and metadata
        """
        try:
            results = self.fallback_search.search(query, num_results)

            # Convert to structured response
            return {
                "results": results,
                "total_results": len(results),
                "response_time": 0.0,  # Will be calculated by caller
                "source": "google_custom_search"
            }

        except Exception as e:
            # Re-raise with context
            raise Exception(f"Web search failed: {str(e)}")

    def search_with_models(self, query: str, num_results: int = 5) -> List[WebResult]:
        """
        Perform web search and return WebResult model objects.

        Args:
            query: Search query text
            num_results: Maximum number of results to return

        Returns:
            List of WebResult objects
        """
        try:
            results = self.fallback_search.search(query, num_results)

            web_results = []
            for result in results:
                try:
                    web_result = WebResult.from_search_result({
                        'title': result['title'],
                        'snippet': result['snippet'],
                        'url': result['url'],
                        'relevance_score': result['confidence']
                    })
                    web_results.append(web_result)
                except Exception as e:
                    # Log but don't fail for individual result errors
                    print(f"Warning: Failed to create WebResult: {e}")
                    continue

            return web_results

        except Exception as e:
            raise Exception(f"Web search with models failed: {str(e)}")

    def validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate web search input."""
        return self.fallback_search.validate_web_input(input_data)

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of web search service."""
        return self.fallback_search.get_health_status()