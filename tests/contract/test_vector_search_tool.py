"""Contract tests for vector_search_tool.

Tests the MCP tool contract specification from contracts/vector_search_tool.json.
These tests must fail initially and pass after implementation.
"""

import json
from typing import Any

import pytest
from pydantic import ValidationError


class TestVectorSearchToolContract:
    """Test contract compliance for vector_search_tool."""

    @pytest.fixture
    def contract_spec(self) -> dict[str, Any]:
        """Load contract specification from JSON file."""
        with open("specs/001-rag-with-intelligent/contracts/vector_search_tool.json") as f:
            return json.load(f)

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Valid input according to contract."""
        return {
            "query": "feature engineering techniques for machine learning",
            "limit": 3,
            "min_confidence": 0.5
        }

    @pytest.fixture
    def minimal_input(self) -> dict[str, Any]:
        """Minimal valid input (only required fields)."""
        return {
            "query": "machine learning overfitting"
        }

    @pytest.fixture
    def invalid_inputs(self) -> list[dict[str, Any]]:
        """Invalid inputs for validation testing."""
        return [
            {},  # Missing required query
            {"query": ""},  # Empty query
            {"query": "x" * 1001},  # Query too long
            {"query": "test", "limit": 0},  # Limit too small
            {"query": "test", "limit": 25},  # Limit too large
            {"query": "test", "min_confidence": -0.1},  # Confidence below minimum
            {"query": "test", "min_confidence": 1.1},  # Confidence above maximum
        ]

    def test_tool_exists_in_server(self):
        """Test that vector_search_tool is registered in MCP server."""
        # This will fail until tool is implemented
        from server import mcp

        # Check if tool is registered
        assert hasattr(mcp, '_tools'), "MCP server should have tools registered"
        tool_names = [tool.name for tool in mcp._tools.values()]
        assert "vector_search_tool" in tool_names, "vector_search_tool should be registered"

    def test_input_validation_valid(self, valid_input: dict[str, Any], minimal_input: dict[str, Any]):
        """Test input validation accepts valid inputs."""
        # This will fail until validation is implemented
        from rag_code import Retriever

        retriever = Retriever(None, None)  # Will be properly initialized in implementation

        # Should not raise for valid inputs
        retriever.validate_vector_input(valid_input)
        retriever.validate_vector_input(minimal_input)

    def test_input_validation_rejects_invalid(self, invalid_inputs: list):
        """Test input validation rejects invalid inputs."""
        from rag_code import Retriever

        retriever = Retriever(None, None)

        for invalid_input in invalid_inputs:
            with pytest.raises((ValidationError, ValueError)):
                retriever.validate_vector_input(invalid_input)

    def test_output_schema_compliance(self, valid_input: dict[str, Any]):
        """Test output matches contract schema."""
        from server import mcp

        # This will fail until implementation exists
        result = mcp.call_tool("vector_search_tool", valid_input)

        # Check required fields exist
        assert "query_id" in result
        assert "results" in result
        assert "total_found" in result
        assert "search_time_seconds" in result

        # Validate types
        assert isinstance(result["query_id"], str)
        assert isinstance(result["results"], list)
        assert isinstance(result["total_found"], int)
        assert isinstance(result["search_time_seconds"], (int, float))

        # Validate result items structure
        for item in result["results"]:
            assert "content" in item
            assert "confidence" in item
            assert "source_document" in item
            assert isinstance(item["content"], str)
            assert isinstance(item["confidence"], (int, float))
            assert 0 <= item["confidence"] <= 1
            assert isinstance(item["source_document"], str)

    def test_confidence_filtering(self):
        """Test min_confidence parameter filters results correctly."""
        from server import mcp

        # Search with different confidence thresholds
        low_threshold_input = {
            "query": "machine learning",
            "min_confidence": 0.3
        }

        high_threshold_input = {
            "query": "machine learning",
            "min_confidence": 0.8
        }

        low_result = mcp.call_tool("vector_search_tool", low_threshold_input)
        high_result = mcp.call_tool("vector_search_tool", high_threshold_input)

        # Higher threshold should return fewer or equal results
        assert len(high_result["results"]) <= len(low_result["results"])

        # All results should meet minimum confidence
        for item in high_result["results"]:
            assert item["confidence"] >= 0.8

    def test_limit_parameter(self):
        """Test limit parameter controls result count."""
        from server import mcp

        input_small_limit = {
            "query": "deep learning",
            "limit": 2
        }

        input_large_limit = {
            "query": "deep learning",
            "limit": 10
        }

        small_result = mcp.call_tool("vector_search_tool", input_small_limit)
        large_result = mcp.call_tool("vector_search_tool", input_large_limit)

        # Small limit should return at most 2 results
        assert len(small_result["results"]) <= 2

        # Large limit should return more results (if available)
        assert len(large_result["results"]) >= len(small_result["results"])

    def test_average_confidence_calculation(self, valid_input: dict[str, Any]):
        """Test average_confidence field is calculated correctly."""
        from server import mcp

        result = mcp.call_tool("vector_search_tool", valid_input)

        if result["results"]:
            # Calculate expected average
            confidences = [item["confidence"] for item in result["results"]]
            expected_avg = sum(confidences) / len(confidences)

            assert "average_confidence" in result
            assert abs(result["average_confidence"] - expected_avg) < 0.001

    def test_vector_db_only_no_web_search(self, valid_input: dict[str, Any]):
        """Test that vector_search_tool only queries vector DB, never web."""
        from server import mcp

        result = mcp.call_tool("vector_search_tool", valid_input)

        # Should not contain any web search results or metadata
        for item in result["results"]:
            assert "url" not in item
            assert "title" not in item
            assert "source_document" in item  # Vector results have source_document

    def test_error_handling_invalid_query(self):
        """Test INVALID_QUERY error handling."""
        from server import mcp

        with pytest.raises(Exception) as exc_info:
            mcp.call_tool("vector_search_tool", {"query": ""})

        error = exc_info.value
        assert hasattr(error, 'error_type')
        assert error.error_type == "INVALID_QUERY"

    def test_error_handling_vector_db_unavailable(self):
        """Test VECTOR_DB_UNAVAILABLE error handling."""
        # This test requires mocking Qdrant to be unavailable
        from unittest.mock import patch

        from server import mcp

        with patch('qdrant_client.QdrantClient') as mock_client:
            mock_client.side_effect = ConnectionError("Cannot connect to Qdrant")

            with pytest.raises(Exception) as exc_info:
                mcp.call_tool("vector_search_tool", {"query": "test"})

            error = exc_info.value
            assert error.error_type == "VECTOR_DB_UNAVAILABLE"
            assert hasattr(error, 'http_status_code')
            assert error.http_status_code == 503

    def test_search_timeout_handling(self):
        """Test SEARCH_TIMEOUT error handling."""
        from unittest.mock import patch

        from server import mcp

        with patch('rag_code.Retriever.search') as mock_search:
            mock_search.side_effect = TimeoutError("Search timed out")

            with pytest.raises(Exception) as exc_info:
                mcp.call_tool("vector_search_tool", {"query": "test"})

            error = exc_info.value
            assert error.error_type == "SEARCH_TIMEOUT"
            assert hasattr(error, 'http_status_code')
            assert error.http_status_code == 408

    def test_performance_requirement(self, valid_input: dict[str, Any]):
        """Test vector search completes within performance requirements."""
        import time

        from server import mcp

        start_time = time.time()
        result = mcp.call_tool("vector_search_tool", valid_input)
        end_time = time.time()

        # Vector search should complete in under 1 second
        assert end_time - start_time < 1.0
        assert result["search_time_seconds"] < 1.0
