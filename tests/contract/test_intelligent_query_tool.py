"""Contract tests for intelligent_query_tool.

Tests the MCP tool contract specification from contracts/intelligent_query_tool.json.
These tests must fail initially and pass after implementation.
"""

import pytest
import json
from typing import Dict, Any
from pydantic import ValidationError


class TestIntelligentQueryToolContract:
    """Test contract compliance for intelligent_query_tool."""

    @pytest.fixture
    def contract_spec(self) -> Dict[str, Any]:
        """Load contract specification from JSON file."""
        with open("specs/001-rag-with-intelligent/contracts/intelligent_query_tool.json", "r") as f:
            return json.load(f)

    @pytest.fixture
    def valid_input(self) -> Dict[str, Any]:
        """Valid input according to contract."""
        return {
            "query": "What is overfitting in machine learning models?",
            "domain_hint": "machine_learning"
        }

    @pytest.fixture
    def invalid_inputs(self) -> list[Dict[str, Any]]:
        """Invalid inputs for validation testing."""
        return [
            {},  # Missing required query
            {"query": ""},  # Empty query
            {"query": "x" * 1001},  # Query too long
            {"query": "test", "domain_hint": "invalid"},  # Invalid domain hint
            {"query": "test", "force_web_search": "not_boolean"},  # Invalid type
        ]

    def test_tool_exists_in_server(self):
        """Test that intelligent_query_tool is registered in MCP server."""
        # This will fail until tool is implemented
        from server import mcp

        # Check if tool is registered
        assert hasattr(mcp, '_tools'), "MCP server should have tools registered"
        tool_names = [tool.name for tool in mcp._tools.values()]
        assert "intelligent_query_tool" in tool_names, "intelligent_query_tool should be registered"

    def test_input_schema_validation(self, contract_spec: Dict[str, Any], valid_input: Dict[str, Any]):
        """Test input validation matches contract schema."""
        # This will fail until validation is implemented
        from intelligent_router import IntelligentRouter

        router = IntelligentRouter()

        # Should not raise for valid input
        result = router.validate_input(valid_input)
        assert result is not None

    def test_input_schema_rejects_invalid(self, contract_spec: Dict[str, Any], invalid_inputs: list):
        """Test input validation rejects invalid inputs."""
        from intelligent_router import IntelligentRouter

        router = IntelligentRouter()

        for invalid_input in invalid_inputs:
            with pytest.raises((ValidationError, ValueError)):
                router.validate_input(invalid_input)

    def test_output_schema_compliance(self, valid_input: Dict[str, Any]):
        """Test output matches contract schema."""
        from server import mcp

        # This will fail until implementation exists
        result = mcp.call_tool("intelligent_query_tool", valid_input)

        # Check required fields exist
        assert "query_id" in result
        assert "combined_results" in result
        assert "routing_info" in result

        # Validate routing_info structure
        routing = result["routing_info"]
        assert "sources_used" in routing
        assert "fallback_triggered" in routing
        assert "response_time_seconds" in routing

        # Validate types
        assert isinstance(result["query_id"], str)
        assert isinstance(result["combined_results"], list)
        assert isinstance(routing["sources_used"], list)
        assert isinstance(routing["fallback_triggered"], bool)
        assert isinstance(routing["response_time_seconds"], (int, float))

    def test_ml_query_scenario(self):
        """Test ML query scenario from quickstart.md."""
        from server import mcp

        query_input = {
            "query": "What is overfitting in machine learning models?",
            "domain_hint": "machine_learning"
        }

        result = mcp.call_tool("intelligent_query_tool", query_input)

        # Should use vector DB first
        routing = result["routing_info"]
        assert "vector_db" in routing["sources_used"]
        assert "ml_keywords_detected" in routing
        assert len(routing["ml_keywords_detected"]) > 0

    def test_general_query_fallback(self):
        """Test general query triggering fallback."""
        from server import mcp

        query_input = {
            "query": "What are the latest quantum computing breakthroughs in 2025?",
            "domain_hint": "general"
        }

        result = mcp.call_tool("intelligent_query_tool", query_input)

        # Should trigger fallback due to low confidence
        routing = result["routing_info"]
        assert routing["fallback_triggered"] is True
        assert "web_search" in routing["sources_used"]

    def test_confidence_threshold_behavior(self):
        """Test 0.7 confidence threshold behavior."""
        from server import mcp

        query_input = {
            "query": "feature engineering techniques for machine learning"
        }

        result = mcp.call_tool("intelligent_query_tool", query_input)

        routing = result["routing_info"]

        if "vector_confidence_avg" in routing and routing["vector_confidence_avg"] is not None:
            if routing["vector_confidence_avg"] >= 0.7:
                assert routing["threshold_met"] is True
            else:
                assert routing["threshold_met"] is False
                assert routing["fallback_triggered"] is True

    def test_response_time_constraint(self, valid_input: Dict[str, Any]):
        """Test 5-second maximum response time."""
        from server import mcp

        result = mcp.call_tool("intelligent_query_tool", valid_input)

        routing = result["routing_info"]
        assert routing["response_time_seconds"] <= 5.0

    def test_error_handling(self):
        """Test error scenarios match contract."""
        from server import mcp

        # Test invalid query error
        with pytest.raises(Exception) as exc_info:
            mcp.call_tool("intelligent_query_tool", {"query": ""})

        # Should return structured error matching contract
        error = exc_info.value
        assert hasattr(error, 'error_type')
        assert error.error_type == "INVALID_QUERY"

    @pytest.mark.slow
    def test_no_results_error(self):
        """Test NO_RESULTS_FOUND error scenario."""
        from server import mcp

        query_input = {
            "query": "xyzabc123 nonexistent impossible query term"
        }

        with pytest.raises(Exception) as exc_info:
            mcp.call_tool("intelligent_query_tool", query_input)

        error = exc_info.value
        assert error.error_type == "NO_RESULTS_FOUND"
        assert hasattr(error, 'sources_attempted')
        assert hasattr(error, 'http_status_code')
        assert error.http_status_code == 404