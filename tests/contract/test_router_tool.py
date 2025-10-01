"""Contract tests for intelligent_query_router_tool."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.mark.contract
class TestRouterToolContract:
    """Contract tests for intelligent_query_router_tool."""

    def test_tool_signature_exists(self):
        """Test that the tool function exists with correct signature."""
        try:
            import inspect

            from mcp_agentic_rag.server.mcp_server import (
                intelligent_query_router_tool,
            )
            sig = inspect.signature(intelligent_query_router_tool)
            params = list(sig.parameters.keys())

            assert "query" in params, "Tool must have 'query' parameter"
            assert sig.return_annotation is dict, "Tool must return dict"

        except ImportError:
            pytest.fail("intelligent_query_router_tool not found - implement in mcp_server.py")

    def test_input_validation(self):
        """Test input validation for router tool."""
        try:
            from mcp_agentic_rag.server.mcp_server import (
                intelligent_query_router_tool,
            )

            with pytest.raises(ValueError):
                intelligent_query_router_tool("")

        except ImportError:
            pytest.fail("intelligent_query_router_tool not found")

    @patch('mcp_agentic_rag.services.intelligent_router.IntelligentRouter')
    def test_returns_unified_response(self, mock_router_class):
        """Test that router returns unified response with routing decision."""
        try:
            from mcp_agentic_rag.server.mcp_server import (
                intelligent_query_router_tool,
            )

            mock_router = Mock()
            mock_router.route_query.return_value = {
                "results": [],
                "sources_used": ["vector"],
                "routing_decision": {
                    "strategy": "vector_only",
                    "reasoning": "ML terminology detected",
                    "domain_detected": True,
                    "ml_keywords_found": ["neural", "network"]
                },
                "total_results": 0,
                "response_time": 1.2,
                "confidence_level": "high",
                "fallback_triggered": False
            }
            mock_router_class.return_value = mock_router

            result = intelligent_query_router_tool("neural network architecture")

            assert isinstance(result, dict)
            assert "routing_decision" in result
            assert "sources_used" in result
            assert "fallback_triggered" in result

        except ImportError:
            pytest.fail("intelligent_query_router_tool not found")
