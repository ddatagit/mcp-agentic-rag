"""Integration tests for ML domain query routing."""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.mark.integration
class TestMLQueryFlow:
    """Integration tests for ML domain query routing."""

    @patch('mcp_agentic_rag.services.intelligent_router.IntelligentRouter')
    def test_ml_query_routes_to_vector_db(self, mock_router):
        """Test that ML queries are routed to vector database."""
        try:
            from mcp_agentic_rag.server.mcp_server import intelligent_query_router_tool

            mock_router_instance = Mock()
            mock_router_instance.route_query.return_value = {
                "sources_used": ["vector"],
                "routing_decision": {"strategy": "vector_only"},
                "results": [],
                "total_results": 0,
                "response_time": 0.5,
                "confidence_level": "high",
                "fallback_triggered": False
            }
            mock_router.return_value = mock_router_instance

            result = intelligent_query_router_tool("What is neural network backpropagation?")
            assert "vector" in result["sources_used"]

        except ImportError:
            pytest.fail("Integration test failed - ML query routing not implemented")