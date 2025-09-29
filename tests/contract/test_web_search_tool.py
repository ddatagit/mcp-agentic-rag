"""Contract tests for bright_data_web_search_tool."""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.mark.contract
class TestWebSearchToolContract:
    """Contract tests for bright_data_web_search_tool."""

    def test_tool_signature_exists(self):
        """Test that the tool function exists with correct signature."""
        try:
            from mcp_agentic_rag.server.mcp_server import bright_data_web_search_tool

            # Check function signature
            import inspect
            sig = inspect.signature(bright_data_web_search_tool)
            params = list(sig.parameters.keys())

            assert "query" in params, "Tool must have 'query' parameter"
            assert sig.return_annotation == dict, "Tool must return dict"

        except ImportError:
            pytest.fail("bright_data_web_search_tool not found - implement in mcp_server.py")

    def test_input_validation_string_required(self):
        """Test that tool validates input is a string."""
        try:
            from mcp_agentic_rag.server.mcp_server import bright_data_web_search_tool

            # Test non-string input raises ValueError
            with pytest.raises(ValueError, match="query must be a string"):
                bright_data_web_search_tool(123)

        except ImportError:
            pytest.fail("bright_data_web_search_tool not found")

    @patch('mcp_agentic_rag.services.web_search.FallbackSearch')
    def test_successful_query_returns_dict(self, mock_fallback_class):
        """Test that valid query returns dict response with correct structure."""
        try:
            from mcp_agentic_rag.server.mcp_server import bright_data_web_search_tool

            # Mock the fallback search to return test data
            mock_search = Mock()
            mock_search.search.return_value = {
                "results": [
                    {
                        "title": "Test Web Result",
                        "snippet": "Test snippet content",
                        "url": "https://example.com/test",
                        "relevance_score": 0.85
                    }
                ],
                "total_results": 1,
                "response_time": 0.5
            }
            mock_fallback_class.return_value = mock_search

            result = bright_data_web_search_tool("machine learning tutorial")

            # Verify response contract
            assert isinstance(result, dict), "Tool must return dict"
            assert "results" in result, "Response must have 'results' field"
            assert "total_results" in result, "Response must have 'total_results' field"
            assert "response_time" in result, "Response must have 'response_time' field"

        except ImportError:
            pytest.fail("bright_data_web_search_tool not found")