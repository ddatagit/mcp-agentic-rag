"""Contract tests for machine_learning_faq_retrieval_tool."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.mark.contract
class TestMLFAQToolContract:
    """Contract tests for machine_learning_faq_retrieval_tool."""

    def test_tool_signature_exists(self):
        """Test that the tool function exists with correct signature."""
        # This will fail until we implement the tool
        try:
            # Check function signature
            import inspect

            from mcp_agentic_rag.server.mcp_server import (
                machine_learning_faq_retrieval_tool,
            )
            sig = inspect.signature(machine_learning_faq_retrieval_tool)
            params = list(sig.parameters.keys())

            assert "query" in params, "Tool must have 'query' parameter"
            assert sig.return_annotation is str, "Tool must return str"

        except ImportError:
            pytest.fail("machine_learning_faq_retrieval_tool not found - implement in mcp_server.py")

    def test_input_validation_string_required(self):
        """Test that tool validates input is a string."""
        try:
            from mcp_agentic_rag.server.mcp_server import (
                machine_learning_faq_retrieval_tool,
            )

            # Test non-string input raises ValueError
            with pytest.raises(ValueError, match="query must be a string"):
                machine_learning_faq_retrieval_tool(123)

            with pytest.raises(ValueError, match="query must be a string"):
                machine_learning_faq_retrieval_tool(None)

        except ImportError:
            pytest.fail("machine_learning_faq_retrieval_tool not found")

    def test_input_validation_non_empty(self):
        """Test that tool validates input is non-empty."""
        try:
            from mcp_agentic_rag.server.mcp_server import (
                machine_learning_faq_retrieval_tool,
            )

            # Test empty string raises ValueError
            with pytest.raises(ValueError, match="query cannot be empty"):
                machine_learning_faq_retrieval_tool("")

            with pytest.raises(ValueError, match="query cannot be empty"):
                machine_learning_faq_retrieval_tool("   ")

        except ImportError:
            pytest.fail("machine_learning_faq_retrieval_tool not found")

    def test_input_validation_max_length(self):
        """Test that tool validates input max length."""
        try:
            from mcp_agentic_rag.server.mcp_server import (
                machine_learning_faq_retrieval_tool,
            )

            # Test query too long raises ValueError
            long_query = "x" * 1001
            with pytest.raises(ValueError, match="query exceeds maximum length"):
                machine_learning_faq_retrieval_tool(long_query)

        except ImportError:
            pytest.fail("machine_learning_faq_retrieval_tool not found")

    @patch('mcp_agentic_rag.services.vector_retrieval.Retriever')
    def test_successful_query_returns_string(self, mock_retriever_class):
        """Test that valid query returns string response."""
        try:
            from mcp_agentic_rag.server.mcp_server import (
                machine_learning_faq_retrieval_tool,
            )

            # Mock the retriever to return test data
            mock_retriever = Mock()
            mock_retriever.search.return_value = "Test ML content about neural networks"
            mock_retriever_class.return_value = mock_retriever

            result = machine_learning_faq_retrieval_tool("What is machine learning?")

            assert isinstance(result, str), "Tool must return string"
            assert len(result) > 0, "Tool must return non-empty string"
            assert "neural networks" in result, "Tool should return relevant content"

        except ImportError:
            pytest.fail("machine_learning_faq_retrieval_tool not found")

    @patch('mcp_agentic_rag.services.vector_retrieval.Retriever')
    def test_database_connection_error_handling(self, mock_retriever_class):
        """Test that database connection errors are handled properly."""
        try:
            from mcp_agentic_rag.server.mcp_server import (
                machine_learning_faq_retrieval_tool,
            )

            # Mock retriever to raise connection error
            mock_retriever = Mock()
            mock_retriever.search.side_effect = ConnectionError("Database unreachable")
            mock_retriever_class.return_value = mock_retriever

            with pytest.raises(ConnectionError, match="Database unreachable"):
                machine_learning_faq_retrieval_tool("What is machine learning?")

        except ImportError:
            pytest.fail("machine_learning_faq_retrieval_tool not found")

    @patch('mcp_agentic_rag.services.vector_retrieval.Retriever')
    def test_timeout_error_handling(self, mock_retriever_class):
        """Test that timeout errors are handled properly."""
        try:
            from mcp_agentic_rag.server.mcp_server import (
                machine_learning_faq_retrieval_tool,
            )

            # Mock retriever to raise timeout error
            mock_retriever = Mock()
            mock_retriever.search.side_effect = TimeoutError("Search timed out")
            mock_retriever_class.return_value = mock_retriever

            with pytest.raises(TimeoutError, match="Search timed out"):
                machine_learning_faq_retrieval_tool("What is machine learning?")

        except ImportError:
            pytest.fail("machine_learning_faq_retrieval_tool not found")

    def test_mcp_tool_decorator_applied(self):
        """Test that the function has MCP tool decorator applied."""
        try:
            from mcp_agentic_rag.server.mcp_server import mcp

            # Check that the tool is registered with MCP
            # This test verifies the @mcp.tool() decorator is applied
            tools = getattr(mcp, '_tools', [])
            tool_names = [tool.name for tool in tools if hasattr(tool, 'name')]

            assert "machine_learning_faq_retrieval_tool" in tool_names, \
                "Tool must be registered with @mcp.tool() decorator"

        except ImportError:
            pytest.fail("MCP server or tool registration not found")

    def test_response_format_compliance(self):
        """Test that response format matches contract specification."""
        try:
            from mcp_agentic_rag.server.mcp_server import (
                machine_learning_faq_retrieval_tool,
            )

            with patch('mcp_agentic_rag.services.vector_retrieval.Retriever') as mock_retriever_class:
                mock_retriever = Mock()
                mock_retriever.search.return_value = "Machine learning is a subset of AI"
                mock_retriever_class.return_value = mock_retriever

                result = machine_learning_faq_retrieval_tool("What is ML?")

                # Verify response contract
                assert isinstance(result, str), "Response must be string type"
                assert len(result.strip()) > 0, "Response must not be empty"
                assert not result.startswith('{"'), "Response must be plain text, not JSON"

        except ImportError:
            pytest.fail("machine_learning_faq_retrieval_tool not found")
