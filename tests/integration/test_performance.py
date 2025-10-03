"""Integration test for performance validation.

Tests Scenario 7 from quickstart.md: Timeout Handling and performance requirements.
This test must fail initially and pass after implementation.
"""

import concurrent.futures
import time

import pytest


class TestPerformance:
    """Test performance validation and timeout handling."""

    def test_vector_search_performance_requirement(self):
        """Test vector-only queries complete in < 1 second."""
        from server import mcp

        query_input = {
            "query": "machine learning overfitting",
            "limit": 5
        }

        start_time = time.time()
        result = mcp.call_tool("vector_search_tool", query_input)
        end_time = time.time()

        # Vector search should complete in under 1 second
        assert end_time - start_time < 1.0
        assert result["search_time_seconds"] < 1.0

    def test_fallback_query_performance_requirement(self):
        """Test fallback queries complete in < 5 seconds total."""
        from server import mcp

        query_input = {
            "query": "latest technology trends 2025"
        }

        start_time = time.time()
        result = mcp.call_tool("intelligent_query_tool", query_input)
        end_time = time.time()

        # Total response should be under 5 seconds
        assert end_time - start_time < 5.0
        assert result["routing_info"]["response_time_seconds"] < 5.0

    def test_error_scenario_performance(self):
        """Test error scenarios complete in < 2 seconds."""
        from server import mcp

        start_time = time.time()
        try:
            mcp.call_tool("intelligent_query_tool", {"query": ""})
        except Exception:
            pass  # Expected to fail
        end_time = time.time()

        # Error handling should be fast
        assert end_time - start_time < 2.0

    def test_timeout_enforcement(self):
        """Test timeout is enforced at exactly 5 seconds."""
        import time
        from unittest.mock import patch

        from server import mcp

        # Mock a slow operation that would exceed 5 seconds
        def slow_search(*args, **kwargs):
            time.sleep(6)  # Simulate 6-second operation
            return "result"

        with patch('fallback_search.FallbackSearch.search', side_effect=slow_search):
            start_time = time.time()

            with pytest.raises(Exception) as exc_info:
                mcp.call_tool("intelligent_query_tool", {"query": "test"})

            end_time = time.time()

            # Should timeout at 5 seconds
            assert end_time - start_time <= 5.5  # Allow small buffer

            error = exc_info.value
            assert error.error_type == "TIMEOUT_ERROR"
            assert error.timeout_seconds == 5

    @pytest.mark.slow
    def test_concurrent_query_performance(self):
        """Test performance under concurrent load."""
        from server import mcp

        def run_query(query_text):
            start_time = time.time()
            result = mcp.call_tool("intelligent_query_tool", {"query": query_text})
            end_time = time.time()
            return end_time - start_time, result

        queries = [
            "machine learning algorithms",
            "deep learning techniques",
            "neural network architectures",
            "data science methods",
            "artificial intelligence trends"
        ]

        # Run queries concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_query, query) for query in queries]
            results = [future.result() for future in futures]

        # All queries should meet performance requirements
        for response_time, _result in results:
            assert response_time < 5.0, f"Query took {response_time}s, should be < 5s"

    def test_confidence_threshold_accuracy(self):
        """Test 0.7 threshold consistently applied."""
        from server import mcp

        # Test multiple queries to verify threshold consistency
        test_queries = [
            "overfitting in machine learning",
            "gradient descent optimization",
            "neural network training",
            "feature engineering techniques"
        ]

        for query in test_queries:
            result = mcp.call_tool("intelligent_query_tool", {"query": query})
            routing = result["routing_info"]

            if routing["vector_confidence_avg"] is not None:
                if routing["vector_confidence_avg"] >= 0.7:
                    assert routing["threshold_met"] is True
                    assert routing["fallback_triggered"] is False
                else:
                    assert routing["threshold_met"] is False
                    assert routing["fallback_triggered"] is True
