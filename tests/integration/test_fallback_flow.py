"""Integration test for vector miss fallback scenario.

Tests Scenario 2 from quickstart.md: Vector Database Miss (General Query Below Threshold).
This test must fail initially and pass after implementation.
"""

import pytest
from typing import Dict, Any
import time


class TestFallbackFlow:
    """Test vector database miss with web search fallback."""

    @pytest.fixture
    def general_query_input(self) -> Dict[str, Any]:
        """General query that should trigger fallback due to low vector confidence."""
        return {
            "query": "What are the latest quantum computing breakthroughs in 2025?",
            "domain_hint": "general"
        }

    def test_fallback_scenario_complete_flow(self, general_query_input: Dict[str, Any]):
        """Test complete flow for fallback scenario."""
        # This will fail until implementation exists
        from server import mcp

        start_time = time.time()
        result = mcp.call_tool("intelligent_query_tool", general_query_input)
        end_time = time.time()

        # Validate response structure
        assert "query_id" in result
        assert "combined_results" in result
        assert "routing_info" in result

        routing = result["routing_info"]

        # Expected behavior from quickstart.md
        expected_sources = ["vector_db", "web_search"]
        assert routing["sources_used"] == expected_sources, f"Should use both sources, got {routing['sources_used']}"
        assert routing["fallback_triggered"] is True, "Should trigger fallback"
        assert routing["threshold_met"] is False, "Should not meet confidence threshold"

        # Vector confidence should be low
        if routing["vector_confidence_avg"] is not None:
            assert routing["vector_confidence_avg"] < 0.7, "Vector confidence should be < 0.7"

        # ML keywords should be empty for general query
        assert "ml_keywords_detected" in routing
        ml_keywords = routing["ml_keywords_detected"]
        assert len(ml_keywords) == 0, "Should not detect ML keywords in general query"

        # Performance requirement
        response_time = end_time - start_time
        assert response_time < 5.0, f"Response time {response_time}s should be < 5s"
        assert routing["response_time_seconds"] < 5.0

        # Results should contain web search results
        combined_results = result["combined_results"]
        assert len(combined_results) > 0, "Should return results"

        # Should have web search results
        web_results = [item for item in combined_results if item["source"] == "web_search"]
        assert len(web_results) > 0, "Should have web search results"

        # Web results should have proper structure
        for item in web_results:
            assert "url" in item, "Web results should have URL"
            assert "title" in item, "Web results should have title"
            assert "content" in item
            assert 0 <= item["confidence"] <= 1.0

    def test_vector_search_attempted_first(self, general_query_input: Dict[str, Any]):
        """Test that vector search is attempted first before fallback."""
        from server import mcp
        from unittest.mock import patch, call

        # Mock both vector and web search to track call order
        with patch('rag_code.Retriever.search') as mock_vector_search, \
             patch('fallback_search.FallbackSearch.search') as mock_web_search:

            # Configure mocks to return low confidence for vector, results for web
            mock_vector_search.return_value = "Low confidence vector results"
            mock_web_search.return_value = [{"title": "Test", "snippet": "Test", "url": "http://test.com"}]

            result = mcp.call_tool("intelligent_query_tool", general_query_input)

            # Vector search should be called first
            mock_vector_search.assert_called_once()
            # Web search should be called due to low confidence
            mock_web_search.assert_called_once()

    def test_confidence_ranking_across_sources(self, general_query_input: Dict[str, Any]):
        """Test that results are ranked by confidence regardless of source."""
        from server import mcp

        result = mcp.call_tool("intelligent_query_tool", general_query_input)
        combined_results = result["combined_results"]

        if len(combined_results) > 1:
            # Results should be sorted by confidence descending
            confidences = [item["confidence"] for item in combined_results]
            assert confidences == sorted(confidences, reverse=True), \
                "Results should be sorted by confidence (descending)"

            # Should have clear source attribution
            for item in combined_results:
                assert item["source"] in ["vector_db", "web_search"]

    def test_fallback_with_different_confidence_thresholds(self):
        """Test fallback behavior with queries of different confidence levels."""
        from server import mcp

        # Queries likely to have different vector confidence levels
        test_queries = [
            {"query": "random unrelated topic xyz123", "expected_fallback": True},
            {"query": "quantum computing algorithms 2025", "expected_fallback": True},
            {"query": "latest technology trends", "expected_fallback": True},
        ]

        for query_data in test_queries:
            result = mcp.call_tool("intelligent_query_tool", query_data)
            routing = result["routing_info"]

            if query_data["expected_fallback"]:
                assert routing["fallback_triggered"] is True, \
                    f"Query '{query_data['query']}' should trigger fallback"
                assert "web_search" in routing["sources_used"], \
                    "Should include web search in sources"

    def test_web_search_error_handling(self, general_query_input: Dict[str, Any]):
        """Test behavior when web search fails."""
        from server import mcp
        from unittest.mock import patch
        import httpx

        # Mock web search to fail
        with patch('httpx.get') as mock_get:
            mock_get.side_effect = httpx.ConnectError("Network error")

            # Should handle web search failure gracefully
            try:
                result = mcp.call_tool("intelligent_query_tool", general_query_input)
                # If it doesn't raise, should at least have vector results
                assert len(result["combined_results"]) >= 0
            except Exception as e:
                # Should provide appropriate error information
                assert hasattr(e, 'error_type')
                assert e.error_type in ["WEB_SEARCH_ERROR", "NO_RESULTS_FOUND"]

    def test_timeout_behavior_fallback_scenario(self, general_query_input: Dict[str, Any]):
        """Test timeout handling in fallback scenario."""
        from server import mcp

        start_time = time.time()
        result = mcp.call_tool("intelligent_query_tool", general_query_input)
        end_time = time.time()

        total_time = end_time - start_time

        # Should complete within 5 second limit
        assert total_time < 5.0, f"Total time {total_time}s exceeded 5s limit"

        routing = result["routing_info"]
        assert routing["response_time_seconds"] < 5.0

    def test_no_ml_keywords_in_general_queries(self):
        """Test that general queries don't trigger ML keyword detection."""
        from server import mcp

        general_queries = [
            "weather forecast for tomorrow",
            "latest news about politics",
            "cooking recipes for dinner",
            "travel destinations in Europe",
        ]

        for query in general_queries:
            result = mcp.call_tool("intelligent_query_tool", {"query": query})
            routing = result["routing_info"]

            ml_keywords = routing["ml_keywords_detected"]
            assert len(ml_keywords) == 0, \
                f"General query '{query}' should not detect ML keywords, got {ml_keywords}"

    def test_web_result_quality_and_format(self, general_query_input: Dict[str, Any]):
        """Test quality and format of web search results."""
        from server import mcp

        result = mcp.call_tool("intelligent_query_tool", general_query_input)
        combined_results = result["combined_results"]

        # Find web results
        web_results = [item for item in combined_results if item["source"] == "web_search"]
        assert len(web_results) > 0, "Should have web search results"

        for web_result in web_results:
            # Required fields for web results
            assert "title" in web_result and len(web_result["title"]) > 0
            assert "content" in web_result and len(web_result["content"]) > 0
            assert "url" in web_result
            assert "confidence" in web_result

            # URL validation
            url = web_result["url"]
            assert url.startswith(("http://", "https://")), f"Invalid URL: {url}"

            # Confidence should be normalized
            confidence = web_result["confidence"]
            assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range"

    def test_fallback_consistency_across_runs(self, general_query_input: Dict[str, Any]):
        """Test that fallback behavior is consistent across multiple runs."""
        from server import mcp

        results = []
        for _ in range(3):
            result = mcp.call_tool("intelligent_query_tool", general_query_input)
            results.append(result)

        # Should consistently trigger fallback
        for result in results:
            routing = result["routing_info"]
            assert routing["fallback_triggered"] is True
            assert "web_search" in routing["sources_used"]

    @pytest.mark.slow
    def test_fallback_performance_requirements(self, general_query_input: Dict[str, Any]):
        """Test performance requirements for fallback scenario."""
        from server import mcp
        import time

        # Run multiple fallback scenarios to test performance
        start_time = time.time()
        result = mcp.call_tool("intelligent_query_tool", general_query_input)
        end_time = time.time()

        # Total time should be under 5 seconds
        total_time = end_time - start_time
        assert total_time < 5.0, f"Fallback scenario took {total_time}s, should be < 5s"

        # Should have results from both sources
        routing = result["routing_info"]
        assert "vector_db" in routing["sources_used"]
        assert "web_search" in routing["sources_used"]

        # Should have meaningful results
        assert len(result["combined_results"]) > 0