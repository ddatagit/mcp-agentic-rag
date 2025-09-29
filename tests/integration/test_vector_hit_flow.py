"""Integration test for vector DB hit scenario.

Tests Scenario 1 from quickstart.md: Vector Database Hit (ML Query with High Confidence).
This test must fail initially and pass after implementation.
"""

import pytest
from typing import Dict, Any
import time


class TestVectorHitFlow:
    """Test vector database hit scenario with high confidence."""

    @pytest.fixture
    def ml_query_input(self) -> Dict[str, Any]:
        """ML query that should hit vector DB with high confidence."""
        return {
            "query": "What is overfitting in machine learning models?",
            "domain_hint": "machine_learning"
        }

    def test_vector_hit_scenario_complete_flow(self, ml_query_input: Dict[str, Any]):
        """Test complete flow for vector DB hit scenario."""
        # This will fail until implementation exists
        from server import mcp

        start_time = time.time()
        result = mcp.call_tool("intelligent_query_tool", ml_query_input)
        end_time = time.time()

        # Validate response structure
        assert "query_id" in result
        assert "combined_results" in result
        assert "routing_info" in result

        routing = result["routing_info"]

        # Expected behavior from quickstart.md
        assert routing["sources_used"] == ["vector_db"], "Should only use vector DB"
        assert routing["fallback_triggered"] is False, "Should not trigger fallback"
        assert routing["threshold_met"] is True, "Should meet confidence threshold"
        assert routing["vector_confidence_avg"] >= 0.7, "Should have confidence >= 0.7"

        # ML keywords detection
        assert "ml_keywords_detected" in routing
        ml_keywords = routing["ml_keywords_detected"]
        assert len(ml_keywords) > 0, "Should detect ML keywords"
        assert any(keyword in ["overfitting", "machine learning"] for keyword in ml_keywords)

        # Performance requirement
        response_time = end_time - start_time
        assert response_time < 2.0, f"Response time {response_time}s should be < 2s"
        assert routing["response_time_seconds"] < 2.0

        # Results should contain only vector DB results
        combined_results = result["combined_results"]
        assert len(combined_results) > 0, "Should return results"

        for item in combined_results:
            assert item["source"] == "vector_db", "All results should be from vector DB"
            assert "source_document" in item, "Vector results should have source_document"
            assert "content" in item
            assert item["confidence"] >= 0.7, "All results should meet confidence threshold"

    def test_vector_hit_no_web_search_called(self, ml_query_input: Dict[str, Any]):
        """Test that web search is not called when vector DB provides sufficient results."""
        from server import mcp
        from unittest.mock import patch

        # Mock web search to ensure it's not called
        with patch('fallback_search.FallbackSearch.search') as mock_web_search:
            result = mcp.call_tool("intelligent_query_tool", ml_query_input)

            # Web search should not be called
            mock_web_search.assert_not_called()

            # Should still get results from vector DB
            assert len(result["combined_results"]) > 0

    def test_ml_keyword_detection_accuracy(self):
        """Test ML keyword detection for various ML queries."""
        from server import mcp

        ml_queries = [
            {"query": "gradient descent optimization algorithms", "expected_keywords": ["gradient descent"]},
            {"query": "neural networks vs transformer architectures", "expected_keywords": ["neural networks"]},
            {"query": "feature engineering techniques", "expected_keywords": ["feature engineering"]},
            {"query": "cross-validation methods", "expected_keywords": ["cross-validation"]},
        ]

        for query_data in ml_queries:
            result = mcp.call_tool("intelligent_query_tool", query_data)
            routing = result["routing_info"]

            detected_keywords = routing["ml_keywords_detected"]
            expected_keywords = query_data["expected_keywords"]

            # Should detect at least one expected keyword
            assert any(keyword in detected_keywords for keyword in expected_keywords), \
                f"Should detect keywords {expected_keywords} in query '{query_data['query']}'"

    def test_vector_confidence_threshold_validation(self, ml_query_input: Dict[str, Any]):
        """Test that confidence threshold is properly applied."""
        from server import mcp

        result = mcp.call_tool("intelligent_query_tool", ml_query_input)
        routing = result["routing_info"]

        if routing["vector_confidence_avg"] is not None:
            # If we have vector results with confidence >= 0.7, threshold should be met
            if routing["vector_confidence_avg"] >= 0.7:
                assert routing["threshold_met"] is True
                assert routing["fallback_triggered"] is False
            else:
                # If confidence < 0.7, threshold not met and fallback should trigger
                assert routing["threshold_met"] is False
                assert routing["fallback_triggered"] is True

    def test_response_metadata_completeness(self, ml_query_input: Dict[str, Any]):
        """Test that all required metadata is present in response."""
        from server import mcp

        result = mcp.call_tool("intelligent_query_tool", ml_query_input)
        routing = result["routing_info"]

        # All required fields should be present
        required_fields = [
            "sources_used", "fallback_triggered", "threshold_met",
            "vector_confidence_avg", "ml_keywords_detected", "response_time_seconds"
        ]

        for field in required_fields:
            assert field in routing, f"Required field '{field}' missing from routing_info"

        # Validate field types and ranges
        assert isinstance(routing["sources_used"], list)
        assert isinstance(routing["fallback_triggered"], bool)
        assert isinstance(routing["threshold_met"], bool)
        assert isinstance(routing["ml_keywords_detected"], list)
        assert isinstance(routing["response_time_seconds"], (int, float))
        assert 0 < routing["response_time_seconds"] <= 5.0

        if routing["vector_confidence_avg"] is not None:
            assert 0.0 <= routing["vector_confidence_avg"] <= 1.0

    def test_vector_results_quality(self, ml_query_input: Dict[str, Any]):
        """Test that vector results are high quality and relevant."""
        from server import mcp

        result = mcp.call_tool("intelligent_query_tool", ml_query_input)
        combined_results = result["combined_results"]

        # Should have at least one result
        assert len(combined_results) > 0

        for item in combined_results:
            # All vector results should have required fields
            assert "content" in item
            assert "confidence" in item
            assert "source" in item
            assert "source_document" in item

            # Content should be non-empty and relevant
            assert len(item["content"]) > 10, "Content should be substantial"
            assert item["source"] == "vector_db"

            # Confidence should be high for ML queries
            assert item["confidence"] >= 0.7, f"Confidence {item['confidence']} should be >= 0.7"

    def test_consistent_behavior_across_runs(self, ml_query_input: Dict[str, Any]):
        """Test that the same query produces consistent results."""
        from server import mcp

        # Run the same query multiple times
        results = []
        for _ in range(3):
            result = mcp.call_tool("intelligent_query_tool", ml_query_input)
            results.append(result)

        # Should consistently use vector DB only
        for result in results:
            routing = result["routing_info"]
            assert routing["sources_used"] == ["vector_db"]
            assert routing["fallback_triggered"] is False
            assert routing["threshold_met"] is True

        # Should have similar confidence scores (within reasonable variance)
        confidence_avgs = [r["routing_info"]["vector_confidence_avg"] for r in results]
        max_confidence = max(confidence_avgs)
        min_confidence = min(confidence_avgs)
        assert (max_confidence - min_confidence) < 0.1, "Confidence should be consistent across runs"

    @pytest.mark.slow
    def test_performance_under_load(self, ml_query_input: Dict[str, Any]):
        """Test performance when handling multiple vector queries."""
        from server import mcp
        import concurrent.futures

        def run_query():
            start_time = time.time()
            result = mcp.call_tool("intelligent_query_tool", ml_query_input)
            end_time = time.time()
            return end_time - start_time, result

        # Run multiple queries concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_query) for _ in range(5)]
            results = [future.result() for future in futures]

        # All queries should complete within performance requirements
        for response_time, result in results:
            assert response_time < 2.0, f"Response time {response_time}s should be < 2s"
            routing = result["routing_info"]
            assert routing["sources_used"] == ["vector_db"]
            assert routing["threshold_met"] is True