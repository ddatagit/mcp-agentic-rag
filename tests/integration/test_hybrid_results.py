"""Integration test for hybrid results scenario.

Tests Scenario 3 from quickstart.md: Hybrid Results (Mixed Confidence Ranking).
This test must fail initially and pass after implementation.
"""

import pytest
from typing import Dict, Any
import time


class TestHybridResults:
    """Test hybrid results with mixed confidence ranking."""

    @pytest.fixture
    def technical_query_input(self) -> Dict[str, Any]:
        """Technical query that should produce hybrid results."""
        return {
            "query": "neural networks vs transformer architectures comparison 2025",
            "domain_hint": "technical"
        }

    def test_hybrid_scenario_complete_flow(self, technical_query_input: Dict[str, Any]):
        """Test complete flow for hybrid results scenario."""
        from server import mcp

        result = mcp.call_tool("intelligent_query_tool", technical_query_input)
        routing = result["routing_info"]

        # Expected behavior from quickstart.md
        expected_sources = ["vector_db", "web_search"]
        assert routing["sources_used"] == expected_sources
        assert routing["fallback_triggered"] is True

        # Should detect ML keywords
        ml_keywords = routing["ml_keywords_detected"]
        assert "neural networks" in ml_keywords

        # Should have mixed results ranked by confidence
        combined_results = result["combined_results"]
        assert len(combined_results) > 0

        # Should have results from both sources
        vector_results = [r for r in combined_results if r["source"] == "vector_db"]
        web_results = [r for r in combined_results if r["source"] == "web_search"]
        assert len(vector_results) > 0
        assert len(web_results) > 0

        # Results should be ranked by confidence
        confidences = [r["confidence"] for r in combined_results]
        assert confidences == sorted(confidences, reverse=True)

    def test_source_attribution_clarity(self, technical_query_input: Dict[str, Any]):
        """Test that source attribution is clear for each result."""
        from server import mcp

        result = mcp.call_tool("intelligent_query_tool", technical_query_input)
        combined_results = result["combined_results"]

        for item in combined_results:
            # Each result must clearly indicate source
            assert "source" in item
            assert item["source"] in ["vector_db", "web_search"]

            # Vector results have source_document, web results have url/title
            if item["source"] == "vector_db":
                assert "source_document" in item
            elif item["source"] == "web_search":
                assert "url" in item
                assert "title" in item