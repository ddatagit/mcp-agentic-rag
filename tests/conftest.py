"""Test configuration and shared fixtures for MCP Agentic RAG tests."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock_client = Mock()
    mock_client.search = AsyncMock()
    mock_client.get_collection = Mock()
    mock_client.create_collection = Mock()
    return mock_client


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing."""
    mock_model = Mock()
    mock_model.encode = Mock(return_value=[[0.1, 0.2, 0.3] * 128])  # 384-dim vector
    return mock_model


@pytest.fixture
def mock_web_search_api():
    """Mock web search API for testing."""
    mock_api = AsyncMock()
    mock_api.search = AsyncMock(return_value={
        "results": [
            {
                "title": "Test Result",
                "snippet": "Test snippet",
                "url": "https://example.com",
                "relevance_score": 0.8
            }
        ],
        "total_results": 1,
        "response_time": 0.5
    })
    return mock_api


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return {
        "text": "What is machine learning?",
        "confidence_threshold": 0.7,
        "max_response_time": 5.0
    }


@pytest.fixture
def sample_vector_results():
    """Sample vector search results for testing."""
    return [
        {
            "document_id": "doc1",
            "content": "Machine learning is a subset of AI...",
            "score": 0.9,
            "source_document": "ml_basics.pdf",
            "metadata": {"topic": "ml_fundamentals"}
        },
        {
            "document_id": "doc2",
            "content": "Supervised learning uses labeled data...",
            "score": 0.8,
            "source_document": "supervised_learning.pdf",
            "metadata": {"topic": "supervised_learning"}
        }
    ]


@pytest.fixture
def sample_web_results():
    """Sample web search results for testing."""
    return [
        {
            "title": "Machine Learning Guide",
            "snippet": "A comprehensive guide to machine learning...",
            "url": "https://example.com/ml-guide",
            "relevance_score": 0.85,
            "source_domain": "example.com"
        },
        {
            "title": "ML Algorithms Overview",
            "snippet": "Overview of common ML algorithms...",
            "url": "https://another.com/algorithms",
            "relevance_score": 0.75,
            "source_domain": "another.com"
        }
    ]


@pytest.fixture
def temp_project_structure(tmp_path):
    """Create temporary project structure for migration testing."""
    # Create flat structure (current state)
    (tmp_path / "models.py").write_text("# Original models")
    (tmp_path / "server.py").write_text("# Original server")
    (tmp_path / "rag_code.py").write_text("# Original RAG code")
    (tmp_path / "config.py").write_text("# Original config")

    # Create tests directory
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_existing.py").write_text("# Existing tests")

    return tmp_path


@pytest.fixture
def mcp_server_mock():
    """Mock MCP server for testing."""
    mock_server = Mock()
    mock_server.tool = Mock()
    mock_server.start = AsyncMock()
    mock_server.stop = AsyncMock()
    return mock_server


# Test markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.contract = pytest.mark.contract
pytest.mark.slow = pytest.mark.slow
pytest.mark.migration = pytest.mark.migration
