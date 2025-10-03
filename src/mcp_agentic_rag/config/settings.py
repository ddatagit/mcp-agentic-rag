"""Configuration management for MCP Agentic RAG system."""

import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load environment variables from .env file
load_dotenv()


class Settings(BaseModel):
    """System configuration with environment variable loading."""

    # Vector database configuration
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant database connection URL")
    vector_confidence_threshold: float = Field(default=0.7, description="Minimum confidence for vector results")

    # Response timing configuration
    max_response_time: float = Field(default=5.0, description="Maximum total response time in seconds")

    # Google Custom Search API configuration
    google_api_key: str | None = Field(default=None, description="Google Custom Search API key")
    google_cx: str | None = Field(default=None, description="Google Custom Search engine ID")

    # ML terminology for domain detection
    ml_keywords: list[str] = Field(
        default=[
            "overfitting", "gradient descent", "neural networks", "machine learning",
            "deep learning", "classification", "regression", "clustering", "supervised",
            "unsupervised", "reinforcement learning", "feature engineering", "cross-validation",
            "ai", "artificial intelligence", "algorithm", "model", "training", "backpropagation",
            "cnn", "rnn", "lstm", "transformer", "attention", "regularization", "dataset",
            "tensor", "pytorch", "tensorflow", "scikit"
        ],
        description="Keywords that trigger ML domain detection"
    )

    # Embedding model configuration
    embedding_model_name: str = Field(
        default="nomic-ai/nomic-embed-text-v1.5",
        description="HuggingFace embedding model name"
    )
    embedding_batch_size: int = Field(default=32, description="Batch size for embedding generation")

    # Vector database configuration
    vector_collection_name: str = Field(default="ml_faq_collection", description="Qdrant collection name")
    vector_dimension: int = Field(default=768, description="Vector dimension")
    vector_batch_size: int = Field(default=512, description="Batch size for vector operations")

    @field_validator('vector_confidence_threshold')
    @classmethod
    def validate_confidence_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        return v

    @field_validator('max_response_time')
    @classmethod
    def validate_response_time(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('Max response time must be positive')
        return v

    @field_validator('embedding_batch_size', 'vector_batch_size')
    @classmethod
    def validate_batch_sizes(cls, v: int) -> int:
        if v <= 0:
            raise ValueError('Batch size must be positive')
        return v

    @field_validator('vector_dimension')
    @classmethod
    def validate_vector_dimension(cls, v: int) -> int:
        if v <= 0:
            raise ValueError('Vector dimension must be positive')
        return v

    @classmethod
    def from_env(cls) -> "Settings":
        """Create configuration from environment variables."""
        return cls(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            vector_confidence_threshold=float(os.getenv("VECTOR_CONFIDENCE_THRESHOLD", "0.7")),
            max_response_time=float(os.getenv("MAX_RESPONSE_TIME_SECONDS", "5.0")),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_cx=os.getenv("GOOGLE_CX"),
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5"),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            vector_collection_name=os.getenv("VECTOR_COLLECTION_NAME", "ml_faq_collection"),
            vector_dimension=int(os.getenv("VECTOR_DIMENSION", "768")),
            vector_batch_size=int(os.getenv("VECTOR_BATCH_SIZE", "512")),
        )

    def get_google_search_config(self) -> dict:
        """Get Google Custom Search configuration."""
        return {
            "api_key": self.google_api_key,
            "cx": self.google_cx,
            "base_url": "https://www.googleapis.com/customsearch/v1"
        }

    def get_vector_config(self) -> dict:
        """Get vector database configuration."""
        return {
            "url": self.qdrant_url,
            "collection_name": self.vector_collection_name,
            "dimension": self.vector_dimension,
            "batch_size": self.vector_batch_size,
            "confidence_threshold": self.vector_confidence_threshold
        }

    def get_embedding_config(self) -> dict:
        """Get embedding model configuration."""
        return {
            "model_name": self.embedding_model_name,
            "batch_size": self.embedding_batch_size,
            "cache_folder": "./hf_cache"
        }


class ConfigManager:
    """Configuration manager for centralized config access."""

    _instance: Settings | None = None

    @classmethod
    def get_config(cls) -> Settings:
        """Get the global configuration instance."""
        if cls._instance is None:
            cls._instance = Settings.from_env()
        return cls._instance

    @classmethod
    def reset_config(cls) -> None:
        """Reset configuration (useful for testing)."""
        cls._instance = None

    @classmethod
    def reload_config(cls) -> Settings:
        """Reload configuration from environment."""
        cls._instance = None
        return cls.get_config()


# Global configuration instance
_config: Settings | None = None


def load_config() -> Settings:
    """Load and return the global configuration."""
    global _config
    if _config is None:
        _config = Settings.from_env()
    return _config


def get_setting(key: str, default: Any = None) -> Any:
    """Get a specific setting value."""
    config = load_config()
    return getattr(config, key, os.getenv(key, default))


def reload_settings() -> Settings:
    """Reload settings from environment variables."""
    global _config
    _config = None
    return load_config()


# For backward compatibility
Configuration = Settings
