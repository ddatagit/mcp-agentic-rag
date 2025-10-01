"""Base model classes for MCP Agentic RAG."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict


class BaseModel(PydanticBaseModel):
    """
    Base model class with common configuration.

    Provides consistent configuration across all models including:
    - Validation on assignment
    - JSON schema generation
    - Modern Python type hints support
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_schema_extra=None,
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    def model_dump_json(self, **kwargs: Any) -> str:
        """Override to ensure consistent JSON serialization."""
        return super().model_dump_json(exclude_none=True, **kwargs)

    @classmethod
    def generate_id(cls) -> str:
        """Generate a unique ID for model instances."""
        return str(uuid.uuid4())

    @classmethod
    def current_timestamp(cls) -> datetime:
        """Get current timestamp in UTC."""
        return datetime.utcnow()


class TimestampedModel(BaseModel):
    """Base model with automatic timestamp tracking."""

    created_at: datetime
    updated_at: datetime | None = None

    def __init__(self, **data: Any):
        if 'created_at' not in data:
            data['created_at'] = self.current_timestamp()
        super().__init__(**data)

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp to current time."""
        self.updated_at = self.current_timestamp()
