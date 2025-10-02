"""Minimal subset of the Pydantic API required for tests."""
from ._stub import BaseModel, Field, ConfigDict, ValidationError

__all__ = ["BaseModel", "Field", "ConfigDict", "ValidationError"]
