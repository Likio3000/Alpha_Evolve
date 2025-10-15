"""
Genetic-programming primitives used by Alpha Evolve.

This package groups operator definitions, type metadata, and helper utilities
required to build and mutate alpha programs.
"""

from .types import (
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
    FINAL_PREDICTION_VECTOR_NAME,
    OP_REGISTRY,
    SAFE_MAX,
    SCALAR_FEATURE_NAMES,
    OpSpec,
    TypeId,
)
from .ops import Op
from .program import AlphaProgram
from . import operators

__all__ = [
    "CROSS_SECTIONAL_FEATURE_VECTOR_NAMES",
    "FINAL_PREDICTION_VECTOR_NAME",
    "OP_REGISTRY",
    "SAFE_MAX",
    "SCALAR_FEATURE_NAMES",
    "OpSpec",
    "TypeId",
    "Op",
    "AlphaProgram",
    "operators",
]
