"""Core engine of the Parametric Deformation plugin (Blender-independent)."""

from .config import (
    SCHEMA_VERSION,
    PackConfig,
    PackError,
    load_data_array,
    load_pack,
    validate_pack,
)
from .interfaces import MeshResult, TemplateAdapter
from .pipeline import run_pipeline

__all__ = [
    "SCHEMA_VERSION",
    "PackConfig",
    "PackError",
    "MeshResult",
    "TemplateAdapter",
    "load_data_array",
    "load_pack",
    "run_pipeline",
    "validate_pack",
]
