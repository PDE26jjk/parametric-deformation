"""Template adapter interface (framework provides interface only)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .config import PackConfig


@dataclass
class MeshResult:
    """Clean mesh output of a template evaluation."""

    meshes: list[tuple[str, Any, Any, Any | None]]  # (name, verts, faces, uvs)
    joints: Any = None
    extra: dict = field(default_factory=dict)


class TemplateAdapter(ABC):
    """Interface every template pack must implement in its own py/driver.py."""

    pack: PackConfig | None = None
    pack_dir: str = ""

    def load(self, pack: PackConfig, pack_dir: str) -> None:
        """Load a validated pack; template code owns any data preparation."""
        self.pack = pack
        self.pack_dir = pack_dir

    @abstractmethod
    def evaluate(self, params: dict, pose: dict) -> MeshResult:
        """Evaluate the template into a clean mesh."""

    def schema(self) -> list[dict]:
        """Parameter schema for UI generation (defaults to pack params)."""
        return list(self.pack.params) if self.pack else []

    def export_map(self) -> dict:
        """Export mapping; interface only until researched."""
        return {"expressions": {}, "humanoid_bones": {}}

    # --- Pipeline hooks (default no-ops) ---------------------------------
    def on_before_semantic(self, params: dict, ctx: dict) -> dict:
        return params

    def on_after_semantic(self, weights: dict, ctx: dict) -> dict:
        return weights

    def on_after_bs_sum(self, vertices: Any, ctx: dict) -> Any:
        return vertices

    def on_before_lbs(self, vertices: Any, joints: Any, ctx: dict) -> tuple:
        return vertices, joints

    def on_after_lbs(self, vertices: Any, ctx: dict) -> Any:
        return vertices

    def on_after_postprocess(self, vertices: Any, ctx: dict) -> Any:
        return vertices

    def on_after_final(self, result: MeshResult, ctx: dict) -> MeshResult:
        return result
