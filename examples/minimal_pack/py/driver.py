"""Example driver for the minimal pack.

It exists to exercise the pack API end to end - driver loading, the generic
pipeline and the detach shape-key hook - without any model data of its own.
Everything except ``detach_shape_keys`` comes from ``GenericAdapter``.
"""

from __future__ import annotations

from parametric_deformation.core.generic import GenericAdapter


class MinimalAdapter(GenericAdapter):
    """One linear slider plus one bone-translation slider."""

    def detach_shape_keys(self) -> list[dict]:
        """Expose the blend-shape slider as a shape key when detaching."""
        for param in (self.pack.params if self.pack else []):
            if param.get("id") == "identity":
                return [{
                    "name": param.get("label", "identity"),
                    "params": {"identity": 1.0},
                }]
        return []
