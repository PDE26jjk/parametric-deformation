"""GenericAdapter: default pipeline implementations for typical templates."""

from __future__ import annotations

import numpy as np

from . import math_utils
from .config import PackError, load_data_array
from .interfaces import MeshResult, TemplateAdapter


class GenericAdapter(TemplateAdapter):
    """Base adapter implementing the common stages; templates override as needed."""

    def __init__(self) -> None:
        self._cache: dict = {}
        self._template_arr: np.ndarray | None = None
        self._bs_data: dict[str, np.ndarray] = {}
        self._weights_arr: np.ndarray | None = None
        self._parents: list[int] = []
        self._joints0_arr: np.ndarray | None = None
        self._template_ref = None
        self._bs_refs: dict[str, object] = {}
        self._faces_ref = None
        self._uvs_ref = None
        self._joints_ref = None
        self._weights_file: str | None = None

    def load(self, pack, pack_dir: str) -> None:
        super().load(pack, pack_dir)
        if pack.mesh_descriptors:
            desc = pack.mesh_descriptors[0]
            self._template_ref = desc["vertices"]
            self._faces_ref = desc.get("faces")
            self._uvs_ref = desc.get("uvs")
        for entry in pack.bs_linear:
            self._bs_refs[entry["id"]] = entry["data"]
        skel = pack.skeleton
        self._joints_ref = skel.get("positions")
        self._parents = skel.get("parent_indices") or [
            j.get("parent", 0) for j in skel.get("joints", [])
        ]
        self._weights_file = pack.weights_file

    # --- Lazy data access -------------------------------------------------
    def _template(self) -> np.ndarray:
        if self._template_arr is None:
            self._template_arr = load_data_array(
                self.pack_dir, self._template_ref, self._cache
            )
        return self._template_arr

    def _bs(self, eid: str) -> np.ndarray:
        arr = self._bs_data.get(eid)
        if arr is None:
            arr = load_data_array(self.pack_dir, self._bs_refs[eid], self._cache)
            self._bs_data[eid] = arr
        return arr

    def _joints0(self) -> np.ndarray:
        if self._joints0_arr is None:
            self._joints0_arr = load_data_array(
                self.pack_dir, self._joints_ref, self._cache
            )
        return self._joints0_arr

    def _weights(self) -> np.ndarray:
        if self._weights_arr is None:
            if not self._weights_file:
                raise PackError(
                    "pack has no lbs/weights.npz: the generic adapter needs "
                    "skinning weights, or the pack must override lbs_skin() "
                    "in its py/driver.py"
                )
            npz = np.load(self._weights_file)
            self._weights_arr = npz[npz.files[0]]
        return self._weights_arr

    def _faces(self) -> np.ndarray | None:
        if self._faces_ref is None:
            return None
        return load_data_array(self.pack_dir, self._faces_ref, self._cache)

    def _uvs(self) -> np.ndarray | None:
        if self._uvs_ref is None:
            return None
        return load_data_array(self.pack_dir, self._uvs_ref, self._cache)

    def evaluate(self, params: dict, pose: dict) -> MeshResult:
        from .pipeline import run_pipeline

        return run_pipeline(self.pack, self, params, pose)

    def resolve_semantic(self, params: dict, ctx: dict) -> dict:
        """Default: pass through direct linear weights from params."""
        weights = {}
        for entry in self.pack.bs_linear:
            if entry["id"] in params:
                weights[entry["id"]] = params[entry["id"]]
        return weights

    def linear_bs_sum(self, weights: dict, ctx: dict) -> np.ndarray:
        vertices = np.asarray(self._template(), dtype=np.float64).copy()
        for eid in self._bs_refs:
            w = weights.get(eid)
            if w is None:
                continue
            w = np.asarray(w, dtype=np.float64)
            if w.ndim == 0:
                vertices = vertices + float(w) * self._bs(eid)
            else:
                vertices = vertices + np.einsum("i,ijk->jk", w, self._bs(eid))
        return vertices

    def _bone_index(self, name: str) -> int:
        for i, j in enumerate(self.pack.skeleton.get("joints", [])):
            if j.get("id") == name:
                return i
        raise KeyError(f"unknown bone {name}")

    def compute_joints(self, params: dict, ctx: dict) -> np.ndarray:
        joints = np.asarray(self._joints0(), dtype=np.float64).copy()
        for param_id, deltas in self.pack.bone_deltas.items():
            value = params.get(param_id, 0.0)
            if not value:
                continue
            for delta in deltas:
                index = self._bone_index(delta["bone"])
                if "t" in delta:
                    joints[index] += value * np.asarray(delta["t"], dtype=np.float64)
        return joints

    def lbs_skin(
        self, vertices, joints, params: dict, pose: dict, ctx: dict
    ) -> np.ndarray:
        rotations = np.asarray(
            pose.get("rotations", np.zeros((len(joints), 3))), dtype=np.float64
        )
        translation = np.asarray(pose.get("translation", np.zeros(3)), dtype=np.float64)
        return math_utils.linear_blend_skinning(
            vertices,
            joints,
            rotations,
            translation,
            self._weights(),
            self._parents,
        )

    def finalize(self, vertices, joints, ctx: dict) -> MeshResult:
        name = "mesh"
        if self.pack.mesh_descriptors:
            name = self.pack.mesh_descriptors[0].get("name", "mesh")
        return MeshResult(
            meshes=[
                (
                    name,
                    np.asarray(vertices, dtype=np.float32),
                    self._faces(),
                    self._uvs(),
                )
            ],
            joints=joints,
        )
