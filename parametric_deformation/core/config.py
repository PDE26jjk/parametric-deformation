"""Template config pack loading and validation (format_version 1).

A template pack is a directory that follows the JSON-first layout defined in
docs/parametric-deformation-framework.md. This module is the schema checker:
it loads and validates every pack so adapters can rely on the structure.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = "1"


class PackError(Exception):
    """Raised when a template pack is missing or malformed."""


@dataclass
class PackConfig:
    """Parsed representation of a template pack directory."""

    pack_dir: str
    manifest: dict
    params: list[dict] = field(default_factory=list)
    bs_linear: list[dict] = field(default_factory=list)
    bs_semantic: list[dict] = field(default_factory=list)
    skeleton: dict = field(default_factory=dict)
    bone_deltas: dict = field(default_factory=dict)
    mesh_descriptors: list[dict] = field(default_factory=list)
    postprocess_steps: list[dict] = field(default_factory=list)
    export_map: dict = field(default_factory=dict)
    weights_file: str | None = None


def _read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        raise PackError(f"cannot read JSON {path}: {exc}") from exc


def _resolve_ref(ref: Any) -> dict:
    """Normalize a data reference into a dict with file/key/index/count."""
    if isinstance(ref, str):
        return {"file": ref, "key": None, "index": None, "count": None}
    if isinstance(ref, dict) and isinstance(ref.get("file"), str):
        return {
            "file": ref["file"],
            "key": ref.get("key"),
            "index": ref.get("index"),
            "count": ref.get("count"),
        }
    raise PackError(f"invalid data reference: {ref!r}")


def load_pack(pack_dir: str) -> PackConfig:
    """Load a template pack directory into a PackConfig."""
    pack_dir = os.path.abspath(pack_dir)
    manifest = _read_json(os.path.join(pack_dir, "manifest.json"))
    if str(manifest.get("format_version", "")) != SCHEMA_VERSION:
        raise PackError(
            f"unsupported format_version {manifest.get('format_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    cfg = PackConfig(pack_dir=pack_dir, manifest=manifest)

    params_path = os.path.join(pack_dir, "params.json")
    if os.path.exists(params_path):
        cfg.params = _read_json(params_path)

    for sub in ("linear", "semantic"):
        base = os.path.join(pack_dir, "bs", sub)
        if os.path.isdir(base):
            for fn in sorted(os.listdir(base)):
                if fn.endswith(".json"):
                    entry = _read_json(os.path.join(base, fn))
                    if sub == "linear":
                        cfg.bs_linear.append(entry)
                    else:
                        cfg.bs_semantic.append(entry)

    skeleton_path = os.path.join(pack_dir, "lbs", "skeleton.json")
    if os.path.exists(skeleton_path):
        cfg.skeleton = _read_json(skeleton_path)
    bd_path = os.path.join(pack_dir, "lbs", "bone_deltas.json")
    if os.path.exists(bd_path):
        cfg.bone_deltas = _read_json(bd_path)
    weights_path = os.path.join(pack_dir, "lbs", "weights.npz")
    if os.path.exists(weights_path):
        cfg.weights_file = weights_path

    meshes_base = os.path.join(pack_dir, "meshes")
    if os.path.isdir(meshes_base):
        for fn in sorted(os.listdir(meshes_base)):
            if fn.endswith(".mesh.json"):
                cfg.mesh_descriptors.append(_read_json(os.path.join(meshes_base, fn)))

    post_base = os.path.join(pack_dir, "postprocess")
    if os.path.isdir(post_base):
        for fn in sorted(os.listdir(post_base)):
            if fn.endswith(".json"):
                cfg.postprocess_steps.append(_read_json(os.path.join(post_base, fn)))

    export_path = os.path.join(pack_dir, "export_map.json")
    if os.path.exists(export_path):
        cfg.export_map = _read_json(export_path)
    return cfg


def resolve_data_path(pack_dir: str, ref: Any) -> str:
    """Absolute path of a data reference inside a pack."""
    r = _resolve_ref(ref)
    return os.path.join(pack_dir, r["file"])


def load_data_array(
    pack_dir: str, ref: Any, cache: dict | None = None
) -> np.ndarray:
    """Load the array referenced by ref, honoring merged file+key+index+count."""
    r = _resolve_ref(ref)
    path = os.path.join(pack_dir, r["file"])
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        npz = None
        if cache is not None and path in cache:
            npz = cache[path]
        if npz is None:
            npz = np.load(path)
            if cache is not None:
                cache[path] = npz
        key = r["key"]
        arr = npz[npz.files[0]] if key is None else npz[key]
    else:
        raise PackError(f"unsupported data file extension: {path}")
    index, count = r["index"], r["count"]
    if index is not None:
        if count is None:
            count = 1
        arr = arr[index : index + count]
    return arr


def _check_ref_bounds(pack_dir: str, ref: Any, issues: list[str], label: str) -> None:
    r = _resolve_ref(ref)
    path = os.path.join(pack_dir, r["file"])
    if not os.path.exists(path):
        issues.append(f"{label}: file missing {path}")
        return
    # Plain (whole-file) references only need an existence check; loading the
    # array is expensive and mmap is unsafe on some virtual drives.
    if r["index"] is None:
        return
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".npz":
            with np.load(path) as npz:
                key = r["key"] or npz.files[0]
                arr = npz[key]
        else:
            arr = np.load(path)
    except Exception as exc:
        issues.append(f"{label}: cannot inspect data: {exc}")
        return
    index = r["index"]
    count = r["count"] or 1
    if index < 0 or index + count > arr.shape[0]:
        issues.append(
            f"{label}: merged range [{index}:{index + count}] out of bounds "
            f"for shape {arr.shape}"
        )


def validate_pack(pack_dir: str) -> list[str]:
    """Validate a pack; returns a list of issue strings (empty means valid)."""
    issues: list[str] = []
    try:
        cfg = load_pack(pack_dir)
    except PackError as exc:
        return [str(exc)]

    manifest = cfg.manifest
    for key in ("id", "name", "version", "format_version"):
        if key not in manifest:
            issues.append(f"manifest missing {key}")
    if str(manifest.get("format_version", "")) != SCHEMA_VERSION:
        issues.append(f"manifest format_version must be {SCHEMA_VERSION}")

    seen_ids: set[str] = set()
    for p in cfg.params:
        pid = p.get("id")
        if not isinstance(pid, str) or not pid:
            issues.append("param entry missing string id")
            continue
        if pid in seen_ids:
            issues.append(f"duplicate param id {pid}")
        seen_ids.add(pid)
        for num in ("min", "max", "default", "step"):
            if num in p and not isinstance(p[num], (int, float)):
                issues.append(f"param {pid}: {num} is not numeric")
        if "kind" in p and p["kind"] not in ("linear", "semantic"):
            issues.append(f"param {pid}: kind {p['kind']} not linear/semantic")

    for entry in cfg.bs_linear + cfg.bs_semantic:
        eid = entry.get("id", "?")
        if entry.get("type") not in ("linear", "semantic"):
            issues.append(f"bs {eid}: type must be linear or semantic")
        data = entry.get("data")
        if data is None:
            issues.append(f"bs {eid}: missing data")
        elif isinstance(data, dict) and "file" in data:
            _check_ref_bounds(pack_dir, data, issues, f"bs {eid}")
        elif isinstance(data, dict):
            for sub_path in data.values():
                if isinstance(sub_path, str) and not os.path.exists(
                    os.path.join(pack_dir, sub_path)
                ):
                    issues.append(f"bs {eid}: referenced file missing {sub_path}")

    joints = cfg.skeleton.get("joints", [])
    if not joints:
        issues.append("lbs/skeleton.json missing joints")
    else:
        for i, j in enumerate(joints):
            if "id" not in j:
                issues.append(f"joint {i} missing id")
            parent = j.get("parent", 0)
            if not isinstance(parent, int) or not (0 <= parent <= i):
                issues.append(f"joint {i}: parent {parent} invalid")

    if cfg.weights_file:
        try:
            npz = np.load(cfg.weights_file)
            w = npz[npz.files[0]]
            if w.ndim != 2 or w.shape[0] != len(joints):
                issues.append(
                    f"weights shape {w.shape} does not match joints {len(joints)}"
                )
        except Exception as exc:
            issues.append(f"cannot load weights: {exc}")

    for step in cfg.postprocess_steps:
        sid = step.get("id", "?")
        if "hook" not in step:
            issues.append(f"postprocess {sid}: missing hook")
        if step.get("kind") not in ("generic", "custom"):
            issues.append(f"postprocess {sid}: kind must be generic or custom")
    return issues
