"""Generic evaluation pipeline with hooks and postprocess steps."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from . import math_utils
from .config import PackConfig, load_data_array
from .interfaces import MeshResult, TemplateAdapter


GENERIC_STEPS: dict[str, Callable] = {}


def register_generic_step(name: str, fn: Callable) -> None:
    GENERIC_STEPS[name] = fn


def _apply_pose_correctives(
    pack: PackConfig, step: dict, vertices, joints, params: dict, pose: dict
) -> np.ndarray:
    """Generic corrective step: regressor data + pose rotations -> vertex delta."""
    regressor = load_data_array(pack.pack_dir, step["data"])
    rotations = np.asarray(pose.get("rotations", np.zeros((len(joints), 3))))
    num_vertices = vertices.shape[0]
    return vertices + math_utils.compute_pose_correctives(
        rotations, regressor, num_vertices
    )


register_generic_step("core.postprocess.pose_correctives", _apply_pose_correctives)


def run_postprocess_step(
    pack: PackConfig,
    adapter: TemplateAdapter,
    step: dict,
    vertices: np.ndarray,
    joints: np.ndarray,
    params: dict,
    pose: dict,
) -> np.ndarray:
    impl = step.get("impl", "")
    if step.get("kind") == "generic":
        fn = GENERIC_STEPS.get(impl)
        if fn is None:
            raise ValueError(f"unknown generic step: {impl}")
        return fn(pack, step, vertices, joints, params, pose)
    if step.get("kind") == "custom":
        # "py.module:function" resolved inside the pack directory.
        module_path, _, func_name = impl.partition(":")
        if not module_path.startswith("py."):
            raise ValueError(f"custom step must reference py.* module: {impl}")
        result = _call_pack_function(pack, module_path, func_name, vertices, joints, params, pose)
        return result
    raise ValueError(f"postprocess step {step.get('id', '?')} has no valid kind")


def _call_pack_function(pack, module_path, func_name, vertices, joints, params, pose):
    import importlib.util
    import os

    module_file = os.path.join(pack.pack_dir, *module_path.split(".")) + ".py"
    spec = importlib.util.spec_from_file_location(module_path, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, func_name)
    return fn(vertices=vertices, joints=joints, params=params, pose=pose, pack=pack)


def run_pipeline(
    pack: PackConfig,
    adapter: TemplateAdapter,
    params: dict,
    pose: dict,
    ctx: dict | None = None,
) -> MeshResult:
    """Evaluate a pack through the generic pipeline with adapter hooks."""
    ctx = dict(ctx or {})
    params = _or_keep(adapter.on_before_semantic(params, ctx), params)
    weights = adapter.resolve_semantic(params, ctx) or {}
    weights = _or_keep(adapter.on_after_semantic(weights, ctx), weights)

    vertices = adapter.linear_bs_sum(weights, ctx)
    vertices = _or_keep(adapter.on_after_bs_sum(vertices, ctx), vertices)

    joints = adapter.compute_joints(params, ctx)
    for step in pack.postprocess_steps:
        if step.get("hook") == "before_lbs":
            vertices = run_postprocess_step(
                pack, adapter, step, vertices, joints, params, pose
            )

    vertices, joints = adapter.on_before_lbs(vertices, joints, ctx)
    vertices = adapter.lbs_skin(vertices, joints, params, pose, ctx)
    vertices = _or_keep(adapter.on_after_lbs(vertices, ctx), vertices)

    for step in pack.postprocess_steps:
        if step.get("hook") == "after_lbs":
            vertices = run_postprocess_step(
                pack, adapter, step, vertices, joints, params, pose
            )
    vertices = _or_keep(adapter.on_after_postprocess(vertices, ctx), vertices)

    result = adapter.finalize(vertices, joints, ctx)
    result = _or_keep(adapter.on_after_final(result, ctx), result)
    return result


def _or_keep(value, fallback):
    """Return value unless it is None (safe for numpy arrays)."""
    return fallback if value is None else value
