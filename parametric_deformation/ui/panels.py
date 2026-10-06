"""Blender UI: template pack loading, collapsible slider groups, live update."""

from __future__ import annotations

import json
import os
import random
import re
import time

import bpy
import numpy as np
from bpy.props import BoolProperty, FloatProperty, StringProperty
from bpy.types import Operator, Panel
from mathutils import Matrix, Vector

from parametric_deformation.core.config import (
    load_data_array,
    load_pack,
    validate_pack,
)
from parametric_deformation.core import actions as pd_actions
from parametric_deformation.tools import load_driver

_state = {
    "pack": None,
    "adapter": None,
    "params": [],
    "prop_names": {},      # param id -> Scene property name
    "groups": [],          # group names ordered by rank
    "group_rank": {},      # group -> min rank
    "group_props": {},     # group -> Scene bool prop (collapse toggle)
    "materials": {},       # material id -> descriptor from materials.json
    "mesh_data": {},       # mesh name -> {slots, bones, weights}
    "rig_joints": [],
    "rig_world": None,
    "rig_remap": None,     # joint index -> kept joint index
    "rig_sign": None,      # (full-rig flag, joint count) of the current rig
    "rig_regroup": False,
    "suspending": False,
    "dirty": False,
    "timer": None,
    "param_snapshot": {},
    "mirror_snapshot": {},  # prop name -> (host value, Scene mirror value)
    "poll_timer": None,
    "rig_stale": False,
    "written": {},         # object name -> last vertex array written
}


# VRoid imports the Unity Y-up skeletons; the template drivers emit Blender
# Z-up vertices, so every matrix has to travel through the same axis change.
_AXIS_TO_BLENDER = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)

_RIG_NAME = "PD_Rig"
_RIG_MODIFIER = "PD_Rig"
#: Bumped whenever the viewport update path changes, so the panel can show
#: which build is actually loaded (a stale add-on reload looks like "the fix
#: did nothing").
BUILD = "2026-10-06d"

# Bones VRoid Studio keeps in its own VRM export (59 skin joints): Root, the
# humanoid J_Bip_* set without *_end markers, the secondary J_Sec_* bones,
# and the two eye bones. Everything else in the runtime skeleton is a
# J_Adj_* shape bone (131 of them) whose effect is already baked into the
# mesh; their weights are collapsed onto the nearest kept ancestor.
_EYE_BONES = ("J_Adj_L_FaceEye", "J_Adj_R_FaceEye")

# Solid-colour fallback for packs without materials.json (textures are a
# later phase; this only keeps the viewport readable).
_MATERIAL_FALLBACK = (
    ("Eyelash", (0.03, 0.03, 0.03, 1.0)),
    ("Eyeline", (0.10, 0.07, 0.06, 1.0)),
    ("Eyebrow", (0.35, 0.21, 0.12, 1.0)),
    ("Brow", (0.35, 0.21, 0.12, 1.0)),
    ("Iris", (0.31, 0.19, 0.10, 1.0)),
    ("EyeWhite", (0.88, 0.88, 0.88, 1.0)),
    ("EyeHighlight", (1.0, 1.0, 1.0, 1.0)),
    ("Eye", (0.55, 0.55, 0.60, 1.0)),
    ("Mouth", (0.80, 0.45, 0.45, 1.0)),
    ("Face", (0.94, 0.80, 0.75, 1.0)),
    ("Body", (0.92, 0.80, 0.72, 1.0)),
    ("Skin", (0.92, 0.80, 0.72, 1.0)),
)


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def _preset_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".parametric_deformation", "presets")


_SUBGROUP_KEYWORDS = (
    ("Chin", ("chin", "jaw")),
    ("Ears", ("ear",)),
    ("Eyes", ("eye", "iris", "gaze", "slant", "scorn")),
    ("Brows", ("brow",)),
    ("Nose", ("nose",)),
    ("Mouth", ("mouth", "lip")),
    ("Teeth", ("tooth", "teeth", "fang")),
    ("Face", ("face", "cheek", "head", "faceline", "mandible", "outline")),
    ("Neck", ("neck", "adam", "collarbone", "clavicle", "throat")),
    ("Chest", ("bust", "chest")),
    ("Waist/Hips", ("waist", "hip", "torso")),
    ("Shoulders", ("shoulder",)),
    ("Arms/Hands", ("arm", "hand", "finger", "thumb")),
    ("Legs/Feet", ("leg", "foot", "thigh", "calf", "knee")),
    ("Body", ("body", "figure", "height", "slender", "whole")),
)


def _subgroup_of(param: dict) -> str:
    text = f"{param.get('id', '')} {param.get('label', '')}".lower()
    for name, keywords in _SUBGROUP_KEYWORDS:
        if any(k in text for k in keywords):
            return name
    return "Other"


#: Slider changes are polled instead of driven by an RNA update callback.
#: A Python-defined property with ``update=`` makes Blender tag its owner on
#: every write and EEVEE then rebuilds the MToon material shaders: measured
#: 204 ms per slider step with the VRM materials versus 6 ms without the
#: callback (docs/vroid.md 7.4).  The same property with no callback is free, so
#: the panel detects changes in a light 20 Hz timer.  The *owner* matters just
#: as much - see _host below.
_PARAM_POLL_SECONDS = 0.05

#: Delay before a widget write is evaluated.  The callback itself must stay
#: cheap (the old code ran the whole evaluation - including a bone mode switch -
#: inside the RNA update callback, which blew up with "mode_set.poll() Context
#: missing active object"), so it only arms this single-shot timer.
_SETTLE_SECONDS = 0.01


#: Widgets must not be bound to *Scene* properties.
#:
#: A UI button always calls ``RNA_property_update()`` (Blender 4.5:
#: ``editors/interface/interface_handlers.cc`` calls it unconditionally, while a
#: Python assignment first tests ``RNA_property_update_check()`` in
#: ``python/intern/bpy_rna.cc``), so touching a Scene-owned widget tags the whole
#: scene; EEVEE then rebuilds the material shaders of the visible meshes, and
#: with the VRoid MToon node trees that costs 1.1-2.4 s per slider step
#: (docs/vroid.md 7.4.9).  The identical property on the master collection costs
#: 11-13 ms, always exists, is saved with the .blend and is invisible in the UI,
#: so the panel binds there and the Scene property is kept as a mirror for
#: scripts, tools and files saved by older revisions.
_FLAG_PROPS = ("pd_pack_path", "pd_preset_name", "pd_live_update", "pd_use_rig",
               "pd_filter")


def _host(context=None):
    """RNA owner the panel's widgets write to (cheap depsgraph tag)."""
    context = context if context is not None else bpy.context
    scene = getattr(context, "scene", None) or bpy.context.scene
    return scene.collection


def _flag(name: str, context=None):
    """Read a UI flag through the host, after reconciling it with the mirror."""
    _sync_host()
    host = _host(context)
    if hasattr(host, name):
        return getattr(host, name)
    scene = getattr(context, "scene", None) or bpy.context.scene
    return getattr(scene, name)


def _host_updated(_self, _context) -> None:
    """RNA ``update`` callback of the widget-facing properties.

    Kept deliberately cheap: mark the state dirty and arm a single-shot timer.
    A burst of widget writes therefore coalesces into one evaluation ~10 ms
    later, instead of waiting for the 20 Hz poll (which is kept as the fallback
    for scripts that write the Scene mirror directly).
    """
    _schedule_flush()


def _schedule_flush() -> None:
    if _state["suspending"]:
        return
    _state["dirty"] = True
    # bpy.app.timers.register() returns None, so the handle *is* the function:
    # always test with is_registered() instead of storing the return value.
    if not bpy.app.timers.is_registered(_flush_timer):
        bpy.app.timers.register(_flush_timer, first_interval=_SETTLE_SECONDS)


def _mirror_names() -> list[str]:
    names = list(_state["prop_names"].values())
    names += list(_state["group_props"].values())
    names += list(_FLAG_PROPS)
    return names


def _push_to_host() -> None:
    """Copy the Scene values onto the widget host and re-baseline the mirror."""
    host = _host()
    scene = bpy.context.scene
    snapshot = {}
    was_suspending = _state["suspending"]
    _state["suspending"] = True  # our own writes must not arm the settle timer
    try:
        for name in _mirror_names():
            value = getattr(scene, name, None)
            if value is None:
                continue
            try:
                setattr(host, name, value)
            except (AttributeError, TypeError, ValueError):
                continue
            snapshot[name] = (value, value)
    finally:
        _state["suspending"] = was_suspending
    _state["mirror_snapshot"] = snapshot


def _sync_host() -> None:
    """Reconcile the widget host and the Scene mirror in both directions.

    Whichever side changed since the last poll wins; if both changed, the host
    wins because that is the side the user is dragging.
    """
    host = _host()
    scene = bpy.context.scene
    previous = _state.get("mirror_snapshot") or {}
    snapshot = {}
    was_suspending = _state["suspending"]
    _state["suspending"] = True  # mirror writes must not arm the settle timer
    try:
        for name in _mirror_names():
            host_value = getattr(host, name, None)
            scene_value = getattr(scene, name, None)
            if host_value is None or scene_value is None:
                continue
            pair = previous.get(name)
            if pair is None:
                # First sight of this property: the Scene copy is the one saved
                # by older revisions, so it wins over a host that is still at
                # its class default.
                if host_value != scene_value:
                    setattr(host, name, scene_value)
                    host_value = scene_value
                snapshot[name] = (host_value, scene_value)
                continue
            host_changed = host_value != pair[0]
            scene_changed = scene_value != pair[1]
            if host_changed and not scene_changed:
                setattr(scene, name, host_value)
                scene_value = host_value
            elif scene_changed and not host_changed:
                setattr(host, name, scene_value)
                host_value = scene_value
            elif host_changed and scene_changed:
                setattr(scene, name, host_value)
                scene_value = host_value
            snapshot[name] = (host_value, scene_value)
    finally:
        _state["suspending"] = was_suspending
    _state["mirror_snapshot"] = snapshot


def _collect_snapshot() -> dict:
    return _collect_params() if _state["prop_names"] else {}


def _sync_param_snapshot() -> None:
    """Remember the current slider values so the poll does not re-fire."""
    if _state["prop_names"]:
        _state["param_snapshot"] = _collect_snapshot()


def _param_poll():
    """Repeating timer: notice slider changes without tagging the scene."""
    try:
        return _param_poll_body()
    except Exception as exc:   # never let the timer die: the panel would stop
        #                          updating without any visible reason
        _state["last_error"] = f"{type(exc).__name__}: {exc}"
        print("parametric_deformation:", _state["last_error"])
        return _PARAM_POLL_SECONDS


def _param_poll_body():
    if _state["adapter"] is None:
        _state["poll_timer"] = None
        return None
    interval = _PARAM_POLL_SECONDS
    if not _state["suspending"]:
        # Widget writes land on the host; pull them onto the Scene before the
        # change test, so the mesh follows the drag.
        _sync_host()
        values = _collect_snapshot()
        if values != _state.get("param_snapshot"):
            _state["param_snapshot"] = values
            if _flag("pd_live_update"):
                # Evaluate right away: the poll itself is the coalescer (20 Hz),
                # and an extra debounce delay only adds latency to the drag.
                _state["dirty"] = False
                start = time.perf_counter()
                try:
                    _evaluate_mesh_impl(bpy.context, live=True)
                except Exception as exc:
                    _state["last_error"] = f"{type(exc).__name__}: {exc}"
                    print("parametric_deformation:", _state["last_error"])
                spent = time.perf_counter() - start
                _state["eval_ms"] = spent * 1000.0
                # Back off only for a pathological step.  The old 15 ms
                # threshold existed for the 1.2 s material-preview stall that
                # the collection host removed (docs/vroid.md 7.4.10); with it,
                # a Solid step (rest rebuild ~20 ms + eval ~8 ms) dropped the
                # poll to 6.7 Hz and the drag felt laggy.
                if spent > 0.1:
                    interval = min(0.2, _PARAM_POLL_SECONDS * 4.0)
        elif _state.get("rig_stale") and not _heavy_shading():
            # Shading dropped back to Solid/Wireframe: apply the rest refresh
            # that was skipped while the material preview was open.
            _schedule_flush()
    return interval


def _start_param_poll() -> None:
    """Make sure the slider poll is running (it may have died earlier)."""
    # bpy.app.timers.register() returns None (the function object *is* the
    # handle), so membership has to be tested against the function itself.
    if bpy.app.timers.is_registered(_param_poll):
        return
    _state["poll_timer"] = bpy.app.timers.register(
        _param_poll, first_interval=_PARAM_POLL_SECONDS)


def _flush_timer():
    """Single-shot timer: coalesce bursts of slider changes into one evaluate."""
    _state["timer"] = None
    if _state.get("dirty") and _state["adapter"] is not None:
        _state["dirty"] = False
        try:
            # The widget wrote the host, so pull it onto the Scene before
            # evaluating and re-baseline the poll snapshot (otherwise the poll
            # would evaluate the same values a second time).
            _sync_host()
            _state["param_snapshot"] = _collect_snapshot()
            _evaluate_mesh_impl(bpy.context, live=True)
        except Exception as exc:
            # Never swallow: a silent failure here looks like "dragging a
            # slider only updates the rig".
            _state["last_error"] = f"{type(exc).__name__}: {exc}"
            print("parametric_deformation:", _state["last_error"])
    return None




def _register_params(params: list[dict]) -> None:
    groups: dict[str, int] = {}
    for p in params:
        group = p.get("group", "Params")
        rank = p.get("rank", 1)
        groups[group] = min(groups.get(group, 10**9), int(rank))
    ordered = sorted(groups, key=lambda g: (groups[g], g))
    _state["groups"] = ordered
    _state["group_rank"] = groups
    for group in ordered:
        prop_name = f"pd_show_{_sanitize(group)}"
        previous = getattr(bpy.context.scene, prop_name, None)
        default = (groups[group] == 0) if previous is None else previous
        for cls in (bpy.types.Scene, bpy.types.Collection):
            if hasattr(cls, prop_name):
                delattr(cls, prop_name)
            kwargs = {"name": group, "description": group, "default": default}
            if cls is bpy.types.Collection:
                # Only the widget-facing copy carries the update callback: on
                # the Scene it would tag the scene (and EEVEE) on every write.
                kwargs["update"] = _host_updated
            setattr(cls, prop_name, BoolProperty(**kwargs))
        if previous is not None:
            setattr(bpy.context.scene, prop_name, previous)
        setattr(_host(), prop_name, default)
        _state["group_props"][group] = prop_name
    for p in params:
        pid = p["id"]
        prop_name = f"pd_{_sanitize(pid)}"
        # Always (re)create the property.  A slider property left behind by an
        # older revision keeps its old ``update=`` callback, and that stale
        # callback keeps driving the old code path - measured in a live session
        # as ~1.2 s of main-thread stall per click, invisible to any hook in the
        # current module (docs/vroid.md 7.4.7).  Values are preserved.
        previous = getattr(bpy.context.scene, prop_name, None)
        for cls in (bpy.types.Scene, bpy.types.Collection):
            if hasattr(cls, prop_name):
                delattr(cls, prop_name)
            kwargs = {
                "name": p.get("label", pid),
                "description": p.get("label", pid),
                "min": p.get("min", -1.0),
                "max": p.get("max", 1.0),
                "default": p.get("default", 0.0),
                # step=p.get("step", 0.01), too slow
            }
            if cls is bpy.types.Collection:
                kwargs["update"] = _host_updated
            setattr(cls, prop_name, FloatProperty(**kwargs))
        _state["prop_names"][pid] = prop_name
        if previous is not None:
            setattr(bpy.context.scene, prop_name, previous)
        setattr(
            _host(), prop_name,
            previous if previous is not None else p.get("default", 0.0),
        )
    # Keep equal sub-group labels contiguous: the panel draws one collapsible
    # block per label run, so an unsorted list splits "Body" into several
    # blocks separated by other sliders. Sorting is stable, so the authored
    # order inside a sub-group is preserved.
    _state["params"] = sorted(
        params, key=lambda p: (p.get("group", "Params"), _subgroup_of(p))
    )
    _push_to_host()
    _sync_param_snapshot()


def _collect_params() -> dict:
    scene = bpy.context.scene
    values = {}
    for pid, pname in _state["prop_names"].items():
        try:
            values[pid] = getattr(scene, pname)
        except AttributeError:
            # A property that was registered by an older revision (or removed
            # with the add-on) must not take the whole poll down.
            continue
    return values


def _set_params(values: dict, suspend: bool = True) -> None:
    if suspend:
        _state["suspending"] = True
    scene = bpy.context.scene
    for pid, pname in _state["prop_names"].items():
        if pid in values:
            setattr(scene, pname, float(values[pid]))
    if suspend:
        _state["suspending"] = False
    # Programmatic writes (pack load, presets, randomize, group ops) go to the
    # Scene; push them to the widget host so the panel shows them immediately.
    _push_to_host()
    _sync_param_snapshot()


# --------------------------------------------------------------------------
# Material slots, UV and rig helpers
# --------------------------------------------------------------------------


def _fallback_color(name: str) -> tuple[float, float, float, float]:
    text = (name or "").lower()
    for key, rgba in _MATERIAL_FALLBACK:
        if key.lower() in text:
            return rgba
    return (0.80, 0.80, 0.80, 1.0)


def _get_material(name: str):
    """Reuse or create a solid-colour material for one submesh slot."""
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    spec = _state["materials"].get(name, {})
    color = spec.get("color")
    rgba = tuple(color) + (1.0,) if color else _fallback_color(name)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
    if node is not None:
        node.inputs["Base Color"].default_value = rgba
        if "Metallic" in node.inputs:
            node.inputs["Metallic"].default_value = 0.0
        if "Roughness" in node.inputs:
            node.inputs["Roughness"].default_value = 0.85
    mat.diffuse_color = rgba
    alpha_mode = str(spec.get("alpha_mode", "")).upper()
    if alpha_mode in ("MASK", "CLIP"):
        try:
            mat.blend_method = "CLIP"
        except (TypeError, AttributeError):
            pass
    elif alpha_mode in ("BLEND", "TRANSPARENT"):
        try:
            mat.blend_method = "BLEND"
        except (TypeError, AttributeError):
            pass
    if spec.get("double_sided") is not None:
        mat.use_backface_culling = not bool(spec["double_sided"])
    return mat


def _configure_mesh(obj, mesh_name: str, uvs) -> None:
    """Write UVs and per-submesh material slots once geometry exists."""
    mesh = obj.data
    # VRoid's own meshes are smooth-shaded; flat facets look wrong on them.
    if len(mesh.polygons):
        mesh.polygons.foreach_set(
            "use_smooth", np.ones(len(mesh.polygons), dtype=bool))
    uv = None if uvs is None else np.asarray(uvs, dtype=np.float32)
    if uv is not None and len(mesh.loops):
        loops = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loops)
        try:
            per_loop = _per_loop_uv(uv, mesh, loops)
        except ValueError as exc:
            print(f"parametric_deformation: {mesh_name}: {exc}")
        else:
            layer = (mesh.uv_layers[0] if mesh.uv_layers
                     else mesh.uv_layers.new(name="UVMap"))
            layer.data.foreach_set("uv", per_loop.ravel())
            mesh.uv_layers.active_index = 0
    slots = (_state["mesh_data"].get(mesh_name) or {}).get("slots") or []
    if slots:
        existing: dict[str, object] = {}
        for material in mesh.materials:
            if material is None:
                continue
            existing.setdefault(material.name, material)
            existing.setdefault(re.sub(r"\.\d{3}$", "", material.name), material)
        mesh.materials.clear()
        for slot in slots:
            name = slot.get("material", mesh_name)
            # Keep materials the user (or a pack action) already put in place.
            mesh.materials.append(existing.get(name) or _get_material(name))
        index = np.zeros(len(mesh.polygons), dtype=np.int32)
        for i, slot in enumerate(slots):
            start = int(slot.get("tri_start", 0))
            count = int(slot.get("tri_count", 0))
            index[start:start + count] = i
        mesh.polygons.foreach_set("material_index", index)
    elif not len(mesh.materials):
        mesh.materials.append(_get_material(mesh_name))
    mesh.update()


def _per_loop_uv(uv: np.ndarray, mesh, loops: np.ndarray) -> np.ndarray:
    """Normalize per-vertex (V,2) or per-corner (F,K,2) UV data to loops."""
    if uv.ndim == 2 and uv.shape[0] == len(mesh.vertices):
        return uv[loops]
    if uv.ndim == 3:
        flat = uv.reshape(-1, 2)
        if len(flat) == len(loops):
            return flat
    if uv.ndim == 2 and len(uv) == len(loops):
        return uv
    raise ValueError(
        f"unsupported UV layout {uv.shape} for {len(mesh.vertices)} verts / "
        f"{len(loops)} loops"
    )


def _ensure_vertex_groups(obj, mesh_name: str) -> None:
    """Create one vertex group per kept bone from the pack weights.

    Weights on dropped shape bones (J_Adj_*, *_end) are merged into their
    nearest kept ancestor, which is what VRoid's own VRM export does: the
    shape is baked into the mesh and only the humanoid bones stay bound.
    """
    entry = _state["mesh_data"].get(mesh_name) or {}
    bones = entry.get("bones")
    indices = entry.get("indices")
    weights = entry.get("weights")
    joints = _state["rig_joints"]
    if bones is None or indices is None or weights is None or not joints:
        return
    weights = np.asarray(weights, dtype=np.float32)
    bones = np.asarray(bones, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    # bones = the mesh's skeleton slots (global joint indices);
    # indices = per-vertex slot index into that list.
    vertex_joints = _state["rig_remap"][bones[indices]] if _state.get(
        "rig_remap") is not None else bones[indices]
    merged: dict[tuple[int, int], float] = {}
    for slot in range(min(4, weights.shape[1])):
        for vertex, (joint, weight) in enumerate(
                zip(vertex_joints[:, slot].tolist(), weights[:, slot].tolist())):
            if weight > 1e-6:
                key = (vertex, int(joint))
                merged[key] = merged.get(key, 0.0) + weight
    per_joint: dict[int, list] = {}
    for (vertex, joint), weight in merged.items():
        per_joint.setdefault(joint, []).append((vertex, weight))
    for joint, items in per_joint.items():
        name = joints[joint]
        group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
        for vertex, weight in items:
            group.add([vertex], weight, "REPLACE")


def _blender_matrix(matrix) -> Matrix:
    """Unity (Y-up) 4x4 -> Blender (Z-up) 4x4."""
    axis = np.eye(4, dtype=np.float64)
    axis[:3, :3] = _AXIS_TO_BLENDER
    return Matrix((axis @ np.asarray(matrix, dtype=np.float64) @ axis.T).tolist())


def _rig_changed(world, joints: list) -> bool:
    cached = _state.get("rig_world")
    if _state.get("rig_joints") != joints:
        return True
    if cached is None or np.shape(cached) != np.shape(world):
        return True
    return not np.allclose(cached, np.asarray(world, dtype=np.float64), atol=1e-6)


def _keep_bone(name: str) -> bool:
    """True if the bone belongs to VRoid's exported (humanoid) skeleton."""
    if name == "Root" or name.startswith("J_Sec_") or name in _EYE_BONES:
        return True
    return name.startswith("J_Bip_") and not name.endswith("_end")


def _rig_subset(joints: list, parents: list):
    """(kept joint indices, joint index -> nearest kept ancestor index)."""
    kept = [i for i, name in enumerate(joints) if _keep_bone(name)]
    kept_set = set(kept)
    remap = np.arange(len(joints), dtype=np.int64)
    for i in range(len(joints)):
        if i in kept_set:
            continue
        parent = parents[i] if i < len(parents) else -1
        while parent >= 0 and parent not in kept_set:
            parent = parents[parent]
        remap[i] = parent if parent >= 0 else i
    return kept, remap


def _view_layer_containing(obj, fallback_scene):
    """Return (scene, view_layer) that really contains ``obj``."""
    scenes = list(obj.users_scene) or [fallback_scene]
    for scene in scenes:
        for layer in scene.view_layers:
            if obj.name in layer.objects:
                return scene, layer
    fallback_scene.collection.objects.link(obj)
    return fallback_scene, fallback_scene.view_layers[0]


def _object_mode_set(context, obj, mode: str) -> None:
    """Switch ``obj`` into ``mode`` without trusting the ambient UI context.

    ``bpy.ops.object.mode_set`` polls the context it is given.  Pressing a
    panel button in a window whose view layer does not resolve our armature (or
    with no active object at all) makes that poll fail with "Context missing
    active object", so the override below names the object and a view layer
    that actually contains it.
    """
    if obj.mode == mode:
        return
    scene, layer = _view_layer_containing(obj, context.scene)
    layer.objects.active = obj
    with context.temp_override(
        scene=scene,
        view_layer=layer,
        object=obj,
        active_object=obj,
        selected_objects=[obj],
        selected_editable_objects=[obj],
    ):
        bpy.ops.object.mode_set(mode=mode)


def _target_scene(context):
    """Scene the avatar lives in (the panel context may point elsewhere)."""
    for obj in bpy.data.objects:
        if obj.name.startswith("PD_Avatar"):
            scenes = list(obj.users_scene)
            if scenes:
                return scenes[0]
    return context.scene


def _shading_types() -> set:
    """Shading mode of every open 3D viewport ("SOLID", "MATERIAL", ...)."""
    modes = set()
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type != "VIEW_3D":
                    continue
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        modes.add(space.shading.type)
    except Exception:
        pass
    return modes


def _heavy_shading() -> bool:
    """True while a material-preview / rendered viewport is open.

    Refreshing the armature rest needs an Edit-mode round trip, and EEVEE then
    re-evaluates every visible skinned mesh: measured 1.24 s per call with the
    VRM MToon materials against 16 ms in Solid/Wireframe (docs/vroid.md 7.4).
    """
    return bool(_shading_types() & {"MATERIAL", "RENDERED"})


def _ensure_armature(context, world, joints: list, parents: list, kept: list):
    """Build or refresh the armature that mirrors the runtime skeleton."""
    arm_obj = bpy.data.objects.get(_RIG_NAME)
    created = arm_obj is None or arm_obj.type != "ARMATURE"
    scene = _target_scene(context)
    if created:
        armature = bpy.data.armatures.new(_RIG_NAME)
        arm_obj = bpy.data.objects.new(_RIG_NAME, armature)
        scene.collection.objects.link(arm_obj)
    # Keep any manual pose edits across a rest-pose refresh.
    basis = (
        {}
        if created
        else {pb.name: pb.matrix_basis.copy() for pb in arm_obj.pose.bones}
    )
    previous = context.view_layer.objects.active
    _object_mode_set(context, arm_obj, "EDIT")
    edit_bones = arm_obj.data.edit_bones
    names = [joints[i] for i in kept]
    wanted = set(names)
    for edit_bone in list(edit_bones):
        if edit_bone.name not in wanted:
            edit_bones.remove(edit_bone)
    for name in names:
        if edit_bones.get(name) is None:
            edit_bones.new(name)
    slot_of = {joint: slot for slot, joint in enumerate(kept)}
    # Nearest kept ancestor of every kept bone (VRoid's export is flattened:
    # the J_Adj_* shape bones are simply removed from the chain).
    flat_parent: dict[int, int] = {}
    for slot, joint in enumerate(kept):
        parent = parents[joint] if joint < len(parents) else -1
        while parent >= 0 and parent not in slot_of:
            parent = parents[parent]
        flat_parent[joint] = parent

    children: dict[int, list] = {}
    heads: dict[int, Vector] = {}
    frames: dict[int, Matrix] = {}
    for slot, joint in enumerate(kept):
        name = joints[joint]
        edit_bone = edit_bones[name]
        parent = flat_parent[joint]
        edit_bone.parent = edit_bones[joints[parent]] if parent >= 0 else None
        edit_bone.use_connect = False
        frames[joint] = _blender_matrix(world[joint])
        edit_bone.head = frames[joint].to_translation()
        heads[joint] = edit_bone.head
        if parent >= 0:
            children.setdefault(parent, []).append(joint)

    # Bone length/direction: the runtime matrices carry the bind *scale*, not
    # a bone length, so a bone points at its child joint. Leaves (dropped
    # *_end markers, eye and bust tips) continue their parent's direction.
    direction: dict[int, Vector] = {}
    for joint in kept:
        edit_bone = edit_bones[joints[joint]]
        head = heads[joint]
        ref_y = frames[joint].col[1].to_3d()
        ref_y = ref_y.normalized() if ref_y.length > 1e-9 else Vector((0, 0, 1))
        kids = children.get(joint, [])
        tail = None
        parent = flat_parent[joint]
        parent_dir = direction.get(parent)
        best_dot = -2.0
        offsets = [(kid, heads[kid] - head) for kid in kids]
        offsets = [(kid, off) for kid, off in offsets if off.length > 1e-5]
        # Symmetric child pairs (the two eyes under the head) must not tilt
        # the bone to one side: use their average direction instead.
        for i in range(len(offsets)):
            for j in range(i + 1, len(offsets)):
                a = offsets[i][1] / offsets[i][1].length
                b = offsets[j][1] / offsets[j][1].length
                if abs(a.x + b.x) < 0.05 and abs(a.y - b.y) < 0.05 \
                        and abs(a.z - b.z) < 0.05:
                    tail = head + (offsets[i][1] + offsets[j][1]) / 2.0
                    break
            if tail is not None:
                break
        for kid, offset in offsets if tail is None else ():
            score = (offset.normalized().dot(parent_dir)
                     if parent_dir is not None else 0.0)
            if tail is None or score > best_dot:
                best_dot, tail = score, heads[kid]
        if tail is None:
            length = max(0.6 * edit_bones[joints[parent]].length, 0.005) \
                if parent >= 0 else 0.05
            tail = head + (parent_dir if parent_dir is not None else ref_y) * length
        edit_bone.tail = tail
        offset = edit_bone.tail - head
        direction[joint] = (offset.normalized() if offset.length > 1e-9
                            else ref_y)
        roll_ref = frames[joint].col[2].to_3d()
        if abs(roll_ref.normalized().dot(direction[joint])) > 0.95:
            roll_ref = frames[joint].col[0].to_3d()
        edit_bone.align_roll(roll_ref)
    _object_mode_set(context, arm_obj, "OBJECT")
    for pose_bone in arm_obj.pose.bones:
        pose_bone.matrix_basis = basis.get(pose_bone.name, Matrix.Identity(4))
    try:
        context.view_layer.objects.active = previous
    except RuntimeError:
        # The previous object may live in another scene/view layer.
        pass
    return arm_obj


def _bind_object(obj, arm_obj, enabled: bool) -> None:
    modifier = obj.modifiers.get(_RIG_MODIFIER)
    if modifier is None:
        modifier = obj.modifiers.new(_RIG_MODIFIER, "ARMATURE")
    # Only write what actually differs: re-assigning the modifier object (or
    # its flags) tags the modifier stack, which makes EEVEE treat the mesh as
    # structurally changed and re-bake its probes/shadows.  Measured ~200 ms
    # per slider step with the VRM MToon materials (see docs/vroid.md 7.4).
    if modifier.object is not arm_obj:
        modifier.object = arm_obj
    enabled = bool(enabled)
    if modifier.show_viewport != enabled:
        modifier.show_viewport = enabled
    if modifier.show_render != enabled:
        modifier.show_render = enabled
    if obj.parent is not arm_obj:
        obj.parent = arm_obj
        obj.matrix_parent_inverse = Matrix.Identity(4)


def _prepare_mesh_data(pack) -> None:
    """Cache the per-mesh data the viewport needs (slots, weights, bones)."""
    _state["materials"] = {
        (m.get("id") or m.get("name")): m
        for m in (pack.materials or [])
        if (m.get("id") or m.get("name"))
    }
    data = {}
    for desc in pack.mesh_descriptors:
        entry = {"slots": desc.get("slots") or []}
        for key in ("bones", "indices", "weights"):
            ref = desc.get(key)
            if ref is None:
                continue
            try:
                entry[key] = load_data_array(pack.pack_dir, ref)
            except Exception:
                entry[key] = None
        data[desc["name"]] = entry
    _state["mesh_data"] = data
    _state["rig_joints"] = []
    _state["rig_world"] = None
    _state["rig_remap"] = None
    _state["rig_sign"] = None


def _evaluate_mesh(context) -> tuple[bool, str]:
    return _evaluate_mesh_impl(context, live=False)


def _evaluate_mesh_impl(context, live: bool) -> tuple[bool, str]:
    if _state["adapter"] is None:
        return False, "load a template pack first"
    params = _collect_params()
    num_joints = len(_state["pack"].skeleton.get("joints", []))
    pose = {
        "rotations": [[0.0, 0.0, 0.0]] * num_joints,
        "translation": [0.0, 0.0, 0.0],
    }
    try:
        result = _state["adapter"].evaluate(params, pose)
    except Exception as exc:
        return False, str(exc)
    try:
        _update_mesh(context, result, live=live)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _state["last_error"] = f"{type(exc).__name__}: {exc}"
        return False, f"mesh update failed: {exc}"
    _state["last_error"] = ""
    return True, "ok"


def _face_list(faces) -> list[tuple[int, ...]]:
    """Python face tuples, only built when a mesh really has to be rebuilt."""
    return [tuple(int(idx) for idx in f) for f in faces]


def _write_vertices(mesh, coords: np.ndarray) -> None:
    """Copy vertex positions in one buffer write (no Python list round trip).

    ``foreach_set`` accepts the numpy buffer directly; going through
    ``tolist()`` cost ~30 ms per slider step on the 17k-triangle packs and
    showed up as lag whenever the viewport had to keep up.
    """
    mesh.vertices.foreach_set("co", coords.ravel())


def _coords_equal(cached, coords: np.ndarray) -> bool:
    """True when a mesh already holds exactly these vertex positions."""
    return (cached is not None and cached.shape == coords.shape
            and np.array_equal(cached, coords))


def _update_mesh(context, result, live: bool = False) -> bpy.types.Object | None:
    use_rig = bool(_flag("pd_use_rig", context))
    extra = getattr(result, "extra", None) or {}
    world = extra.get("world")
    joints = list(extra.get("joints") or [])
    parents = list(extra.get("parents") or [])
    sign = len(joints)
    arm_obj = None
    if world is not None and joints:
        rig_obj = bpy.data.objects.get(_RIG_NAME)
        needs_rig = (_rig_changed(world, joints) or _state.get("rig_sign") != sign
                     or rig_obj is None)
        # Defer only the *rest refresh* of an existing rig, and only while a
        # material preview is open: the first build and every explicit evaluate
        # stay immediate.
        if (needs_rig and live and rig_obj is not None
                and _state.get("rig_sign") == sign and _heavy_shading()):
            _state["rig_stale"] = True
            arm_obj = rig_obj
            needs_rig = False
        if needs_rig:
            kept, remap = _rig_subset(joints, parents)
            if _state.get("rig_sign") != sign:
                _state["rig_regroup"] = True
            arm_obj = _ensure_armature(context, world, joints, parents, kept)
            _state["rig_remap"] = remap
            _state["rig_sign"] = sign
            _state["rig_joints"] = joints
            _state["rig_world"] = np.array(world, dtype=np.float64)
            _state["rig_stale"] = False
        else:
            arm_obj = rig_obj

    regroup = _state.get("rig_regroup", False)
    _state["rig_regroup"] = False
    first = None
    for i, (name, verts, faces, uvs) in enumerate(result.meshes):
        obj_name = "PD_Avatar" if i == 0 else f"PD_Avatar_{name}"
        coords = np.ascontiguousarray(verts, dtype=np.float32)
        obj = bpy.data.objects.get(obj_name)
        if obj is not None and obj.name not in context.scene.objects:
            # The datablock survived but is not linked to this scene (deleted
            # from the view layer, or it belongs to another scene): without
            # this the mesh would be updated invisibly forever.
            context.scene.collection.objects.link(obj)
        rebuilt = False
        if obj is None:
            mesh = bpy.data.meshes.new(name)
            mesh.from_pydata(coords.tolist(), [], _face_list(faces))
            mesh.update()
            obj = bpy.data.objects.new(obj_name, mesh)
            context.scene.collection.objects.link(obj)
            rebuilt = True
        else:
            mesh = obj.data
            if (len(mesh.vertices) != len(coords)
                    or len(mesh.polygons) != len(faces)):
                mesh.clear_geometry()
                mesh.from_pydata(coords.tolist(), [], _face_list(faces))
                mesh.update()
                rebuilt = True
            elif not _coords_equal(_state["written"].get(obj_name), coords):
                # Only touch meshes whose vertices really changed: rewriting a
                # mesh (or even calling Mesh.update on it) makes EEVEE
                # re-evaluate it, and with the VRM materials that costs ~7 ms
                # per mesh plus ~5 ms for its outline modifier.
                _write_vertices(mesh, coords)
                mesh.update()
        _state["written"][obj_name] = coords
        if rebuilt:
            _configure_mesh(obj, name, uvs)
        if rebuilt or regroup:
            obj.vertex_groups.clear()
            _ensure_vertex_groups(obj, name)
        elif uvs is not None and not len(mesh.uv_layers):
            _configure_mesh(obj, name, uvs)
        if arm_obj is not None:
            _bind_object(obj, arm_obj, use_rig)
        if first is None:
            first = obj
    return first


class PD_OT_load_pack(Operator):
    bl_idname = "pd.load_pack"
    bl_label = "Load Template Pack"

    pack_path: StringProperty(name="Pack Path", subtype="FILE_PATH", default="")

    def execute(self, context):
        path = self.pack_path or _flag("pd_pack_path", context)
        issues = validate_pack(path)
        if issues:
            self.report({"ERROR"}, "; ".join(issues[:3]))
            return {"CANCELLED"}
        pack = load_pack(path)
        adapter = load_driver(path)
        adapter.load(pack, path)
        _state.update(pack=pack, adapter=adapter)
        _prepare_mesh_data(pack)
        _register_params(pack.params)
        _start_param_poll()
        self.report(
            {"INFO"},
            f"loaded {pack.manifest['name']} ({len(pack.params)} params)",
        )
        return {"FINISHED"}


class PD_OT_evaluate(Operator):
    bl_idname = "pd.evaluate"
    bl_label = "Evaluate / Update Mesh"

    def execute(self, context):
        ok, message = _evaluate_mesh(context)
        if not ok:
            self.report({"ERROR"}, message)
            return {"CANCELLED"}
        total = sum(
            len(o.data.vertices)
            for o in bpy.data.objects
            if o.name.startswith("PD_Avatar")
        )
        arm = bpy.data.objects.get(_RIG_NAME)
        bones = f", {len(arm.data.bones)} bones" if arm is not None else ""
        self.report(
            {"INFO"}, f"{total} verts across PD_Avatar meshes{bones}"
        )
        return {"FINISHED"}


class PD_OT_run_action(Operator):
    bl_idname = "pd.run_action"
    bl_label = "Template Action"

    action_id: StringProperty(default="")

    def execute(self, context):
        if _state["pack"] is None:
            return {"CANCELLED"}
        action = next(
            (a for a in _state["pack"].actions if a.get("id") == self.action_id),
            None,
        )
        if action is None:
            self.report({"ERROR"}, f"unknown action {self.action_id}")
            return {"CANCELLED"}
        objects = {}
        for mesh_name in _state["mesh_data"]:
            obj = bpy.data.objects.get(
                "PD_Avatar" if mesh_name == "face" else f"PD_Avatar_{mesh_name}")
            if obj is not None:
                objects[mesh_name] = obj
        ctx = {"objects": objects, "adapter": _state["adapter"]}
        try:
            message = pd_actions.run_action(_state["pack"], action, ctx)
        except Exception as exc:  # surface template/add-on problems in the UI
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, message)
        return {"FINISHED"}


def _finish_evaluate(operator, context) -> set:
    """Evaluate and turn a failure into a visible operator error."""
    ok, message = _evaluate_mesh(context)
    if not ok:
        operator.report({"ERROR"}, message)
        return {"CANCELLED"}
    # The values this evaluation used are now the reference for the poll.
    _sync_param_snapshot()
    return {"FINISHED"}


class PD_OT_randomize(Operator):
    bl_idname = "pd.randomize"
    bl_label = "Randomize"

    def execute(self, context):
        _state["suspending"] = True
        scene = context.scene
        for p in _state["params"]:
            pname = _state["prop_names"].get(p["id"])
            if not pname:
                continue
            lo, hi = p.get("min", -1.0), p.get("max", 1.0)
            setattr(scene, pname, random.uniform(lo, hi))
        _state["suspending"] = False
        return _finish_evaluate(self, context)


class PD_OT_randomize_group(Operator):
    bl_idname = "pd.randomize_group"
    bl_label = "Randomize Group"

    group: StringProperty(default="")

    def execute(self, context):
        _state["suspending"] = True
        scene = context.scene
        for p in _state["params"]:
            if p.get("group", "Params") != self.group:
                continue
            pname = _state["prop_names"].get(p["id"])
            if not pname:
                continue
            lo, hi = p.get("min", -1.0), p.get("max", 1.0)
            setattr(scene, pname, random.uniform(lo, hi))
        _state["suspending"] = False
        return _finish_evaluate(self, context)


class PD_OT_zero_group(Operator):
    bl_idname = "pd.zero_group"
    bl_label = "Reset Group"

    group: StringProperty(default="")

    def execute(self, context):
        _state["suspending"] = True
        scene = context.scene
        for p in _state["params"]:
            if p.get("group", "Params") != self.group:
                continue
            pname = _state["prop_names"].get(p["id"])
            if not pname:
                continue
            setattr(scene, pname, p.get("default", 0.0))
        _state["suspending"] = False
        return _finish_evaluate(self, context)


class PD_OT_toggle_outline(Operator):
    bl_idname = "pd.toggle_outline"
    bl_label = "Toggle Viewport Outline"
    bl_description = (
        "Enable or disable the outline Geometry Nodes modifiers in the "
        "viewport. They re-evaluate on every slider step, which is a large "
        "part of the material-mode lag; rendering is unaffected"
    )

    def execute(self, context):
        modifiers = [
            mod
            for obj in bpy.data.objects
            if obj.type == "MESH" and obj.name.startswith("PD_Avatar")
            for mod in obj.modifiers
            if mod.type == "NODES"
        ]
        if not modifiers:
            self.report({"INFO"}, "no outline modifiers on the avatar")
            return {"CANCELLED"}
        enable = not any(mod.show_viewport for mod in modifiers)
        for mod in modifiers:
            mod.show_viewport = enable
        _state["outline_viewport"] = enable
        self.report({"INFO"},
                    f"viewport outline {'on' if enable else 'off'} "
                    f"({len(modifiers)} modifiers)")
        return {"FINISHED"}


class PD_OT_save_preset(Operator):
    bl_idname = "pd.save_preset"
    bl_label = "Save Preset"

    preset_name: StringProperty(name="Preset", default="preset")

    def execute(self, context):
        preset_dir = _preset_dir()
        os.makedirs(preset_dir, exist_ok=True)
        path = os.path.join(preset_dir, f"{self.preset_name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_collect_params(), fh, indent=2)
        self.report({"INFO"}, f"preset saved to {path}")
        return {"FINISHED"}


class PD_OT_load_preset(Operator):
    bl_idname = "pd.load_preset"
    bl_label = "Load Preset"

    preset_name: StringProperty(name="Preset", default="preset")

    def execute(self, context):
        path = os.path.join(_preset_dir(), f"{self.preset_name}.json")
        if not os.path.exists(path):
            self.report({"ERROR"}, f"preset not found: {path}")
            return {"CANCELLED"}
        with open(path, "r", encoding="utf-8") as fh:
            _set_params(json.load(fh))
        return _finish_evaluate(self, context)


class PD_OT_delete_preset(Operator):
    bl_idname = "pd.delete_preset"
    bl_label = "Delete Preset"

    preset_name: StringProperty(name="Preset", default="preset")

    def execute(self, context):
        path = os.path.join(_preset_dir(), f"{self.preset_name}.json")
        if not os.path.exists(path):
            self.report({"ERROR"}, f"preset not found: {path}")
            return {"CANCELLED"}
        os.remove(path)
        self.report({"INFO"}, f"preset deleted: {path}")
        return {"FINISHED"}


class PD_PT_main(Panel):
    bl_label = "参数化变形 Parametric Deformation"
    bl_idname = "PD_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PD"

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        # Bind every widget to the collection, never to the Scene: see _host.
        host = _host(context)
        # Self-heal: the panel is drawn from a real UI context, so if the poll
        # timer died (an exception, a stale revision) bring it back here —
        # otherwise slider changes would silently stop updating the mesh.
        if _state.get("pack") is not None:
            _start_param_poll()
        col.prop(host, "pd_pack_path")
        col.operator("pd.load_pack", text="Load Pack").pack_path = (
            host.pd_pack_path
        )
        if _state["pack"] is None:
            return
        col.separator()
        col.prop(host, "pd_preset_name")
        row = col.row(align=True)
        row.operator("pd.save_preset", text="Save").preset_name = (
            host.pd_preset_name
        )
        row.operator("pd.load_preset", text="Load").preset_name = (
            host.pd_preset_name
        )
        row.operator("pd.delete_preset", text="Delete").preset_name = (
            host.pd_preset_name
        )
        col.prop(host, "pd_live_update", text="Live Update")
        col.prop(host, "pd_use_rig", text="Deform With Bones")
        if _state.get("last_error"):
            box = col.box()
            box.label(text="Last error:", icon="ERROR")
            box.label(text=_state["last_error"][:120])
        rig = bpy.data.objects.get(_RIG_NAME)
        if rig is not None:
            col.label(
                text=f"Rig: {len(rig.data.bones)} bones (VRoid humanoid set)"
            )
            if _state.get("rig_stale"):
                col.label(
                    text="Rig rest updates in Solid mode or on Evaluate",
                    icon="INFO",
                )
        outline_mods = [
            mod
            for obj in bpy.data.objects
            if obj.type == "MESH" and obj.name.startswith("PD_Avatar")
            for mod in obj.modifiers
            if mod.type == "NODES"
        ]
        if outline_mods:
            shown = any(mod.show_viewport for mod in outline_mods)
            col.operator(
                "pd.toggle_outline",
                text=f"Viewport Outline: {'On' if shown else 'Off'}",
                icon="HIDE_OFF" if shown else "HIDE_ON",
            )
        col.operator("pd.evaluate", text="Evaluate / Update Mesh")
        for action in _state["pack"].actions:
            col.operator(
                "pd.run_action", text=action.get("label", action.get("id", "Action"))
            ).action_id = action.get("id", "")
        col.operator("pd.randomize", text="Randomize")
        col.separator()
        col.prop(host, "pd_filter", text="Filter (regex)")

        pattern = None
        filter_text = host.pd_filter.strip()
        if filter_text:
            try:
                pattern = re.compile(filter_text, re.IGNORECASE)
            except re.error:
                pattern = None
        for group in _state["groups"]:
            box = col.box()
            show_prop = _state["group_props"].get(group)
            if not show_prop:
                continue
            expanded = getattr(host, show_prop)
            header = box.row()
            header.prop(
                host,
                show_prop,
                text=group,
                icon="TRIA_DOWN" if expanded else "TRIA_RIGHT",
                emboss=False,
            )
            group_row = header.row(align=True)
            group_row.operator("pd.randomize_group", text="R").group = group
            group_row.operator("pd.zero_group", text="0").group = group
            if not expanded:
                continue
            last_subgroup = None
            subgroup_drawn = False
            for p in _state["params"]:
                if p.get("group", "Params") != group:
                    continue
                if pattern is not None:
                    text = f"{p.get('id', '')} {p.get('label', '')}"
                    if not pattern.search(text):
                        continue
                subgroup = _subgroup_of(p)
                if subgroup != last_subgroup:
                    box.label(text=f"  {subgroup}", icon="LAYER_ACTIVE")
                    last_subgroup = subgroup
                    subgroup_drawn = True
                pname = _state["prop_names"].get(p["id"])
                if pname:
                    box.prop(host, pname)


CLASSES = (
    PD_OT_load_pack,
    PD_OT_evaluate,
    PD_OT_run_action,
    PD_OT_randomize,
    PD_OT_randomize_group,
    PD_OT_zero_group,
    PD_OT_toggle_outline,
    PD_OT_save_preset,
    PD_OT_load_preset,
    PD_OT_delete_preset,
    PD_PT_main,
)
