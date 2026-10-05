"""Blender UI: template pack loading, collapsible slider groups, live update."""

from __future__ import annotations

import json
import os
import random
import re

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
}


# VRoid imports the Unity Y-up skeletons; the template drivers emit Blender
# Z-up vertices, so every matrix has to travel through the same axis change.
_AXIS_TO_BLENDER = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)

_RIG_NAME = "PD_Rig"
_RIG_MODIFIER = "PD_Rig"

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


def _param_updated(self, context) -> None:
    """Live-update callback for slider props (debounced via a timer)."""
    if _state["suspending"] or _state["adapter"] is None:
        return
    if not context.scene.pd_live_update:
        return
    _state["dirty"] = True
    if _state["timer"] is None:
        _state["timer"] = bpy.app.timers.register(_flush_timer, first_interval=0.05)


def _flush_timer():
    """Single-shot timer: coalesce bursts of slider changes into one evaluate."""
    _state["timer"] = None
    if _state.get("dirty") and _state["adapter"] is not None:
        _state["dirty"] = False
        try:
            _evaluate_mesh(bpy.context)
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
        if not hasattr(bpy.types.Scene, prop_name):
            setattr(
                bpy.types.Scene,
                prop_name,
                BoolProperty(
                    name=group,
                    description=group,
                    default=(groups[group] == 0),
                ),
            )
        _state["group_props"][group] = prop_name
    for p in params:
        pid = p["id"]
        prop_name = f"pd_{_sanitize(pid)}"
        if hasattr(bpy.types.Scene, prop_name):
            _state["prop_names"][pid] = prop_name
            continue
        setattr(
            bpy.types.Scene,
            prop_name,
            FloatProperty(
                name=p.get("label", pid),
                description=p.get("label", pid),
                min=p.get("min", -1.0),
                max=p.get("max", 1.0),
                default=p.get("default", 0.0),
                # step=p.get("step", 0.01), too slow
                update=_param_updated,
            ),
        )
        _state["prop_names"][pid] = prop_name
    # Keep equal sub-group labels contiguous: the panel draws one collapsible
    # block per label run, so an unsorted list splits "Body" into several
    # blocks separated by other sliders. Sorting is stable, so the authored
    # order inside a sub-group is preserved.
    _state["params"] = sorted(
        params, key=lambda p: (p.get("group", "Params"), _subgroup_of(p))
    )


def _collect_params() -> dict:
    scene = bpy.context.scene
    return {
        pid: getattr(scene, pname) for pid, pname in _state["prop_names"].items()
    }


def _set_params(values: dict, suspend: bool = True) -> None:
    if suspend:
        _state["suspending"] = True
    scene = bpy.context.scene
    for pid, pname in _state["prop_names"].items():
        if pid in values:
            setattr(scene, pname, float(values[pid]))
    if suspend:
        _state["suspending"] = False


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
    modifier.object = arm_obj
    modifier.show_viewport = enabled
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
        _update_mesh(context, result)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _state["last_error"] = f"{type(exc).__name__}: {exc}"
        return False, f"mesh update failed: {exc}"
    _state["last_error"] = ""
    return True, "ok"


def _update_mesh(context, result) -> bpy.types.Object | None:
    use_rig = bool(getattr(context.scene, "pd_use_rig", True))
    extra = getattr(result, "extra", None) or {}
    world = extra.get("world")
    joints = list(extra.get("joints") or [])
    parents = list(extra.get("parents") or [])
    sign = len(joints)
    arm_obj = None
    if world is not None and joints:
        if (_rig_changed(world, joints) or _state.get("rig_sign") != sign
                or bpy.data.objects.get(_RIG_NAME) is None):
            kept, remap = _rig_subset(joints, parents)
            if _state.get("rig_sign") != sign:
                _state["rig_regroup"] = True
            arm_obj = _ensure_armature(context, world, joints, parents, kept)
            _state["rig_remap"] = remap
            _state["rig_sign"] = sign
        else:
            arm_obj = bpy.data.objects.get(_RIG_NAME)
        _state["rig_joints"] = joints
        _state["rig_world"] = np.array(world, dtype=np.float64)

    regroup = _state.get("rig_regroup", False)
    _state["rig_regroup"] = False
    first = None
    for i, (name, verts, faces, uvs) in enumerate(result.meshes):
        obj_name = "PD_Avatar" if i == 0 else f"PD_Avatar_{name}"
        verts = np.asarray(verts, dtype=np.float64)
        flat_verts = verts.ravel().tolist()
        face_list = [tuple(int(idx) for idx in f) for f in faces]
        obj = bpy.data.objects.get(obj_name)
        if obj is not None and obj.name not in context.scene.objects:
            # The datablock survived but is not linked to this scene (deleted
            # from the view layer, or it belongs to another scene): without
            # this the mesh would be updated invisibly forever.
            context.scene.collection.objects.link(obj)
        rebuilt = False
        if obj is None:
            mesh = bpy.data.meshes.new(name)
            mesh.from_pydata(verts.tolist(), [], face_list)
            mesh.update()
            obj = bpy.data.objects.new(obj_name, mesh)
            context.scene.collection.objects.link(obj)
            rebuilt = True
        else:
            mesh = obj.data
            if (len(mesh.vertices) != len(verts)
                    or len(mesh.polygons) != len(face_list)):
                mesh.clear_geometry()
                mesh.from_pydata(verts.tolist(), [], face_list)
                rebuilt = True
            else:
                mesh.vertices.foreach_set("co", flat_verts)
            mesh.update()
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
        path = self.pack_path or context.scene.pd_pack_path
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
        col.prop(context.scene, "pd_pack_path")
        col.operator("pd.load_pack", text="Load Pack").pack_path = (
            context.scene.pd_pack_path
        )
        if _state["pack"] is None:
            return
        col.separator()
        col.prop(context.scene, "pd_preset_name")
        row = col.row(align=True)
        row.operator("pd.save_preset", text="Save").preset_name = (
            context.scene.pd_preset_name
        )
        row.operator("pd.load_preset", text="Load").preset_name = (
            context.scene.pd_preset_name
        )
        row.operator("pd.delete_preset", text="Delete").preset_name = (
            context.scene.pd_preset_name
        )
        col.prop(context.scene, "pd_live_update", text="Live Update")
        col.prop(context.scene, "pd_use_rig", text="Deform With Bones")
        if _state.get("last_error"):
            box = col.box()
            box.label(text="Last error:", icon="ERROR")
            box.label(text=_state["last_error"][:120])
        rig = bpy.data.objects.get(_RIG_NAME)
        if rig is not None:
            col.label(
                text=f"Rig: {len(rig.data.bones)} bones (VRoid humanoid set)"
            )
        col.operator("pd.evaluate", text="Evaluate / Update Mesh")
        for action in _state["pack"].actions:
            col.operator(
                "pd.run_action", text=action.get("label", action.get("id", "Action"))
            ).action_id = action.get("id", "")
        col.operator("pd.randomize", text="Randomize")
        col.separator()
        col.prop(context.scene, "pd_filter", text="Filter (regex)")

        pattern = None
        filter_text = context.scene.pd_filter.strip()
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
            expanded = getattr(context.scene, show_prop)
            header = box.row()
            header.prop(
                context.scene,
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
                    box.prop(context.scene, pname)


CLASSES = (
    PD_OT_load_pack,
    PD_OT_evaluate,
    PD_OT_run_action,
    PD_OT_randomize,
    PD_OT_randomize_group,
    PD_OT_zero_group,
    PD_OT_save_preset,
    PD_OT_load_preset,
    PD_OT_delete_preset,
    PD_PT_main,
)
