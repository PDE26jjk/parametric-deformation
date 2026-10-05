"""Pack-declared actions (custom buttons) and their generic implementations.

A template pack lists buttons in ``manifest.json``::

    "actions": [
      {"id": "import_materials", "label": "导入 VRM 材质",
       "type": "import_vrm_materials", "data": "models/model_clear.vrm"}
    ]

``type`` selects a built-in implementation below; ``handler``
(``"py.module:function"``) calls pack-owned code instead.  Built-ins are
deliberately template-agnostic so every pack can reuse them.
"""

from __future__ import annotations

import importlib.util
import os
import re

import bpy

ACTIONS: dict[str, callable] = {}


def register_action(name: str, fn) -> None:
    ACTIONS[name] = fn


def run_action(pack, action: dict, ctx: dict) -> str:
    """Execute one pack action and return a status message."""
    handler = action.get("handler")
    if handler:
        return _call_pack_handler(pack, handler, action, ctx)
    kind = action.get("type")
    fn = ACTIONS.get(kind)
    if fn is None:
        raise ValueError(f"unknown action type: {kind!r}")
    return fn(pack, action, ctx)


def _call_pack_handler(pack, handler: str, action: dict, ctx: dict) -> str:
    module_path, _, func_name = handler.partition(":")
    if not module_path.startswith("py.") or not func_name:
        raise ValueError(f"action handler must look like 'py.module:function': {handler}")
    module_file = os.path.join(pack.pack_dir, *module_path.split(".")) + ".py"
    spec = importlib.util.spec_from_file_location(module_path, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = getattr(module, func_name)(pack=pack, action=action, ctx=ctx)
    return result or f"{action.get('label', action.get('id'))}: ok"


# ---------------------------------------------------------------------------
# Built-in: import the pack's VRM materials with the "VRM format" add-on
# ---------------------------------------------------------------------------

VRM_ADDON_MODULES = (
    "bl_ext.blender_org.vrm",   # Blender 4.2+ extension
    "io_scene_vrm",             # classic add-on id
)

_OUTLINE_PREFIX = "MToon Outline"


def _vrm_operator_ready() -> bool:
    try:
        bpy.ops.import_scene.vrm.get_rna_type()
    except (AttributeError, KeyError, RuntimeError):
        return False
    return True


def _enable_vrm_addon() -> str | None:
    """Try to enable the VRM add-on; returns its module name or None."""
    if _vrm_operator_ready():
        return "already enabled"
    import addon_utils

    for module in VRM_ADDON_MODULES:
        # The operator form also creates the add-on preferences the VRM
        # importer expects ("No add-on preferences for ..." otherwise).
        try:
            bpy.ops.preferences.addon_enable(module=module)
        except Exception:
            pass
        if _vrm_operator_ready():
            return module
        try:
            addon_utils.enable(module, default_set=False, persistent=True)
        except Exception:
            continue
        if _vrm_operator_ready():
            return module
    return None


def _base_material_name(name: str) -> str:
    return re.sub(r"\.\d{3}$", "", name)


def _copy_outline_modifiers(source, target) -> int:
    """Copy the VRM add-on's MToon outline modifiers from one object to another."""
    copied = 0
    for modifier in source.modifiers:
        if modifier.type != "NODES" or modifier.node_group is None:
            continue
        new = target.modifiers.new(modifier.name, "NODES")
        new.node_group = modifier.node_group
        # Geometry Nodes settings live in the modifier's socket properties.
        # Sockets that point at the (temporary) source object must be rebound
        # to the mesh we keep, otherwise they go null when the import is
        # deleted and the outline disappears.
        for key in modifier.keys():
            try:
                value = modifier[key]
                new[key] = target if value is source else value
            except Exception:
                pass
        copied += 1
    return copied


def _resolve_vrm_path(pack, action: dict) -> str:
    data = action.get("data") or "models"
    path = os.path.join(pack.pack_dir, data)
    if os.path.isdir(path):
        candidates = sorted(f for f in os.listdir(path) if f.lower().endswith(".vrm"))
        if not candidates:
            raise FileNotFoundError(f"no .vrm in {path}")
        return os.path.join(path, candidates[0])
    if not os.path.exists(path):
        raise FileNotFoundError(f"VRM not found: {path}")
    return path


def import_vrm_materials(pack, action: dict, ctx: dict) -> str:
    """Import a VRM and put its MToon materials on this pack's meshes.

    The VRM is imported into the current scene, the materials are copied onto
    the matching mesh slots (by material name), and the imported objects are
    then removed again - so the result is exactly what importing the VRM by
    hand gives, without a duplicate model in the scene.
    """
    objects = ctx.get("objects") or {}
    if not objects:
        raise RuntimeError("no evaluated meshes: press 'Evaluate / Update Mesh' first")
    vrm_path = _resolve_vrm_path(pack, action)
    enabled = _enable_vrm_addon()
    if enabled is None:
        raise RuntimeError(
            "VRM format add-on not available; enable it in "
            "Edit > Preferences > Add-ons (extension 'VRM format' by lunapark)"
        )

    before_objects = set(bpy.data.objects)
    before_materials = set(bpy.data.materials)
    before_images = set(bpy.data.images)
    bpy.ops.import_scene.vrm(filepath=vrm_path)

    imported = {}
    for material in bpy.data.materials:
        if material in before_materials:
            continue
        if material.name.startswith(_OUTLINE_PREFIX):
            continue
        imported.setdefault(_base_material_name(material.name), material)

    assigned = 0
    replaced = set()
    for mesh_name, obj in objects.items():
        descriptor = next(
            (d for d in pack.mesh_descriptors if d["name"] == mesh_name), None)
        slots = (descriptor or {}).get("slots") or []
        for index, slot in enumerate(slots):
            material = imported.get(slot.get("material"))
            if material is None or index >= len(obj.data.materials):
                continue
            replaced.add(obj.data.materials[index])
            obj.data.materials[index] = material
            assigned += 1

    # Optional: mirror the add-on's inverted-hull outline (a Geometry Nodes
    # modifier) onto our meshes. The mapping is declared by the pack because
    # the VRM merges parts (Face = face+ear, Body = body+head+hand+foot).
    outlines = 0
    for source_name, mesh_names in (action.get("outline") or {}).items():
        source = bpy.data.objects.get(source_name)
        if source is None:
            continue
        for mesh_name in mesh_names:
            target = objects.get(mesh_name)
            if target is not None:
                outlines += _copy_outline_modifiers(source, target)

    # Drop the imported model again; keep only its materials, images and the
    # node groups they reference.
    for obj in [o for o in bpy.data.objects if o not in before_objects]:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if isinstance(data, bpy.types.Mesh) and data.users == 0:
            bpy.data.meshes.remove(data)
        elif isinstance(data, bpy.types.Armature) and data.users == 0:
            bpy.data.armatures.remove(data)
    for material in list(replaced):
        if material is not None and material.users == 0:
            bpy.data.materials.remove(material)
    # The placeholders were named like the VRM materials, so the import had to
    # suffix the new ones; restore the plain names now that the slots point at
    # the imported materials.
    for base, material in imported.items():
        if material.name != base and material.name.startswith(base):
            try:
                material.name = base
            except Exception:
                pass
    for material in bpy.data.materials:
        cleaned = re.sub(r"\.\d{3}\)$", ")", material.name)
        if cleaned != material.name:
            try:
                material.name = cleaned
            except Exception:
                pass

    images = [i for i in bpy.data.images if i not in before_images]
    packed = sum(1 for i in images if i.packed_file is not None)
    return (f"{len(imported)} VRM materials on {assigned} slots, "
            f"{outlines} outline modifiers "
            f"({packed}/{len(images)} images packed)"
            + ("" if enabled == "already enabled" else f", enabled {enabled}"))


register_action("import_vrm_materials", import_vrm_materials)
