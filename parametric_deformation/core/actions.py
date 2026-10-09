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


#: The VRM importer is a thin wrapper around the bundled glTF importer, so a
#: disabled glTF add-on makes ``bpy.ops.import_scene.vrm`` fail *inside* the
#: VRM add-on with "Calling operator bpy.ops.import_scene.gltf error, could not
#: be found" - which reads like a VRM problem but is not.
GLTF_ADDON_MODULE = "io_scene_gltf2"


def _gltf_operator_ready() -> bool:
    try:
        bpy.ops.import_scene.gltf.get_rna_type()
    except (AttributeError, KeyError, RuntimeError):
        return False
    return True


def _ensure_gltf_addon() -> str:
    """Make sure the glTF importer the VRM add-on delegates to is enabled."""
    if _gltf_operator_ready():
        return "ok"
    try:
        bpy.ops.preferences.addon_enable(module=GLTF_ADDON_MODULE)
    except Exception as exc:  # noqa: BLE001
        return f"failed ({type(exc).__name__}: {exc})"
    return "enabled" if _gltf_operator_ready() else "unavailable"


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
        _copy_modifier_socket_values(modifier, new, source, target)
        copied += 1
    return copied


def _copy_modifier_socket_values(
        source_modifier, target_modifier, source_object, target_object) -> int:
    """Copy Geometry Nodes socket values, rebinding the source object.

    Sockets that point at the (temporary) source object must be rebound to the
    mesh we keep, otherwise they go null when the import is deleted and the
    outline disappears.

    Blender < 5.2 keeps these values as (system) ID properties on the modifier
    (``modifier["Socket_2"]``). Blender 5.2 moved them to runtime RNA exposed
    as ``modifier.properties.inputs``, where the ID property API raises
    "this type doesn't support IDProperties", so both routes are tried.
    """
    copied = 0
    try:
        items = list(source_modifier.items())
    except TypeError:
        items = []
    if items:
        for key, value in items:
            try:
                target_modifier[key] = (
                    target_object if value is source_object else value)
                copied += 1
            except Exception:
                pass
        return copied

    source_inputs = getattr(
        getattr(source_modifier, "properties", None), "inputs", None)
    target_inputs = getattr(
        getattr(target_modifier, "properties", None), "inputs", None)
    if source_inputs is None or target_inputs is None:
        return copied
    try:
        keys = list(source_inputs.keys())
    except Exception:
        return copied
    for key in keys:
        source_socket = getattr(source_inputs, key, None)
        target_socket = getattr(target_inputs, key, None)
        if source_socket is None or target_socket is None:
            continue
        # The modifier's interface sockets are read through runtime RNA:
        # `type` selects value/attribute, `value` holds the setting itself.
        for attr in ("type", "attribute_name", "value"):
            if not (hasattr(source_socket, attr) and hasattr(target_socket, attr)):
                continue
            try:
                value = getattr(source_socket, attr)
                setattr(target_socket, attr,
                        target_object if value is source_object else value)
                copied += 1
            except Exception:
                pass
    return copied


def rebind_object_sockets(obj, source, target) -> int:
    """Point Geometry Nodes object sockets that reference ``source`` at ``target``.

    Copying an object carries its modifiers along, but a socket that holds the
    original object keeps using it - the VRM add-on's outline modifier stores
    exactly that. Returns the number of sockets that were repointed.
    """
    fixed = 0
    for modifier in obj.modifiers:
        if modifier.type != "NODES":
            continue
        try:
            items = list(modifier.items())
        except TypeError:
            items = []
        for key, value in items:
            if value is source:
                try:
                    modifier[key] = target
                    fixed += 1
                except Exception:
                    pass
        if items:
            continue
        # Blender 5.2 exposes the same values as runtime RNA instead of ID
        # properties (see _copy_modifier_socket_values).
        inputs = getattr(getattr(modifier, "properties", None), "inputs", None)
        if inputs is None:
            continue
        try:
            keys = list(inputs.keys())
        except Exception:
            continue
        for key in keys:
            socket = getattr(inputs, key, None)
            if socket is None:
                continue
            try:
                if getattr(socket, "value", None) is source:
                    socket.value = target
                    fixed += 1
            except Exception:
                pass
    return fixed


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
    gltf = _ensure_gltf_addon()
    if gltf == "unavailable":
        raise RuntimeError(
            "the VRM add-on needs the glTF importer, but "
            "'Import-Export: glTF 2.0 format' could not be enabled "
            "(Edit > Preferences > Add-ons)"
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
            + ("" if enabled == "already enabled" else f", enabled {enabled}")
            + ("" if gltf in ("ok", "enabled") else f", glTF {gltf}"))


register_action("import_vrm_materials", import_vrm_materials)
