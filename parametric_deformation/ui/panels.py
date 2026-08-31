"""Blender UI: template pack loading, collapsible slider groups, live update."""

from __future__ import annotations

import json
import os
import random

import bpy
from bpy.props import BoolProperty, FloatProperty, StringProperty
from bpy.types import Operator, Panel

from parametric_deformation.core.config import load_pack, validate_pack
from parametric_deformation.tools import load_driver

_state = {
    "pack": None,
    "adapter": None,
    "params": [],
    "prop_names": {},      # param id -> Scene property name
    "groups": [],          # group names ordered by rank
    "group_rank": {},      # group -> min rank
    "group_props": {},     # group -> Scene bool prop (collapse toggle)
    "suspending": False,
}


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def _preset_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".parametric_deformation", "presets")


def _param_updated(self, context) -> None:
    """Live-update callback for every slider property."""
    if _state["suspending"] or _state["adapter"] is None:
        return
    if context.scene.pd_live_update:
        _evaluate_mesh(context)


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
                step=p.get("step", 0.01),
                update=_param_updated,
            ),
        )
        _state["prop_names"][pid] = prop_name
    _state["params"] = params


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
    _update_mesh(context, result)
    return True, "ok"


def _update_mesh(context, result) -> bpy.types.Object:
    name, verts, faces, _uvs = result.meshes[0]
    import numpy as np

    flat_verts = np.asarray(verts, dtype=np.float64).ravel().tolist()
    faces = [tuple(int(i) for i in f) for f in faces]
    obj = bpy.data.objects.get("PD_Avatar")
    if obj is None:
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(np.asarray(verts, dtype=np.float64).tolist(), [], faces)
        mesh.update()
        obj = bpy.data.objects.new("PD_Avatar", mesh)
        context.scene.collection.objects.link(obj)
    else:
        mesh = obj.data
        if len(mesh.vertices) != len(verts):
            mesh.clear_geometry()
            mesh.from_pydata(np.asarray(verts, dtype=np.float64).tolist(), [], faces)
        else:
            mesh.vertices.foreach_set("co", flat_verts)
        mesh.update()
    return obj


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
        obj = bpy.data.objects.get("PD_Avatar")
        self.report(
            {"INFO"},
            f"{obj.name if obj else 'mesh'}: {len(obj.data.vertices) if obj else 0} verts",
        )
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
        _evaluate_mesh(context)
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
        _evaluate_mesh(context)
        return {"FINISHED"}


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
        col.operator("pd.evaluate", text="Evaluate / Update Mesh")
        col.operator("pd.randomize", text="Randomize")
        col.separator()

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
            if not expanded:
                continue
            for p in _state["params"]:
                if p.get("group", "Params") != group:
                    continue
                pname = _state["prop_names"].get(p["id"])
                if pname:
                    box.prop(context.scene, pname)


CLASSES = (
    PD_OT_load_pack,
    PD_OT_evaluate,
    PD_OT_randomize,
    PD_OT_save_preset,
    PD_OT_load_preset,
    PD_OT_delete_preset,
    PD_PT_main,
)
