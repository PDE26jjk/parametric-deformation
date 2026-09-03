"""Blender UI: template pack loading, collapsible slider groups, live update."""

from __future__ import annotations

import json
import os
import random
import re

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
    "dirty": False,
    "timer": None,
}


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def _preset_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".parametric_deformation", "presets")


_SUBGROUP_KEYWORDS = (
    ("Chin", ("chin", "jaw")),
    ("Ears", ("ear",)),
    ("Eyes", ("eye",)),
    ("Brows", ("brow",)),
    ("Nose", ("nose",)),
    ("Mouth", ("mouth", "lip")),
    ("Face", ("face", "cheek", "head", "faceline", "mandible")),
    ("Neck", ("neck",)),
    ("Chest", ("bust", "chest")),
    ("Waist/Hips", ("waist", "hip")),
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
        except Exception:
            pass
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


def _update_mesh(context, result) -> bpy.types.Object | None:
    import numpy as np

    first = None
    for i, (name, verts, faces, _uvs) in enumerate(result.meshes):
        obj_name = "PD_Avatar" if i == 0 else f"PD_Avatar_{name}"
        flat_verts = np.asarray(verts, dtype=np.float64).ravel().tolist()
        face_list = [tuple(int(idx) for idx in f) for f in faces]
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            mesh = bpy.data.meshes.new(name)
            mesh.from_pydata(
                np.asarray(verts, dtype=np.float64).tolist(), [], face_list
            )
            mesh.update()
            obj = bpy.data.objects.new(obj_name, mesh)
            context.scene.collection.objects.link(obj)
        else:
            mesh = obj.data
            if len(mesh.vertices) != len(verts):
                mesh.clear_geometry()
                mesh.from_pydata(np.asarray(verts, dtype=np.float64).tolist(), [], face_list)
            else:
                mesh.vertices.foreach_set("co", flat_verts)
            mesh.update()
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
        self.report({"INFO"}, f"{total} verts across PD_Avatar meshes")
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
        _evaluate_mesh(context)
        return {"FINISHED"}


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
    PD_OT_randomize,
    PD_OT_randomize_group,
    PD_OT_zero_group,
    PD_OT_save_preset,
    PD_OT_load_preset,
    PD_OT_delete_preset,
    PD_PT_main,
)
