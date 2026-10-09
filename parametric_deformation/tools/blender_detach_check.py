"""Headless check for the Detach button (independent mesh copies).

Usage:
  blender.exe --background --factory-startup --python \
    parametric_deformation/tools/blender_detach_check.py -- <pack_dir>

Evaluates the pack, detaches the meshes, and verifies that the copies are
independent (own object and mesh data, renamed away from the PD_ prefix), that
the add-on keeps working on its own objects, and that template-provided shape
keys carry the template's sliders.
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402

import parametric_deformation  # noqa: E402
from parametric_deformation.ui.panels import _state  # noqa: E402


def main() -> None:
    if "--" not in sys.argv or len(sys.argv) < sys.argv.index("--") + 2:
        raise SystemExit("usage: blender --python blender_detach_check.py -- <pack>")
    pack_dir = os.path.abspath(sys.argv[sys.argv.index("--") + 1])

    parametric_deformation.register()
    # The Pack Path widget must load the pack by itself: writing the property
    # arms the auto-load timer, and the timer does the actual load.
    from parametric_deformation.ui.panels import _auto_load_pack

    bpy.context.scene.collection.pd_pack_path = pack_dir
    assert bpy.app.timers.is_registered(_auto_load_pack), (
        "Pack Path did not arm the auto-load timer")
    _auto_load_pack()
    assert _state["pack"] is not None, "auto load did not install the pack"
    assert _state["pack_path"] == os.path.abspath(pack_dir)
    print("AUTO LOAD OK")
    assert bpy.ops.pd.evaluate() == {"FINISHED"}

    source_objects = {o.name for o in bpy.data.objects
                      if o.name.startswith("PD_Avatar")}
    assert source_objects, "evaluate produced no PD_Avatar meshes"
    source_meshes = {o.name: o.data for o in bpy.data.objects
                     if o.name.startswith("PD_Avatar")}

    # A slider value the detached copy has to preserve.
    expression_ids = [p["id"] for p in _state["pack"].params
                      if p["id"].startswith("expr_")]
    if expression_ids:
        setattr(bpy.context.scene, _state["prop_names"][expression_ids[0]], 0.5)

    assert bpy.ops.pd.detach() == {"FINISHED"}

    detached = {o.name: o for o in bpy.data.objects
                if o.name.startswith("Detached_")}
    assert detached, "detach produced no objects"
    print(f"DETACHED {sorted(detached)}")

    sources = [bpy.data.objects[n] for n in sorted(source_objects)]
    for name, obj in detached.items():
        if obj.type != "MESH":
            continue
        source = next((s for s in sources
                       if len(s.data.vertices) == len(obj.data.vertices)), None)
        assert source is not None, f"{name}: no PD mesh with that vertex count"
        assert obj.data is not source.data, f"{name}: mesh data is shared"
        assert obj.data.materials, f"{name}: lost its materials"
        assert len(obj.data.uv_layers) == len(source.data.uv_layers)
        assert obj.vertex_groups.keys() == source.vertex_groups.keys()

    # Moving a slider must deform the add-on's mesh and leave the copy alone.
    # Pair copies with their sources by vertex count (the driver's mesh order
    # is its own, and the counts are unique per pack) and prefer the copy that
    # carries shape keys, since that is the interesting one.
    pairs = []
    for name, obj in detached.items():
        if obj.type != "MESH":
            continue
        source = next((s for s in sources
                       if len(s.data.vertices) == len(obj.data.vertices)), None)
        assert source is not None, f"{name}: no PD mesh with that vertex count"
        pairs.append((obj, source))
    copy_obj, source_obj = max(
        pairs, key=lambda p: (p[0].data.shape_keys is not None,
                              len(p[0].data.vertices)))

    moved = False
    copy_before = [v.co.copy() for v in copy_obj.data.vertices[:64]]
    for candidate in ("Eye Size X", "Nose Size", "Body Size", "Arm Length",
                      *(p["id"] for p in _state["pack"].params[:4])):
        prop = _state["prop_names"].get(candidate)
        if prop is None:
            continue
        before = [v.co.copy() for v in source_obj.data.vertices[:64]]
        setattr(bpy.context.scene, prop, 0.5)
        assert bpy.ops.pd.evaluate() == {"FINISHED"}
        after = [v.co.copy() for v in source_obj.data.vertices[:64]]
        if after != before:
            moved = True
            print(f"  slider {candidate} moved {source_obj.name}")
            break
    assert moved, "no slider moved the add-on's mesh"
    copy_after = [v.co.copy() for v in copy_obj.data.vertices[:64]]
    assert copy_after == copy_before, "evaluate still drives the detached copy"
    print("INDEPENDENT OK evaluate leaves the detached copy untouched")

    # A detached copy must reproduce the add-on's own shape exactly - including
    # the case where several expression sliders are active at once, which is
    # what catches keys that were measured on top of the other active sliders.
    def evaluated(obj):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated_obj = obj.evaluated_get(depsgraph)
        mesh = evaluated_obj.to_mesh()
        coords = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", coords)
        evaluated_obj.to_mesh_clear()
        return coords.reshape(-1, 3)

    active = [p["id"] for p in _state["pack"].params
              if p["id"].startswith("expr_")][:3]
    for index, param_id in enumerate(active):
        setattr(bpy.context.scene, _state["prop_names"][param_id],
                (0.7, 1.0, 0.35)[index % 3])
    assert bpy.ops.pd.evaluate() == {"FINISHED"}
    assert bpy.ops.pd.detach() == {"FINISHED"}
    detached = {o.name: o for o in bpy.data.objects
                if o.name.startswith("Detached_")}
    face = next((o for o in detached.values()
                 if o.type == "MESH" and o.data.shape_keys), None)
    if face is not None:
        source = next(s for s in (bpy.data.objects[n] for n in sorted(source_objects))
                      if len(s.data.vertices) == len(face.data.vertices))
        gap = np.linalg.norm(evaluated(source) - evaluated(face), axis=1)
        assert gap.max() < 1e-4, (
            "detached copy differs from the add-on by %.4f mm "
            "(shape keys were measured with other sliders active)"
            % (gap.max() * 1000))
        print("SHAPE MATCH OK max gap %.5f mm" % (gap.max() * 1000))

    if face is not None and face.data.shape_keys:
        keys = [k.name for k in face.data.shape_keys.key_blocks]
        assert keys[0] == "Basis", f"first shape key is {keys[0]!r}"
        print(f"SHAPE KEYS {len(keys) - 1}")
        print(f"  first: {keys[1:4]}")
        # Every key must carry the slider value it was detached at - including
        # the sliders sitting at 0 (shape_key_add creates keys at 1.0).
        asserted = 0
        for param in _state["pack"].params:
            if not param["id"].startswith("expr_"):
                continue
            block = face.data.shape_keys.key_blocks.get(param["label"])
            assert block is not None, f"missing shape key {param['label']!r}"
            expected = getattr(bpy.context.scene, _state["prop_names"][param["id"]])
            assert abs(block.value - expected) < 1e-4, (
                f"{param['label']} value {block.value}, "
                f"expected the slider value {expected}")
            asserted += 1
        print(f"SLIDER VALUES OK {asserted} keys match their sliders")

    # Re-running replaces the previous copies instead of piling up .001 names.
    assert bpy.ops.pd.detach() == {"FINISHED"}
    again = {o.name for o in bpy.data.objects if o.name.startswith("Detached_")}
    assert again == set(detached), f"second detach changed the names: {again}"
    print("IDEMPOTENT OK")
    print("DETACH CHECK OK")
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
