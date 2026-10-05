"""Headless check for pack actions (custom template buttons).

Usage:
  blender.exe --background --factory-startup --python \
    parametric_deformation/tools/blender_action_check.py -- <pack_dir>

Runs the pack's "import_materials" action and verifies that the VRM materials
land on the evaluated meshes without leaving a duplicate model behind.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402

import parametric_deformation  # noqa: E402
from parametric_deformation.ui.panels import _state  # noqa: E402


def main() -> None:
    if "--" not in sys.argv or len(sys.argv) < sys.argv.index("--") + 2:
        raise SystemExit("usage: blender --python blender_action_check.py -- <pack>")
    pack_dir = os.path.abspath(sys.argv[sys.argv.index("--") + 1])

    parametric_deformation.register()
    assert bpy.ops.pd.load_pack(pack_path=pack_dir) == {"FINISHED"}
    assert bpy.ops.pd.evaluate() == {"FINISHED"}
    actions = [a.get("id") for a in _state["pack"].actions]
    print(f"ACTIONS {actions}")
    if not actions:
        print("ACTION CHECK SKIP pack declares no actions")
        bpy.ops.wm.quit_blender()
        return

    avatar_objects = {o.name for o in bpy.data.objects
                      if o.name.startswith("PD_Avatar")}
    objects_before = set(bpy.data.objects)
    slots_before = len(bpy.data.objects["PD_Avatar"].material_slots)

    assert bpy.ops.pd.run_action(action_id=actions[0]) == {"FINISHED"}
    face = bpy.data.objects["PD_Avatar"]
    assert set(bpy.data.objects) == objects_before, "action left extra objects"
    assert {o.name for o in bpy.data.objects} - set() >= avatar_objects
    assert len(face.material_slots) == slots_before, "slot count changed"

    skin = next(m for m in face.data.materials if m and "SKIN" in m.name)
    nodes = len(skin.node_tree.nodes)
    groups = sorted({n.node_tree.name for n in skin.node_tree.nodes
                     if n.type == "GROUP" and n.node_tree})
    assert nodes > 20, f"{skin.name}: only {nodes} nodes, VRM import failed?"
    print(f"MATERIAL OK {skin.name} nodes={nodes}")
    print(f"  node groups: {groups[:3]}")
    images = [i for i in bpy.data.images if i.users and i.size[0]]
    packed = [i for i in images if i.packed_file is not None]
    assert packed, "no packed images came with the materials"
    print(f"IMAGES OK {len(packed)}/{len(images)} packed")

    # Running the action twice must not duplicate anything.
    assert bpy.ops.pd.run_action(action_id=actions[0]) == {"FINISHED"}
    assert set(bpy.data.objects) == objects_before
    assert len(face.material_slots) == slots_before
    print("IDEMPOTENT OK")
    print("ACTION CHECK OK")
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
