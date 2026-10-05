"""Headless check: panel operators survive hostile UI contexts.

Usage:
  blender.exe --background --factory-startup --python \
    parametric_deformation/tools/blender_operator_check.py -- <pack_dir>

The randomize / reset / preset buttons rebuild the armature, and that rebuild
used to depend on the ambient context having an active object.  Pressing the
button from a window whose view layer did not resolve our armature raised
"RuntimeError: Operator bpy.ops.object.mode_set.poll() Context missing active
object".  This check runs every panel operator with no active object, with the
avatar deselected and from a foreign view layer, and requires all of them to
finish.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402

import parametric_deformation  # noqa: E402
from parametric_deformation.ui.panels import _state  # noqa: E402


def clear_selection() -> None:
    for obj in bpy.data.objects:
        obj.select_set(False)
    bpy.context.view_layer.objects.active = None


def run(label: str, call) -> list:
    before = _state.get("last_error", "")
    try:
        result = call()
    except RuntimeError as exc:
        print(f"FAIL {label}: raised {exc}")
        return ["raised"]
    error = _state.get("last_error", "")
    if error and error != before:
        print(f"FAIL {label}: {error}")
        return [error]
    if result != {"FINISHED"}:
        print(f"FAIL {label}: {result}")
        return [str(result)]
    print(f"OK   {label}")
    return []


def main() -> int:
    pack_dir = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
    parametric_deformation.register()
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    assert bpy.ops.pd.load_pack(pack_path=pack_dir) == {"FINISHED"}
    assert bpy.ops.pd.evaluate() == {"FINISHED"}

    failures: list = []
    groups = sorted({p.get("group", "Params") for p in _state["params"]})

    clear_selection()
    failures += run("evaluate (no active object)", bpy.ops.pd.evaluate)
    failures += run("randomize", bpy.ops.pd.randomize)
    clear_selection()
    for group in groups:
        failures += run(f"randomize_group {group!r}",
                        lambda g=group: bpy.ops.pd.randomize_group(group=g))
    clear_selection()
    for group in groups:
        failures += run(f"zero_group {group!r}",
                        lambda g=group: bpy.ops.pd.zero_group(group=g))

    clear_selection()
    failures += run("save_preset", lambda: bpy.ops.pd.save_preset(
        preset_name="operator_check"))
    failures += run("load_preset", lambda: bpy.ops.pd.load_preset(
        preset_name="operator_check"))
    failures += run("delete_preset", lambda: bpy.ops.pd.delete_preset(
        preset_name="operator_check"))

    action_ids = [a.get("id") for a in (_state["pack"].actions or [])]
    for action_id in action_ids:
        clear_selection()
        failures += run(f"run_action {action_id!r}",
                        lambda a=action_id: bpy.ops.pd.run_action(action_id=a))

    # A panel click whose context resolves to a view layer that does not contain
    # the armature (second window / different scene).
    other = bpy.data.scenes.new("OperatorCheckScene")
    try:
        with bpy.context.temp_override(scene=other,
                                       view_layer=other.view_layers[0]):
            failures += run("randomize_group 'Face' (foreign view layer)",
                            lambda: bpy.ops.pd.randomize_group(group="Face"))
            failures += run("evaluate (foreign view layer)",
                            bpy.ops.pd.evaluate)
    finally:
        bpy.data.scenes.remove(other)

    print("--- summary")
    for line in failures:
        print("FAIL", line)
    print("OPERATOR CHECK " + ("FAILED" if failures else "OK"))
    parametric_deformation.unregister()
    bpy.ops.wm.quit_blender()
    return 0


if __name__ == "__main__":
    main()
