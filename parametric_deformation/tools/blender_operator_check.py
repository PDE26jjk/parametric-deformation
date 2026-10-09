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
import numpy as np  # noqa: E402

import parametric_deformation  # noqa: E402
from parametric_deformation.ui import panels  # noqa: E402
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

    # This check is written against an avatar-style pack: it needs a slider that
    # really moves a separate body mesh.  Skip cleanly for anything else so the
    # same command can run against every pack in CI.
    if ("Arm Length" not in _state["prop_names"]
            or bpy.data.objects.get("PD_Avatar_body") is None):
        print("OPERATOR CHECK SKIP pack has no 'Arm Length' slider / body mesh")
        parametric_deformation.unregister()
        bpy.ops.wm.quit_blender()
        return 0

    failures: list = []
    groups = sorted({p.get("group", "Params") for p in _state["params"]})

    clear_selection()
    failures += run("evaluate (no active object)", bpy.ops.pd.evaluate)
    failures += run("randomize", bpy.ops.pd.randomize)

    # The sliders carry no RNA update callback any more (a Scene property with
    # one makes EEVEE re-bake on every step, see docs/vroid.md 7.4), so a
    # change has to be picked up by the poll timer.  Drive that path directly:
    # set a value outside the "suspending" window, run the poll and the
    # debounced evaluate, and require the mesh to actually move.
    prop = _state["prop_names"]["Arm Length"]
    body = bpy.data.objects["PD_Avatar_body"]
    before = np.empty(len(body.data.vertices) * 3, dtype=np.float64)
    body.data.vertices.foreach_get("co", before)
    setattr(bpy.context.scene, prop, 0.8)
    panels._param_poll()          # the poll evaluates on the spot
    panels._flush_timer()         # (kept as a no-op safety net)
    after = np.empty(len(body.data.vertices) * 3, dtype=np.float64)
    body.data.vertices.foreach_get("co", after)
    moved = float(np.abs(after - before).max())
    if moved > 1e-4:
        print(f"OK   poll picks up a slider change (moved {moved * 1000:.1f} mm)")
    else:
        failures.append(f"poll did not pick up the slider change ({moved})")
        print(f"FAIL poll pickup moved={moved}")
    setattr(bpy.context.scene, prop, 0.0)
    panels._param_poll()
    panels._flush_timer()

    # The rest pose must keep following the shape during a live drag (the
    # original behaviour): the armature bones are rebuilt as part of the same
    # step that rewrites the mesh.
    rig = bpy.data.objects["PD_Rig"]
    head_before = rig.data.bones["J_Bip_L_UpperArm"].head_local.copy()
    setattr(bpy.context.scene, prop, 0.7)
    panels._param_poll()
    panels._flush_timer()
    head_after = rig.data.bones["J_Bip_L_UpperArm"].head_local.copy()
    moved_rest = (head_after - head_before).length
    if moved_rest > 1e-4:
        print(f"OK   rest pose follows the shape while dragging (moved "
              f"{moved_rest * 1000:.2f} mm)")
    else:
        failures.append(f"rest pose did not follow the shape (moved {moved_rest})")
        print(f"FAIL rest pose did not follow the shape ({moved_rest})")
    setattr(bpy.context.scene, prop, 0.0)
    panels._param_poll()
    panels._flush_timer()
    bpy.ops.pd.evaluate()

    # Material-preview rule: while a heavy viewport is open the rest refresh is
    # skipped during a drag (it costs ~1.2 s there) and applied once the
    # shading is cheap again.  Headless has no 3D viewport, so patch the probe.
    original_heavy = panels._heavy_shading
    try:
        panels._heavy_shading = lambda: True
        head_before = rig.data.bones["J_Bip_L_UpperArm"].head_local.copy()
        setattr(bpy.context.scene, prop, 0.9)
        panels._param_poll()
        panels._flush_timer()
        skipped = _state.get("rig_stale") and (
            rig.data.bones["J_Bip_L_UpperArm"].head_local - head_before).length < 1e-9
        panels._heavy_shading = lambda: False
        panels._param_poll()             # poll notices the cheap shading
        panels._flush_timer()
        refreshed = not _state.get("rig_stale") and (
            rig.data.bones["J_Bip_L_UpperArm"].head_local - head_before).length > 1e-4
        if skipped and refreshed:
            print("OK   material-mode drag skips the rest refresh, Solid applies it")
        else:
            failures.append(f"material-mode rest rule wrong ({skipped}, {refreshed})")
            print(f"FAIL material-mode rest rule skipped={skipped} refreshed={refreshed}")
    finally:
        panels._heavy_shading = original_heavy
        setattr(bpy.context.scene, prop, 0.0)
        panels._param_poll()
        panels._flush_timer()
        bpy.ops.pd.evaluate()
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

    # Viewport outline toggle: only meaningful once the action added the
    # outline Geometry Nodes modifiers, and it must survive being pressed twice.
    if _state["pack"].actions:
        failures += run("toggle_outline (off)", bpy.ops.pd.toggle_outline)
        failures += run("toggle_outline (on)", bpy.ops.pd.toggle_outline)

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
