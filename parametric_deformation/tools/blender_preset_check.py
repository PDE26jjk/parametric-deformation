"""Headless check for the Preset Path widget (browse, save, auto load).

Usage:
  blender.exe --background --factory-startup --python \
    parametric_deformation/tools/blender_preset_check.py -- <pack_dir>

The preset field is a file path now: saving writes to it, picking a file in the
file browser loads it without pressing Load (the widget arms a timer, exactly
like the Pack Path field).
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402

import parametric_deformation  # noqa: E402
from parametric_deformation.ui.panels import (  # noqa: E402
    _auto_load_preset,
    _load_pack_path,
    _state,
)


def main() -> None:
    if "--" not in sys.argv or len(sys.argv) < sys.argv.index("--") + 2:
        raise SystemExit("usage: blender --python blender_preset_check.py -- <pack>")
    pack_dir = os.path.abspath(sys.argv[sys.argv.index("--") + 1])

    parametric_deformation.register()
    ok, message = _load_pack_path(pack_dir)
    assert ok, message
    assert bpy.ops.pd.evaluate() == {"FINISHED"}

    scene = bpy.context.scene
    host = scene.collection
    param_id = _state["pack"].params[0]["id"]
    prop = _state["prop_names"][param_id]

    target = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".test_tmp",
                     "preset_path_check.json"))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.exists(target):
        os.remove(target)

    # Save writes to the path in the widget and points the widget at the file.
    setattr(scene, prop, 0.42)
    assert bpy.ops.pd.save_preset(preset_path=target) == {"FINISHED"}
    assert os.path.exists(target), f"save did not create {target}"
    assert os.path.abspath(host.pd_preset_path) == target, (
        f"widget still points at {host.pd_preset_path!r}")
    print("SAVE OK", target)

    # Picking that file must load it on its own: the widget arms the timer.
    setattr(scene, prop, -0.5)
    _state["preset_path"] = None
    host.pd_preset_path = target
    assert bpy.app.timers.is_registered(_auto_load_preset), (
        "Preset Path did not arm the auto-load timer")
    _auto_load_preset()
    # Slider properties are float32, so compare with a matching tolerance.
    assert abs(getattr(scene, prop) - 0.42) < 1e-6, (
        f"auto load left {param_id} at {getattr(scene, prop)}")
    print("AUTO LOAD OK", param_id, "=", getattr(scene, prop))

    # A second pass over the same file is a no-op, not a reload loop.
    setattr(scene, prop, 0.1)
    _auto_load_preset()
    assert abs(getattr(scene, prop) - 0.1) < 1e-6, "same file loaded twice"
    print("NO RELOAD LOOP OK")

    # The button path still works, and a bare name still means presets/<name>.
    assert bpy.ops.pd.load_preset(preset_path=target) == {"FINISHED"}
    assert abs(getattr(scene, prop) - 0.42) < 1e-6
    assert bpy.ops.pd.save_preset(preset_name="preset_path_check_default") == {
        "FINISHED"}
    assert bpy.ops.pd.load_preset(preset_name="preset_path_check_default") == {
        "FINISHED"}
    assert bpy.ops.pd.delete_preset(preset_name="preset_path_check_default") == {
        "FINISHED"}
    print("NAME FALLBACK OK")

    assert bpy.ops.pd.delete_preset(preset_path=target) == {"FINISHED"}
    assert not os.path.exists(target)
    print("DELETE OPERATOR STILL AVAILABLE (no longer a panel button)")
    print("PRESET CHECK OK")
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
