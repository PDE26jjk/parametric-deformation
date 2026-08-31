"""Headless Blender smoke test for the Parametric Deformation addon (generic).

Usage:
  blender.exe --background --factory-startup --python \
    parametric_deformation/tools/blender_smoke.py -- <pack_dir>
"""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402

import parametric_deformation  # noqa: E402


def _state():
    from parametric_deformation.ui.panels import _state as st

    return st


def main() -> None:
    if "--" not in sys.argv or len(sys.argv) < sys.argv.index("--") + 2:
        raise SystemExit("usage: blender --python blender_smoke.py -- <pack_dir>")
    pack_dir = os.path.abspath(sys.argv[sys.argv.index("--") + 1])

    parametric_deformation.register()
    assert hasattr(bpy.types.Scene, "pd_pack_path"), "scene prop missing"

    result = bpy.ops.pd.load_pack(pack_path=pack_dir)
    assert result == {"FINISHED"}, result
    result = bpy.ops.pd.evaluate()
    assert result == {"FINISHED"}, result
    obj = bpy.data.objects.get("PD_Avatar")
    assert obj is not None, "PD_Avatar mesh object not created"
    print(f"SMOKE OK verts={len(obj.data.vertices)} polys={len(obj.data.polygons)}")

    # Cached-basis evaluation timing (after first load).
    t0 = time.perf_counter()
    for _ in range(20):
        bpy.ops.pd.evaluate()
    elapsed_ms = (time.perf_counter() - t0) / 20 * 1000.0
    print(f"EVAL_MS {elapsed_ms:.1f}")

    # Live update: setting a slider prop refreshes the mesh without an operator.
    scene = bpy.context.scene
    scene.pd_live_update = True
    identity_id = next(
        p["id"] for p in _state()["params"] if p.get("group") == "Identity"
    )
    prop_name = _state()["prop_names"][identity_id]
    before = [tuple(v.co) for v in obj.data.vertices[:5]]
    setattr(scene, prop_name, getattr(scene, prop_name) + 0.5)
    after = [tuple(v.co) for v in obj.data.vertices[:5]]
    assert before != after, "live update did not refresh the mesh"
    print("LIVE OK")

    # Preset save -> change -> restore -> delete round-trip.
    scene.pd_preset_name = "smoke"
    original = getattr(scene, prop_name)
    bpy.ops.pd.save_preset(preset_name="smoke")
    setattr(scene, prop_name, original + 0.5)
    bpy.ops.pd.load_preset(preset_name="smoke")
    assert abs(getattr(scene, prop_name) - original) < 1e-9, "preset restore failed"
    bpy.ops.pd.delete_preset(preset_name="smoke")
    print("PRESET OK")

    parametric_deformation.unregister()
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
