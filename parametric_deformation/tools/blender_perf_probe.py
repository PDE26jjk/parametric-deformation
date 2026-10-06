"""Measure the real per-step cost of the panel in *this* Blender session.

Run it from the Scripting workspace (Text Editor -> Open -> Run Script) with a
template pack already loaded, or from a terminal with a visible window:

  blender.exe --factory-startup --python \
    parametric_deformation/tools/blender_perf_probe.py -- <pack_dir>

It reports the active engine / shading mode / samples / raytracing state, then
times full window redraws for:
  idle            - nothing changed (steady frame cost)
  live step       - one slider step through the poll + debounce path
  step, no outline- same with the outline Geometry Nodes modifiers disabled
  step, solid     - same in Solid shading
  explicit eval   - one "Evaluate / Update Mesh" (includes the rest refresh)
and writes the numbers to .previews/perf_probe_<timestamp>.json so they can be
compared across machines.

Everything is restored afterwards (slider value, outline modifiers, shading
mode).
"""

from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from parametric_deformation.ui import panels  # noqa: E402
from parametric_deformation.ui.panels import _state  # noqa: E402


def find_space():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                return area, area.spaces.active
    return None, None


def frame_ms() -> float:
    t0 = time.perf_counter()
    bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    return (time.perf_counter() - t0) * 1000.0


def stats(values) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {"min": float(arr.min()), "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)), "max": float(arr.max())}


def measure_idle(frames: int = 20) -> dict:
    return stats([frame_ms() for _ in range(frames)])


def measure_steps(scene, prop, frames: int = 12, live: bool = True,
                  pause: float = 0.0, undo_push: bool = False) -> dict:
    times = []
    cpu = []
    undo = []
    for i in range(frames):
        t0 = time.perf_counter()
        setattr(scene, prop, 0.05 + 0.6 * (i / max(frames - 1, 1)))
        if live:
            panels._param_poll()      # what the 20 Hz timer does
            panels._flush_timer()     # what the debounce timer does
        else:
            bpy.ops.pd.evaluate()
        cpu.append((time.perf_counter() - t0) * 1000.0)
        if undo_push:
            t0 = time.perf_counter()
            try:
                bpy.ops.ed.undo_push(message="pd perf probe")
                undo.append((time.perf_counter() - t0) * 1000.0)
            except RuntimeError:
                pass
        if pause:
            time.sleep(pause)
        times.append(frame_ms())
    out = stats(times)
    out["cpu_median"] = float(np.median(cpu))
    out["cpu_max"] = float(max(cpu))
    if undo:
        out["undo_push_median"] = float(np.median(undo))
        out["undo_push_max"] = float(max(undo))
    return out


def outline_modifiers():
    return [
        mod
        for obj in bpy.data.objects
        if obj.type == "MESH" and obj.name.startswith("PD_Avatar")
        for mod in obj.modifiers
        if mod.type == "NODES"
    ]


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    pack_dir = os.path.abspath(args[0]) if args else ""

    import parametric_deformation  # noqa: E402

    registered_here = False
    report_action = None
    if _state.get("adapter") is None:
        if not pack_dir:
            raise SystemExit(
                "no pack loaded: pass one, e.g. -- templates/vroid_m00")
        parametric_deformation.register()
        registered_here = True
        assert bpy.ops.pd.load_pack(pack_path=pack_dir) == {"FINISHED"}
        assert bpy.ops.pd.evaluate() == {"FINISHED"}
        # Match the state the user complains about: the pack's material action
        # (VRM/MToon + outline modifiers) applied.
        if any(a.get("id") == "import_materials"
               for a in (_state["pack"].actions or [])):
            bpy.ops.pd.run_action(action_id="import_materials")
            report_action = "import_materials"

    if _state.get("adapter") is None:
        raise SystemExit("no template pack loaded")

    area, space = find_space()
    scene = bpy.context.scene
    render = scene.render
    eevee = getattr(scene, "eevee", None)
    report = {
        "action": report_action,
        "blender": bpy.app.version_string,
        "build": getattr(panels, "BUILD", "?"),
        "pack": os.path.basename(_state["pack"].pack_dir),
        "blend_file": bpy.data.filepath,
        "engine": render.engine,
        "shading": space.shading.type if space else None,
        "overlays": bool(space.overlay.show_overlays) if space else None,
        "studio_light": getattr(space.shading, "studio_light", None) if space else None,
        "use_scene_lights": bool(getattr(space.shading, "use_scene_lights", False)) if space else None,
        "use_scene_world": bool(getattr(space.shading, "use_scene_world", False)) if space else None,
        "show_shadows": bool(getattr(space.shading, "show_shadows", False)) if space else None,
        "show_cavity": bool(getattr(space.shading, "show_cavity", False)) if space else None,
        "viewport_aa": getattr(space.shading, "render_pass" if False else "type", None) if space else None,
        "viewport": [area.width, area.height] if area else None,
        "resolution_percentage": render.resolution_percentage,
        "taa_samples": getattr(eevee, "taa_render_samples", None),
        "use_raytracing": bool(getattr(eevee, "use_raytracing", False)) if eevee else None,
        "use_shadows": bool(getattr(eevee, "use_shadows", False)) if eevee else None,
        "objects": len([o for o in bpy.data.objects if o.type == "MESH"]),
        "lights": len([o for o in bpy.data.objects if o.type == "LIGHT"]),
        "worlds": len(bpy.data.worlds),
        "materials": len(bpy.data.materials),
        "images": len(bpy.data.images),
        "gn_modifiers": len(outline_modifiers()),
        "gpu": (bpy.context.preferences.system.gpu_backend
                if hasattr(bpy.context.preferences.system, "gpu_backend") else None),
    }
    try:
        report["gpu_name"] = bpy.context.preferences.system.gpu_device_name \
            if hasattr(bpy.context.preferences.system, "gpu_device_name") else ""
    except Exception:
        pass

    prop = _state["prop_names"].get("Body Size") or next(
        iter(_state["prop_names"].values()))
    original = getattr(scene, prop)
    original_shading = space.shading.type if space else "SOLID"

    try:
        if space:
            space.shading.type = "MATERIAL"
        for _ in range(40):
            frame_ms()
        report["idle_material"] = measure_idle()
        report["step_material"] = measure_steps(scene, prop, live=True)
        report["step_material_undo"] = measure_steps(
            scene, prop, live=True, frames=6, undo_push=True)

        mods = outline_modifiers()
        if mods:
            for mod in mods:
                mod.show_viewport = False
            for _ in range(10):
                frame_ms()
            report["step_material_no_outline"] = measure_steps(scene, prop, live=True)
            for mod in mods:
                mod.show_viewport = True

        if space:
            space.shading.type = "SOLID"
            for _ in range(10):
                frame_ms()
            report["step_solid"] = measure_steps(scene, prop, live=True)
            space.shading.type = original_shading

        report["step_explicit"] = measure_steps(scene, prop, live=False, frames=6)
    finally:
        setattr(scene, prop, original)
        if space:
            space.shading.type = original_shading
            space.overlay.show_overlays = True
        if registered_here:
            parametric_deformation.unregister()

    out_dir = os.path.join(ROOT, ".previews")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, "perf_probe_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)

    print("=== PD perf probe ===")
    for key in ("blender", "build", "pack", "engine", "shading", "viewport",
                "objects", "materials", "images", "gn_modifiers",
                "taa_samples", "use_raytracing", "use_shadows"):
        print(f"  {key:<18} {report.get(key)}")
    for key in ("idle_material", "step_material", "step_material_no_outline",
                "step_material_undo", "step_solid", "step_explicit"):
        entry = report.get(key)
        if not entry:
            continue
        print(f"  {key:<24} median {entry['median']:7.1f} ms   "
              f"p90 {entry['p90']:7.1f}   max {entry['max']:7.1f}"
              + (f"   cpu {entry['cpu_median']:.1f} ms" if "cpu_median" in entry else "")
              + (f"   undo_push {entry['undo_push_median']:.1f} ms"
                 if "undo_push_median" in entry else ""))
    print(f"  written to {out_path}")


if __name__ == "__main__":
    main()
