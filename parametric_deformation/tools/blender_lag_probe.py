"""A few lines to run in Blender: is the lag the UI, or the material?

DROP-IN SNIPPET (Scripting workspace -> New -> paste -> Run Script):

    import bpy, time
    S = bpy.context.scene
    p = "pd_Eye_Size_X"          # any slider property name from the N panel
    sp = next(s for w in bpy.context.window_manager.windows
              for a in w.screen.areas if a.type == "VIEW_3D"
              for s in a.spaces if s.type == "VIEW_3D")

    def run(label, fn):
        t0 = time.perf_counter(); fn()
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
        print(f"{label:22s} {1000 * (time.perf_counter() - t0):7.0f} ms")

    for mode in ("MATERIAL", "SOLID"):
        sp.shading.type = mode
        run(mode + " slider", lambda: setattr(S, p, getattr(S, p) + 0.02))
        run(mode + " cursor", lambda: setattr(
            S.cursor, "location", (S.cursor.location[0] + 0.001, 0, 0)))
        run(mode + " redraw only", lambda: None)

What the output means:
  * slider ~= cursor ~= 1.2 s in MATERIAL and ~= 10-20 ms in SOLID
      -> the cost is the MToon material / EEVEE redraw, not the panel or the UI
         click path (Blender's own 3D cursor shows it too).
  * slider slow but cursor fast
      -> then it *is* our slider/evaluation path and worth another look.

Root cause of the ~1.2 s step (docs/vroid.md 7.4.9/7.4.10): the UI writes a
property, which always calls RNA_property_update() and tags the depsgraph (a
Python assignment skips that via RNA_property_update_check); EEVEE then rebuilds
the MToon material shaders.  A property owned by the *Scene* tags the whole
scene and costs 1.1-2.4 s per step, while the same property owned by the master
collection costs 11-13 ms - which is why the panel binds its widgets there and
keeps the Scene property as a mirror (see panels._host).

Setting ``Preferences > System > Shader Compilation Method`` to ``Subprocess``
was measured at 24 ms once but did not reproduce (1101-1756 ms on a later run
with the identical setup), so it is not a usable workaround.  Blender 5.x
measures 175-280 ms with the same file and script.

This file mirrors the probe used in docs/vroid.md 7.4.8; running the module
itself (blender --python blender_lag_probe.py) does the same measurement.
"""

from __future__ import annotations

import time

import bpy


def _viewport_space():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        return space
    return None


def _run(label: str, fn) -> None:
    t0 = time.perf_counter()
    fn()
    bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    print(f"{label:22s} {1000 * (time.perf_counter() - t0):7.0f} ms")


def main() -> None:
    scene = bpy.context.scene
    props = [p for p in dir(scene)
             if p.startswith("pd_") and not p.startswith("pd_show")]
    prop = "pd_Eye_Size_X" if hasattr(scene, "pd_Eye_Size_X") else (props[0] if props else None)
    if prop is None:
        raise SystemExit("no parametric_deformation slider found on this scene")
    space = _viewport_space()
    original = space.shading.type if space else None
    print(f"prop={prop}  shading={original}  "
          f"shader_compilation={bpy.context.preferences.system.shader_compilation_method}  "
          f"engine={scene.render.engine}")

    try:
        for mode in ("MATERIAL", "SOLID"):
            if space is not None:
                space.shading.type = mode
            _run(f"{mode} slider", lambda: setattr(scene, prop, getattr(scene, prop) + 0.02))
            _run(f"{mode} cursor", lambda: setattr(
                scene.cursor, "location",
                (scene.cursor.location[0] + 0.001, 0.0, 0.0)))
            _run(f"{mode} redraw only", lambda: None)
    finally:
        if space is not None and original is not None:
            space.shading.type = original


if __name__ == "__main__":
    main()
