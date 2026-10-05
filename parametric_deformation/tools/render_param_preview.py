"""Headless preview render of a template pack at chosen slider values.

Usage:
  blender.exe --background --factory-startup --python \
    parametric_deformation/tools/render_param_preview.py -- \
    <pack_dir> <out.png> [view] ["Slider=Value" ...]

Views: ``front`` (full body), ``neck`` (side close-up of the throat and
collar, the view the VRoid body morphs need), ``face`` (head close-up).
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402

import parametric_deformation  # noqa: E402
from parametric_deformation.ui.panels import _evaluate_mesh, _set_params  # noqa: E402

# (camera location, look-at target, lens mm) per view.  The avatar faces -Y
# and is about 1.5 m tall, so the neck sits near z = 1.45.
VIEWS = {
    "front": ((0.0, -3.2, 0.85), (0.0, 0.0, 0.80), 50.0),
    "neck": ((0.85, -0.35, 1.48), (0.0, 0.0, 1.44), 60.0),
    "throat": ((0.62, -0.30, 1.47), (0.0, 0.02, 1.44), 80.0),
    "face": ((0.0, -0.75, 1.55), (0.0, 0.0, 1.52), 60.0),
}


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) < 2:
        raise SystemExit("usage: render_param_preview.py -- <pack_dir> <out.png>"
                         " [front|neck|throat|face] [Slider=Value ...]")
    pack_dir, out_png = os.path.abspath(args[0]), os.path.abspath(args[1])
    view = args[2] if len(args) > 2 and args[2] in VIEWS else "front"
    values = {}
    for spec in args[3 if len(args) > 2 and args[2] in VIEWS else 2:]:
        key, _, raw = spec.partition("=")
        values[key] = float(raw)

    parametric_deformation.register()
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    assert bpy.ops.pd.load_pack(pack_path=pack_dir) == {"FINISHED"}
    assert bpy.ops.pd.evaluate() == {"FINISHED"}
    if values:
        _set_params(values)
        ok, message = _evaluate_mesh(bpy.context)
        if not ok:
            raise SystemExit(f"evaluate failed: {message}")

    material = bpy.data.materials.new("PreviewMat")
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs[0].default_value = (0.86, 0.78, 0.73, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.75
    for obj in bpy.data.objects:
        if obj.type == "MESH" and not len(obj.data.materials):
            obj.data.materials.append(material)

    location, target, lens = VIEWS[view]
    camera = bpy.data.objects.new("PreviewCam", bpy.data.cameras.new("PreviewCam"))
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.location = location
    camera.data.lens = lens
    empty = bpy.data.objects.new("PreviewTarget", None)
    bpy.context.scene.collection.objects.link(empty)
    empty.location = target
    track = camera.constraints.new("TRACK_TO")
    track.target = empty
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    for name, direction, energy in (("KeyLight", (0.7, 0.9, 0.4), 3.0),
                                    ("FillLight", (-0.9, 0.6, 0.2), 1.4)):
        light_obj = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
        bpy.context.scene.collection.objects.link(light_obj)
        light_obj.rotation_euler = direction
        light_obj.data.energy = energy

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 900
    scene.render.resolution_y = 1100
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = out_png
    bpy.ops.render.render(write_still=True)
    print(f"RENDERED {out_png}")
    parametric_deformation.unregister()
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
