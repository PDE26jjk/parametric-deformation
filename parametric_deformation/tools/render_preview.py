"""Headless EEVEE preview render of a template pack (generic).

Usage:
  blender.exe --background --factory-startup --python \
    parametric_deformation/tools/render_preview.py -- <pack_dir> <out.png>
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402

import parametric_deformation  # noqa: E402


def main() -> None:
    if "--" not in sys.argv or len(sys.argv) < sys.argv.index("--") + 3:
        raise SystemExit("usage: render_preview.py -- <pack_dir> <out.png>")
    i = sys.argv.index("--")
    pack_dir = os.path.abspath(sys.argv[i + 1])
    out_png = os.path.abspath(sys.argv[i + 2])

    parametric_deformation.register()
    assert bpy.ops.pd.load_pack(pack_path=pack_dir) == {"FINISHED"}
    assert bpy.ops.pd.evaluate() == {"FINISHED"}

    material = bpy.data.materials.new("PreviewMat")
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs[0].default_value = (0.82, 0.74, 0.7, 1.0)
    for obj in bpy.data.objects:
        if obj.name.startswith("PD_Avatar") and obj.data:
            obj.data.materials.append(material)

    camera = bpy.data.objects.new("PreviewCam", bpy.data.cameras.new("PreviewCam"))
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.location = (0.0, -2.8, 1.05)
    camera.rotation_euler = (1.5708, 0.0, 0.0)

    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = (0.9, 0.0, 0.7)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 800
    scene.render.resolution_y = 1100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = out_png
    bpy.ops.render.render(write_still=True)
    print(f"RENDERED {out_png}")
    parametric_deformation.unregister()
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
