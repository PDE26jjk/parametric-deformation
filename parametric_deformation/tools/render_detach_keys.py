"""Render detached shape keys next to the sliders they came from.

Usage:
  blender.exe --background --factory-startup --python \
    parametric_deformation/tools/render_detach_keys.py -- <pack_dir> <out_dir> [key ...]

Evaluates the pack, detaches the meshes, hides the add-on's own objects and
renders the detached face once per requested shape key.  With no key names it
renders the detached basis plus three keys that must land in different regions.
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


def frame_camera(obj) -> None:
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    centre = co.mean(0)
    height = max(co[:, 2].max() - co[:, 2].min(), 0.05)
    camera = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = float(height) * 1.6
    camera.location = (float(centre[0]), float(centre[1]) - 0.6, float(centre[2]))
    camera.rotation_euler = (1.5708, 0.0, 0.0)
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = (0.9, 0.0, 0.7)
    scene = bpy.context.scene
    engines = {item.identifier
               for item in scene.render.bl_rna.properties["engine"].enum_items}
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        if candidate in engines:
            scene.render.engine = candidate
            break
    scene.render.resolution_x = 600
    scene.render.resolution_y = 800
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"


def render(path: str) -> None:
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("RENDERED", path)


def main() -> None:
    if "--" not in sys.argv or len(sys.argv) < sys.argv.index("--") + 3:
        raise SystemExit("usage: render_detach_keys.py -- <pack> <out_dir> [key ...]")
    index = sys.argv.index("--")
    pack_dir = os.path.abspath(sys.argv[index + 1])
    out_dir = os.path.abspath(sys.argv[index + 2])
    wanted = sys.argv[index + 3:]
    os.makedirs(out_dir, exist_ok=True)

    parametric_deformation.register()
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    assert bpy.ops.pd.load_pack(pack_path=pack_dir) == {"FINISHED"}
    assert bpy.ops.pd.evaluate() == {"FINISHED"}
    assert bpy.ops.pd.detach() == {"FINISHED"}

    for obj in list(bpy.data.objects):
        if obj.name.startswith("PD_Avatar"):
            bpy.data.objects.remove(obj, do_unlink=True)

    face = bpy.data.objects["Detached_Face"]
    material = bpy.data.materials.new("PreviewMat")
    material.use_nodes = True
    material.node_tree.nodes.get("Principled BSDF").inputs[0].default_value = (
        0.82, 0.74, 0.70, 1.0)
    face.data.materials.clear()
    face.data.materials.append(material)
    frame_camera(face)

    blocks = face.data.shape_keys.key_blocks
    for block in blocks:
        block.value = 0.0
    render(os.path.join(out_dir, "00_basis.png"))

    if not wanted:
        labels = [b.name for b in blocks][1:]
        picked = []
        for token in ("Eyebrow", "Eye.", "Mouth."):
            match = next((n for n in labels if token in n), None)
            if match:
                picked.append(match)
        wanted = picked

    for i, name in enumerate(wanted, 1):
        block = blocks.get(name)
        if block is None:
            print("MISSING KEY", name)
            continue
        for other in blocks:
            other.value = 0.0
        block.value = 1.0
        render(os.path.join(out_dir, "%02d_%s.png" % (i, name.replace(".", "_"))))

    parametric_deformation.unregister()
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
