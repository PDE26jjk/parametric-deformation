"""参数化变形 (Parametric Deformation) - Blender addon."""

from __future__ import annotations

bl_info = {
    "name": "参数化变形 (Parametric Deformation)",
    "author": "cyber_persona project",
    "version": (0, 1, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > PD",
    "description": "Unified parametric deformation template framework with "
                   "user-installed template packs.",
    "category": "Mesh",
}


def register() -> None:
    import bpy
    from bpy.props import BoolProperty, StringProperty

    from .ui.panels import CLASSES

    bpy.types.Scene.pd_pack_path = StringProperty(
        name="Pack Path", subtype="FILE_PATH", default=""
    )
    bpy.types.Scene.pd_preset_name = StringProperty(
        name="Preset", default="preset"
    )
    bpy.types.Scene.pd_live_update = BoolProperty(
        name="Live Update", default=True
    )
    bpy.types.Scene.pd_filter = StringProperty(
        name="Filter", description="Regex filter for sliders", default=""
    )
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    import bpy

    from .ui.panels import CLASSES, _state

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    for pname in list(_state["prop_names"].values()):
        if hasattr(bpy.types.Scene, pname):
            delattr(bpy.types.Scene, pname)
    if hasattr(bpy.types.Scene, "pd_pack_path"):
        del bpy.types.Scene.pd_pack_path
    if hasattr(bpy.types.Scene, "pd_preset_name"):
        del bpy.types.Scene.pd_preset_name
    if hasattr(bpy.types.Scene, "pd_live_update"):
        del bpy.types.Scene.pd_live_update
    if hasattr(bpy.types.Scene, "pd_filter"):
        del bpy.types.Scene.pd_filter
    for pname in list(_state["group_props"].values()):
        if hasattr(bpy.types.Scene, pname):
            delattr(bpy.types.Scene, pname)


if __name__ == "__main__":
    register()
