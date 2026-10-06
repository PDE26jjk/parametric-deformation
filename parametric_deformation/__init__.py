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

    from .ui.panels import CLASSES, _host_updated

    # The widget-facing copies of these flags live on the *collection* the panel
    # binds to (see panels._host), because a Scene-owned property tags the whole
    # scene on every widget change and EEVEE then rebuilds the VRoid MToon
    # material shaders: 1096-2276 ms per step versus 11-13 ms on the collection
    # (docs/vroid.md 7.4.9).  The Scene copies stay authoritative for scripts,
    # tools and older .blend files; panels._sync_host keeps both in step.
    flags = (
        ("pd_pack_path", StringProperty,
         {"name": "Pack Path", "subtype": "FILE_PATH", "default": ""}),
        ("pd_preset_name", StringProperty, {"name": "Preset", "default": "preset"}),
        ("pd_live_update", BoolProperty, {"name": "Live Update", "default": True}),
        ("pd_use_rig", BoolProperty,
         {"name": "Deform With Bones",
          "description": "Let the imported armature deform the mesh; idle pose "
                         "keeps the evaluated shape",
          "default": True}),
        ("pd_filter", StringProperty,
         {"name": "Filter", "description": "Regex filter for sliders",
          "default": ""}),
    )
    for name, ctor, kwargs in flags:
        for cls in (bpy.types.Scene, bpy.types.Collection):
            if hasattr(cls, name):
                delattr(cls, name)
        setattr(bpy.types.Scene, name, ctor(**kwargs))
        host_kwargs = dict(kwargs, update=_host_updated)
        setattr(bpy.types.Collection, name, ctor(**host_kwargs))
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    import bpy

    from .ui.panels import CLASSES, _flush_timer, _param_poll, _state

    # Timers hold the *function object*; a module reload (importlib.reload)
    # replaces the module's names but leaves the registered timer running the
    # old code, which then keeps driving the old evaluation path invisibly.
    if bpy.app.timers.is_registered(_param_poll):
        bpy.app.timers.unregister(_param_poll)
    if bpy.app.timers.is_registered(_flush_timer):
        bpy.app.timers.unregister(_flush_timer)
    _state["poll_timer"] = None
    _state["timer"] = None
    _state["dirty"] = False
    # Stop the pending timers (poll, debounce, rest refresh) from touching
    # deleted scene properties after the add-on is gone.
    _state.update(adapter=None, pack=None, dirty=False, rig_deferred=None,
                  param_snapshot={}, mirror_snapshot={}, suspending=False)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    for pname in list(_state["prop_names"].values()):
        if hasattr(bpy.types.Scene, pname):
            delattr(bpy.types.Scene, pname)
        if hasattr(bpy.types.Collection, pname):
            delattr(bpy.types.Collection, pname)
    for pname in ("pd_pack_path", "pd_preset_name", "pd_live_update",
                  "pd_use_rig", "pd_filter"):
        if hasattr(bpy.types.Scene, pname):
            delattr(bpy.types.Scene, pname)
        if hasattr(bpy.types.Collection, pname):
            delattr(bpy.types.Collection, pname)
    for pname in list(_state["group_props"].values()):
        if hasattr(bpy.types.Scene, pname):
            delattr(bpy.types.Scene, pname)
        if hasattr(bpy.types.Collection, pname):
            delattr(bpy.types.Collection, pname)


if __name__ == "__main__":
    register()
