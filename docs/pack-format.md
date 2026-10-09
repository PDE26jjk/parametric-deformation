# Template pack format (format_version 1)

A template pack is a directory that the add-on loads through **Pack Path**. The
schema checker lives in `parametric_deformation/core/config.py`; this document
is the contract packs can rely on.

## Compatibility

`manifest.json` must declare `"format_version": "1"`. The add-on refuses any
other value, so a breaking change to this document means bumping the string and
teaching the loader to accept both. Additive changes (new optional keys, new
optional files) keep the version.

## Layout

```
<pack>/
├─ manifest.json           # required: id, name, version, format_version
├─ params.json             # optional: slider table
├─ bs/linear/*.json        # optional: linear blend-shape bases
├─ bs/semantic/*.json      # optional: semantic bases
├─ lbs/skeleton.json       # joints + parent indices + bind transforms
├─ lbs/bone_deltas.json    # optional: slider -> per-bone translation deltas
├─ lbs/semantic_map.json   # optional
├─ lbs/weights.npz         # optional: (joints, vertices) skinning weights
├─ meshes/*.mesh.json      # one descriptor per mesh
├─ materials.json          # optional: solid-colour / slot metadata
├─ postprocess/*.json      # optional: pipeline steps
├─ export_map.json         # optional
└─ py/driver.py            # optional: the pack's TemplateAdapter subclass
```

## manifest.json

```json
{
  "id": "my_pack",
  "name": "My Pack",
  "version": "0.1.0",
  "format_version": "1",
  "dependencies": ["numpy"],
  "actions": [
    {"id": "import_materials", "label": "Import materials",
     "type": "import_vrm_materials", "data": "models/model.vrm"}
  ]
}
```

`actions` become buttons in the panel. Each entry needs either a built-in
`type` (see `parametric_deformation/core/actions.py`) or a pack-owned
`"handler": "py.module:function"`; the handler is called with
`pack`, `action` and `ctx` (which carries `objects` - the evaluated meshes by
name - and `adapter`).

## params.json

A list of slider descriptors:

```json
[{"id": "Arm Length", "label": "Arm Length", "group": "Body",
  "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01, "rank": 1}]
```

- `id` is the key the driver sees in `params`; it must be unique.
- `group` and `rank` drive the collapsible panel sections.
- `kind` is `"linear"` or `"semantic"` when the parameter feeds a blend-shape
  basis; the matching `bs/` entry id is the same string.

## Data references

Anything that points at numeric data uses one of:

```json
"vertices": {"file": "data/vertices.npy"}
"region":   {"file": "data/merged.npz", "key": "face", "index": 0, "count": 12}
```

`.npy` and `.npz` are supported. With `index`/`count` a slice of the array is
used, which lets several small references share one merged file.

## meshes/*.mesh.json

```json
{"name": "face",
 "vertices": {"file": "data/face_vertices.npy"},
 "faces":    {"file": "data/face_faces.npy"},
 "uvs":      {"file": "data/face_uv.npy"},
 "slots": [{"material": "Face_SKIN", "tri_start": 0, "tri_count": 120}]}
```

The mesh `name` decides the object name in Blender: the first mesh in the
evaluated result becomes `PD_Avatar`, the others `PD_Avatar_<name>`. The detach
button renames its copies `Detached_<Name>`.

## py/driver.py

The pack may ship a driver. It is imported as `_pd_template.driver` and must
expose a class deriving from `parametric_deformation.core.interfaces.
TemplateAdapter`; the first matching class is instantiated. Drivers may import
`parametric_deformation...` even when the add-on runs as an extension, because
the package publishes an alias for that name.

Useful overrides:

| Hook | Purpose |
|---|---|
| `load(pack, pack_dir)` | load pack data; call `super().load(...)` first |
| `evaluate(params, pose)` | full evaluation (default runs the generic pipeline) |
| `resolve_semantic` / `linear_bs_sum` | blend-shape stages |
| `compute_joints` / `lbs_skin` | skeleton stage |
| `on_before_lbs` / `on_after_lbs` | patch vertices or joints mid-pipeline |
| `detach_shape_keys()` | shape keys to bake when the user detaches |
| `export_map()` | export mapping metadata |

Packs without a driver fall back to `GenericAdapter`, which is enough for a
single-mesh pack driven by a linear basis plus optional bone deltas.

`detach_shape_keys()` returns
`[{"name": <shape key name>, "params": {<param id>: <value>}}, ...]`. The
detach button evaluates each entry from the basis (that parameter switched on,
every other shape-keyed parameter at zero) and writes the difference as a shape
key, so the keys stay correct when several sliders are active at once.
