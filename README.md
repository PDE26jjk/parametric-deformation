# Parametric Deformation (Blender add-on)

A template-pack driven deformation framework for Blender: slider panels are
generated from a pack's parameter table, and evaluation (blend shapes, linear
blend skinning, postprocess hooks) runs through a small adapter the pack
provides.

The add-on ships **no model data**. Every avatar, mesh, morph basis and
skeleton lives in a separate *template pack* that the user installs and points
the add-on at. That split is deliberate: it keeps the add-on itself free of
third-party assets and their licenses.

## Requirements

- Blender 4.5 or newer (developed against 4.5 LTS and 5.2 LTS)
- `numpy` (bundled with Blender's Python)

## Install

As an extension (Blender 4.2+):

```sh
blender --command extension build --source-dir parametric_deformation --output-dir dist
# then Edit > Preferences > Get Extensions > Install from Disk...
```

Or as a legacy add-on: put the `parametric_deformation` directory into your
Blender `scripts/addons/` directory and enable it in the preferences.

For development, do not copy the package into the install directory by hand -
point Blender's script directory at this repository (or link the package into
`scripts/addons/`) so there is only ever one copy of the code.

## Use

1. Open the `PD` tab in the 3D viewport sidebar.
2. Set **Pack Path** to a template pack directory; it loads as soon as the path
   is accepted.
3. Drag sliders - the viewport updates live (see **Live Update**).
4. **Preset Path** points at a JSON file: Save writes the current slider values
   there, and picking a file loads it right away.
5. **Detach (Separate Mesh)** copies the evaluated meshes out of the add-on as
   independent objects. Templates may contribute shape keys; for a template
   that drives expressions from a linear blend-shape basis, one key per slider
   is written and the current slider values are applied to them.

## Template packs

A pack is a directory with a JSON-first layout; see
[docs/pack-format.md](docs/pack-format.md) for the schema and
`format_version` compatibility rules. `examples/minimal_pack` is a generated,
asset-free pack that exercises the whole pipeline and is what the headless
checks below run against.

## Checks

The `parametric_deformation/tools/blender_*_check.py` scripts run inside
Blender and print a `... CHECK OK` line on success:

```sh
blender --background --factory-startup --python \
  parametric_deformation/tools/blender_smoke.py -- examples/minimal_pack
```

Available checks: `blender_smoke` (load, evaluate, live update, preset),
`blender_operator_check`, `blender_rig_check`, `blender_morph_check`,
`blender_detach_check`, `blender_preset_check`, `blender_action_check`.

## License

GPL-2.0-or-later, the same license Blender's own add-on template uses. See
[LICENSE](LICENSE).

Template packs are separate works with their own licenses and are **not**
covered by this repository's license.
