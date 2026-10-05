"""Headless check: blend-shape driven sliders actually deform the mesh.

Usage:
  blender.exe --background --factory-startup --python
    parametric_deformation/tools/blender_morph_check.py -- <pack_dir>

For every binding in ``data/morph_sliders.json`` the slider is pushed to both
ends of its range and the evaluated meshes are compared against the rest pose.
A binding passes when the owning mesh moves (and nothing else moves).
It also cross-checks the eye expressions against the eye bones, so a left/right
naming mistake cannot hide behind a plausible looking face.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import parametric_deformation  # noqa: E402
from parametric_deformation.ui.panels import _evaluate_mesh, _set_params  # noqa: E402

MESH_ORDER = ["face", "head", "ear", "body", "hand", "foot"]


def object_vertices(obj) -> np.ndarray:
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def snapshot() -> dict[str, np.ndarray]:
    out = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        name = obj.name
        if not name.startswith("PD_Avatar"):
            continue
        suffix = name[len("PD_Avatar"):].lstrip("_").lower()
        # The face mesh keeps the bare "PD_Avatar" object name.
        out[suffix or "face"] = object_vertices(obj)
    return out


def main() -> int:
    if "--" not in sys.argv:
        raise SystemExit("usage: blender_morph_check.py -- <pack_dir>")
    pack_dir = os.path.abspath(sys.argv[sys.argv.index("--") + 1])

    parametric_deformation.register()
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    assert bpy.ops.pd.load_pack(pack_path=pack_dir) == {"FINISHED"}
    assert bpy.ops.pd.evaluate() == {"FINISHED"}

    with open(os.path.join(pack_dir, "data", "morph_sliders.json"),
              encoding="utf-8") as fh:
        table = json.load(fh)
    params = json.load(open(os.path.join(pack_dir, "params.json"), encoding="utf-8"))
    limits = {p["id"]: (p["min"], p["max"]) for p in params}

    rest = snapshot()
    print("snapshot keys:", sorted(rest))

    failures: list[str] = []
    # Regression: leaving zero used to switch the whole pack from the legacy
    # V1 path to the runtime-basis path, which shifted the rest shape (M00 head
    # +0.6 mm, F00 face +5.3 mm) the moment the first slider moved.
    _set_params({table["bindings"][0]["slider"]: 1e-6})
    _evaluate_mesh(bpy.context)
    nudged = snapshot()
    worst = max(float(np.abs(nudged[m] - rest[m]).max()) for m in rest)
    status = "OK " if worst < 1e-6 else "FAIL"
    if worst >= 1e-6:
        failures.append(f"rest pose jumps when a slider leaves zero ({worst})")
    print(f"{status} rest path consistency  max drift {worst * 1000:.5f} mm")
    _set_params({table["bindings"][0]["slider"]: 0.0})
    _evaluate_mesh(bpy.context)

    print("--- slider bindings")
    for binding in table["bindings"]:
        slider = binding["slider"]
        low, high = limits.get(slider, (0.0, 100.0))
        side = -1.0 if binding.get("negative_morph") else 1.0
        value = high if side > 0 else low
        _set_params({slider: value})
        ok, message = _evaluate_mesh(bpy.context)
        if not ok:
            failures.append(f"{slider}: evaluate failed ({message})")
            continue
        posed = snapshot()
        moved = {}
        for mesh, base in rest.items():
            if mesh not in posed:
                continue
            delta = np.linalg.norm(posed[mesh] - base, axis=1)
            if delta.max() > 1e-4:
                moved[mesh] = (delta.max(), int((delta > 1e-4).sum()))
        owned = moved.get(binding["mesh"])
        others = [f"{m}:{moved[m][0] * 1000:.3f}mm/{moved[m][1]}v"
                  for m in sorted(moved) if m != binding["mesh"]]
        status = "OK " if owned and not others else "FAIL"
        if status == "FAIL":
            failures.append(f"{slider}: moved={moved}")
        print(f"{status} {slider:<28} -> {binding['mesh']:<5} "
              f"{binding['morph']}@{value:g}  "
              f"max_delta={0.0 if not owned else owned[0] * 1000:.2f} mm "
              f"verts={0 if not owned else owned[1]}  others={others}")
        _set_params({slider: 0.0})
        _evaluate_mesh(bpy.context)

    print("--- eye expression sides")
    armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    names_path = os.path.join(pack_dir, "data", "expression_names.json")
    if armature is not None and os.path.exists(names_path):
        with open(names_path, encoding="utf-8") as fh:
            names = json.load(fh)
        bones = {}
        for name in ("J_Adj_L_FaceEye", "J_Adj_R_FaceEye"):
            bone = armature.data.bones.get(name)
            if bone is not None:
                bones[name] = armature.matrix_world @ bone.head_local
        for index, name in enumerate(names):
            short = name.rsplit(".", 1)[-1]
            if short not in ("JoyL", "JoyR"):
                continue
            _set_params({f"expr_{index:03d}": 1.0})
            _evaluate_mesh(bpy.context)
            posed = snapshot()
            delta = np.linalg.norm(posed["face"] - rest["face"], axis=1)
            idx = np.nonzero(delta > 1e-4)[0]
            if idx.size == 0:
                print(f"FAIL {name}: no face vertices moved")
                failures.append(f"{name}: no movement")
                _set_params({f"expr_{index:03d}": 0.0})
                _evaluate_mesh(bpy.context)
                continue
            weight = delta[idx]
            centroid = np.stack(
                [posed["face"][idx, 0] * weight]).sum() / weight.sum()
            want = "J_Adj_L_FaceEye" if short == "JoyL" else "J_Adj_R_FaceEye"
            other = "J_Adj_R_FaceEye" if short == "JoyL" else "J_Adj_L_FaceEye"
            if want in bones and other in bones:
                near = (abs(centroid - bones[want].x)
                        < abs(centroid - bones[other].x))
                status = "OK " if near else "FAIL"
                if not near:
                    failures.append(f"{name}: moves the wrong eye")
                print(f"{status} {name:<20} moved x={centroid:+.4f}  "
                      f"n={idx.size} max={delta.max() * 1000:.2f}mm  "
                      f"{want} x={bones[want].x:+.4f}  {other} x={bones[other].x:+.4f}")
            _set_params({f"expr_{index:03d}": 0.0})
            _evaluate_mesh(bpy.context)

    print("--- summary")
    if failures:
        for line in failures:
            print("FAIL", line)
        print("MORPH CHECK FAILED")
    else:
        print("MORPH CHECK OK")
    parametric_deformation.unregister()
    bpy.ops.wm.quit_blender()
    return 0


if __name__ == "__main__":
    main()
