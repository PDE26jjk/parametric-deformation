"""Headless check: UV, material slots, imported bones and rig deformation.

Usage:
  blender.exe --background --factory-startup --python
    parametric_deformation/tools/blender_rig_check.py -- <pack_dir>

Verifies that one evaluated pack produces meshes with a UV layer matching the
pack's UV array, one material slot per submesh (meshes/*.mesh.json "slots"),
an armature carrying the pack skeleton plus vertex groups, an idle pose that
reproduces the evaluated mesh exactly, and a local deformation when a limb
bone is rotated.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Euler  # noqa: E402

import parametric_deformation  # noqa: E402
from parametric_deformation.ui.panels import _state  # noqa: E402

MESH_ORDER = ["face", "head", "ear", "body", "hand", "foot"]


def object_vertices(obj) -> np.ndarray:
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def evaluated_vertices(context, obj) -> np.ndarray:
    depsgraph = context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    evaluated.to_mesh_clear()
    return co.reshape(-1, 3)


def check_weights_sum(obj) -> None:
    """Every vertex must still be fully driven by the kept bones."""
    totals = np.zeros(len(obj.data.vertices), dtype=np.float64)
    for vertex in obj.data.vertices:
        for group in vertex.groups:
            totals[vertex.index] += group.weight
    worst = float(np.abs(totals - 1.0).max())
    assert worst < 1e-3, f"{obj.name}: weights not normalized (max {worst})"
    print(f"WEIGHTS OK {obj.name} max_dev={worst:.2e}")


def check_bone_anatomy(arm) -> None:
    """Bone length/direction must be anatomical, not matrix scale/axes.

    Regression guard: the runtime bone matrices carry the bind *scale* in
    their 3x3, so a naive implementation gives every bone the same length
    (~1.05 m) pointing along its matrix Y axis.
    """
    lengths = sorted({round(b.length, 4) for b in arm.data.bones})
    assert len(lengths) >= 10, f"bone lengths look degenerate: {lengths}"
    for bone in arm.data.bones:
        if bone.name != "Root":
            assert bone.length < 0.5, f"{bone.name} is {bone.length:.3f} m long"

    def direction(name):
        bone = arm.data.bones[name]
        head = np.array(bone.head_local)
        return (np.array(bone.tail_local) - head) / max(bone.length, 1e-9)

    spine = direction("J_Bip_C_Spine")
    assert spine[2] > 0.9, f"spine is not upright: {np.round(spine, 3)}"
    arm_dir = direction("J_Bip_L_UpperArm")
    # X is mirrored into the VRM export's convention: the left arm sits at +x.
    assert arm_dir[0] > 0.3 and arm_dir[2] < -0.3, (
        f"left upper arm is not hanging out/down: {np.round(arm_dir, 3)}")
    leg_dir = direction("J_Bip_L_UpperLeg")
    assert leg_dir[2] < -0.9, f"leg is not pointing down: {np.round(leg_dir, 3)}"
    hand = arm.data.bones["J_Bip_L_Hand"]
    assert 0.02 < hand.length < 0.15, f"hand bone {hand.length:.3f} m"
    print(f"ANATOMY OK {len(lengths)} distinct lengths, "
          f"spine={np.round(spine, 2).tolist()} arm={np.round(arm_dir, 2).tolist()}")


def check_uv(obj, pack_dir: str, mesh_name: str) -> None:
    mesh = obj.data
    assert len(mesh.uv_layers) == 1, f"{mesh_name}: missing UV layer"
    loops = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loops)
    got = np.empty(len(mesh.loops) * 2, dtype=np.float64)
    mesh.uv_layers[0].data.foreach_get("uv", got)
    got = got.reshape(-1, 2)
    want = np.load(os.path.join(pack_dir, "data", f"{mesh_name}_uv.npy"))
    error = float(np.abs(got - want[loops]).max())
    assert error < 1e-5, f"{mesh_name}: UV mismatch {error}"
    print(f"UV OK {mesh_name} loops={len(mesh.loops)} err={error:.2e}")


def check_slots(obj, pack_dir: str, mesh_name: str) -> None:
    mesh = obj.data
    path = os.path.join(pack_dir, "meshes", f"{mesh_name}.mesh.json")
    with open(path, encoding="utf-8") as fh:
        slots = json.load(fh).get("slots") or []
    assert len(mesh.materials) == len(slots), (
        f"{mesh_name}: {len(mesh.materials)} slots, expected {len(slots)}")
    if not slots:
        return
    index = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("material_index", index)
    assert len(set(index.tolist())) == len(slots), f"{mesh_name}: unused slots"
    for i, slot in enumerate(slots):
        start = int(slot["tri_start"])
        end = start + int(slot["tri_count"])
        assert set(index[start:end].tolist()) == {i}, f"{mesh_name}: slot {i}"
    names = [m.name for m in mesh.materials]
    print(f"SLOTS OK {mesh_name} n={len(slots)} first={names[:2]}")


def collect_params() -> dict:
    scene = bpy.context.scene
    return {pid: getattr(scene, pname)
            for pid, pname in _state["prop_names"].items()}


def zero_pose() -> dict:
    joints = len(_state["pack"].skeleton.get("joints", []))
    return {"rotations": [[0.0, 0.0, 0.0]] * joints,
            "translation": [0.0, 0.0, 0.0]}


def check_rig(context, pack_dir: str) -> None:
    arm = bpy.data.objects.get("PD_Rig")
    assert arm is not None, "PD_Rig armature missing"
    joints = _state["pack"].skeleton["joints"]
    # Default rig = the 59 joints VRoid itself exports (Root + humanoid +
    # secondary + the two eye bones), not the 200-bone runtime skeleton.
    names = {b.name for b in arm.data.bones}
    assert len(names) == 59, f"expected 59 humanoid bones, got {len(names)}"
    leftover = {n for n in names
                if n.startswith("J_Adj_")
                and n not in ("J_Adj_L_FaceEye", "J_Adj_R_FaceEye")}
    assert not leftover, f"shape bones leaked into the rig: {sorted(leftover)}"
    assert not [n for n in names if n.endswith("_end")], "leaf markers in the rig"
    face = bpy.data.objects["PD_Avatar"]
    modifier = face.modifiers.get("PD_Rig")
    assert modifier is not None and modifier.object is arm, "modifier not bound"
    assert face.parent is arm, "mesh not parented to the rig"
    # VRoid's export weights the face to head + the two eye bones only.
    got = {g.name for g in face.vertex_groups}
    assert got == {"J_Bip_C_Head", "J_Adj_L_FaceEye", "J_Adj_R_FaceEye"}, got
    body = bpy.data.objects["PD_Avatar_body"]
    assert not [g.name for g in body.vertex_groups if "J_Adj_" in g.name], (
        "body still weighted to shape bones")
    print(f"RIG OK bones={len(names)} face_groups={len(got)} "
          f"body_groups={len(body.vertex_groups)}")
    for mesh_name in MESH_ORDER:
        obj = bpy.data.objects.get(
            "PD_Avatar" if mesh_name == "face" else f"PD_Avatar_{mesh_name}")
        if obj is None:
            continue
        print(f"  groups {mesh_name}: {len(obj.vertex_groups)}")
        check_weights_sum(obj)
    chain = []
    bone = arm.data.bones.get("J_Bip_L_LowerArm")
    while bone is not None:
        chain.append(bone.name)
        bone = bone.parent
    print("CHAIN " + " <- ".join(chain))
    assert "J_Bip_L_UpperArm" in chain, "arm chain not flattened"
    check_bone_anatomy(arm)

    for pose_bone in arm.pose.bones:
        assert pose_bone.matrix_basis.is_identity, (
            f"{pose_bone.name}: idle pose is not identity")
    result = _state["adapter"].evaluate(collect_params(), zero_pose())
    for name, verts, _faces, _uv in result.meshes:
        obj_name = "PD_Avatar" if name == "face" else f"PD_Avatar_{name}"
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        error = float(np.abs(object_vertices(obj) - np.asarray(verts)).max())
        assert error < 1e-6, f"{name}: idle rig moved the mesh by {error}"
    print("IDLE OK rig is identity at rest")

    body = bpy.data.objects.get("PD_Avatar_body")
    parts = [bpy.data.objects.get(n) for n in
             ("PD_Avatar_body", "PD_Avatar_hand", "PD_Avatar_foot")]
    parts = [o for o in parts if o is not None]
    if parts:
        points = np.vstack([object_vertices(o) for o in parts])
        low, high = points.min(axis=0), points.max(axis=0)
        for bone_name in ("J_Bip_C_Hips", "J_Bip_L_Hand", "J_Bip_L_Foot",
                          "J_Bip_C_Head"):
            bone = arm.data.bones.get(bone_name)
            if bone is None:
                continue
            head = np.array(bone.head_local)
            inside = bool(((head >= low - 0.05) & (head <= high + 0.05)).all())
            print(f"BONE {bone_name} head={np.round(head, 3).tolist()} "
                  f"inside={inside}")
    pose_bone = arm.pose.bones.get("J_Bip_L_UpperArm")
    if body is None or pose_bone is None:
        print("POSE SKIP no body or upper-arm bone")
        return
    before = object_vertices(body)
    foot = bpy.data.objects.get("PD_Avatar_foot")
    foot_before = evaluated_vertices(context, foot) if foot is not None else None
    pose_bone.rotation_mode = "XYZ"
    pose_bone.rotation_euler = Euler((0.0, 0.0, math.radians(60.0)), "XYZ")
    context.view_layer.update()
    after = evaluated_vertices(context, body)
    moved = np.linalg.norm(after - before, axis=1)
    print(f"POSE OK max={moved.max() * 1000:.1f}mm "
          f"moved={(moved > 1e-5).sum()}/{len(moved)}")
    assert moved.max() > 0.01, "rotating the upper arm did nothing"
    if foot is not None:
        foot_after = evaluated_vertices(context, foot)
        foot_moved = float(np.abs(foot_after - foot_before).max())
        assert foot_moved < 1e-6, (
            f"arm rotation leaked into the foot mesh ({foot_moved:.2e})")
        print("LOCALITY OK foot mesh unaffected by the arm rotation")

    # Resetting the pose and moving a shape slider must rebuild the rest pose.
    pose_bone.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
    context.view_layer.update()
    target = next((p for p in _state["params"]
                   if p.get("label") == "Arm Length"), None)
    if target is None:
        return
    prop = _state["prop_names"][target["id"]]
    head_before = arm.data.bones["J_Bip_L_UpperArm"].head_local.copy()
    context.scene.__setattr__(prop, 1.0)
    start = time.perf_counter()
    bpy.ops.pd.evaluate()
    elapsed = (time.perf_counter() - start) * 1000
    head_after = arm.data.bones["J_Bip_L_UpperArm"].head_local.copy()
    print(f"REST REFRESH OK {elapsed:.1f}ms "
          f"shoulder_moved={(head_after - head_before).length * 1000:.2f}mm")


def main() -> None:
    if "--" not in sys.argv or len(sys.argv) < sys.argv.index("--") + 2:
        raise SystemExit("usage: blender --python blender_rig_check.py -- <pack>")
    pack_dir = os.path.abspath(sys.argv[sys.argv.index("--") + 1])

    parametric_deformation.register()
    assert bpy.ops.pd.load_pack(pack_path=pack_dir) == {"FINISHED"}
    assert bpy.ops.pd.evaluate() == {"FINISHED"}
    # Avatar-shaped expectations (a 'face' mesh carrying UVs and material
    # slots, humanoid bone names): skip instead of failing for other packs.
    face = next((d for d in _state["pack"].mesh_descriptors
                 if d.get("name") == "face"), None)
    if face is None or not face.get("uv"):
        print("RIG CHECK SKIP pack has no avatar-style 'face' mesh with UVs")
        parametric_deformation.unregister()
        bpy.ops.wm.quit_blender()
        return
    for i, name in enumerate(MESH_ORDER):
        obj_name = "PD_Avatar" if i == 0 else f"PD_Avatar_{name}"
        obj = bpy.data.objects.get(obj_name)
        if obj is not None:
            check_uv(obj, pack_dir, name)
            check_slots(obj, pack_dir, name)
    check_rig(bpy.context, pack_dir)
    t0 = time.perf_counter()
    for _ in range(3):
        bpy.ops.pd.evaluate()
    print(f"EVAL_MS {(time.perf_counter() - t0) / 3 * 1000:.1f}")
    print("RIG CHECK OK")
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
