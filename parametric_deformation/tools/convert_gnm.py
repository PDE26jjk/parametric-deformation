"""Convert the official GNM Head npz into a format-v1 template config pack.

The emitted pack lives in templates/gnm_head (a separate git repository).
Large numeric arrays are stored as .npy and referenced from JSON descriptors.
The pack's py/ code is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

DEFAULT_NPZ = os.path.join(
    os.path.dirname(__file__), "..", "..", "GNM", "gnm", "shape", "data",
    "versions", "v3_0", "gnm_head.npz",
)
DEFAULT_H5_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "GNM", "gnm", "shape", "data",
    "semantic_sampler",
)
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "..", "templates", "gnm_head")


def _write_npy(path: str, arr: np.ndarray) -> None:
    np.save(path, arr.astype(np.float32, copy=False) if arr.dtype.kind == "f" else arr)


def build_params(names_identity, names_expression) -> list[dict]:
    """Slider schema: 253 identity + 383 expression + semantic controls."""
    params: list[dict] = []
    for i, name in enumerate(names_identity):
        params.append(
            {
                "id": f"identity_{i:03d}",
                "label": str(name),
                "group": "Identity",
                "min": -3.0, "max": 3.0, "default": 0.0, "step": 0.01,
                "kind": "linear", "semantic_alias": None, "mirrored": False,
            }
        )
    for i, name in enumerate(names_expression):
        params.append(
            {
                "id": f"expression_{i:03d}",
                "label": str(name),
                "group": "Expression",
                "min": -3.0, "max": 3.0, "default": 0.0, "step": 0.01,
                "kind": "linear", "semantic_alias": None, "mirrored": False,
            }
        )
    semantic = [
        ("semantic_mode", "Semantic Mode (0=direct, 1=sample)", 0, 1, 0, 1, "semantic_mode"),
        ("gender", "Gender (0=Female, 1=Male)", 0, 1, 0.5, 0.01, "gender"),
        ("ethnicity_0", "Ethnicity: Middle Eastern", 0, 1, 0.0, 0.01, "ethnicity"),
        ("ethnicity_1", "Ethnicity: Asian", 0, 1, 1.0, 0.01, "ethnicity"),
        ("ethnicity_2", "Ethnicity: White", 0, 1, 0.0, 0.01, "ethnicity"),
        ("ethnicity_3", "Ethnicity: Black", 0, 1, 0.0, 0.01, "ethnicity"),
        ("expression_class", "Expression Class", 0, 19, 0, 1, "expression_class"),
        ("expression_intensity", "Expression Intensity", 0, 2, 1.0, 0.01, "expression_intensity"),
        ("seed", "Sampling Seed", 0, 2**31 - 1, 0, 1, "seed"),
    ]
    for pid, label, lo, hi, default, step, alias in semantic:
        params.append(
            {
                "id": pid,
                "label": label,
                "group": "Semantic",
                "min": float(lo), "max": float(hi),
                "default": float(default), "step": float(step),
                "kind": "semantic", "semantic_alias": alias, "mirrored": False,
            }
        )
    return params


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", default=DEFAULT_NPZ)
    parser.add_argument("--h5-dir", default=DEFAULT_H5_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    data_dir = os.path.join(out, "data")
    sem_dir = os.path.join(data_dir, "semantic")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(sem_dir, exist_ok=True)
    os.makedirs(os.path.join(out, "bs", "linear"), exist_ok=True)
    os.makedirs(os.path.join(out, "bs", "semantic"), exist_ok=True)
    os.makedirs(os.path.join(out, "lbs"), exist_ok=True)
    os.makedirs(os.path.join(out, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(out, "postprocess"), exist_ok=True)

    with np.load(args.npz, allow_pickle=False) as data:
        keys = set(data.files)
        _write_npy(os.path.join(data_dir, "template_vertices.npy"), data["template_vertex_positions"])
        _write_npy(os.path.join(data_dir, "identity_basis.npy"), data["vertex_identity_basis"])
        _write_npy(os.path.join(data_dir, "expression_basis.npy"), data["expression_basis"])
        _write_npy(os.path.join(data_dir, "template_joints.npy"), data["template_joint_positions"])
        _write_npy(os.path.join(data_dir, "joint_identity_basis.npy"), data["joint_identity_basis"])
        _write_npy(os.path.join(data_dir, "pose_correctives.npy"), data["pose_correctives_regressor"])
        _write_npy(os.path.join(data_dir, "quads.npy"), data["quads"])
        _write_npy(os.path.join(data_dir, "quad_uvs.npy"), data["quad_uvs"])
        _write_npy(os.path.join(data_dir, "vertex_groups.npy"), data["vertex_groups"])
        np.savez(
            os.path.join(out, "lbs", "weights.npz"),
            skinning_weights=data["skinning_weights"].astype(np.float32),
        )
        joint_names = [str(n) for n in data["joint_names"]]
        parent_indices = [int(p) for p in data["joint_parent_indices"]]
        identity_names = [str(n) for n in data["identity_names"]]
        expression_names = [str(n) for n in data["expression_names"]]
        vertex_group_names = [str(n) for n in data["vertex_group_names"]]
        mesh_component_names = [str(n) for n in data["mesh_component_names"]]

    # Copy semantic sampler models (small, Apache-2.0).
    for src_name in ("identity_decoder_model.h5", "expression_decoder_model.h5"):
        src = os.path.join(args.h5_dir, src_name)
        dst = os.path.join(sem_dir, src_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    json_files = {
        "manifest.json": {
            "id": "gnm_head",
            "name": "GNM Head",
            "version": "0.1.0",
            "format_version": "1",
            "dependencies": ["numpy", "h5py"],
        },
        "params.json": build_params(identity_names, expression_names),
        "bs/linear/identity.bs.json": {
            "id": "identity",
            "type": "linear",
            "target_mesh": "skin",
            "data": {"file": "data/identity_basis.npy"},
            "region": "head",
        },
        "bs/linear/expression.bs.json": {
            "id": "expression",
            "type": "linear",
            "target_mesh": "skin",
            "data": {"file": "data/expression_basis.npy"},
            "region": "face",
        },
        "bs/semantic/samplers.bs.json": {
            "id": "semantic_samplers",
            "type": "semantic",
            "data": {
                "identity_decoder": "data/semantic/identity_decoder_model.h5",
                "expression_decoder": "data/semantic/expression_decoder_model.h5",
            },
        },
        "lbs/skeleton.json": {
            "joints": [{"id": n} for n in joint_names],
            "parent_indices": parent_indices,
            "positions": {"file": "data/template_joints.npy"},
            "joint_identity_basis": {"file": "data/joint_identity_basis.npy"},
        },
        "lbs/bone_deltas.json": {},
        "meshes/template.mesh.json": {
            "name": "gnm_head",
            "vertices": {"file": "data/template_vertices.npy"},
            "faces": {"file": "data/quads.npy"},
            "uvs": {"file": "data/quad_uvs.npy"},
            "vertex_groups": {"file": "data/vertex_groups.npy"},
            "vertex_group_names": vertex_group_names,
            "mesh_component_names": mesh_component_names,
        },
        "postprocess/pose_correctives.step.json": {
            "id": "pose_correctives",
            "type": "corrective",
            "hook": "before_lbs",
            "kind": "generic",
            "impl": "core.postprocess.pose_correctives",
            "data": {"file": "data/pose_correctives.npy"},
        },
        "export_map.json": {"expressions": {}, "humanoid_bones": {}},
    }
    for rel, payload in json_files.items():
        path = os.path.join(out, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"pack written to {out}")
    print(f"joints={len(joint_names)} parents={parent_indices} "
          f"identity={len(identity_names)} expression={len(expression_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
