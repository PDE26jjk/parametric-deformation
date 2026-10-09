"""Generate ``examples/minimal_pack``.

The pack ships no third-party assets: it is an eight-vertex box, two joints,
one linear blend-shape slider and one bone-translation slider. It exists so
the headless checks and CI have something to run without pulling in model data
that has its own license.

Usage:

    python examples/build_minimal_pack.py
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(HERE, "minimal_pack")
DATA = os.path.join(PACK, "data")


def write_json(relative: str, payload) -> None:
    path = os.path.join(PACK, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")


def main() -> None:
    os.makedirs(DATA, exist_ok=True)

    # A 0.2 m tall box: four bottom corners, four top corners.
    vertices = np.array(
        [
            [-0.1, -0.1, 0.0],
            [0.1, -0.1, 0.0],
            [0.1, 0.1, 0.0],
            [-0.1, 0.1, 0.0],
            [-0.1, -0.1, 0.2],
            [0.1, -0.1, 0.2],
            [0.1, 0.1, 0.2],
            [-0.1, 0.1, 0.2],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ],
        dtype=np.int32,
    )

    # Two joints, one per half of the mesh.
    joints = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]], dtype=np.float32)
    weights = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )

    # Identity slider: bottom corners widen, top corners narrow, so the change
    # is visible on the very first vertices the smoke test samples.
    identity = np.zeros_like(vertices)
    identity[:4, 0] = -0.05
    identity[4:, 0] = 0.05

    np.save(os.path.join(DATA, "vertices.npy"), vertices)
    np.save(os.path.join(DATA, "faces.npy"), faces)
    np.save(os.path.join(DATA, "joints.npy"), joints)
    np.save(os.path.join(DATA, "identity.npy"), identity)
    # The generic adapter reads the skinning weights from lbs/weights.npz.
    os.makedirs(os.path.join(PACK, "lbs"), exist_ok=True)
    np.savez(os.path.join(PACK, "lbs", "weights.npz"), weights=weights)

    write_json(
        "manifest.json",
        {
            "id": "minimal_pack",
            "name": "Minimal Example Pack",
            "version": "1.0.0",
            "format_version": "1",
            "dependencies": ["numpy"],
        },
    )
    write_json(
        "params.json",
        [
            {
                "id": "identity",
                "label": "Identity",
                "group": "Shape",
                "min": 0.0,
                "max": 1.0,
                "default": 0.0,
                "step": 0.01,
                "rank": 1,
                "kind": "linear",
            },
            {
                "id": "stretch",
                "label": "Stretch",
                "group": "Skeleton",
                "min": -1.0,
                "max": 1.0,
                "default": 0.0,
                "step": 0.01,
                "rank": 2,
            },
        ],
    )
    write_json(
        "bs/linear/identity.json",
        {
            "id": "identity",
            "type": "linear",
            "data": {"file": "data/identity.npy"},
        },
    )
    write_json(
        "lbs/skeleton.json",
        {
            "joints": [
                {"id": "Root", "parent": 0},
                {"id": "Top", "parent": 0},
            ],
            "parent_indices": [0, 0],
            "positions": {"file": "data/joints.npy"},
        },
    )
    write_json(
        "lbs/bone_deltas.json",
        {"stretch": [{"bone": "Top", "t": [0.0, 0.0, 0.1]}]},
    )
    write_json(
        "meshes/box.mesh.json",
        {
            "name": "box",
            "vertices": {"file": "data/vertices.npy"},
            "faces": {"file": "data/faces.npy"},
        },
    )
    print("wrote", PACK)


if __name__ == "__main__":
    main()
