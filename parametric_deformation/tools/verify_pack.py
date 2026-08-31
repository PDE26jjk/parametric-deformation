"""End-to-end verification of the GNM pack against the official GNM forward."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GNM_ROOT = os.path.join(ROOT, "GNM")


def run_compare(pack_dir: str, tolerance: float = 1e-3) -> None:
    sys.path.insert(0, GNM_ROOT)
    from gnm.shape import gnm_numpy  # official reference (numpy backend)

    gnm = gnm_numpy.GNM.from_local(
        version=gnm_numpy.GNMMajorVersion.V3,
        variant=gnm_numpy.GNMVariant.HEAD,
    )

    rng = np.random.default_rng(7)
    identity = rng.normal(0.0, 0.4, size=gnm.identity_dim)
    expression = rng.normal(0.0, 0.3, size=gnm.expression_dim)
    rotations = rng.uniform(-0.15, 0.15, size=(gnm.num_joints, 3))
    translation = rng.uniform(-0.02, 0.02, size=(3,))

    reference = gnm(
        identity=identity,
        expression=expression,
        rotations=rotations,
        translation=translation,
    )

    sys.path.insert(0, ROOT)
    from parametric_deformation.core.config import load_pack, validate_pack
    from parametric_deformation.core.pipeline import run_pipeline
    from parametric_deformation.tools import load_driver

    issues = validate_pack(pack_dir)
    if issues:
        raise SystemExit(f"pack validation failed:\n" + "\n".join(issues))
    pack = load_pack(pack_dir)
    adapter = load_driver(pack_dir)
    adapter.load(pack, pack_dir)

    params = {"semantic_mode": 0}
    params.update({f"identity_{i:03d}": float(v) for i, v in enumerate(identity)})
    params.update({f"expression_{i:03d}": float(v) for i, v in enumerate(expression)})
    pose = {"rotations": rotations.tolist(), "translation": translation.tolist()}

    result = run_pipeline(pack, adapter, params, pose)
    name, vertices, faces, uvs = result.meshes[0]
    diff = np.abs(np.asarray(vertices, dtype=np.float64) - np.asarray(reference))
    print(f"mesh={name} vertices={vertices.shape} faces={faces.shape} max_diff={diff.max():.3e}")
    if diff.max() > tolerance:
        raise SystemExit(f"max diff {diff.max():.3e} exceeds tolerance {tolerance}")
    print("OK: pack evaluation matches the official GNM forward pass")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack", default=os.path.join(ROOT, "templates", "gnm_head")
    )
    args = parser.parse_args()
    run_compare(args.pack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
