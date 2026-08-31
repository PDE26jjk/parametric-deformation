"""Plain unit tests for the Parametric Deformation core.

Run: python parametric_deformation/tools/test_core.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from parametric_deformation.core import math_utils  # noqa: E402
from parametric_deformation.core.config import (  # noqa: E402
    PackError,
    load_pack,
    validate_pack,
)
from parametric_deformation.core.interfaces import (  # noqa: E402
    MeshResult,
    TemplateAdapter,
)
from parametric_deformation.core.pipeline import run_pipeline  # noqa: E402
from parametric_deformation.core.generic import GenericAdapter  # noqa: E402


def _tmp_dir() -> str:
    """Sandbox-safe temp root inside the project."""
    path = os.path.join(ROOT, ".test_tmp")
    os.makedirs(path, exist_ok=True)
    return path


def _make_test_dir(tag: str) -> str:
    """Create a fresh test directory (tempfile ACLs are unusable in sandbox)."""
    path = os.path.join(_tmp_dir(), f"{tag}_{os.getpid()}")
    os.makedirs(path, exist_ok=True)
    return path


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def make_mini_pack(root: str) -> str:
    """Build a tiny valid pack: 3-vertex triangle, 2 bs entries, 2 joints."""
    pack = os.path.join(root, "mini")
    os.makedirs(os.path.join(pack, "data"), exist_ok=True)
    os.makedirs(os.path.join(pack, "lbs"), exist_ok=True)
    os.makedirs(os.path.join(pack, "meshes"), exist_ok=True)
    template = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    basis = np.array(
        [
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0]],
            [[0, 0, 0.1], [0, 0, 0], [0.1, 0.1, 0]],
        ],
        dtype=np.float32,
    )
    np.save(os.path.join(pack, "data", "template.npy"), template)
    np.save(os.path.join(pack, "data", "basis.npy"), basis)
    np.save(os.path.join(pack, "data", "joints.npy"), np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32))
    np.save(os.path.join(pack, "data", "faces.npy"), np.array([[0, 1, 2]], dtype=np.int64))
    np.savez(
        os.path.join(pack, "lbs", "weights.npz"),
        weights=np.array([[1, 1, 0], [0, 0, 1]], dtype=np.float32),
    )
    _write_json(os.path.join(pack, "manifest.json"), {
        "id": "mini", "name": "Mini", "version": "0.0.1", "format_version": "1",
        "dependencies": [],
    })
    _write_json(os.path.join(pack, "params.json"), [
        {"id": "bs", "label": "Basis", "group": "Shape", "min": -3, "max": 3,
         "default": 0, "step": 0.1, "kind": "linear"},
    ])
    _write_json(os.path.join(pack, "bs", "linear", "bs.bs.json"), {
        "id": "bs", "type": "linear", "target_mesh": "mesh",
        "data": {"file": "data/basis.npy"},
    })
    _write_json(os.path.join(pack, "lbs", "skeleton.json"), {
        "joints": [{"id": "root"}, {"id": "tip"}],
        "parent_indices": [0, 0],
        "positions": {"file": "data/joints.npy"},
    })
    _write_json(os.path.join(pack, "lbs", "bone_deltas.json"), {})
    _write_json(os.path.join(pack, "meshes", "mesh.mesh.json"), {
        "name": "mini", "vertices": {"file": "data/template.npy"},
        "faces": {"file": "data/faces.npy"},
    })
    _write_json(os.path.join(pack, "export_map.json"), {"expressions": {}, "humanoid_bones": {}})
    return pack


def test_config_valid_and_invalid() -> None:
    pack = make_mini_pack(_make_test_dir("cfg"))
    assert validate_pack(pack) == [], validate_pack(pack)
    cfg = load_pack(pack)
    assert cfg.manifest["id"] == "mini"
    assert len(cfg.bs_linear) == 1
    # Break format version -> invalid.
    manifest = os.path.join(pack, "manifest.json")
    with open(manifest, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    payload["format_version"] = "99"
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    assert validate_pack(pack), "format_version mismatch must be reported"


class StubAdapter(TemplateAdapter):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def load(self, pack, pack_dir: str) -> None:
        super().load(pack, pack_dir)

    def evaluate(self, params, pose) -> MeshResult:
        self.calls.append("evaluate")
        return MeshResult(meshes=[("stub", np.zeros((3, 3)), None, None)])

    def on_after_lbs(self, vertices, ctx):
        self.calls.append("on_after_lbs")
        return vertices


def test_interface_stub_hook() -> None:
    adapter = StubAdapter()
    result = adapter.evaluate({}, {})
    assert adapter.calls == ["evaluate"]
    assert result.meshes[0][0] == "stub"


def test_math_axis_angle() -> None:
    r = math_utils.axis_angle_to_rotation_matrix(np.array([0, 0, np.pi / 2]))
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    assert np.allclose(r, expected, atol=1e-9)


def test_math_lbs_identity_pose() -> None:
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    joints = np.array([[0.0, 0, 0], [1, 0, 0]])
    weights = np.array([[1.0, 1, 0], [0, 0, 1]])
    out = math_utils.linear_blend_skinning(
        vertices, joints, np.zeros((2, 3)), np.zeros(3), weights, [0, 0]
    )
    assert np.allclose(out, vertices, atol=1e-9)


def test_math_correctives_zero() -> None:
    regressor = np.zeros((18, 9))
    out = math_utils.compute_pose_correctives(np.zeros((2, 3)), regressor, 3)
    assert np.allclose(out, 0.0)


def test_quat_roundtrip() -> None:
    v = np.array([0.2, -0.3, 0.5])
    q = math_utils.exp_quaternion(v)
    assert np.allclose(math_utils.log_quaternion(q), v, atol=1e-9)


def test_pipeline_minimal() -> None:
    pack_dir = make_mini_pack(_make_test_dir("pipe"))
    pack = load_pack(pack_dir)
    adapter = GenericAdapter()
    adapter.load(pack, pack_dir)
    params = {"bs": np.array([1.0, 2.0])}
    result = run_pipeline(pack, adapter, params, {"rotations": np.zeros((2, 3)), "translation": np.zeros(3)})
    name, vertices, faces, _ = result.meshes[0]
    assert name == "mini"
    assert faces.shape == (1, 3)
    expected = np.array([[0.1, 0, 0.2], [1.0, 0.1, 0], [0.2, 1.2, 0]])
    assert np.allclose(vertices, expected, atol=1e-6)


def main() -> int:
    tests = [
        test_config_valid_and_invalid,
        test_interface_stub_hook,
        test_math_axis_angle,
        test_math_lbs_identity_pose,
        test_math_correctives_zero,
        test_quat_roundtrip,
        test_pipeline_minimal,
    ]
    for fn in tests:
        fn()
        print(f"ok: {fn.__name__}")
    print(f"all {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
