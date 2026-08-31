"""Vectorized math helpers (NumPy) shared by template implementations.

The linear algebra follows the standard parametric-model conventions used by
GNM/FLAME-style models: additive blendshape basis, axis-angle joints,
forward-kinematics LBS, and pose-corrective regressors.
"""

from __future__ import annotations

import numpy as np


def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Rodrigues' formula; input (..., 3), output (..., 3, 3)."""
    aa = np.asarray(axis_angle, dtype=np.float64)
    norm_sq = np.sum(np.square(aa), axis=-1, keepdims=True)
    angle = np.sqrt(np.maximum(norm_sq, 1e-8))
    axis = aa / angle
    sin_a, cos_a = np.sin(angle)[..., None], np.cos(angle)[..., None]
    eye = np.broadcast_to(
        np.eye(3, dtype=np.float64), (*aa.shape[:-1], 3, 3)
    )
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    skew = np.stack(
        [
            np.stack([zero, -z, y], axis=-1),
            np.stack([z, zero, -x], axis=-1),
            np.stack([-y, x, zero], axis=-1),
        ],
        axis=-2,
    )
    return eye + sin_a * skew + (1.0 - cos_a) * (skew @ skew)


def sum_basis(template: np.ndarray, basis: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """template (V,3) + einsum('i,ijk->jk', weights, basis)."""
    return np.asarray(template, dtype=np.float64) + np.einsum(
        "i,ijk->jk", np.asarray(weights, dtype=np.float64), basis
    )


def joint_transforms_world(
    joints: np.ndarray,
    rotations: np.ndarray,
    translation: np.ndarray,
    joint_parent_indices: list[int],
) -> np.ndarray:
    """Forward kinematics: world 4x4 transforms for each joint."""
    joints = np.asarray(joints, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    num_joints = len(joint_parent_indices)
    local_rot = axis_angle_to_rotation_matrix(rotations)
    root_t = joints[0] + np.asarray(translation, dtype=np.float64)
    non_root_t = joints[1:] - joints[np.asarray(joint_parent_indices[1:])]
    local_t = np.concatenate([root_t[None], non_root_t], axis=0)
    local = np.concatenate(
        [np.concatenate([local_rot, local_t[..., None]], axis=-1),
         np.zeros((num_joints, 1, 4), dtype=np.float64)], axis=-2
    )
    local[..., -1, -1] = 1.0
    world = [local[0]]
    for i in range(1, num_joints):
        world.append(world[joint_parent_indices[i]] @ local[i])
    return np.stack(world, axis=0)


def linear_blend_skinning(
    vertices: np.ndarray,
    joints: np.ndarray,
    rotations: np.ndarray,
    translation: np.ndarray,
    skinning_weights: np.ndarray,
    joint_parent_indices: list[int],
) -> np.ndarray:
    """Linear blend skinning; matches GNM's convention."""
    vertices = np.asarray(vertices, dtype=np.float64)
    joints = np.asarray(joints, dtype=np.float64)
    world = joint_transforms_world(joints, rotations, translation, joint_parent_indices)
    r_world = world[:, :3, :3]
    t_world = world[:, :3, 3]
    t_bind = t_world - (r_world @ joints[..., None])[..., 0]
    per_vertex_r = np.einsum("jv,jmn->vmn", skinning_weights, r_world)
    per_vertex_t = np.einsum("jv,jm->vm", skinning_weights, t_bind)
    return (per_vertex_r @ vertices[..., None])[..., 0] + per_vertex_t


def compute_pose_correctives(
    rotations: np.ndarray,
    regressor: np.ndarray,
    num_vertices: int,
) -> np.ndarray:
    """Pose-corrective offsets from (R - I) features; shape (V, 3)."""
    rotations = np.asarray(rotations, dtype=np.float64)
    rot = axis_angle_to_rotation_matrix(rotations)
    eye = np.broadcast_to(np.eye(3, dtype=np.float64), rot.shape)
    features = (rot - eye).reshape(-1)  # (J*9,), row-major per joint
    deltas = np.einsum("f,fv->v", features, np.asarray(regressor, dtype=np.float64))
    return deltas.reshape(num_vertices, 3)


def log_quaternion(q: np.ndarray) -> np.ndarray:
    """Logarithm of a unit quaternion (w, x, y, z) -> rotation vector."""
    q = np.asarray(q, dtype=np.float64)
    w, v = q[0], q[1:]
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(v_norm, w)
    return angle * v / v_norm


def exp_quaternion(v: np.ndarray) -> np.ndarray:
    """Exponential of a rotation vector -> unit quaternion (w, x, y, z)."""
    v = np.asarray(v, dtype=np.float64)
    angle = np.linalg.norm(v)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = v / angle
    return np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * axis])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two unit quaternions (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )
