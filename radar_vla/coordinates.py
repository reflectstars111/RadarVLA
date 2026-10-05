"""SE(3) geometry with explicit source→destination transforms (column convention)."""
from __future__ import annotations
import numpy as np


def validate_transform(value, name='transform'):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f'{name} must be a finite 4x4 matrix')
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f'{name} must be a homogeneous transform')
    rotation = matrix[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
        raise ValueError(f'{name} rotation must be orthonormal with determinant +1')
    return matrix


def xyz(value, name='coordinates'):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] not in (2, 3) or not np.isfinite(points).all():
        raise ValueError(f'{name} must have finite final dimension 2 or 3')
    if points.shape[-1] == 2:
        points = np.concatenate((points, np.zeros((*points.shape[:-1], 1))), axis=-1)
    return points


def transform_points(points, transform):
    matrix = validate_transform(transform)
    return xyz(points) @ matrix[:3, :3].T + matrix[:3, 3]


def transform_vectors(vectors, transform):
    matrix = validate_transform(transform)
    return xyz(vectors) @ matrix[:3, :3].T


def transform_yaw(yaw, transform):
    yaw = np.asarray(yaw, np.float64)
    if not np.isfinite(yaw).all():
        raise ValueError('yaw must be finite')
    direction = np.stack((np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)), axis=-1)
    rotated = transform_vectors(direction, transform)
    if np.any(np.linalg.norm(rotated[..., :2], axis=-1) < 1e-8):
        raise ValueError('heading is perpendicular to the destination ground plane')
    return np.arctan2(rotated[..., 1], rotated[..., 0])


def sensor_velocity(ego_velocity, yaw_rate, transform):
    """Sensor origin velocity = ego origin velocity + omega × sensor lever arm."""
    matrix = validate_transform(transform)
    if not np.isfinite(yaw_rate):
        raise ValueError('yaw_rate must be finite')
    return xyz(ego_velocity) + np.cross([0., 0., float(yaw_rate)], matrix[:3, 3])
