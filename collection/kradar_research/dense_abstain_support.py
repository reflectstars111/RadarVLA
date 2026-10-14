"""Support functions for the K-Radar dense unfolding A pipeline."""
from __future__ import annotations
import csv
import re
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates

CACHE = Path('/storage/data/metaiot_data/lxin/KRadar/visualization_cache/seq1_first50')
POSES = Path('/home/luoxin/data/public_dataset/KRadar/resources/odometry/gt/gt_01.txt')
OFFSET_LIDAR_TO_RADAR = np.array([-2.54, 0.30, 0.70], dtype=np.float64)
EXPECTED_DREA = (64, 256, 37, 107)
SPEED_LIMIT_MPS = 35.0
PATCH = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
         (0, 1), (1, -1), (1, 0), (1, 1))

def read_labels(path: Path) -> tuple[float, dict[int, dict]]:
    lines = path.read_text(encoding='utf-8').splitlines()
    match = re.search(r'timestamp=([0-9.]+)', lines[0])
    if not match:
        raise ValueError(f'missing timestamp: {path}')
    labels = {}
    for line in lines[1:]:
        v = [x.strip() for x in line.split(',')]
        if len(v) < 11:
            raise ValueError(f'bad label: {path}: {line}')
        tid = int(v[2])
        if tid < 0:
            continue
        if tid in labels:
            raise ValueError(f'duplicate track ID {tid}: {path}')
        labels[tid] = dict(kind=v[3], xyz=np.array(list(map(float, v[4:7]))),
                           yaw_deg=float(v[7]), half_length=float(v[8]),
                           half_width=float(v[9]), half_height=float(v[10]))
    return float(match.group(1)), labels


def pose_matrix(row: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :] = row.reshape(3, 4)
    return out


def radar_to_world(xyz_r: np.ndarray, pose: np.ndarray) -> np.ndarray:
    xyz_l = xyz_r - OFFSET_LIDAR_TO_RADAR
    return xyz_l @ pose[:3, :3].T + pose[:3, 3]


def world_to_radar(xyz_w: np.ndarray, pose: np.ndarray) -> np.ndarray:
    xyz_l = (xyz_w - pose[:3, 3]) @ pose[:3, :3]
    return xyz_l + OFFSET_LIDAR_TO_RADAR


def candidate_orders(vf: np.ndarray, period: float, limit: float = SPEED_LIMIT_MPS) -> np.ndarray:
    bound = int(np.ceil((limit + np.max(np.abs(vf))) / period))
    return np.arange(-bound, bound + 1, dtype=np.int16)


def polar_xyz(r: np.ndarray, a_deg: np.ndarray, e_deg: np.ndarray) -> np.ndarray:
    a, e = np.deg2rad(a_deg), np.deg2rad(e_deg)
    ce = np.cos(e)
    return np.column_stack((r * ce * np.cos(a), -r * ce * np.sin(a), r * np.sin(e)))


def sample_patches(log_rae: np.ndarray, rb: np.ndarray, eb: np.ndarray,
                   ab: np.ndarray) -> np.ndarray:
    """Bilinear RA samples at fixed elevation; out-of-view samples are NaN."""
    result = np.empty((len(rb), len(PATCH)), dtype=np.float32)
    for j, (dr, da) in enumerate(PATCH):
        result[:, j] = map_coordinates(log_rae, [rb + dr, eb, ab + da],
                                        order=1, mode="constant", cval=np.nan,
                                        prefilter=False)
    return result


def neighbor_coordinates(r: np.ndarray, a: np.ndarray, e: np.ndarray,
                         pose_c: np.ndarray | None, pose_n: np.ndarray | None,
                         assisted: bool) -> tuple[np.ndarray, np.ndarray]:
    if not assisted:
        return a, e
    assert pose_c is not None and pose_n is not None
    xyz = polar_xyz(r, a, e)
    warped = world_to_radar(radar_to_world(xyz, pose_c), pose_n)
    az = np.rad2deg(np.arctan2(-warped[:, 1], warped[:, 0]))
    el = np.rad2deg(np.arctan2(warped[:, 2], np.hypot(warped[:, 0], warped[:, 1])))
    return az, el


def static_radial_prior(r: np.ndarray, a: np.ndarray, e: np.ndarray,
                        pose_c: np.ndarray, pose_p: np.ndarray,
                        pose_n: np.ndarray, dt: float) -> np.ndarray:
    xyz = polar_xyz(r, a, e)
    world = radar_to_world(xyz, pose_c)
    rp = np.linalg.norm(world_to_radar(world, pose_p), axis=1)
    rn = np.linalg.norm(world_to_radar(world, pose_n), axis=1)
    return (rn - rp) / dt


def read_anchors(path: Path) -> dict[tuple[int, int, int], dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 9297:
        raise ValueError(f"expected 9297 anchor candidates, got {len(rows)}")
    return {(int(x["frame_index"]), int(x["range_bin"]), int(x["azimuth_bin"])): x for x in rows}


def corrected_bins(raw_bins: np.ndarray, n_bins: int = 64) -> np.ndarray:
    if n_bins % 2:
        raise ValueError("an even Doppler length is required")
    return (np.asarray(raw_bins) + n_bins // 2) % n_bins
