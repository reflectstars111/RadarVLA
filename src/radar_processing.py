"""Calibrated RA geometry, explicit-prior Doppler de-aliasing and observation association.

Doppler is positive when receding. It stays in its original sensor radial basis;
warping a velocity raster does not rotate a scalar radial measurement correctly.
"""
from __future__ import annotations
import numpy as np
from .coordinates import transform_points, transform_vectors, validate_transform


def load_numeric(path, *, shape=None, complex_allowed=False, name='array'):
    array = np.load(path, allow_pickle=False)
    if (not isinstance(array, np.ndarray) or not np.issubdtype(array.dtype, np.number)
            or (not complex_allowed and np.iscomplexobj(array)) or not np.isfinite(array).all()
            or (shape is not None and tuple(array.shape) != tuple(shape))):
        raise ValueError(f'{name} must be a finite numeric array with shape {shape}')
    return array


def unfold_doppler(folded, velocity_prior, max_unambiguous_velocity, *, max_prior_error=None):
    """Choose v_folded + 2*v_max*k nearest an independent calibrated velocity prior.

    A prior is mandatory: a single aliased map cannot determine its ambiguity
    integer. The supplied prior must be available at inference time (never future
    GT). Ambiguous half-period ties and excessive prior residuals are masked.
    """
    folded = np.asarray(folded, np.float64)
    vmax = float(max_unambiguous_velocity)
    if velocity_prior is None or not np.isfinite(vmax) or vmax <= 0:
        raise ValueError('unfolding requires a velocity prior and positive unambiguous velocity')
    prior = np.asarray(velocity_prior, np.float64)
    if prior.shape != folded.shape or not np.isfinite(prior).all() or not np.isfinite(folded).all():
        raise ValueError('Doppler prior and folded map must have matching finite shapes')
    if np.any(np.abs(folded) > vmax + 1e-6):
        raise ValueError('folded Doppler exceeds its unambiguous interval')
    error_limit = vmax * .5 if max_prior_error is None else float(max_prior_error)
    if not np.isfinite(error_limit) or not 0 < error_limit < vmax:
        raise ValueError('max_prior_error must lie strictly between zero and unambiguous velocity')
    integer = np.rint((prior - folded) / (2. * vmax))
    unfolded = folded + integer * 2. * vmax
    valid = np.abs(unfolded - prior) <= error_limit
    return unfolded.astype(np.float32), valid


def radar_geometry(range_m, azimuth_rad, poses):
    ranges, angles = np.asarray(range_m, float), np.asarray(azimuth_rad, float)
    rr, aa = np.meshgrid(ranges, angles, indexing='ij')
    los = np.stack((np.cos(aa), np.sin(aa), np.zeros_like(aa)), axis=-1)
    sensor_xyz = rr[..., None] * los
    positions, directions = [], []
    for pose in poses:
        positions.append(transform_points(sensor_xyz, pose)[..., :2])
        directions.append(transform_vectors(los, pose)[..., :2])
    return np.asarray(positions, np.float32), np.asarray(directions, np.float32)


def associate_doppler(power, doppler, ranges, angles, pose, agents, *, doppler_valid=None,
                      min_power=0., min_cells=1):
    """Power-weighted measured Doppler within each GT oriented BEV box.

    This is a training association, not a detector. Overlapping boxes compete for
    each cell by nearest normalized box-center distance to prevent double labels.
    No observed radial target is synthesized from annotation velocities.
    """
    if min_cells < 1 or min_power < 0:
        raise ValueError('association thresholds must be nonnegative and min_cells >= 1')
    power, doppler = np.asarray(power, float), np.asarray(doppler, float)
    xy, grid_los = radar_geometry(ranges, angles, [pose]); xy, grid_los = xy[0], grid_los[0]
    if power.shape != xy.shape[:-1] or doppler.shape != power.shape:
        raise ValueError('radar association shape mismatch')
    valid = np.isfinite(power) & np.isfinite(doppler) & (power > min_power)
    if doppler_valid is not None:
        valid &= np.asarray(doppler_valid, bool)
    count = len(agents)
    observed, mask, confidence = np.zeros(count), np.zeros(count, bool), np.zeros(count)
    projection = np.zeros((count, 2))
    los = np.zeros((count, 2)); sensor_origin = validate_transform(pose)[:2, 3]
    scores = np.full((count, *power.shape), np.inf)
    for i, agent in enumerate(agents):
        delta = xy - np.asarray(agent['position'])
        heading = agent['heading']; c, s = np.cos(heading), np.sin(heading)
        local = delta @ np.array([[c, -s], [s, c]])
        half_size = np.asarray(agent['size']) / 2.
        inside = (np.abs(local) <= half_size + 1e-6).all(-1) & valid
        scores[i, inside] = ((local / half_size) ** 2).sum(-1)[inside]
        direction = np.asarray(agent['position']) - sensor_origin
        los[i] = direction / max(np.linalg.norm(direction), 1e-8)
    if count:
        assignment = scores.argmin(0)
        for i in range(count):
            selected = np.isfinite(scores[i]) & (assignment == i)
            n = int(selected.sum())
            if n < min_cells:
                continue
            weights = power[selected]; values = doppler[selected]
            observed[i] = np.average(values, weights=weights)
            projection[i] = np.average(grid_los[selected], axis=0, weights=weights)
            variance = np.average((values - observed[i]) ** 2, weights=weights)
            # A bounded dispersion diagnostic, not a calibrated correctness probability.
            confidence[i] = 1. / (1. + variance)
            mask[i] = True
    return {'observed': observed.astype(np.float32), 'mask': mask,
            'confidence': confidence.astype(np.float32), 'los': los.astype(np.float32),
            'projection': projection.astype(np.float32)}
