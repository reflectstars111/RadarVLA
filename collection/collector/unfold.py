"""Two-dimensional, pose-assisted dense Doppler order hypotheses.

Finite values at weak cells are hypotheses, not ground truth. The separate
reliable mask requires direct agreement of both temporal neighbors.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import map_coordinates
from road_map import radar_pose


SOURCE = {"direct_two_sided": 1, "spatial_neighbor": 3, "zero_order": 5}
SPEED_LIMIT_MPS = 35.0
MIN_PEAK_POWER_FRACTION = 1e-4
MAX_MATCH_COST = 0.5
MIN_MATCH_MARGIN = 0.1


def _warped_azimuth(range_axis, azimuth_axis, center_frame, adjacent_frame, sensor):
    cp, crot = radar_pose(center_frame["ego"]["pose"], sensor)
    np_, nrot = radar_pose(adjacent_frame["ego"]["pose"], sensor)
    r, a = np.meshgrid(range_axis, np.deg2rad(azimuth_axis), indexing="ij")
    local = np.stack([r*np.cos(a), r*np.sin(a), np.zeros_like(r)], axis=0)
    world = np.einsum("ij,jra->ira", crot, local) + np.asarray(cp)[:, None, None]
    delta = world - np.asarray(np_)[:, None, None]
    neighbor = np.einsum("ji,jra->ira", nrot, delta)
    return np.rad2deg(np.arctan2(neighbor[1], neighbor[0]))


def _costs(center_power, other_power, folded, orders, period, axes,
           center_frame, other_frame, sensor):
    range_axis, azimuth_axis = axes["range_m"], axes["azimuth_deg"]
    dt = other_frame["timestamp_s"] - center_frame["timestamp_s"]
    if not 0 < abs(dt) <= 0.5:
        raise ValueError("non-adjacent radar frames")
    az_next = _warped_azimuth(range_axis, azimuth_axis,
                              center_frame, other_frame, sensor)
    ai = (az_next - azimuth_axis[0]) / np.median(np.diff(azimuth_axis))
    rr = range_axis[:, None]
    dr = float(np.median(np.diff(range_axis)))
    center = np.log1p(center_power.astype(float))
    other = np.log1p(other_power.astype(float))
    costs = np.full((*folded.shape, len(orders)), np.inf, dtype=np.float32)
    for j, order in enumerate(orders):
        speed = folded + order * period
        ri = (rr + speed * dt - range_axis[0]) / dr
        valid = ((np.abs(speed) <= SPEED_LIMIT_MPS) & (ri >= 1) &
                 (ri < len(range_axis)-1) & (ai >= 1) & (ai < len(azimuth_axis)-1))
        if not np.any(valid):
            continue
        cost = np.zeros(folded.shape, dtype=float)
        for ro, ao in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            observed = map_coordinates(other, [ri + ro, ai + ao], order=1,
                                       mode="constant", cval=0.0)
            baseline = np.roll(center, (ro, ao), axis=(0, 1))
            cost += np.abs(observed - baseline)
        costs[..., j] = np.where(valid, cost / 5, np.inf)
    return costs


def unfold_frame(index, frames, products, axes, sensor):
    """Return finite dense map and source/reliability arrays for one frame."""
    center = products[index]
    folded = np.asarray(center["doppler_at_peak_ra_mps"], dtype=np.float32)
    power = np.asarray(center["power_peak_ra"], dtype=np.float32)
    if folded.shape != (256, 107) or power.shape != folded.shape:
        raise ValueError("expected 256×107 RA products")
    dv = float(np.median(np.diff(axes["velocity_mps"])))
    period = dv * len(axes["velocity_mps"])
    max_order = int(math.ceil((SPEED_LIMIT_MPS + np.max(np.abs(folded))) / period))
    orders = np.arange(-max_order, max_order + 1, dtype=np.int16)
    stride = 3 if index >= 3 and index + 3 < len(frames) else 1
    side = []
    for neighbor in (index-stride, index+stride):
        if 0 <= neighbor < len(frames):
            if abs(frames[neighbor]["carla_frame"] - frames[index]["carla_frame"]) != stride:
                side.append(None)
            else:
                side.append(_costs(power, products[neighbor]["power_peak_ra"],
                                   folded, orders, period, axes, frames[index],
                                   frames[neighbor], sensor))
        else:
            side.append(None)
    source = np.full(folded.shape, SOURCE["zero_order"], dtype=np.uint8)
    chosen = np.zeros(folded.shape, dtype=np.int8)
    margin = np.zeros(folded.shape, dtype=np.float32)
    reliable = np.zeros(folded.shape, dtype=bool)
    winners = []
    for costs in side:
        if costs is None:
            winners.append(None)
            continue
        idx = np.argmin(costs, axis=-1)
        best = np.take_along_axis(costs, idx[..., None], axis=-1)[..., 0]
        second = np.partition(costs, 1, axis=-1)[..., 1]
        with np.errstate(invalid="ignore"):
            gap = second - best
        unique = (np.isfinite(best) & (best < MAX_MATCH_COST) &
                  (gap > MIN_MATCH_MARGIN) &
                  (power >= float(np.max(power)) * MIN_PEAK_POWER_FRACTION))
        winners.append((orders[idx], best, np.where(np.isfinite(gap), gap, 0), unique))
    if winners[0] is not None and winners[1] is not None:
        a, b = winners
        both = a[3] & b[3] & (a[0] == b[0])
        chosen[both] = a[0][both]
        source[both] = SOURCE["direct_two_sided"]
        margin[both] = np.minimum(a[2][both], b[2][both])
        # Weak noise cells must never become reliable GT by mere finite fill.
        threshold = float(np.max(power)) * 1e-4
        reliable[both] = power[both] >= threshold
    # Uncorroborated single-sided matches are often dominated by multipath.
    # Retain the folded (k=0) finite hypothesis instead of inventing an order.
    # One synchronous vote pass from bilateral direct neighbors only.
    direct = source == SOURCE["direct_two_sided"]
    for ri, ai in np.argwhere(source == SOURCE["zero_order"]):
        r0, r1 = max(0, ri-1), min(256, ri+2)
        a0, a1 = max(0, ai-1), min(107, ai+2)
        votes = chosen[r0:r1, a0:a1][direct[r0:r1, a0:a1]]
        if len(votes) >= 3:
            values, counts = np.unique(votes, return_counts=True)
            if counts.max() >= 3 and (counts == counts.max()).sum() == 1:
                candidate = values[np.argmax(counts)]
                if abs(folded[ri, ai] + candidate * period) <= SPEED_LIMIT_MPS:
                    chosen[ri, ai] = candidate
                    source[ri, ai] = SOURCE["spatial_neighbor"]
    dense = folded + chosen.astype(np.float32) * period
    if not np.isfinite(dense).all():
        raise ValueError("nonfinite dense unfolded Doppler")
    return {"doppler_unfolded_dense_ra_mps": dense.astype(np.float32),
            "ambiguity_order": chosen, "source_code": source,
            "reliable_mask": reliable, "match_margin": margin,
            "fold_period_mps": period}
