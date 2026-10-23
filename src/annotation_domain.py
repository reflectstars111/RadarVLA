"""Explicit BEV annotation coverage, independent of radar input extent.

The sector is expressed about the latest radar pose. Missing metadata retains
legacy full-scene semantics; a fifty-metre extent is never inferred from data.
This BEV domain does not certify visibility, class coverage, or model accuracy.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .geometry import _corners


def detection_domain_values(domain):
    """Validate a manifest sector and return numeric values plus availability."""
    if domain is None:
        return np.zeros(3, np.float32), np.asarray(False)
    if not isinstance(domain, dict) or domain.get('type') != 'radar_bev_sector':
        raise ValueError('annotation domain must be an explicit radar_bev_sector')
    values = [domain.get(key) for key in ('range_m', 'azimuth_min_rad', 'azimuth_max_rad')]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) for value in values):
        raise ValueError('annotation domain bounds must be finite numbers')
    radius, lower, upper = values
    if radius <= 0 or not -math.pi <= lower < upper <= math.pi or upper-lower > math.pi:
        raise ValueError('annotation domain requires positive range and an ordered sector of at most pi radians')
    return np.asarray(values, np.float32), np.asarray(True)


def annotation_domain_mask(centers, batch):
    """Return [B,Q] background-supervision coverage for current-ego XY anchors.

    Positive matched annotations must be retained by the caller, including
    boxes crossing the sector boundary with an out-of-sector centre.
    """
    if centers.ndim != 3 or centers.shape[-1] != 2:
        raise ValueError('annotation domain centers must have shape [B,Q,2]')
    mask = torch.ones(centers.shape[:2], dtype=torch.bool, device=centers.device)
    enabled = batch.get('detection_annotation_domain_mask')
    if enabled is None:
        if 'detection_annotation_domain' in batch:
            raise ValueError('annotation domain requires its availability mask')
        return mask
    enabled = enabled.to(device=centers.device, dtype=torch.bool)
    if enabled.shape != centers.shape[:1]:
        raise ValueError('annotation domain mask must have shape [B]')
    if not enabled.any():
        return mask
    limits = batch.get('detection_annotation_domain')
    pose = batch.get('radar_pose')
    if limits is None or limits.shape != (len(centers), 3) or pose is None:
        raise ValueError('annotation domain requires [B,3] bounds and radar_pose')
    limits, pose = limits.to(centers), pose[:, -1].to(centers)
    yaw = torch.atan2(pose[:, 1, 0], pose[:, 0, 0])
    delta = centers - pose[:, None, :2, 3]
    x = delta[..., 0] * yaw.cos()[:, None] + delta[..., 1] * yaw.sin()[:, None]
    y = -delta[..., 0] * yaw.sin()[:, None] + delta[..., 1] * yaw.cos()[:, None]
    angle = torch.atan2(y, x)
    inside = ((torch.hypot(x, y) <= limits[:, None, 0]) &
              (angle >= limits[:, None, 1]) & (angle <= limits[:, None, 2]))
    return mask & (inside | ~enabled[:, None])


def configured_candidate_mask(centers, batch, limits):
    """Observation-only eligibility shared by assignment and decoding.

    Limits come from the checkpoint's fixed sensor coverage, never GT availability.
    Keep this calculation in float32 even under autocast, including translated sensors.
    """
    if 'radar_pose' not in batch:
        raise ValueError('Candidate sector requires calibrated radar_pose')
    if limits.shape != (len(centers), 3) or not torch.isfinite(limits).all():
        raise ValueError('Candidate limits must be finite [B,3]')
    geometry = dict(radar_pose=batch['radar_pose'].float(),
        detection_annotation_domain=limits.float(),
        detection_annotation_domain_mask=torch.ones(len(centers), dtype=torch.bool, device=centers.device))
    with torch.autocast(device_type=centers.device.type, enabled=False):
        return annotation_domain_mask(centers.float(), geometry)


def box_intersects_annotation_domain(center, size, heading, limits, radar_pose):
    """Exact rectangle/2D-sector intersection, retaining boundary-touching boxes.

    First clip the box by the sector's two radial half-planes, then test the
    resulting polygon against the circle. Edge intersections count even when
    neither the centre nor any original corner lies in the annotated sector.
    """
    radius, lower, upper = np.asarray(limits, float)
    pose = np.asarray(radar_pose, float)
    yaw = math.atan2(pose[1, 0], pose[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    polygon = (_corners(center, heading, size) - pose[:2, 3]) @ np.array([[c, -s], [s, c]])
    for angle, sign in ((lower, 1.), (upper, -1.)):
        normal = sign * np.array([-math.sin(angle), math.cos(angle)])
        incoming, clipped = polygon, []
        if not len(incoming):
            return False
        previous = incoming[-1]
        prev_side = float(previous @ normal)
        for point in incoming:
            side = float(point @ normal)
            if (side >= 0.) != (prev_side >= 0.):
                clipped.append(previous + (point-previous) * (prev_side/(prev_side-side)))
            if side >= -1e-10:
                clipped.append(point)
            previous, prev_side = point, side
        polygon = np.asarray(clipped)
    if not len(polygon):
        return False
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = end-start
        fraction = np.clip(-float(start @ edge) / max(float(edge @ edge), 1e-20), 0., 1.)
        if np.linalg.norm(start + fraction*edge) <= radius + 1e-8:
            return True
    return False


def ignored_region_mask(points, batch, *, ensure_nearest=False):
    """Ignored bbox interiors OR the existing 2m centre-matching neighbourhood.

    Training also ignores the nearest anchor cell for tiny boxes with no anchor
    in the region. Matched retained positives always override this background
    mask. Evaluation never uses the nearest-anchor extension.
    """
    boxes = batch.get('detection_ignore_boxes')
    known = batch.get('detection_ignore_mask')
    result = torch.zeros(points.shape[:2],dtype=torch.bool,device=points.device)
    if boxes is None or known is None or boxes.shape[1] == 0:
        return result
    boxes=boxes.to(points);known=known.to(device=points.device,dtype=torch.bool)
    if boxes.ndim!=3 or boxes.shape[-1]!=5 or known.shape!=boxes.shape[:2] or len(boxes)!=len(points):
        raise ValueError('Ignored regions require [B,I,5] boxes and [B,I] masks')
    margin=batch.get('detection_ignore_margin_m',points.new_full((len(points),),2.)).to(points)
    if margin.shape!=(len(points),) or not bool(torch.isfinite(margin).all()) or bool((margin<0).any()):
        raise ValueError('Ignored region margin must be finite nonnegative [B]')
    delta=points[:,:,None]-boxes[:,None,:,:2]
    yaw=boxes[:,None,:,4]
    x=delta[...,0]*yaw.cos()+delta[...,1]*yaw.sin()
    y=-delta[...,0]*yaw.sin()+delta[...,1]*yaw.cos()
    distance=torch.linalg.vector_norm(delta,dim=-1)
    inside=(x.abs()<=boxes[:,None,:,2]/2)&(y.abs()<=boxes[:,None,:,3]/2)
    region=inside|(distance<=margin[:,None,None])
    if ensure_nearest and points.shape[1]:
        region=region|(distance<=distance.amin(dim=1,keepdim=True)+1e-6)
    return (region&known[:,None]).any(dim=-1)
