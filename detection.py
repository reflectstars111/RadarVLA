"""One-stage dense RA detection with metric boxes and per-detection motion.

Local encoder features remain at stride four. Each cell has multiple hypotheses;
metric centers are decoded relative to calibrated current-ego cell positions.
No annotation is read by the head or the confidence/NMS decoder.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .categories import DETECTION_CLASSES
from .geometry import _corners


def cell_centers(batch, height, width):
    """Pool native calibrated XY geometry, preserving sensor translation/rotation."""
    if 'radar_cartesian' in batch:
        xy = batch['radar_cartesian'][:, -1]
    else:
        radius, angle = batch['range_m'][:, :, None], batch['azimuth_rad'][:, None, :]
        xy = torch.stack((radius * angle.cos(), radius * angle.sin()), -1)
        if 'radar_pose' in batch:
            pose = batch['radar_pose'][:, -1]
            xy = torch.einsum('bij,braj->brai', pose[:, :2, :2], xy) + pose[:, None, None, :2, 3]
    return F.adaptive_avg_pool2d(xy.permute(0, 3, 1, 2), (height, width)).permute(0, 2, 3, 1).float()


class DenseRadarDetector(nn.Module):
    def __init__(self, hidden_dim, horizon_steps, slots_per_cell=2):
        super().__init__()
        self.slots, self.horizon = slots_per_cell, horizon_steps
        self.neck = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                                  nn.GroupNorm(math.gcd(8, hidden_dim), hidden_dim), nn.GELU())
        self.geometry = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.context = nn.Linear(hidden_dim * 2, hidden_dim)
        self.object_head = nn.Conv2d(hidden_dim, slots_per_cell, 1)
        self.class_head = nn.Conv2d(hidden_dim, slots_per_cell * len(DETECTION_CLASSES), 1)
        # center offset x/y, log length/width, heading in radians
        self.box_head = nn.Conv2d(hidden_dim, slots_per_cell * 5, 1)
        # vr/vx/vy, future position residuals, future velocity residuals
        self.motion_head = nn.Conv2d(hidden_dim, slots_per_cell * (3 + 4 * horizon_steps), 1)
        nn.init.constant_(self.object_head.bias, math.log(.01 / .99))

    def forward(self, spatial, tokens, ego, batch):
        b, _, h, w = spatial.shape
        anchors = cell_centers(batch, h, w)
        context = self.context(torch.cat((tokens.mean(1), ego), -1))
        features = self.neck(spatial + context[:, :, None, None]
                             + self.geometry(anchors / 100.).permute(0, 3, 1, 2))

        def flat(head, fields):
            value = head(features).reshape(b, self.slots, fields, h, w)
            return value.permute(0, 3, 4, 1, 2).reshape(b, h * w * self.slots, fields).float()

        anchors = anchors[:, :, :, None].expand(-1, -1, -1, self.slots, -1).reshape(b, -1, 2)
        box, motion = flat(self.box_head, 5), flat(self.motion_head, 3 + 4 * self.horizon)
        center = anchors + box[..., :2]
        size = box[..., 2:4].clamp(-5., 5.).exp()
        heading = torch.atan2(box[..., 4].sin(), box[..., 4].cos())
        state = torch.cat((center, motion[..., :3]), -1)
        residual = motion[..., 3:3+2*self.horizon].reshape(b, -1, self.horizon, 2)
        velocity_residual = motion[..., 3+2*self.horizon:].reshape(b, -1, self.horizon, 2)
        future = center[:, :, None] + state[:, :, None, 3:5] * batch['future_times_s'][:, None, :, None] + residual
        return dict(detection_logits=flat(self.object_head, 1).squeeze(-1),
                    detection_class_logits=flat(self.class_head, len(DETECTION_CLASSES)),
                    detection_state=state, detection_size=size, detection_heading=heading,
                    detection_future=future,
                    detection_future_velocity=state[:, :, None, 3:5] + velocity_residual,
                    detection_anchors=anchors)


def rotated_iou(a, b):
    """IoU of [x,y,length,width,yaw] BEV boxes using convex polygon clipping."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if (a.shape != (5,) or b.shape != (5,) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or min(*a[2:4], *b[2:4]) <= 0):
        raise ValueError('boxes require finite x/y/yaw and positive length/width')
    pa, pb = _corners(a[:2], a[4], a[2:4]), _corners(b[:2], b[4], b[2:4])
    if np.any(pa.max(0) < pb.min(0)) or np.any(pb.max(0) < pa.min(0)):
        return 0.

    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]

    polygon = list(pa)
    for start, end in zip(pb, np.roll(pb, -1, axis=0)):
        if not polygon:
            return 0.
        incoming, polygon = polygon, []
        previous = incoming[-1]
        prev_side = cross(end-start, previous-start)
        for point in incoming:
            side = cross(end-start, point-start)
            if (side >= 0) != (prev_side >= 0):
                polygon.append(previous + (point-previous) * (prev_side / (prev_side-side)))
            if side >= 0:
                polygon.append(point)
            previous, prev_side = point, side
    if len(polygon) < 3:
        return 0.
    p = np.asarray(polygon)
    intersection = abs(np.sum(p[:, 0]*np.roll(p[:, 1], -1) - p[:, 1]*np.roll(p[:, 0], -1))) / 2
    return float(np.clip(intersection / max(a[2]*a[3]+b[2]*b[3]-intersection, 1e-12), 0., 1.))


@torch.no_grad()
def decode_detections(output, config, future_times_s=None):
    """Return variable-length semantic detections; counts use surviving boxes only.

    Confidence threshold applies to objectness. Categories below the separate
    class-confidence threshold remain unknown. NMS is class-aware, except that
    unknown hypotheses can be duplicates of a known category. Truncation at any
    capacity limit is explicit. This function never substitutes a GT count.
    """
    if 'detection_logits' not in output:
        raise ValueError('Checkpoint has no dense detector; train a detection-enabled model')
    keys = ('detection_logits', 'detection_class_logits', 'detection_state', 'detection_size',
            'detection_heading', 'detection_future', 'detection_future_velocity')
    values = {k: output[k].detach().float().cpu() for k in keys}
    if any(not torch.isfinite(v).all() for v in values.values()):
        raise ValueError('Nonfinite dense detection output')
    if not (values['detection_size'] > 0).all():
        raise ValueError('Detection box sizes must be positive')
    probabilities = values['detection_logits'].sigmoid()
    class_probabilities = values['detection_class_logits'].softmax(-1)
    if class_probabilities.shape[-1] != len(DETECTION_CLASSES):
        raise ValueError('Detection class vocabulary differs')
    if future_times_s is not None:
        times = torch.as_tensor(future_times_s).float()
        if (times.shape != (probabilities.shape[0], values['detection_future'].shape[2])
                or not torch.isfinite(times).all() or (times <= 0).any() or (times.diff(dim=-1) <= 0).any()):
            raise ValueError('Detection future_times_s must match the positive increasing horizon')
    rows = []
    for b in range(len(probabilities)):
        scores = probabilities[b]
        candidates = (scores >= config.detection_confidence).nonzero().flatten().tolist()
        candidates.sort(key=lambda i: (-float(scores[i]), i))
        truncated = (config.detection_max_candidates is not None
                     and len(candidates) > config.detection_max_candidates)
        candidates = candidates[:config.detection_max_candidates]
        boxes = torch.cat((values['detection_state'][b, :, :2], values['detection_size'][b],
                           values['detection_heading'][b, :, None]), -1).numpy()
        confidence, ids = class_probabilities[b].max(-1)
        labels = [DETECTION_CLASSES[int(ids[i])] if confidence[i] >= config.detection_class_confidence
                  else 'unknown' for i in range(len(scores))]
        keep = []
        for i in candidates:
            if any((labels[i] == labels[j] or 'unknown' in (labels[i], labels[j]))
                   and rotated_iou(boxes[i], boxes[j]) > config.detection_nms_iou for j in keep):
                continue
            keep.append(i)
            if config.detection_max_objects is not None and len(keep) > config.detection_max_objects:
                truncated = True
                keep.pop()
                break
        detections, counts = [], dict.fromkeys((*DETECTION_CLASSES, 'unknown'), 0)
        for local_id, i in enumerate(keep):
            state = values['detection_state'][b, i]
            counts[labels[i]] += 1
            detections.append(dict(local_id=local_id, category=labels[i], confidence=float(scores[i]),
                class_confidence=float(confidence[i]),
                class_probabilities=dict(zip(DETECTION_CLASSES, class_probabilities[b, i].tolist())),
                position_m=state[:2].tolist(), size_m=values['detection_size'][b, i].tolist(),
                heading_rad=float(values['detection_heading'][b, i]), radial_velocity_mps=float(state[2]),
                velocity_mps=state[3:5].tolist(), future_xy=values['detection_future'][b, i].tolist(),
                future_velocity_mps=values['detection_future_velocity'][b, i].tolist(),
                future_times_s=times[b].tolist() if future_times_s is not None else None))
        rows.append(dict(detection_status='available', num_agents=len(detections),
                         num_vehicles=counts['vehicle'], class_counts=counts, detections=detections,
                         detection_truncated=truncated,
                         detection_thresholds=dict(objectness=config.detection_confidence,
                             classification=config.detection_class_confidence, nms_iou=config.detection_nms_iou)))
    return rows
