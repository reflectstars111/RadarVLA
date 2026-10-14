"""Dense radar lane instances with metric polylines and learned lateral topology.

There is one hypothesis per encoder spatial cell, with no fixed lane-count cap.
Topology scores are evaluated only after assignment or instance selection, so
the encoder never allocates a matrix over every pair of radar cells. Inference
reads model predictions only; map annotations are used exclusively by the loss.
"""
import math

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from .detection import cell_centers


class DenseLaneDetector(nn.Module):
    def __init__(self, hidden_dim, samples=16):
        super().__init__()
        if samples < 2:
            raise ValueError('Lane polylines need at least two samples')
        self.samples = samples
        self.relation_dim = max(4, hidden_dim // 4)
        self.neck = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(math.gcd(8, hidden_dim), hidden_dim), nn.GELU())
        self.anchor_embedding = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(),
                                              nn.Linear(hidden_dim, hidden_dim))
        self.context = nn.Linear(hidden_dim * 2, hidden_dim)
        self.object_head = nn.Conv2d(hidden_dim, 1, 1)
        self.geometry_head = nn.Conv2d(hidden_dim, 3 * samples * 2, 1)
        self.boundary_head = nn.Conv2d(hidden_dim, 2, 1)
        self.adjacency_source = nn.Conv2d(hidden_dim, 2 * self.relation_dim, 1)
        self.adjacency_target = nn.Conv2d(hidden_dim, self.relation_dim, 1)
        self.adjacency_none = nn.Conv2d(hidden_dim, 2, 1)
        self.geometry_embedding = nn.Sequential(nn.Linear(3 * samples * 2, hidden_dim),
                                                nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        nn.init.constant_(self.object_head.bias, math.log(.01 / .99))
        nn.init.constant_(self.boundary_head.bias, math.log(.01 / .99))
        nn.init.constant_(self.adjacency_none.bias, math.log(.01 / .99))

    def forward(self, spatial, tokens, ego, batch):
        b, _, h, w = spatial.shape
        anchors = cell_centers(batch, h, w)
        context = self.context(torch.cat((tokens.mean(1), ego), -1))
        features = self.neck(spatial + context[:, :, None, None]
            + self.anchor_embedding(anchors / 100.).permute(0, 3, 1, 2))

        def flat(head):
            return head(features).flatten(2).transpose(1, 2).float()

        residual = flat(self.geometry_head).reshape(b, h * w, 3, self.samples, 2)
        points = anchors.reshape(b, h * w, 1, 1, 2) + residual
        lane_features = (features.flatten(2).transpose(1, 2)
                         + self.geometry_embedding(points.flatten(2) / 100.))
        return dict(lane_logits=flat(self.object_head).squeeze(-1), lane_points=points,
            lane_boundary_logits=flat(self.boundary_head),
            lane_adjacency_source=flat(self.adjacency_source).reshape(b, h * w, 2, self.relation_dim),
            lane_adjacency_target=flat(self.adjacency_target),
            lane_adjacency_none_logits=flat(self.adjacency_none), lane_features=lane_features)


def _pair_logits(source, target):
    return torch.einsum('isd,jd->ijs', source, target) / math.sqrt(target.shape[-1])


def _masked_mean(value, mask, zero):
    return value[mask].mean() if mask.any() else zero


def lane_losses(output, batch, weights=None):
    """Match every observed lane and supervise only annotated geometry/topology.

    Missing map coverage, boundaries and lateral relations are unknown, not
    negative labels. Explicit null boundaries can be supervised through
    ``lane_boundary_known``. Known null neighbors mark the full adjacency row
    (including its diagonal), preserving that information for singleton scenes.
    Boundary polyline direction is immaterial; centerline direction is retained.
    """
    cfg = dict(lane_objectness=1., lane_geometry=1., lane_boundary=1., lane_adjacency=1.,
               lane_scale_position=10.)
    unknown = set(weights or {}) - set(cfg)
    if unknown:
        raise ValueError(f'Unknown lane loss weights: {sorted(unknown)}')
    cfg.update(weights or {})
    if any(not math.isfinite(v) or v < 0 for v in cfg.values()) or cfg['lane_scale_position'] <= 0:
        raise ValueError('Lane loss weights must be finite and nonnegative, with positive scale')
    zero = sum(value.sum() * 0 for key, value in output.items()
               if key.startswith('lane_') and torch.is_tensor(value) and value.is_floating_point())
    losses = {key: zero for key in cfg if key != 'lane_scale_position'}
    logits = output['lane_logits']
    if 'lane_mask' not in batch:
        return dict(losses, lane_total=zero)
    points, truth = output['lane_points'], batch['lane_points']
    active = batch['lane_mask'].bool()
    supervision = batch.get('lane_supervision_mask', torch.zeros(logits.shape[0],
                                  device=logits.device, dtype=torch.bool)).bool()
    valid = batch['lane_point_mask'].bool() & torch.isfinite(truth).all(-1)
    if (truth.ndim != 5 or truth.shape[2:] != points.shape[2:] or active.shape != truth.shape[:2]
            or valid.shape != truth.shape[:-1] or supervision.shape != logits.shape[:1]):
        raise ValueError('Lane labels must match [B,L,3,S,2] geometry and its masks')
    object_target = torch.zeros_like(logits)
    geometry_losses, boundary_losses, relation_losses = [], [], []
    scale = cfg['lane_scale_position']
    for b in range(points.shape[0]):
        if not supervision[b]:
            continue
        targets = active[b].nonzero().flatten()
        if len(targets) > points.shape[1]:
            raise ValueError('More annotated lanes than spatial candidates; increase radar spatial resolution')
        if not len(targets):
            continue
        if (valid[b, targets, 0].sum(-1) < 2).any():
            raise ValueError('Each active lane requires at least two finite centerline points')
        # Canonical ordering makes matching ties independent of annotation order.
        targets = torch.tensor(sorted(targets.tolist(), key=lambda t:
            tuple(torch.nan_to_num(truth[b, t]).flatten().tolist())), device=points.device)
        safe_truth = torch.where(valid[b, targets, :, :, None], truth[b, targets], 0.)
        with torch.no_grad():
            error = (points[b, :, None, 0] - safe_truth[None, :, 0]).abs().sum(-1)
            cost = ((error * valid[b, None, targets, 0]).sum(-1)
                    / valid[b, targets, 0].sum(-1)[None])
            predicted, assigned = linear_sum_assignment(cost.detach().cpu().numpy())
        pi = torch.as_tensor(predicted, device=points.device)
        ti = targets[torch.as_tensor(assigned, device=points.device)]
        object_target[b, pi] = 1.
        target = torch.where(valid[b, ti, :, :, None], truth[b, ti], 0.)
        prediction = points[b, pi]
        mask = valid[b, ti]
        for part in range(3):
            direct = F.smooth_l1_loss(prediction[:, part] / scale, target[:, part] / scale,
                                     reduction='none').mean(-1)
            if part:
                reverse = F.smooth_l1_loss(prediction[:, part] / scale,
                    target[:, part].flip(-2) / scale, reduction='none').mean(-1)
                normal_loss = (direct * mask[:, part]).sum(-1) / mask[:, part].sum(-1).clamp_min(1)
                reversed_loss = (reverse * mask[:, part].flip(-1)).sum(-1) / mask[:, part].sum(-1).clamp_min(1)
                chosen = torch.minimum(normal_loss, reversed_loss)
            else:
                chosen = (direct * mask[:, part]).sum(-1) / mask[:, part].sum(-1).clamp_min(1)
            observed = mask[:, part].any(-1)
            if observed.any():
                geometry_losses.append(chosen[observed])
        boundary_present = mask[:, 1:].any(-1)
        known = batch.get('lane_boundary_known')
        boundary_known = known[b, ti].bool() if known is not None else boundary_present
        if boundary_known.any():
            bce = F.binary_cross_entropy_with_logits(output['lane_boundary_logits'][b, pi],
                boundary_present.float(), reduction='none')
            boundary_losses.append(bce[boundary_known])
        if 'lane_adjacency' in batch and 'lane_adjacency_mask' in batch:
            labels = batch['lane_adjacency'][b][ti][:, ti]
            known = batch['lane_adjacency_mask'][b][ti][:, ti].bool()
            if labels.shape != (len(pi), len(pi), 2) or known.shape != labels.shape:
                raise ValueError('Lane adjacency and mask must have shape [B,L,L,2]')
            if ((labels[known] < 0) | (labels[known] > 1) | ~torch.isfinite(labels[known])).any():
                raise ValueError('Known lane adjacency targets must lie in [0,1]')
            diagonal = torch.eye(len(pi), dtype=torch.bool, device=points.device)[..., None]
            edges_known = known & ~diagonal
            pair = _pair_logits(output['lane_adjacency_source'][b, pi],
                                output['lane_adjacency_target'][b, pi])
            safe_labels = torch.where(known, labels, 0.)
            if edges_known.any():
                bce = F.binary_cross_entropy_with_logits(pair, safe_labels, reduction='none')
                relation_losses.append(bce[edges_known])
            any_neighbor = ((safe_labels > .5) & edges_known).any(1)
            row_known = known.any(1) & (known | diagonal).all(1)
            none_known = any_neighbor | row_known
            if none_known.any():
                bce = F.binary_cross_entropy_with_logits(output['lane_adjacency_none_logits'][b, pi],
                    (~any_neighbor).float(), reduction='none')
                relation_losses.append(bce[none_known])
    object_mask = supervision[:, None].expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, object_target, reduction='none')
    p = logits.sigmoid()
    losses['lane_objectness'] = (_masked_mean((1 - p).square() * bce,
                                object_mask & object_target.bool(), zero)
                              + _masked_mean(p.square() * bce,
                                object_mask & ~object_target.bool(), zero))
    for key, values in (('lane_geometry', geometry_losses), ('lane_boundary', boundary_losses),
                        ('lane_adjacency', relation_losses)):
        if values:
            losses[key] = torch.cat(values).mean() + zero
    losses['lane_total'] = sum(cfg[key] * value for key, value in losses.items())
    return losses


def _duplicate(a, b, boundary_a, boundary_b, distance):
    if distance <= 0:
        return False
    ta, tb = a[0, -1] - a[0, 0], b[0, -1] - b[0, 0]
    norm = float(ta.norm() * tb.norm())
    if norm < 1e-6 or float(ta @ tb) / norm < .95:
        return False
    if float((a[0] - b[0]).norm(dim=-1).mean()) >= distance:
        return False
    # Sharing opposite boundaries is evidence of separate adjacent lanes, even
    # when their centerlines are closer than the configured dedup distance.
    shared_tolerance = min(.25, distance * .25)
    for ai, bi in ((1, 2), (2, 1)):
        if boundary_a[ai - 1] and boundary_b[bi - 1]:
            delta = min(float((a[ai] - b[bi]).norm(dim=-1).mean()),
                        float((a[ai] - b[bi].flip(0)).norm(dim=-1).mean()))
            if delta <= shared_tolerance:
                return False
    return True


def _lateral_side(source, target):
    tangent = source[-1] - source[0]
    target_tangent = target[-1] - target[0]
    length, target_length = float(tangent.norm()), float(target_tangent.norm())
    if min(length, target_length) < 1e-6:
        return None
    # Crossing or sequential road segments are not left/right neighbors.
    if abs(float(tangent @ target_tangent) / (length * target_length)) < .5:
        return None
    direction = tangent / length
    source_span, target_span = source @ direction, target @ direction
    overlap = min(float(source_span.max()), float(target_span.max())) - max(float(source_span.min()), float(target_span.min()))
    if overlap <= 1e-4:
        return None
    displacement = target.mean(0) - source.mean(0)
    cross = float(tangent[0] * displacement[1] - tangent[1] * displacement[0])
    if abs(cross) < 1e-6:
        return None
    return 0 if cross > 0 else 1


@torch.no_grad()
def decode_lanes(output, config):
    """Decode variable lane instances; topology IDs always refer to kept lanes.

    A relation is emitted only when both directional predictions agree with the
    geometric side. Greedy reciprocal assignment avoids contradictory lateral
    edges and dangling references. Low confidence relations remain unknown;
    learned explicit absence is reported separately as ``none``.
    """
    tensor = next((v for v in output.values() if torch.is_tensor(v) and v.ndim), None)
    batch_size = len(output['lane_logits']) if 'lane_logits' in output else (len(tensor) if tensor is not None else 1)
    base = dict(schema='radar_vla_lanes_v1', coordinate_frame='current_ego', distance_unit='m')
    unavailable = lambda: dict(base, available=False, lanes=[], num_lanes=0, candidate_count=0)
    if 'lane_logits' not in output:
        return [unavailable() for _ in range(batch_size)]
    for key in ('lane_confidence', 'lane_boundary_confidence', 'lane_adjacency_confidence'):
        value = getattr(config, key)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{key} must lie in [0,1]')
    if not math.isfinite(config.lane_nms_distance_m) or config.lane_nms_distance_m < 0:
        raise ValueError('lane_nms_distance_m must be finite and nonnegative')
    keys = ('lane_points', 'lane_logits', 'lane_boundary_logits', 'lane_adjacency_source',
            'lane_adjacency_target', 'lane_adjacency_none_logits')
    values = {key: output[key].detach().float().cpu() for key in keys}
    if any(not torch.isfinite(v).all() for v in values.values()):
        raise ValueError('Lane predictions must be finite')
    probabilities = values['lane_logits'].sigmoid()
    boundary_probabilities = values['lane_boundary_logits'].sigmoid()
    none_probabilities = values['lane_adjacency_none_logits'].sigmoid()
    available = output.get('lane_available', torch.ones(batch_size, dtype=torch.bool))

    def side_availability(key):
        value = output.get(key)
        if value is None:
            return torch.ones(batch_size, 2, dtype=torch.bool)
        value = value.detach().bool().cpu()
        if value.shape == (batch_size,):
            value = value[:, None].expand(-1, 2)
        if value.shape != (batch_size, 2):
            raise ValueError(f'{key} must have shape [B,2]')
        return value

    topology_available = side_availability('lane_topology_available')
    boundary_available = side_availability('lane_boundary_available')
    rows = []
    for b in range(batch_size):
        if not bool(available[b]):
            rows.append(unavailable())
            continue
        points, scores = values['lane_points'][b], probabilities[b]
        observed_boundary = ((boundary_probabilities[b] >= config.lane_boundary_confidence)
                             & boundary_available[b])
        candidates = (scores >= config.lane_confidence).nonzero().flatten().tolist()
        candidates.sort(key=lambda i: (-float(scores[i]), i))
        keep = []

        def duplicate_without_topology(i, j):
            if not _duplicate(points[i], points[j], observed_boundary[i], observed_boundary[j],
                              config.lane_nms_distance_m):
                return False
            side_i, side_j = _lateral_side(points[i, 0], points[j, 0]), _lateral_side(points[j, 0], points[i, 0])
            if (side_i is not None and side_j is not None
                    and topology_available[b, side_i] and topology_available[b, side_j]):
                # Evaluate just this close pair, never an all-cell relation matrix.
                source, target = values['lane_adjacency_source'][b], values['lane_adjacency_target'][b]
                scale = math.sqrt(target.shape[-1])
                relation_i = float(((source[i, side_i] @ target[j]) / scale).sigmoid())
                relation_j = float(((source[j, side_j] @ target[i]) / scale).sigmoid())
                if (min(relation_i, relation_j) > config.lane_adjacency_confidence
                        and relation_i > float(none_probabilities[b, i, side_i])
                        and relation_j > float(none_probabilities[b, j, side_j])):
                    return False
            return True

        for i in candidates:
            # Degenerate centerlines do not define a drivable lane or its sides.
            if float((points[i, 0, 1:] - points[i, 0, :-1]).norm(dim=-1).sum()) < 1e-4:
                continue
            if not any(duplicate_without_topology(i, j) for j in keep):
                keep.append(i)
        selected = torch.tensor(keep, dtype=torch.long)
        pair = _pair_logits(values['lane_adjacency_source'][b, selected],
                            values['lane_adjacency_target'][b, selected]).sigmoid()
        none = none_probabilities[b, selected]
        proposals = []
        for i in range(len(keep)):
            for j in range(i + 1, len(keep)):
                side_i = _lateral_side(points[keep[i], 0], points[keep[j], 0])
                side_j = _lateral_side(points[keep[j], 0], points[keep[i], 0])
                if (side_i is None or side_j is None
                        or not topology_available[b, side_i] or not topology_available[b, side_j]):
                    continue
                confidence = min(float(pair[i, j, side_i]), float(pair[j, i, side_j]))
                if (confidence > config.lane_adjacency_confidence
                        and confidence > float(none[i, side_i]) and confidence > float(none[j, side_j])):
                    proposals.append((confidence, i, side_i, j, side_j))
        edges = {}
        for confidence, i, side_i, j, side_j in sorted(proposals, reverse=True):
            if (i, side_i) not in edges and (j, side_j) not in edges:
                edges[i, side_i] = (j, confidence)
                edges[j, side_j] = (i, confidence)
        lanes = []
        for local_id, i in enumerate(keep):
            lane = dict(lane_id=local_id, confidence=float(scores[i]), centerline=points[i, 0].tolist())
            for side, name in enumerate(('left', 'right')):
                lane[name + '_boundary'] = points[i, side + 1].tolist() if observed_boundary[i, side] else None
                lane[name + '_boundary_confidence'] = (float(boundary_probabilities[b, i, side])
                                                       if boundary_available[b, side] else None)
                edge = edges.get((local_id, side))
                lane[name + '_neighbor_id'] = edge[0] if edge else None
                lane[name + '_neighbor_status'] = ('linked' if edge else 'none'
                    if topology_available[b, side] and float(none[local_id, side]) > config.lane_adjacency_confidence else 'unknown')
                lane[name + '_neighbor_confidence'] = (edge[1] if edge else float(none[local_id, side])
                                                       if topology_available[b, side] else None)
            lanes.append(lane)
        rows.append(dict(base, available=True, lanes=lanes, num_lanes=len(lanes), candidate_count=len(scores)))
    return rows
