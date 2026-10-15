"""Separate reward-only annotation horizons from the actor's native protocol.

Later prepared rows are evidence only when their ego-to-world poses permit an
explicit transform. We interpolate inside observed GT intervals, never beyond
them, and insert invalid samples across coverage gaps so downstream interpolation
cannot join two disconnected observed intervals into invented supervision.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .coordinates import validate_transform, transform_points, transform_yaw
from .evaluation import _interpolate


def _times(record):
    times = np.asarray(record.get('future_times_s'), dtype=float)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or times[0] <= 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('Reward annotation future times must be finite and increasing')
    return times


def _future(body, times, *, heading=0.):
    n = len(times)
    xy = np.asarray(body.get('future_xy', np.zeros((n, 2))), dtype=float)
    valid = np.asarray(body.get('future_valid', ['future_xy' in body] * n), dtype=bool)
    yaw = np.asarray(body.get('future_yaw', [heading] * n), dtype=float)
    if xy.shape != (n, 2) or valid.shape != (n,) or yaw.shape != (n,):
        raise ValueError('Reward annotation future arrays must align in shape with future_times_s')
    if not np.isfinite(xy).all() or not np.isfinite(yaw).all():
        raise ValueError('Reward annotation future arrays must be finite')
    return xy, yaw, valid


def _coverage(record, times):
    default = record.get('schema_version') != 'radar_vla_v2'
    result = np.asarray(record.get('tracking_coverage_valid', [default] * len(times)), dtype=bool)
    if result.shape != times.shape:
        raise ValueError('Tracking coverage shape must align with reward annotation times')
    return result


def _explicit(record, payload, max_horizon_s):
    if not isinstance(payload, dict):
        raise ValueError('response_evaluation must be a reward annotation object')
    original, times = _times(record), _times(payload)
    if len(times) < len(original) or not np.allclose(times[:len(original)], original, atol=1e-8, rtol=0):
        raise ValueError('Explicit reward annotations must preserve the actor future-time prefix')
    if payload.get('coordinate_frame', 'current_ego') != 'current_ego':
        raise ValueError('Explicit reward annotations must use the current ego frame')
    for key in ('scene_id', 'split', 'sample_id', 'timestamp_s'):
        if key in payload and payload[key] != record.get(key):
            raise ValueError('Explicit reward annotations must belong to the current observation')
    result = copy.deepcopy(record)
    keep = times <= max(max_horizon_s, original[-1]) + 1e-8
    result['future_times_s'] = times[keep].tolist()
    coverage = _coverage(payload, times)
    if not np.array_equal(coverage[:len(original)], _coverage(record, original)):
        raise ValueError('Explicit reward annotations must preserve original tracking coverage')
    result['tracking_coverage_valid'] = coverage[keep].tolist()

    def check_prefix(before, after):
        for field in ('future_xy', 'future_yaw', 'future_valid'):
            if field in before and (field not in after or not np.array_equal(
                    np.asarray(before[field]), np.asarray(after[field])[:len(original)])):
                raise ValueError('Explicit reward annotations must preserve original GT arrays')

    ego = {**record['ego'], **payload.get('ego', {})}
    xy, yaw, valid = _future(ego, times)
    check_prefix(record['ego'], ego)
    result['ego'].update(future_xy=xy[keep].tolist(), future_yaw=yaw[keep].tolist(), future_valid=valid[keep].tolist())
    agents = payload.get('risk_agents', payload.get('agents'))
    if not isinstance(agents, list):
        raise ValueError('Explicit reward annotations require an annotated agents list')
    existing = {str(a['id']): a for a in record.get('risk_agents', record.get('agents', []))}
    if not set(existing).issubset({str(a['id']) for a in agents}):
        raise ValueError('Explicit reward annotations cannot omit an original track')
    result['risk_agents'] = []
    for agent in agents:
        xy, yaw, valid = _future(agent, times, heading=agent['heading'])
        before = existing.get(str(agent['id']))
        if before is not None:
            check_prefix(before, agent)
        output = copy.deepcopy(before if before is not None else agent)
        if before is None:
            output['present_valid'] = False
        output.update(future_xy=xy[keep].tolist(), future_yaw=yaw[keep].tolist(), future_valid=valid[keep].tolist())
        result['risk_agents'].append(output)
    current = {str(a['id']) for a in record.get('agents', [])}
    result['agents'] = [copy.deepcopy(a) for a in result['risk_agents'] if str(a['id']) in current]
    result['response_annotation_provenance'] = dict(source='explicit_response_evaluation',
        actor_horizon_s=float(original[-1]), reward_horizon_s=float(result['future_times_s'][-1]))
    return result


def _source(row, origin_inverse, observation_time):
    times = _times(row)
    transform = origin_inverse @ validate_transform(row['ego']['pose'], 'reward source ego pose')
    offset = float(row['timestamp_s']) - observation_time
    knots = np.r_[offset, offset + times]
    coverage = np.r_[row.get('agent_supervision_available', row.get('agent_supervision', True)),
                     _coverage(row, times)]
    bodies = {}
    all_bodies = [(None, row['ego'])] + [(str(a['id']), a) for a in row.get('risk_agents', row.get('agents', []))]
    for identifier, body in all_bodies:
        heading = 0. if identifier is None else float(body['heading'])
        xy, yaw, valid = _future(body, times, heading=heading)
        position = [0., 0.] if identifier is None else body['position']
        present = True if identifier is None else body.get('present_valid', True)
        points = transform_points(np.vstack((position, xy)), transform)[:, :2]
        headings = transform_yaw(np.r_[heading, yaw], transform)
        bodies[identifier] = dict(points=points, yaw=np.unwrap(headings),
                                  valid=np.r_[present, valid], original=body)
    return dict(knots=knots, coverage=coverage, bodies=bodies, transform=transform,
                sample_id=row.get('sample_id'), offset=offset)


def _sample(sources, moment, identifier='coverage'):
    """A value is known only inside an individual observed source interval."""
    for source in reversed(sources):
        if identifier == 'coverage':
            _, mask = _interpolate(source['knots'], np.zeros(len(source['knots'])),
                                   source['coverage'], [moment])
            if mask[0]:
                return None, None, True
        elif identifier in source['bodies']:
            body = source['bodies'][identifier]
            point, mask = _interpolate(source['knots'], body['points'], body['valid'], [moment])
            if mask[0]:
                yaw, _ = _interpolate(source['knots'], body['yaw'], body['valid'], [moment])
                return point[0], float(yaw[0]), True
    return np.zeros(2), 0., False


def reward_record(record, scene_records, max_horizon_s=10.):
    """Deep-copy and extend only reward GT, within this scene and split.

    Without a pose, the original coverage is retained. ``response_evaluation``
    can explicitly supply a longer current-frame GT record in that case. Missing
    entrants' past positions remain invalid; absence is never encoded as a fake
    distant position. All extensions are marked with their source observations.
    """
    if not math.isfinite(max_horizon_s) or max_horizon_s <= 0:
        raise ValueError('Reward annotation horizon must be finite and positive')
    if 'response_evaluation' in record:
        return _explicit(record, record['response_evaluation'], max_horizon_s)
    result = copy.deepcopy(record)
    native = _times(record)
    provenance = dict(source='native_annotations', actor_horizon_s=float(native[-1]),
                      reward_horizon_s=float(native[-1]), source_sample_ids=[record.get('sample_id')])
    result['response_annotation_provenance'] = provenance
    if record['ego'].get('pose') is None or max_horizon_s <= native[-1]:
        return result
    inverse = np.linalg.inv(validate_transform(record['ego']['pose'], 'reward origin ego pose'))
    stamp = float(record['timestamp_s'])
    rows = [record] + [row for row in scene_records
        if row.get('scene_id') == record.get('scene_id') and row.get('split') == record.get('split')
        and stamp < float(row['timestamp_s']) <= stamp + max_horizon_s
        and row.get('ego', {}).get('pose') is not None]
    rows.sort(key=lambda r: r['timestamp_s'])
    sources = [_source(row, inverse, stamp) for row in rows]
    extra = sorted({float(t) for source in sources for t in source['knots']
                    if native[-1] + 1e-8 < t <= max_horizon_s + 1e-8})
    if not extra:
        return result
    identifiers = set().union(*(source['bodies'].keys() for source in sources))
    # Source knots partition every validity interval. A midpoint in a gap is
    # explicitly invalid, preventing endpoint interpolation from hiding it.
    boundaries = [float(native[-1])] + extra
    gap_points = []
    for left, right in zip(boundaries, boundaries[1:]):
        midpoint = (left + right) / 2
        if (not _sample(sources, midpoint)[2]
                or any(not _sample(sources, midpoint, identifier)[2] for identifier in identifiers)):
            gap_points.append(midpoint)
    extra = sorted(set(extra + gap_points))
    result['future_times_s'] = native.tolist() + extra
    result['tracking_coverage_valid'] = _coverage(record, native).tolist() + [_sample(sources, t)[2] for t in extra]
    original_bodies = {str(a['id']): a for a in record.get('risk_agents', record.get('agents', []))}

    def extend(identifier):
        original = record['ego'] if identifier is None else original_bodies.get(identifier)
        if original is not None:
            output = copy.deepcopy(original)
            xy, yaw, valid = _future(original, native, heading=original.get('heading', 0.))
        else:
            source = next(source for source in sources if identifier in source['bodies'])
            entry = source['bodies'][identifier]
            output = copy.deepcopy(entry['original'])
            output.update(position=entry['points'][0].tolist(), heading=float(entry['yaw'][0]),
                          velocity=(np.asarray(output['velocity']) @ source['transform'][:2, :2].T).tolist(),
                          present_valid=False)
            observed = [_sample(sources, t, identifier) for t in native]
            xy = np.asarray([value[0] for value in observed])
            yaw = np.asarray([value[1] for value in observed])
            valid = np.asarray([value[2] for value in observed], dtype=bool)
        values = [_sample(sources, t, identifier) for t in extra]
        output.update(future_xy=xy.tolist() + [v[0].tolist() for v in values],
                      future_yaw=yaw.tolist() + [v[1] for v in values],
                      future_valid=valid.tolist() + [v[2] for v in values])
        return output

    result['ego'] = extend(None)
    ordered_ids = list(original_bodies) + sorted(identifiers - {None} - set(original_bodies))
    result['risk_agents'] = [extend(identifier) for identifier in ordered_ids]
    current = {str(a['id']) for a in record.get('agents', [])}
    result['agents'] = [copy.deepcopy(a) for a in result['risk_agents'] if str(a['id']) in current]
    provenance.update(source='stitched_prepared_gt', reward_horizon_s=float(extra[-1]),
                      source_sample_ids=list(dict.fromkeys(source['sample_id'] for source in sources)))
    return result
